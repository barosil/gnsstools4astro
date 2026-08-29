"""Propagadores de órbita a partir de efemérides de transmissão (RINEX NAV).

* GPS/GALILEO/BEIDOU: algoritmo do Apêndice II do ICD-GPS-200 (elementos
  Kepler), vetorizado em numpy;
* GLONASS: propagação cinemática (posição/velocidade/aceleração) com
  correção da rotação da Terra (aproximação do ICD-GLONASS).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
from gnsstools4astro.domain.satellite import BroadcastEphemeris, GlonassEphemeris

# Constantes físicas (ICD-GPS-200)
MU = 3.986005e14  # constante gravitacional da Terra (m³/s²)
OMEGA_E_DOT = 7.2921151467e-5  # taxa de rotação da Terra (rad/s)
SPEED_OF_LIGHT = 2.99792458e8  # m/s
F_REL = -4.442807633e-10  # s/sqrt(m) - correção relativística

_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=UTC)
_WEEK_S = 7 * 86400.0

# GLONASS (ICD): raio equatorial e taxa de rotação (para a correção).
_J2 = 1.0826257e-3
_GLONASS_EARTH_RADIUS_M = 6378136.0
_GLONASS_MU = 3.9860044e14


def _gps_seconds(dt: datetime) -> float:
    """Segundos desde a época GPS (1980-01-06 UTC)."""
    return (dt - _GPS_EPOCH).total_seconds()


def _solve_kepler(m: np.ndarray, e: float) -> np.ndarray:
    """Resolve E - e*sin(E) = M por iteração de Newton (vetorizado)."""
    e_arr = e
    ek = m.copy()
    for _ in range(10):
        f = ek - e_arr * np.sin(ek) - m
        fp = 1.0 - e_arr * np.cos(ek)
        ek = ek - f / fp
    return ek


def clock_correction(eph: BroadcastEphemeris, t: datetime) -> float:
    """Correção de relógio do satélite (s) no instante ``t`` (UTC)."""
    dt = (t - eph.toc).total_seconds()
    return eph.af0 + eph.af1 * dt + eph.af2 * dt * dt


def propagate_kepler(eph: BroadcastEphemeris, times) -> np.ndarray:
    """Posição ECEF (m) nos instantes dados (ICD-GPS-200).

    Parameters
    ----------
    eph : BroadcastEphemeris
        Efemérides de transmissão (GPS/GALILEO/BEIDOU).
    times : datetime or sequence[datetime]
        Instantes de observação (UTC).

    Returns
    -------
    ndarray
        Array (N, 3) com posições ECEF em metros.
    """
    single = isinstance(times, datetime)
    t_list = [times] if single else list(times)
    t_arr = np.array([_gps_seconds(t) for t in t_list])

    # Correção de relógio e tempo de transmissão (t - dt_sv).
    toc = _gps_seconds(eph.toc)
    toe = _gps_seconds(eph.toe)
    toe_sow = toe % _WEEK_S  # segundos de semana (termo da rotação da Terra)

    dt_sv = (
        eph.af0
        + eph.af1 * (t_arr - toc)
        + eph.af2 * (t_arr - toc) ** 2
    )
    # Correção relativística: F · e · sqrt(a) · sin(Ek)
    a = eph.sqrt_a**2
    n0 = math.sqrt(MU / a**3)
    n = n0 + eph.delta_n
    mk = eph.m0 + n * (t_arr - dt_sv - toe)
    ek = _solve_kepler(mk, eph.e)
    dt_sv = dt_sv + F_REL * eph.e * eph.sqrt_a * np.sin(ek)
    t_trans = t_arr - dt_sv

    tk = t_trans - toe
    # Mantém t_k dentro de ±meia-semana (rollover de semana).
    tk = np.where(tk > _WEEK_S / 2, tk - _WEEK_S, tk)
    tk = np.where(tk < -_WEEK_S / 2, tk + _WEEK_S, tk)

    # Anomalia verdadeira e argumento de latitude.
    nu = np.arctan2(
        np.sqrt(1.0 - eph.e**2) * np.sin(ek),
        np.cos(ek) - eph.e,
    )
    phi = nu + eph.omega

    # Correções harmônicas.
    cu = eph.cuc * np.cos(2 * phi) + eph.cus * np.sin(2 * phi)
    cr = eph.crc * np.cos(2 * phi) + eph.crs * np.sin(2 * phi)
    ci = eph.cic * np.cos(2 * phi) + eph.cis * np.sin(2 * phi)

    u = phi + cu
    r = a * (1.0 - eph.e * np.cos(ek)) + cr
    i = eph.i0 + eph.idot * tk + ci

    # Posição no plano orbital.
    xp = r * np.cos(u)
    yp = r * np.sin(u)

    # Longitude do nodo ascendente corrigida.
    omega = eph.omega0 + (eph.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * toe_sow

    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_i = np.cos(i)
    sin_i = np.sin(i)

    x = xp * cos_omega - yp * cos_i * sin_omega
    y = xp * sin_omega + yp * cos_i * cos_omega
    z = yp * sin_i

    out = np.column_stack((x, y, z))
    return out[0] if single else out


def propagate_glonass(eph: GlonassEphemeris, times) -> np.ndarray:
    r"""Posição ECEF (m) nos instantes dados (integração RK4, ICD-GLONASS).

    O estado (posição/velocidade/aceleração) é fornecido em m, m/s e m/s²
    (unidades do georinex, referenciadas ao instante ``tb`` no referencial
    terrestre PZ-90).

    O algoritmo converte o estado para o referencial inercial (adicionando a
    rotação da Terra :math:`\vec\Omega x \vec r` à velocidade), integra
    numericamente com **Runge-Kutta de 4ª ordem** incluindo o campo
    gravitacional central + perturbação J2 (mais a aceleração de perturbação
    transmitida ``a0``) e retorna ao ECEF no instante pedido.

    O modelo linear de primeira ordem (usado antes) divergia ~1.600 km em
    30 min porque desprezava :math:`\vec\Omega x \vec r` e a curvatura
    orbital.
    """
    single = isinstance(times, datetime)
    t_list = [times] if single else list(times)
    tau = np.array([(t - eph.tb).total_seconds() for t in t_list], dtype=float)

    r0 = np.array([eph.x, eph.y, eph.z])  # m
    v0 = np.array([eph.vx, eph.vy, eph.vz])  # m/s (ECEF em tb)
    a0 = np.array([eph.ax, eph.ay, eph.az])  # m/s² (perturbação transmitida)

    # Estado inercial em tb: a velocidade ECEF ganha o termo Omega x r.
    v_inertial = v0 + np.array([-OMEGA_E_DOT * r0[1], OMEGA_E_DOT * r0[0], 0.0])
    state0 = np.concatenate([r0, v_inertial])

    positions = np.empty((len(tau), 3))
    for i, dt in enumerate(tau):
        if abs(dt) < 1e-6:
            positions[i] = r0
            continue
        n = max(1, round(abs(dt) / 30.0))
        h = dt / n
        state = state0.copy()
        for _ in range(n):
            k1 = _glonass_dynamics(state, a0)
            k2 = _glonass_dynamics(state + 0.5 * h * k1, a0)
            k3 = _glonass_dynamics(state + 0.5 * h * k2, a0)
            k4 = _glonass_dynamics(state + h * k3, a0)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        positions[i] = _inertial_to_ecef(state, dt)[:3]

    out = positions
    return out[0] if single else out


def _glonass_dynamics(state: np.ndarray, a0: np.ndarray) -> np.ndarray:
    """Derivada de estado [dr/dt, dv/dt] no referencial inercial (gravidade + J2)."""
    r = state[:3]
    v = state[3:]
    s = float(np.sqrt(np.dot(r, r)))
    f = 1.5 * _J2 * (_GLONASS_EARTH_RADIUS_M / s) ** 2
    q = 5.0 * (r[2] / s) ** 2
    ax = -_GLONASS_MU * r[0] / s**3 * (1.0 + f * (1.0 - q))
    ay = -_GLONASS_MU * r[1] / s**3 * (1.0 + f * (1.0 - q))
    az = -_GLONASS_MU * r[2] / s**3 * (1.0 + f * (3.0 - q))
    return np.concatenate([v, np.array([ax, ay, az]) + a0])


def _inertial_to_ecef(state: np.ndarray, tau: float) -> np.ndarray:
    """Converte estado inercial (em tb) para ECEF no instante tb + tau.

    O ECEF em t está rotacionado de +omega_e·tau em relação ao inercial em tb,
    logo a conversão usa Rz(-omega_e·tau); a velocidade recebe a correção
    -Omega x r.
    """
    r = state[:3]
    v = state[3:]
    theta = -OMEGA_E_DOT * tau
    ct = float(np.cos(theta))
    st = float(np.sin(theta))

    rx = r[0] * ct - r[1] * st
    ry = r[0] * st + r[1] * ct
    rz = r[2]

    vx = v[0] * ct - v[1] * st + OMEGA_E_DOT * ry
    vy = v[0] * st + v[1] * ct - OMEGA_E_DOT * rx
    vz = v[2]
    return np.array([rx, ry, rz, vx, vy, vz])


def propagate(eph: BroadcastEphemeris | GlonassEphemeris, times) -> np.ndarray:
    """Propaga uma efeméride (Kepler ou GLONASS) para os instantes dados."""
    if isinstance(eph, GlonassEphemeris):
        return propagate_glonass(eph, times)
    return propagate_kepler(eph, times)
