"""Integração Skyfield: converte posições ECEF em coordenadas de céu.

As efemérides de transmissão (broadcast) não são lidas nativamente pelo
Skyfield; portanto calculamos a posição ECEF com o propagador próprio e a
embrulhamos num objeto Geocentric do Skyfield para obter alt/az/RA/Dec
topocêntrico relativo a uma estação.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from gnsstools4astro.domain.station import GNSSStation
from skyfield.api import load, wgs84
from skyfield.framelib import itrs
from skyfield.positionlib import Geocentric

_AU = 149597870700.0  # metros por unidade astronômica
_SECONDS_PER_DAY = 86400.0


@dataclass(frozen=True)
class SkyPosition:
    """Posição topocêntrica de um satélite num instante."""

    time: datetime
    alt_deg: float
    az_deg: float
    ra_deg: float
    dec_deg: float
    distance_km: float


class SkyfieldTracks:
    """Constrói coordenadas de céu a partir de um propagador ECEF."""

    def __init__(self, station: GNSSStation) -> None:
        self._ts = load.timescale()
        self._station = wgs84.latlon(
            station.lat_deg,
            station.lon_deg,
            station.height_m,
        )

    def ecef_to_sky(
        self,
        ecef: Callable[[datetime], np.ndarray],
        t: datetime,
        dt_s: float = 0.5,
    ) -> SkyPosition:
        """Converte uma posição ECEF (m) no instante ``t`` para o céu.

        Parameters
        ----------
        ecef : Callable[[datetime], ndarray]
            Função que retorna a posição ECEF (m) num instante (propagador).
        t : datetime
            Instante de observação (UTC).
        dt_s : float
            Passo (s) para a derivada por diferenças finitas.

        Returns
        -------
        SkyPosition
            Alt/az (graus), RA/Dec (graus, ICRS) e distância topocêntrica (km).
        """
        p0 = np.asarray(ecef(t - timedelta(seconds=dt_s)), dtype=float)
        p1 = np.asarray(ecef(t + timedelta(seconds=dt_s)), dtype=float)
        pos_m = np.asarray(ecef(t), dtype=float)
        vel_m_s = (p1 - p0) / (2.0 * dt_s)

        sky_time = self._ts.from_datetime(t)

        # Posição topocêntrica (satélite - estação) em **ECEF/ITRS**: a estação
        # deve ser tomada no mesmo referencial do satélite (``frame_xyz(itrs)``);
        # usar ``.xyz`` (ICRF) aqui misturava referenciais e corrompia alt/az.
        station_ecef = np.asarray(self._station.at(sky_time).frame_xyz(itrs).m)
        topo_m = pos_m - station_ecef

        # O Geocentric do Skyfield espera o vetor em ICRF: converte o vetor
        # topocêntrico (e a velocidade) de ITRS para ICRF.
        rot = itrs.rotation_at(sky_time)  # ICRF → ITRS
        topo_icrf = rot.T @ (topo_m / _AU)
        vel_icrf = rot.T @ (vel_m_s * _SECONDS_PER_DAY / _AU)

        geo = Geocentric(
            topo_icrf,
            vel_icrf,
            sky_time,
            center=self._station,
            target=self._station.target,
        )

        alt, az, distance = geo.altaz()
        ra, dec, _ = geo.radec()

        return SkyPosition(
            time=t,
            alt_deg=float(alt.degrees),
            az_deg=float(az.degrees),
            ra_deg=float(ra.hours * 15.0),
            dec_deg=float(dec.degrees),
            distance_km=float(distance.km),
        )
