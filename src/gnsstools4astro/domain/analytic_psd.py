"""PSDs analíticas (closed-form) de modulações GNSS — código compartilhado.

Este módulo é a **fonte única** das PSDs espectrais usadas tanto pelos
satélites TLE (referência ``signals.json``) quanto pelos satélites RINEX
(dados extraídos do arquivo + link budget). Substitui a simulação no domínio
do tempo (PRN + Welch) por expressões fechadas:

* determinístico e suave (sem sequência pseudo-aleatória);
* sem **piso de ruído alto em banda larga** (o PSD analítico decai como
  ``sinc²`` longe da portadora — não há "chão" de Welch);
* normalizado (integral unitária) — a amplitude em W/Hz é obtida pela
  potência total (EIRP) via :func:`psd_w_per_hz`.

Referência: Kaplan & Hegarty, *Understanding GPS/GNSS: Principles and
Applications*, Apêndice — espectros de densidade de potência.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

# Chip rate de referência GPS (1.023 Mcps)
CHIP_RATE_HZ = 1.023e6

# Piso numérico (dB) para evitar -inf em representações logarítmicas.
LOW_FLOOR_DB = -300.0

# Filtro de transmissão do satélite: banda ~24 MHz (GPS ICD; cobre o lóbulo
# principal do BPSK(10) e o CBOC), queda Butterworth e piso fora de banda
# espalhado numa banda de referência de 370 MHz (a cauda sinc² não existe
# após o filtro/amplificador).
TX_FILTER_BW_HZ = 24.0e6
TX_FILTER_ORDER = 16
OOB_POWER_DBC = -60.0
OOB_REFERENCE_BW_HZ = 370e6


def _sinc(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """sinc normalizado ``sin(πx)/(πx)`` (``np.sinc``)."""
    return np.sinc(x)


def bpsk_r_psd(
    freq_hz: NDArray[np.floating],
    chip_rate_hz: float,
) -> NDArray[np.floating]:
    r"""PSD normalizada (:math:`\int G\,\mathrm{d}f = 1`) de BPSK-R(n).

    .. math::

        G(f) = \frac{1}{R_c}\,\mathrm{sinc}^2\left(\frac{f}{R_c}\right)

    Parameters
    ----------
    freq_hz : array
        Frequências (Hz), relativas à portadora (f - f_c).
    chip_rate_hz : float
        Taxa de chips ``R_c = n \cdot 1.023`` MHz.

    Returns
    -------
    ndarray
        PSD normalizada (1/Hz).
    """
    x = np.asarray(freq_hz, dtype=float) / chip_rate_hz
    return _sinc(x) ** 2 / chip_rate_hz


def boc_sin_psd(
    freq_hz: NDArray[np.floating],
    chip_rate_hz: float,
    subcarrier_hz: float,
) -> NDArray[np.floating]:
    r"""PSD normalizada de BOC(m,n) sine-phased (integral unitária).

    .. math::

        G(f) = R_c
        \left[\frac{\sin(\pi f/R_c)\,\sin(\pi f/(2 R_s))}
        {\pi f\,\cos(\pi f/(2 R_s))}\right]^2,
        \qquad R_s = m \cdot 1.023\,\mathrm{MHz}

    BOC tem **nulo espectral na portadora** (``G(0)=0``); os pontos de
    ambiguidade ``0/0`` (que coincidem com nulos do numerador) são tratados
    numericamente (nan/zero).

    Parameters
    ----------
    freq_hz : array
        Frequências (Hz), relativas à portadora (f - f_c).
    chip_rate_hz : float
        Taxa de chips ``R_c = n \cdot 1.023`` MHz.
    subcarrier_hz : float
        Frequência da subportadora ``R_s = m \cdot 1.023`` MHz.

    Returns
    -------
    ndarray
        PSD normalizada (1/Hz).
    """
    f = np.asarray(freq_hz, dtype=float)
    x = np.pi * f / chip_rate_hz
    y = np.pi * f / (2.0 * subcarrier_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(x) * np.sin(y)
        den = np.pi * f * np.cos(y)
        psd = np.where(f == 0.0, 0.0, chip_rate_hz * (num / den) ** 2)
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
    return psd


def cboc_psd(
    freq_hz: NDArray[np.floating],
    chip_rate_hz: float,
    subcarriers_hz: list[float],
    weights: list[float],
) -> NDArray[np.floating]:
    r"""PSD normalizada de CBOC = combinação ponderada de BOC.

    .. math::

        G(f) = \sum_i w_i\, G_{\mathrm{BOC}(m_i, n)}(f), \qquad \sum_i w_i = 1

    Parameters
    ----------
    freq_hz : array
        Frequências (Hz), relativas à portadora.
    chip_rate_hz : float
        Taxa de chips ``R_c = n \cdot 1.023`` MHz (comum aos BOC).
    subcarriers_hz : list[float]
        Frequências de subportadora de cada BOC.
    weights : list[float]
        Pesos (somam 1).

    Returns
    -------
    ndarray
        PSD normalizada (1/Hz).
    """
    total = np.zeros_like(np.asarray(freq_hz, dtype=float))
    for w, sub in zip(weights, subcarriers_hz, strict=True):
        total = total + w * boc_sin_psd(freq_hz, chip_rate_hz, sub)
    return total


def parse_modulation(modulation: str) -> dict:
    """Analisa a string de modulação (ex.: ``'BPSK(10)'``, ``'BOC(1,1)'``).

    Returns
    -------
    dict
        Chaves ``kind`` e os parâmetros específicos (``n``/``m``, ou
        ``chip_rate``/``subcarriers``/``weights`` para CBOC).
    """
    m = modulation.strip()
    if m.startswith("BPSK"):
        n = float(m.split("(")[1].split(")")[0])
        return {"kind": "bpsk", "n": n}
    if m.startswith("QPSK"):
        # QPSK-R tem o mesmo formato espectral do BPSK-R (I+Q em quadratura).
        n = float(m.split("(")[1].split(")")[0])
        return {"kind": "qpsk", "n": n}
    if m.startswith("BOC"):
        parts = m.split("(")[1].split(")")[0].split(",")
        return {"kind": "boc", "m": float(parts[0]), "n": float(parts[1])}
    if m.startswith("CBOC"):
        # CBOC(m_high, n, w_high) → w_high·BOC(m_high,n) + (1-w_high)·BOC(1,n)
        parts = m.split("(")[1].split(")")[0].split(",")
        m_high = float(parts[0])
        n = float(parts[1])
        w_high = float(Fraction(parts[2])) if len(parts) > 2 else 1.0 / 11.0
        return {
            "kind": "cboc",
            "chip_rate": n * CHIP_RATE_HZ,
            "subcarriers": [CHIP_RATE_HZ, m_high * CHIP_RATE_HZ],
            "weights": [1.0 - w_high, w_high],
        }
    msg = f"Modulação {modulation} não suportada"
    raise NotImplementedError(msg)


def analytic_psd(
    modulation: str,
    freq_hz: NDArray[np.floating],
    fc_hz: float,
) -> NDArray[np.floating]:
    """PSD analítica normalizada (1/Hz, :math:`\\int G\\,\\mathrm{d}f = 1`).

    Avalia a forma espectral closed-form da modulação no grid ``freq_hz``,
    centrada na portadora ``fc_hz``. Usada pelo mesmo código para TLE e RINEX.

    Parameters
    ----------
    modulation : str
        Ex.: ``'BPSK(1)'``, ``'BPSK(10)'``, ``'QPSK(0.5)'``, ``'BOC(1,1)'``,
        ``'CBOC(6,1,1/11)'``.
    freq_hz : array
        Grid de frequências absolutas (Hz).
    fc_hz : float
        Frequência da portadora (Hz).

    Returns
    -------
    ndarray
        PSD normalizada (1/Hz) avaliada em ``freq_hz``.
    """
    f = np.asarray(freq_hz, dtype=float) - float(fc_hz)
    p = parse_modulation(modulation)
    if p["kind"] in ("bpsk", "qpsk"):
        g = bpsk_r_psd(f, p["n"] * CHIP_RATE_HZ)
    elif p["kind"] == "boc":
        g = boc_sin_psd(f, p["n"] * CHIP_RATE_HZ, p["m"] * CHIP_RATE_HZ)
    elif p["kind"] == "cboc":
        g = cboc_psd(f, p["chip_rate"], p["subcarriers"], p["weights"])
    else:
        msg = f"Modulação {modulation} não suportada"
        raise NotImplementedError(msg)
    return apply_tx_filter(g, f)


def apply_tx_filter(
    psd: NDArray[np.floating],
    offset_hz: NDArray[np.floating],
    *,
    bandwidth_hz: float = TX_FILTER_BW_HZ,
    order: int = TX_FILTER_ORDER,
    oob_dbc: float = OOB_POWER_DBC,
    reference_bw_hz: float = OOB_REFERENCE_BW_HZ,
) -> NDArray[np.floating]:
    """Aplica o filtro de transmissão e o piso fora de banda à PSD normalizada.

    ``G_out = G · |H(f)|² + 10^(oob_dbc/10) / reference_bw_hz``, com
    ``|H|² = 1/(1+(2|f|/BW)^(2·order))`` (Butterworth passa-baixas em banda base).
    O piso é constante (independente do grid): potência fora de banda de
    ``oob_dbc`` espalhada uniformemente em ``reference_bw_hz``.
    """
    x = 2.0 * np.abs(np.asarray(offset_hz, dtype=float)) / bandwidth_hz
    with np.errstate(over="ignore"):
        h2 = 1.0 / (1.0 + x ** (2 * order))
    floor = 10.0 ** (oob_dbc / 10.0) / reference_bw_hz
    return np.asarray(psd, dtype=float) * h2 + floor


def psd_w_per_hz(
    psd_normalized: NDArray[np.floating],
    power_w: float,
) -> NDArray[np.floating]:
    """PSD em W/Hz a partir da forma normalizada e da potência total (W)."""
    return power_w * np.asarray(psd_normalized, dtype=float)


def psd_db(
    psd_w_hz: NDArray[np.floating],
    floor_db: float = LOW_FLOOR_DB,
) -> NDArray[np.floating]:
    """Converte PSD (W/Hz) para dBW/Hz com piso numérico (evita -inf)."""
    p = np.asarray(psd_w_hz, dtype=float)
    with np.errstate(divide="ignore"):
        out = 10.0 * np.log10(p)
    return np.maximum(out, floor_db).astype(np.float64)
