"""Orçamento de enlace: perda de percurso em espaço livre (FSPL) e EIRP.

Modelo clássico de link budget para a comparação entre a potência recebida
(estimada do C/N0) e o EIRP nominal de referência:

.. math::

    \\text{FSPL (dB)} = 20\\log_{10}(d_{\\text{km}}) + 20\\log_{10}(f_{\\text{MHz}}) + 32.45

    \\text{EIRP}_{\\text{est}} (\\text{dBW}) = P_{\\text{rec}} (\\text{dBW}) + \\text{FSPL}

onde ``d`` é a distância de enlace (slant range) e ``f`` a frequência da banda
medida. A expressão assume ganho unitário no receptor (medição isotrópica).
"""

from __future__ import annotations

from math import log10

from .satellite import Constellation

# Frequência de referência (MHz) da banda medida por constelação.
# Corresponde ao sinal cujo C/N0 é registrado na observação:
#   GPS/Galileo → L1/E1 (S1C); Beidou → B1I (S2I); GLONASS → L1 (S1C).
REFERENCE_FREQUENCY_MHZ: dict[Constellation, float] = {
    Constellation.GPS: 1575.42,
    Constellation.GALILEO: 1575.42,
    Constellation.BEIDOU: 1561.098,
    Constellation.GLONASS: 1602.0,
}


def free_space_path_loss_db(distance_km: float, frequency_mhz: float) -> float:
    r"""Perda de percurso em espaço livre (dB).

    .. math::

        \\text{FSPL} = 20\\log_{10}(d) + 20\\log_{10}(f) + 32.45

    Parameters
    ----------
    distance_km : float
        Distância de enlace (slant range) em km.
    frequency_mhz : float
        Frequência em MHz.

    Returns
    -------
    float
        Perda em dB (valor positivo).
    """
    return 20.0 * log10(distance_km) + 20.0 * log10(frequency_mhz) + 32.45


def received_power_dbw(power_w: float) -> float:
    """Converte potência (W) para dBW."""
    return 10.0 * log10(power_w)


def eirp_estimate_dbw(
    received_power_w: float,
    distance_km: float,
    frequency_mhz: float,
) -> float:
    r"""EIRP estimado a partir da potência recebida e da FSPL.

    .. math::

        \\text{EIRP}_{\\text{est}} = 10\\log_{10}(P_{\\text{rec}}) + \\text{FSPL}

    Parameters
    ----------
    received_power_w : float
        Potência recebida estimada (W) — ver ``received_power_watts``.
    distance_km : float
        Distância de enlace em km.
    frequency_mhz : float
        Frequência em MHz.

    Returns
    -------
    float
        EIRP estimado em dBW.
    """
    return received_power_dbw(received_power_w) + free_space_path_loss_db(
        distance_km, frequency_mhz
    )


def reference_frequency_mhz(constellation: Constellation) -> float:
    """Frequência (MHz) da banda medida para uma constelação."""
    return REFERENCE_FREQUENCY_MHZ[constellation]
