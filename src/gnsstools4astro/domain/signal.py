"""Estimativas de sinal transmitido: C/N0 → potência → PSD."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .satellite import Constellation

# Constantes físicas
BOLTZMANN = 1.380649e-23  # constante de Boltzmann (J/K)
T_NOISE_K = 290.0  # temperatura de ruído de referência (K)

# Largura de banda de correlação nominal dos receptores GNSS (Hz)
DEFAULT_BANDWIDTH_HZ = 2.0e6


class SignalSample(BaseModel):
    """Amostra de sinal (C/N0) para um satélite numa época."""

    time: datetime
    sv: str
    cn0_dbhz: float | None = Field(None, description="C/N0 em dB-Hz")

    @property
    def valid(self) -> bool:
        """True se há C/N0 medido."""
        return self.cn0_dbhz is not None


class FrequencyBand(BaseModel):
    """Banda de frequência de uma constelação GNSS."""

    name: str
    frequency_hz: float
    modulation: str | None = None


class ConstellationSignals(BaseModel):
    """Dados de referência de uma constelação (frequências + EIRP)."""

    constellation: Constellation
    eirp_dbw: float = Field(..., description="EIRP nominal (dBW)")
    rhcp: bool = Field(True, description="Polarização RHCP (True) / LHCP (False)")
    signals: list[FrequencyBand] = Field(default_factory=list)


class SignalEstimate(BaseModel):
    """Estimativa de sinal transmitido para uma amostra C/N0."""

    time: datetime
    sv: str
    constellation: Constellation
    cn0_dbhz: float
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    received_power_w: float = Field(..., description="P_rec estimada (W)")
    psd_w_hz: float = Field(..., description="PSD estimada (W/Hz)")
    eirp_dbw: float | None = Field(None, description="EIRP nominal de referência")
    frequency_mhz: float | None = Field(None, description="Frequência da banda associada")


def received_power_watts(cn0_dbhz: float, t_noise_k: float = T_NOISE_K) -> float:
    r"""Potência recebida estimada a partir do C/N0.

    .. math::

        P_\text{rec} = k\,T\,10^{(C/N_0)/10}

    Parameters
    ----------
    cn0_dbhz : float
        Relação portadora-ruído em dB-Hz.
    t_noise_k : float
        Temperatura de ruído de referência (K).

    Returns
    -------
    float
        Potência recebida em watts.
    """
    n0 = BOLTZMANN * t_noise_k  # W/Hz
    return n0 * (10.0 ** (cn0_dbhz / 10.0))


def power_spectral_density(
    cn0_dbhz: float,
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
    t_noise_k: float = T_NOISE_K,
) -> float:
    r"""Densidade espectral de potência estimada (W/Hz).

    .. math::

        \text{PSD} = \frac{k\,T\,10^{(C/N_0)/10}}{B}

    Parameters
    ----------
    cn0_dbhz : float
        Relação portadora-ruído em dB-Hz.
    bandwidth_hz : float
        Largura de banda de correlação (Hz).
    t_noise_k : float
        Temperatura de ruído de referência (K).
    """
    return received_power_watts(cn0_dbhz, t_noise_k) / bandwidth_hz
