"""Satélites, efemérides de transmissão e tracks."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class Constellation(str, Enum):
    """Constelações GNSS suportadas."""

    GPS = "GPS"
    GALILEO = "GALILEO"
    GLONASS = "GLONASS"
    BEIDOU = "BEIDOU"


_SV_PREFIX = {
    "G": Constellation.GPS,
    "E": Constellation.GALILEO,
    "R": Constellation.GLONASS,
    "C": Constellation.BEIDOU,
}


def constellation_from_sv(sv: str) -> Constellation:
    """Mapeia o prefixo do SV (ex.: ``G01``) para a constelação."""
    prefix = sv[0].upper()
    try:
        return _SV_PREFIX[prefix]
    except KeyError:
        msg = f"prefixo de SV desconhecido: {sv!r}"
        raise ValueError(msg) from None


class Satellite(BaseModel):
    """Um satélite GNSS identificado por SV/PRN."""

    sv: str = Field(..., description="Identificador do satélite, ex.: 'G01'")
    constellation: Constellation

    @field_validator("sv")
    @classmethod
    def _normalize_sv(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) < 2 or not v[1:].isdigit():
            msg = f"SV inválido: {v!r}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> Satellite:
        expected = constellation_from_sv(self.sv)
        if expected is not self.constellation:
            msg = f"constelação {self.constellation.value} incompatível com SV {self.sv!r}"
            raise ValueError(msg)
        return self

    @property
    def prn(self) -> int:
        """Número PRN do satélite."""
        return int(self.sv[1:])


class BroadcastEphemeris(BaseModel):
    """Efemérides de transmissão (RINEX NAV) — elementos Kepler (ICD-GPS-200).

    Suporta GPS, GALILEO e BEIDOU (modelo Kepleriano). GLONASS usa vetor de
    estado (posição/velocidade/aceleração) e é tratado separadamente.
    """

    sv: str
    toc: datetime  # tempo de relógio
    toe: datetime  # época das efemérides

    # Parâmetros de relógio (s, s/s, s/s²)
    af0: float = 0.0
    af1: float = 0.0
    af2: float = 0.0

    # Elementos Kepler (ICD-GPS-200)
    sqrt_a: float  # sqrt(A) em sqrt(m)
    e: float  # excentricidade
    i0: float  # inclinação na época (rad)
    omega0: float  # longitude do nodo ascendente na época (rad)
    omega: float  # argumento do perigeu (rad)
    m0: float  # anomalia média na época (rad)

    # Taxas / correções
    delta_n: float = 0.0  # correção do movimento médio (rad/s)
    idot: float = 0.0  # taxa de inclinação (rad/s)
    omega_dot: float = 0.0  # taxa do nodo ascendente (rad/s)

    # Amplitudes das correções harmônicas (m, m, rad, rad, rad, rad)
    crc: float = 0.0
    crs: float = 0.0
    cuc: float = 0.0
    cus: float = 0.0
    cic: float = 0.0
    cis: float = 0.0

    health: int = 0
    ura: float = 0.0

    @field_validator("sv")
    @classmethod
    def _normalize_sv(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def constellation(self) -> Constellation:
        """Constelação inferida do SV."""
        return constellation_from_sv(self.sv)


class GlonassEphemeris(BaseModel):
    """Efemérides GLONASS (RINEX NAV) — vetor de estado na época ``tb``.

    O modelo GLONASS de transmissão fornece posição (km), velocidade (km/s) e
    aceleração (km/s²) na época de referência (ICD-GLONASS). A propagação é
    cinemática com correção da rotação da Terra.
    """

    sv: str
    tb: datetime  # época de referência (UTC)

    # Estado em km / km/s / km/s² (ICD-GLONASS)
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    ax: float
    ay: float
    az: float

    freq_num: int = 0  # número de frequência (canais L1/L2)

    @field_validator("sv")
    @classmethod
    def _normalize_sv(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def constellation(self) -> Constellation:
        return Constellation.GLONASS
