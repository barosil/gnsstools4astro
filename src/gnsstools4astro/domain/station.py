"""Estações GNSS de referência."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GNSSStation(BaseModel):
    """Estação GNSS com posição conhecida.

    Attributes:
        name: Identificador da estação (ex.: ``braz``, 3 a 9 caracteres).
        lat_deg: Latitude geodésica (graus, positiva ao norte).
        lon_deg: Longitude (graus, positiva a leste).
        height_m: Altitude elipsoidal (metros).
        source: Fonte RINEX associada (``rbmc`` | ``cddis``).
    """

    name: str
    lat_deg: float
    lon_deg: float
    height_m: float = 0.0
    source: str = Field(default="rbmc", description="Fonte RINEX (rbmc | cddis)")

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not (3 <= len(v) <= 9):
            msg = "identificador de estação deve ter entre 3 e 9 caracteres"
            raise ValueError(msg)
        return v


# ----------------------------------------------------------------------
# Estação padrão (RBMC)
# ----------------------------------------------------------------------

# Estação RBMC usada como padrão por CLI, exemplos e notebooks.
# PBCG fica na região do BINGO (próxima ao Uirapuru).
DEFAULT_STATION = "pbcg"

DEFAULT_STATION_POSITION = GNSSStation(
    name=DEFAULT_STATION,
    lat_deg=-7.21368,
    lon_deg=-35.90714,
    height_m=534.1,
)
