"""Modelos de domínio para solicitações e arquivos RINEX."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum

from pydantic import BaseModel, field_validator, model_validator


class RinexVersion(str, Enum):
    """Versões de RINEX suportadas."""

    V2 = "2.11"
    V3 = "3.03"


class RinexFileType(str, Enum):
    """Tipos de arquivo RINEX."""

    OBS = "obs"
    NAV = "nav"


class RinexSource(str, Enum):
    """Fontes de dados RINEX suportadas."""

    RBMC = "rbmc"  # Rede Brasileira de Monitoramento Contínuo (IBGE)
    CDDIS = "cddis"  # Crustal Dynamics Data Information System (NASA)


class RinexRequest(BaseModel):
    """Pedido de download de dados RINEX para uma estação num intervalo.

    Attributes:
        station: Identificador da estação (ex.: ``braz``).
        start: Início do intervalo (UTC).
        end: Fim do intervalo (UTC).
        source: Fonte de dados (``rbmc`` padrão | ``cddis``).
        file_types: Tipos de arquivo a baixar (obs, nav).
    """

    station: str
    start: datetime
    end: datetime
    source: RinexSource = RinexSource.RBMC
    file_types: tuple[RinexFileType, ...] = (
        RinexFileType.OBS,
        RinexFileType.NAV,
    )

    @field_validator("station")
    @classmethod
    def _normalize_station(cls, v: str) -> str:
        v = v.strip().lower()
        if not (3 <= len(v) <= 9):
            msg = "identificador de estação deve ter entre 3 e 9 caracteres"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_interval(self) -> RinexRequest:
        if self.end <= self.start:
            msg = "end deve ser posterior a start"
            raise ValueError(msg)
        return self

    @property
    def duration(self) -> timedelta:
        """Duração total do intervalo pedido."""
        return self.end - self.start

    def days(self) -> list[datetime]:
        """Datas (meia-noite UTC) de cada dia coberto pelo intervalo."""
        days: list[datetime] = []
        day = self.start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day < self.end:
            days.append(day)
            day += timedelta(days=1)
        return days
