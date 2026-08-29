"""Ports (Protocols) da aplicação.

Define os contratos consumidos pelos casos de uso, permitindo trocar
implementações de infraestrutura (download, leitura, efemérides) sem
alterar a aplicação.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import xarray as xr

from gnsstools4astro.domain.rinex import RinexFileType, RinexRequest
from gnsstools4astro.domain.satellite import (
    BroadcastEphemeris,
    GlonassEphemeris,
)
from gnsstools4astro.domain.signal import SignalSample


class RinexSourcePort(Protocol):
    """Fonte que constrói URLs RINEX para um pedido e um dia."""

    def build_url(
        self,
        request: RinexRequest,
        file_type: RinexFileType,
        day: datetime,
    ) -> str:
        """Retorna a URL de um arquivo RINEX para ``day``."""
        ...


class RinexDownloaderPort(Protocol):
    """Baixa arquivos RINEX de uma URL para o disco."""

    def fetch(self, url: str, *, dest: str | Path | None = None) -> list[Path]:
        """Baixa ``url`` e retorna os caminhos locais dos arquivos RINEX.

        Um zip pode conter vários arquivos (obs + nav); o downloader
        descomprime e devolve todos os arquivos RINEX encontrados.
        """
        ...


class RinexParserPort(Protocol):
    """Lê arquivos RINEX (obs/nav) e mapeia para modelos do domínio."""

    def load_nav(self, path: str | Path) -> xr.Dataset:
        ...

    def load_obs(self, path: str | Path) -> xr.Dataset:
        ...

    def to_ephemerides(self, nav: xr.Dataset) -> list[BroadcastEphemeris]:
        ...

    def to_glonass_ephemerides(self, nav: xr.Dataset) -> list[GlonassEphemeris]:
        ...

    def to_signal_samples(
        self,
        obs: xr.Dataset,
        signal: str = "S1C",
    ) -> list[SignalSample]:
        ...


class EphemerisPort(Protocol):
    """Propaga uma efeméride para instantes dados, retornando ECEF (m)."""

    def propagate(
        self,
        eph: BroadcastEphemeris | GlonassEphemeris,
        times,
    ) -> np.ndarray:
        """Retorna posições ECEF (m), shape (N, 3)."""
        ...
