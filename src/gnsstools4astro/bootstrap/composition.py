"""Composição de dependências (composition root).

Constrói os casos de uso com as implementações concretas de infraestrutura,
de forma análoga ao ``bootstrap`` do radioskysims.
"""

from __future__ import annotations

from pathlib import Path

from gnsstools4astro.application.services.rinex_download_service import (
    RinexDownloadService,
)
from gnsstools4astro.application.services.signal_estimator import SignalEstimator
from gnsstools4astro.application.services.track_builder import TrackBuilder
from gnsstools4astro.application.use_cases import (
    BuildTracksUseCase,
    DownloadRinexUseCase,
    EstimateSignalUseCase,
)
from gnsstools4astro.domain.constellations import get_constellation_signals
from gnsstools4astro.infrastructure.ephemeris.broadcast import propagate
from gnsstools4astro.infrastructure.rinex.downloader import RinexDownloader
from gnsstools4astro.infrastructure.rinex.reader import RinexReader
from gnsstools4astro.infrastructure.rinex.sources import RinexSourceAdapter


def build_download_use_case(cache_dir: str | Path) -> DownloadRinexUseCase:
    """Constrói o caso de uso de download com fonte padrão e downloader httpx."""
    source = RinexSourceAdapter()
    downloader = RinexDownloader(cache_dir)
    service = RinexDownloadService(downloader=downloader, source=source)
    return DownloadRinexUseCase(service)


def build_tracks_use_case() -> BuildTracksUseCase:
    """Constrói o caso de uso de tracks com reader georinex e propagador Kepler."""
    parser = RinexReader()
    builder = TrackBuilder(parser=parser)
    return BuildTracksUseCase(builder)


def build_signal_use_case() -> EstimateSignalUseCase:
    """Constrói o caso de uso de estimativa de sinal com a referência do pacote."""
    signals = get_constellation_signals()
    estimator = SignalEstimator(signals_by_constellation=signals)
    return EstimateSignalUseCase(estimator=estimator, signals_by_constellation=signals)


# Reexportado para conveniência (utilizado por testes e exemplos).
__all__ = [
    "build_download_use_case",
    "build_signal_use_case",
    "build_tracks_use_case",
    "propagate",
]
