"""Caso de uso: construir tracks de satélites a partir de NAV."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gnsstools4astro.application.services.track_builder import (
    TrackBuilder,
    TracksResult,
)
from gnsstools4astro.domain.station import GNSSStation


class BuildTracksUseCase:
    """Constrói tracks (alt/az/RA/Dec/ECEF) dos satélites do NAV."""

    def __init__(self, track_builder: TrackBuilder) -> None:
        self._builder = track_builder

    def run(
        self,
        nav_paths: list[str | Path],
        station: GNSSStation,
        *,
        times: list[datetime] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        step_s: float = 600.0,
        min_alt_deg: float | None = None,
    ) -> TracksResult:
        """Executa a construção dos tracks."""
        return self._builder.build(
            nav_paths,
            station,
            times=times,
            start=start,
            end=end,
            step_s=step_s,
            min_alt_deg=min_alt_deg,
        )
