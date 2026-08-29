"""Serviço de construção de tracks de satélites a partir de NAV + Skyfield."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr

from gnsstools4astro.application.ports import RinexParserPort
from gnsstools4astro.domain.satellite import (
    Satellite,
    constellation_from_sv,
)
from gnsstools4astro.domain.station import GNSSStation
from gnsstools4astro.infrastructure.ephemeris.broadcast import propagate
from gnsstools4astro.infrastructure.ephemeris.skyfield import SkyfieldTracks
from gnsstools4astro.infrastructure.logging_config import get_logger

logger = get_logger("gnsstools4astro.application.tracks")


@dataclass
class TracksResult:
    """Tracks de satélites: dataset xarray + lista de satélites."""

    dataset: xr.Dataset
    satellites: list[Satellite]

    def to_netcdf(self, path: str | Path) -> Path:
        """Salva o dataset em NetCDF e retorna o caminho."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.dataset.to_netcdf(out)
        return out


def _default_times(ephemerides, step_s: float) -> list[datetime]:
    """Grade de tempo padrão a partir do intervalo das efemérides."""
    starts = [_epoch(e) for e in ephemerides]
    ends = [_epoch(e) for e in ephemerides]
    start = min(starts)
    end = max(ends) + timedelta(hours=1)
    step = timedelta(seconds=step_s)
    times: list[datetime] = []
    t = start
    while t <= end:
        times.append(t)
        t += step
    return times


class TrackBuilder:
    """Constrói tracks (alt/az/RA/Dec/ECEF) dos satélites num intervalo."""

    def __init__(
        self,
        parser: RinexParserPort,
        skyfactory: Callable[[GNSSStation], SkyfieldTracks] = SkyfieldTracks,
        min_alt_deg: float = 0.0,
    ) -> None:
        self._parser = parser
        self._skyfactory = skyfactory
        self._min_alt = min_alt_deg

    def build(
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
        """Constrói os tracks dos satélites do NAV.

        Parameters
        ----------
        nav_paths : list[Path]
            Arquivos de navegação (GPS/GALILEO/BEIDOU e GLONASS).
        station : GNSSStation
            Estação de referência para alt/az.
        times : list[datetime], optional
            Grade de tempo explícita (prioritária).
        start, end : datetime, optional
            Intervalo da grade de tempo (default: derivado das efemérides).
        step_s : float
            Passo da grade de tempo em segundos (default 600 s).
        min_alt_deg : float, optional
            Elevação mínima (graus); sobrepõe o valor do construtor.
        """
        min_alt = self._min_alt if min_alt_deg is None else min_alt_deg

        ephemerides = self._load_all_ephemerides(nav_paths)
        if not ephemerides:
            msg = "nenhuma efeméride válida encontrada nos arquivos NAV"
            raise ValueError(msg)

        if times is None:
            if start is None or end is None:
                grid = _default_times(ephemerides, step_s)
                ref: datetime | None = None
            else:
                grid = _grid(start, end, step_s)
                ref = start + (end - start) / 2
        else:
            grid = times
            ref = times[0] + (times[-1] - times[0]) / 2

        # Para cada SV escolhe a efeméride mais próxima da janela observada
        # (em vez da última do arquivo), evitando extrapolações longas.
        ephemerides = _select_per_sv(ephemerides, ref)

        sky = self._skyfactory(station)

        records: list[tuple[datetime, str, float, float, float, float, float]] = []
        for eph in ephemerides:
            sv = eph.sv
            ecef_fn = lambda t, e=eph: propagate(e, t)  # noqa: E731
            for t in grid:
                try:
                    sp = sky.ecef_to_sky(ecef_fn, t)
                except Exception as exc:
                    logger.debug("sky_position_failed", sv=sv, time=t.isoformat(), error=str(exc))
                    continue
                if sp.alt_deg < min_alt:
                    continue
                ecef = np.asarray(ecef_fn(t), dtype=float)
                records.append(
                    (
                        t,
                        sv,
                        sp.alt_deg,
                        sp.az_deg,
                        sp.ra_deg,
                        sp.dec_deg,
                        sp.distance_km,
                        ecef[0],
                        ecef[1],
                        ecef[2],
                    )
                )

        dataset = _build_dataset(records)
        satellites = [
            Satellite(sv=sv, constellation=constellation_from_sv(sv))
            for sv in sorted({e.sv for e in ephemerides})
        ]
        return TracksResult(dataset=dataset, satellites=satellites)

    def _load_all_ephemerides(self, nav_paths: list[str | Path]) -> list:
        """Carrega todas as efemérides (sem deduplicar por SV)."""
        ephemerides: list = []
        for path in nav_paths:
            try:
                nav = self._parser.load_nav(path)
            except Exception as exc:
                logger.warning("nav_load_failed", path=str(path), error=str(exc))
                continue
            ephemerides.extend(self._parser.to_ephemerides(nav))
            ephemerides.extend(self._parser.to_glonass_ephemerides(nav))
        return ephemerides


def _select_per_sv(
    ephemerides: list,
    ref: datetime | None,
) -> list:
    """Mantém uma efeméride por SV, preferindo a mais próxima de ``ref``.

    Quando ``ref`` é ``None`` (grade derivada das próprias efemérides),
    mantém a mais recente por SV (comportamento original). Caso contrário
    escolhe a de época mais próxima do meio da janela observada, evitando
    extrapolações longas (ex.: GLONASS cuja última efeméride do arquivo pode
    estar fora da janela).
    """
    if not ephemerides:
        return []

    def distance(eph) -> float:
        if ref is None:
            return 0.0
        return abs((_epoch(eph) - ref).total_seconds())

    best: dict[str, object] = {}
    for eph in ephemerides:
        key = eph.sv
        current = best.get(key)
        if current is None:
            best[key] = eph
            continue
        newer = (
            _epoch(eph) >= _epoch(current) if ref is None else distance(eph) < distance(current)
        )
        if newer:
            best[key] = eph
    return list(best.values())


def _epoch(eph) -> datetime:
    toe = getattr(eph, "toe", None)
    if toe is not None:
        return toe
    return eph.tb


def _grid(start: datetime, end: datetime, step_s: float) -> list[datetime]:
    times: list[datetime] = []
    t = start
    while t <= end:
        times.append(t)
        t += timedelta(seconds=step_s)
    return times


def _build_dataset(records: list[tuple]) -> xr.Dataset:
    """Monta um dataset 2D ``(time, sv)`` a partir dos registros de tracks.

    Satélites abaixo da elevação mínima (ou sem dado) ficam como NaN.
    """
    if not records:
        return xr.Dataset()

    times = sorted({r[0] for r in records})
    svs = sorted({r[1] for r in records})
    time_idx = {t: i for i, t in enumerate(times)}
    sv_idx = {s: j for j, s in enumerate(svs)}
    nt, ns = len(times), len(svs)

    def grid() -> np.ndarray:
        return np.full((nt, ns), np.nan)

    alt = grid()
    az = grid()
    ra = grid()
    dec = grid()
    dist = grid()
    ex = grid()
    ey = grid()
    ez = grid()

    for r in records:
        t, sv = r[0], r[1]
        i, j = time_idx[t], sv_idx[sv]
        alt[i, j] = r[2]
        az[i, j] = r[3]
        ra[i, j] = r[4]
        dec[i, j] = r[5]
        dist[i, j] = r[6]
        ex[i, j] = r[7]
        ey[i, j] = r[8]
        ez[i, j] = r[9]

    return xr.Dataset(
        {
            "alt_deg": (("time", "sv"), alt),
            "az_deg": (("time", "sv"), az),
            "ra_deg": (("time", "sv"), ra),
            "dec_deg": (("time", "sv"), dec),
            "distance_km": (("time", "sv"), dist),
            "ecef_x_m": (("time", "sv"), ex),
            "ecef_y_m": (("time", "sv"), ey),
            "ecef_z_m": (("time", "sv"), ez),
        },
        coords={
            "time": np.array(
                [t.replace(tzinfo=None) for t in times],
                dtype="datetime64[ns]",
            ),
            "sv": np.array(svs),
        },
    )