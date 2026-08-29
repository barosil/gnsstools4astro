"""Tracks de satélites GNSS obtidos por **TLE** (SGP4/Skyfield).

Reutiliza as utilidades do ``radioskysims``:

* catálogo NORAD das constelações GNSS → banco (``satid``) ou recurso local;
* ``SpaceTrackClient`` → TLEs **históricas** (janela ±5 dias da época pedida),
  ideal para comparar com efemérides de transmissão da mesma data.

O dataset de saída tem o mesmo esquema dos tracks por broadcast
(``alt_deg/az_deg/ra_deg/dec_deg/distance_km/ecef_*``, coords ``time``/``sv``),
permitindo comparar posicionamento e orçamento de enlace entre os dois modelos.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from gnsstools4astro.domain.satellite import (
    Constellation,
    Satellite,
    constellation_from_sv,
)
from gnsstools4astro.domain.station import GNSSStation
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.framelib import itrs

_AU = 149597870700.0  # metros por unidade astronômica

_CATALOG_RESOURCE = "resources/gnss_norad.json"

# Nomes de constelação no banco do radioskysims → enum do domínio.
_DB_CONSTELLATION = {
    "GPS": Constellation.GPS,
    "GLONASS": Constellation.GLONASS,
    "GALILEO": Constellation.GALILEO,
    "BEIDOU": Constellation.BEIDOU,
}

_GNSS_CONSTELLATIONS = tuple(Constellation)


@dataclass(frozen=True)
class CatalogEntry:
    """Entrada do catálogo NORAD (sem TLE)."""

    norad_id: int
    name: str


@dataclass(frozen=True)
class TleSatellite:
    """Satélite GNSS com TLE (duas linhas) e época de referência."""

    norad_id: int
    name: str
    constellation: Constellation
    epoch: datetime
    tle_line1: str
    tle_line2: str

    @property
    def tle_id(self) -> str:
        """Identificador estável do satélite TLE (usado como chave)."""
        return f"N{self.norad_id:05d}"


class GnssCatalog:
    """Catálogo NORAD das constelações GNSS.

    Carrega do banco do ``radioskysims`` (``satid``) quando disponível e cai
    para o recurso local ``resources/gnss_norad.json`` caso contrário.
    """

    @staticmethod
    def from_db() -> dict[Constellation, list[CatalogEntry]] | None:
        try:
            engine_mod = importlib.import_module("radioskysims_core.db.engine")
            models_mod = importlib.import_module(
                "radioskysims_sources.infrastructure.db.satellite.models"
            )
            Database = engine_mod.Database
            SatID = models_mod.SatID
            from sqlmodel import select  # noqa: PLC0415

            db = Database()
            out: dict[Constellation, list[CatalogEntry]] = {c: [] for c in _GNSS_CONSTELLATIONS}
            with db.session() as s:
                rows = s.exec(
                    select(SatID).where(SatID.constellation.in_(list(_DB_CONSTELLATION)))
                ).all()
            for row in rows:
                const = _DB_CONSTELLATION.get(row.constellation)
                if const is not None:
                    out[const].append(CatalogEntry(norad_id=row.norad_id, name=row.name))
            if all(out.values()):
                return out
            return None
        except Exception:
            return None

    @staticmethod
    def from_resource() -> dict[Constellation, list[CatalogEntry]] | None:
        try:
            raw = files("gnsstools4astro").joinpath(_CATALOG_RESOURCE).read_text(encoding="utf-8")
            data = json.loads(raw)
            out: dict[Constellation, list[CatalogEntry]] = {c: [] for c in _GNSS_CONSTELLATIONS}
            for name, entries in data.get("constellations", {}).items():
                const = _DB_CONSTELLATION.get(name.upper())
                if const is None:
                    continue
                out[const] = [
                    CatalogEntry(norad_id=int(e["norad_id"]), name=e["name"])
                    for e in entries
                ]
            return out
        except Exception:
            return None

    @classmethod
    def load(cls) -> dict[Constellation, list[CatalogEntry]]:
        """Catálogo (DB primeiro, recurso local como fallback)."""
        catalog = cls.from_db()
        if catalog is None:
            catalog = cls.from_resource()
        if catalog is None or not any(catalog.values()):
            msg = "catálogo NORAD GNSS indisponível (DB e recurso local)"
            raise RuntimeError(msg)
        return catalog


def fetch_historical_tles(
    catalog: dict[Constellation, list[CatalogEntry]],
    epoch: datetime,
    *,
    cache_path: str | Path | None = None,
    max_age_days: float = 6.0,
) -> list[TleSatellite]:
    """Busca as TLEs válidas mais próximas de ``epoch`` (SpaceTrack, histórico).

    Para cada NORAD ID do catálogo, seleciona a TLE cuja época está mais
    próxima de ``epoch`` dentro da janela do ``SpaceTrackClient`` (±5 dias).

    Parameters
    ----------
    catalog : dict[Constellation, list[CatalogEntry]]
        Catálogo NORAD (ver :class:`GnssCatalog`).
    epoch : datetime
        Época de referência (ex.: início da janela de observação).
    cache_path : str | Path, optional
        Se fornecido, grava/lê um cache JSON das TLEs (evita re-buscar).
    max_age_days : float
        Distância máxima aceita entre a época da TLE e ``epoch`` (dias).
    """
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        return _tles_from_json(cache_path.read_text(encoding="utf-8"))

    try:
        st_mod = importlib.import_module(
            "radioskysims_sources.infrastructure.external.satellite_downloader"
        )
        SpaceTrackClient = st_mod.SpaceTrackClient
    except Exception as exc:
        msg = f"SpaceTrackClient do radioskysims indisponível: {exc}"
        raise RuntimeError(msg) from None

    entries = [e for group in catalog.values() for e in group]
    client = SpaceTrackClient(epoch=epoch)
    tle_df = client.fetch_elements_sync([e.norad_id for e in entries])
    if tle_df.empty:
        msg = "SpaceTrack não retornou TLEs para a janela (limite diário ou credenciais)"
        raise RuntimeError(msg)

    entry_by_id = {e.norad_id: e for e in entries}
    const_by_id = {e.norad_id: c for c, group in catalog.items() for e in group}
    target = epoch.replace(tzinfo=None)

    out: list[TleSatellite] = []
    for norad_raw, group in tle_df.groupby("norad_id"):
        norad = int(norad_raw)
        entry = entry_by_id.get(norad)
        if entry is None:
            continue
        const = const_by_id.get(norad)
        if const is None:
            continue
        best = None
        best_epoch: datetime | None = None
        best_delta = float("inf")
        for _, row in group.iterrows():
            tle_epoch = pd.Timestamp(row["epoch"]).to_pydatetime()
            delta = abs((tle_epoch - target).total_seconds()) / 86400.0
            if delta < best_delta:
                best_delta = delta
                best = row
                best_epoch = tle_epoch
        if best is None or best_delta > max_age_days:
            continue
        tle1 = str(best["tle_line1"]).strip()
        tle2 = str(best["tle_line2"]).strip()
        if not tle1.startswith("1 ") or not tle2.startswith("2 "):
            continue
        out.append(
            TleSatellite(
                norad_id=norad,
                name=entry.name,
                constellation=const,
                epoch=best_epoch,
                tle_line1=tle1,
                tle_line2=tle2,
            )
        )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(_tles_to_json(out), encoding="utf-8")

    return out


def _tles_to_json(satellites: list[TleSatellite]) -> str:
    payload = [
        {
            "norad_id": s.norad_id,
            "name": s.name,
            "constellation": s.constellation.value,
            "epoch": s.epoch.isoformat(),
            "tle_line1": s.tle_line1,
            "tle_line2": s.tle_line2,
        }
        for s in satellites
    ]
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _tles_from_json(raw: str) -> list[TleSatellite]:
    out: list[TleSatellite] = []
    for item in json.loads(raw):
        epoch = datetime.fromisoformat(item["epoch"])
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=UTC)
        out.append(
            TleSatellite(
                norad_id=int(item["norad_id"]),
                name=item["name"],
                constellation=Constellation(item["constellation"]),
                epoch=epoch,
                tle_line1=item["tle_line1"],
                tle_line2=item["tle_line2"],
            )
        )
    return out


class TleTracksBuilder:
    """Constrói tracks (alt/az/RA/Dec/ECEF) por TLE via Skyfield/SGP4."""

    def __init__(self, station: GNSSStation, *, min_alt_deg: float = 0.0) -> None:
        self._ts = load.timescale()
        self._station = wgs84.latlon(
            station.lat_deg,
            station.lon_deg,
            station.height_m,
        )
        self._min_alt = min_alt_deg

    def build(
        self,
        satellites: list[TleSatellite],
        *,
        times: list[datetime],
        min_alt_deg: float | None = None,
    ) -> xr.Dataset:
        """Dataset 2D ``(time, tle_id)`` com os tracks dos satélites TLE.

        A chave de satélite é ``TleSatellite.tle_id`` (``N<norad>``); use
        :func:`match_tle_svs` para mapear ao SV do RINEX antes de comparar.
        """
        min_alt = self._min_alt if min_alt_deg is None else min_alt_deg
        sky_times = self._ts.from_datetimes([_as_utc(t) for t in times])
        station_t = self._station.at(sky_times)

        records: list[tuple] = []
        for sat in satellites:
            earth = EarthSatellite(sat.tle_line1, sat.tle_line2, sat.name, self._ts)
            pos = earth.at(sky_times)
            topo = pos - station_t
            alt, az, dist = topo.altaz()
            ra, dec, _ = topo.radec()
            ecef = pos.frame_xyz(itrs).m  # (3, N) metros
            for i, t in enumerate(times):
                if float(alt.degrees[i]) < min_alt:
                    continue
                records.append(
                    (
                        t,
                        sat.tle_id,
                        float(alt.degrees[i]),
                        float(az.degrees[i]),
                        float(ra.hours[i] * 15.0),
                        float(dec.degrees[i]),
                        float(dist.km[i]),
                        float(ecef[0, i]),
                        float(ecef[1, i]),
                        float(ecef[2, i]),
                    )
                )
        return _dataset_from_records(records)

    def ecef_at(
        self,
        satellites: list[TleSatellite],
        t_ref: datetime,
    ) -> dict[str, np.ndarray]:
        """Posições ECEF (m) de cada satélite TLE num instante (chave ``tle_id``)."""
        st = self._ts.from_datetime(_as_utc(t_ref))
        out: dict[str, np.ndarray] = {}
        for sat in satellites:
            earth = EarthSatellite(sat.tle_line1, sat.tle_line2, sat.name, self._ts)
            out[sat.tle_id] = np.asarray(earth.at(st).frame_xyz(itrs).m, dtype=float)
        return out


def broadcast_ecef_at(
    nav_paths: list[str | Path],
    t_ref: datetime,
) -> dict[str, np.ndarray]:
    """Posições ECEF (m) de cada SV no instante ``t_ref`` (efemérides broadcast).

    Usa o propagador ICD-GPS-200 (Kepler) e o modelo cinemático GLONASS, os
    mesmos da construção dos tracks por broadcast — serve de referência para o
    casamento TLE→SV por distância mínima. Para cada SV seleciona a efeméride
    com época mais próxima de ``t_ref`` (evita extrapolar a última do arquivo).
    """
    from gnsstools4astro.infrastructure.ephemeris.broadcast import (  # noqa: PLC0415
        propagate,
    )
    from gnsstools4astro.infrastructure.rinex.reader import RinexReader  # noqa: PLC0415

    reader = RinexReader()
    all_ephs: list = []
    for path in nav_paths:
        try:
            nav = reader.load_nav(path)
        except Exception:
            continue
        for sv_raw in nav["sv"].to_numpy():
            sv = str(sv_raw)
            all_ephs.extend(reader.to_ephemeris_series(nav, sv))
            all_ephs.extend(reader.to_glonass_ephemeris_series(nav, sv))

    # Uma efeméride por SV, a mais próxima de t_ref.
    best: dict[str, object] = {}

    def _epoch(eph) -> datetime:
        toe = getattr(eph, "toe", None)
        return toe if toe is not None else eph.tb

    for eph in all_ephs:
        current = best.get(eph.sv)
        if current is None or abs((_epoch(eph) - t_ref).total_seconds()) < abs(
            (_epoch(current) - t_ref).total_seconds()
        ):
            best[eph.sv] = eph

    out: dict[str, np.ndarray] = {}
    for sv, eph in best.items():
        out[sv] = np.asarray(propagate(eph, t_ref), dtype=float)
    return out


def match_tle_svs(
    broadcast_ecef: dict[str, np.ndarray],
    tle_ecef: dict[str, np.ndarray],
    tle_constellation: dict[str, Constellation],
    *,
    max_distance_km: float = 100.0,
) -> dict[str, str]:
    """Casa satélites TLE → SV por menor distância ECEF (por constelação).

    Ignora a convenção de nomes (que não é 1:1 com os SV do RINEX) e usa a
    posição orbital no instante de referência — GNSS da mesma constelação são
    bem separados, então o vizinho mais próximo é o par correto.

    Returns
    -------
    dict[str, str]
        Mapeamento ``{tle_id: sv}`` apenas para os pares dentro do limiar.
    """
    mapping: dict[str, str] = {}
    for const in _GNSS_CONSTELLATIONS:
        b_svs = [sv for sv in broadcast_ecef if constellation_from_sv(sv) is const]
        t_ids = [tid for tid in tle_ecef if tle_constellation.get(tid) is const]
        if not b_svs or not t_ids:
            continue
        b_pos = np.stack([broadcast_ecef[sv] for sv in b_svs])  # (nB, 3)
        t_pos = np.stack([tle_ecef[tid] for tid in t_ids])  # (nT, 3)
        dist_km = np.linalg.norm(b_pos[:, None, :] - t_pos[None, :, :], axis=2) / 1000.0
        for _ in range(min(len(b_svs), len(t_ids))):
            flat = np.nanargmin(dist_km)
            i, j = np.unravel_index(flat, dist_km.shape)
            if dist_km[i, j] > max_distance_km:
                break
            mapping[t_ids[j]] = b_svs[i]
            dist_km[i, :] = np.nan
            dist_km[:, j] = np.nan
    return mapping


def matched_satellites(
    tle_satellites: list[TleSatellite],
    mapping: dict[str, str],
) -> list[Satellite]:
    """Satélites TLE casados, renomeados para o SV do RINEX correspondente."""
    by_id = {s.tle_id: s for s in tle_satellites}
    out: list[Satellite] = []
    for tle_id, sv in mapping.items():
        sat = by_id.get(tle_id)
        if sat is None:
            continue
        out.append(Satellite(sv=sv, constellation=sat.constellation))
    return out


def _as_utc(dt: datetime) -> datetime:
    """Normaliza o datetime para UTC (tz-aware), como o Skyfield exige."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _dataset_from_records(records: list[tuple]) -> xr.Dataset:
    """Monta um dataset 2D ``(time, sv)`` a partir dos registros de tracks."""
    if not records:
        return xr.Dataset()
    times = sorted({r[0] for r in records})
    svs = sorted({r[1] for r in records})
    time_idx = {t: i for i, t in enumerate(times)}
    sv_idx = {s: j for j, s in enumerate(svs)}
    nt, ns = len(times), len(svs)

    def grid() -> np.ndarray:
        return np.full((nt, ns), np.nan)

    alt, az, ra, dec, dist = grid(), grid(), grid(), grid(), grid()
    ex, ey, ez = grid(), grid(), grid()

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
