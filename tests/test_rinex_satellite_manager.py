"""\
Módulo: tests/test_rinex_satellite_manager.py

Descrição:
    Testes do ``RinexSatelliteManager`` (satélites criados a partir de RINEX
    NAV) usando um dataset NAV sintético (sem rede nem arquivos externos).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy import units as u

from gnsstools4astro.application.services import RinexSatelliteManager
from gnsstools4astro.domain.analytic_psd import LOW_FLOOR_DB
from radioskysims_core.domain.common import FrequencyRange, TimeRange


def _synthetic_nav() -> xr.Dataset:
    """Dataset NAV GPS sintético (2 SVs, 2 épocas)."""
    times = np.array(
        ["2024-06-10T00:00:00", "2024-06-10T01:00:00"], dtype="datetime64[ns]"
    )
    svs = ["G01", "G02"]

    def full(value: float):
        return (("sv", "time"), np.full((2, 2), value))

    return xr.Dataset(
        {
            "sqrtA": full(5153.76),
            "Toe": full(344736.0),  # segundos de semana (GPS week 2316)
            "GPSWeek": full(2316.0),
            "Eccentricity": full(0.0),
            "M0": (("sv", "time"), np.array([[1.0, 1.2], [-1.0, -1.2]])),
            "Omega0": (("sv", "time"), np.array([[0.5, 0.5], [0.7, 0.7]])),
            "Io": (("sv", "time"), np.array([[0.96, 0.96], [0.90, 0.90]])),
            "omega": full(0.0),
        },
        coords={"time": times, "sv": svs},
    )


def test_from_nav_dataset_creates_satellites():
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    assert len(mgr) == 2
    df = mgr.list_satellites()
    assert set(df["sv"]) == {"G01", "G02"}
    assert set(df["constellation"]) == {"GPS"}
    assert set(df["prn"]) == {1, 2}


def test_from_nav_dataset_ephemerides():
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    assert len(mgr.ephemerides) == 4  # 2 SVs × 2 épocas
    assert "sv" in mgr.list_ephemerides().columns


def test_build_tracks_shape():
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    eph = mgr._best_ephemeris_per_sv()["G01"]
    times = [eph.toe, eph.toe + timedelta(hours=1), eph.toe + timedelta(hours=2)]
    ds = mgr.build_tracks(times=times)
    assert ds.sizes["sv"] == 2
    assert ds.sizes["time"] == 3
    for v in ["alt_deg", "az_deg", "ra_deg", "dec_deg", "distance_km"]:
        assert v in ds.data_vars
    # pelo menos alguma posição finita (órbita circular válida)
    assert np.isfinite(ds["alt_deg"].values).any()


def test_build_tracks_empty_manager():
    mgr = RinexSatelliteManager(satellites=[], ephemerides=[])
    ds = mgr.build_tracks()
    assert ds.sizes["sv"] == 0
    assert "time" in ds.coords


def test_build_assembled_tracks_and_signals():
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    tr = TimeRange.from_iso(
        "2024-06-10T00:00:00", "2024-06-10T01:00:00", 60, u.s
    )
    # FR cobrindo as portadoras GPS L5 (1176.45) e L2C (1227.60) MHz
    fr = FrequencyRange(min_mhz=1100.0, max_mhz=1300.0, resolution_mhz=1.0)

    asm = mgr.build(tr, fr)

    # API compatível com SatelliteAssembled (TLE)
    assert asm.tracks.sizes["sv"] == 2
    assert "time" in asm.tracks.coords
    for v in ["alt_deg", "az_deg", "ra_deg", "dec_deg", "distance_km"]:
        assert v in asm.tracks.data_vars

    # sinais: PSD analítica sobre o grid do FR (mesmo código do TLE)
    assert dict(asm.signals.sizes) == {"sv": 2, "freq": 201}
    freqs = np.asarray(asm.signals["freq"])
    psd = asm.signals["PSD"].values
    assert set(asm.signals["constellation"].values) == {"GPS"}
    # picos nas portadoras GPS dentro do FR (L5/L2C), piso longe delas
    for fc_mhz in (1176.45, 1227.60):
        j = int(np.argmin(np.abs(freqs - fc_mhz)))
        assert psd[:, j].mean() > -50.0
    # entre as portadoras (longe de ambas) a PSD decai (sem piso plano alto)
    gap = (np.abs(freqs - 1176.45) > 40.0) & (np.abs(freqs - 1227.60) > 40.0)
    if gap.any():
        assert psd[:, gap].mean() < -50.0

    assert set(asm.sv_ids) == {"G01", "G02"}
    assert "RinexAssembled(2 satélites" in asm.summary()


def test_build_signals_empty_when_fr_outside_gnss():
    """FR sem portadora GNSS dentro → PSD toda no piso (sem artefato)."""
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    tr = TimeRange.from_iso(
        "2024-06-10T00:00:00", "2024-06-10T01:00:00", 60, u.s
    )
    fr = FrequencyRange(min_mhz=1000.0, max_mhz=1010.0, resolution_mhz=5.0)
    asm = mgr.build(tr, fr)
    assert asm.signals.sizes["sv"] == 2
    assert asm.signals.sizes["freq"] == 3  # 1000, 1005, 1010 MHz
    assert (asm.signals["PSD"].values == LOW_FLOOR_DB).all()


def test_estimate_link_budget_from_obs():
    """Link budget: obs (C/N0) + tracks (nav) → estimativas de EIRP."""
    mgr = RinexSatelliteManager.from_nav_dataset(_synthetic_nav())
    tr = TimeRange.from_iso(
        "2024-06-10T00:00:00", "2024-06-10T03:00:00", 60, u.s
    )
    mgr.build_tracks(time_range=tr)  # cacheia distâncias

    times = np.array(
        ["2024-06-10T00:00:00", "2024-06-10T01:00:00"], dtype="datetime64[ns]"
    )
    obs = xr.Dataset(
        {"S1C": (("time", "sv"), np.array([[45.0, 43.0], [46.0, 44.0]]))},
        coords={"time": times, "sv": ["G01", "G02"]},
    )
    estimates = mgr.estimate_link_budget(obs)
    assert estimates
    assert all(np.isfinite(e.eirp_dbw) for e in estimates)
    assert {e.sv for e in estimates} <= {"G01", "G02"}


def test_build_empty_manager_raises():
    mgr = RinexSatelliteManager(satellites=[], ephemerides=[])
    tr = TimeRange.from_iso(
        "2024-06-10T00:00:00", "2024-06-10T01:00:00", 60, u.s
    )
    fr = FrequencyRange(min_mhz=1000.0, max_mhz=1010.0, resolution_mhz=5.0)
    with pytest.raises(RuntimeError, match="for_station"):
        mgr.build(tr, fr)


def test_for_station_config():
    mgr = RinexSatelliteManager.for_station(
        "pbcg",
        ref_date=datetime(2024, 6, 10),
        cache_dir="/tmp/rinex_test_cache",
    )
    assert mgr._station == "pbcg"
    assert mgr.satellites == []  # dados carregados sob demanda no build


def test_download_nav_cache_hit(tmp_path, monkeypatch):
    """download_nav usa cache local e carrega os satélites do arquivo."""
    cache = tmp_path / "rinex"
    cache.mkdir()
    fake = cache / "pbcg2131.26n"
    fake.write_text("dummy")

    mgr = RinexSatelliteManager.for_station(
        "pbcg", ref_date=datetime(2026, 8, 1), cache_dir=cache
    )
    called: list[Path] = []

    def fake_from_nav(cls, path):
        called.append(path)
        return cls.from_nav_dataset(_synthetic_nav())

    monkeypatch.setattr(
        RinexSatelliteManager, "from_nav", classmethod(fake_from_nav)
    )
    nav = mgr.download_nav()
    assert nav == fake
    assert called == [fake]
    assert len(mgr) == 2  # satélites carregados


def test_download_obs_cache_hit(tmp_path):
    """download_obs usa cache local (.o ou .d)."""
    cache = tmp_path / "rinex"
    cache.mkdir()
    fake = cache / "pbcg2131.26o"
    fake.write_text("dummy")

    mgr = RinexSatelliteManager.for_station(
        "pbcg", ref_date=datetime(2026, 8, 1), cache_dir=cache
    )
    assert mgr.download_obs() == fake


def test_download_obs_unavailable_returns_none(tmp_path, monkeypatch):
    """Falha pontual no download de obs → None (não levanta)."""
    cache = tmp_path / "rinex"
    mgr = RinexSatelliteManager.for_station(
        "pbcg", ref_date=datetime(2026, 8, 1), cache_dir=cache
    )

    def boom(self, file_types, label, doy):  # noqa: ARG001
        msg = "sem obs para a data"
        raise RuntimeError(msg)

    monkeypatch.setattr(RinexSatelliteManager, "_download", boom)
    assert mgr.download_obs() is None
