"""Testes do propagador de efemérides (ICD-GPS-200) e do leitor RINEX."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import xarray as xr
from gnsstools4astro.infrastructure.ephemeris.broadcast import (
    propagate_glonass,
    propagate_kepler,
)
from gnsstools4astro.infrastructure.rinex.reader import RinexReader

# Faixa plausível do raio orbital (km): GPS ~26560, GLONASS ~25510, Galileo ~29600.
GPS_ORBIT_KM = (25000.0, 28000.0)
GLONASS_ORBIT_KM = (24500.0, 26500.0)


@pytest.fixture(scope="module")
def reader() -> RinexReader:
    return RinexReader()


def test_read_gps_ephemerides(gps_nav, reader: RinexReader) -> None:
    nav = reader.load_nav(gps_nav)
    ephs = reader.to_ephemerides(nav)
    assert len(ephs) > 10
    svs = {e.sv for e in ephs}
    assert all(s.startswith("G") for s in svs)


def test_kepler_orbit_radius(gps_nav, reader: RinexReader) -> None:
    nav = reader.load_nav(gps_nav)
    eph = next(e for e in reader.to_ephemerides(nav) if e.sv == "G01")
    t = eph.toe
    pos = propagate_kepler(eph, t)
    radius_km = float(np.linalg.norm(pos)) / 1000.0
    assert GPS_ORBIT_KM[0] <= radius_km <= GPS_ORBIT_KM[1]


def test_kepler_vectorized_shape(gps_nav, reader: RinexReader) -> None:
    nav = reader.load_nav(gps_nav)
    eph = next(e for e in reader.to_ephemerides(nav) if e.sv == "G01")
    times = [eph.toe + timedelta(minutes=10 * i) for i in range(6)]
    pos = propagate_kepler(eph, times)
    assert pos.shape == (6, 3)


def test_glonass_orbit_radius(glonass_nav, reader: RinexReader) -> None:
    nav = reader.load_nav(glonass_nav)
    ephs = reader.to_glonass_ephemerides(nav)
    assert len(ephs) > 0
    eph = ephs[0]
    pos = propagate_glonass(eph, eph.tb)
    radius_km = float(np.linalg.norm(pos)) / 1000.0
    assert GLONASS_ORBIT_KM[0] <= radius_km <= GLONASS_ORBIT_KM[1]


def test_galileo_ephemerides(reader: RinexReader) -> None:
    """Mapeia um dataset Galileo (Kepler) para efemérides.

    Usa um dataset sintético porque o arquivo ``.24l`` de exemplo tem os
    valores NaN (limitação do georinex com esse NAV RINEX 2).
    """
    nav = xr.Dataset(
        {
            "SVclockBias": (("time", "sv"), [[1e-4, 2e-4]]),
            "sqrtA": (("time", "sv"), [[5153.7, 5153.8]]),
            "Eccentricity": (("time", "sv"), [[0.01, 0.011]]),
            "Io": (("time", "sv"), [[0.96, 0.96]]),
            "Omega0": (("time", "sv"), [[0.3, 0.31]]),
            "omega": (("time", "sv"), [[1.0, 1.01]]),
            "M0": (("time", "sv"), [[0.1, 0.11]]),
            "Toe": (("time", "sv"), [[172800.0, 172800.0]]),
            "GALWeek": (("time", "sv"), [[1288.0, 1288.0]]),
        },
        coords={
            "time": np.array(["2024-06-10T12:00:00"], dtype="datetime64[ns]"),
            "sv": ["E02", "E11"],
        },
    )
    ephs = reader.to_ephemerides(nav)
    assert len(ephs) == 2
    assert all(e.sv.startswith("E") for e in ephs)
    assert ephs[0].sqrt_a == pytest.approx(5153.7)


def test_obs_load_and_samples(obs_file, reader: RinexReader) -> None:
    obs = reader.load_obs(obs_file)
    # RINEX 2.11 da RBMC pode não conter S*; garante apenas que carrega.
    samples = reader.to_signal_samples(obs)
    assert isinstance(samples, list)


def test_obs_samples_synthetic() -> None:
    """C/N0 presente deve gerar amostras (dataset sintético)."""
    times = np.array(["2024-06-10T12:00:00", "2024-06-10T12:00:30"], dtype="datetime64[ns]")
    obs = xr.Dataset(
        {"S1C": (("time", "sv"), np.array([[45.0, np.nan], [44.0, 40.0]]))},
        coords={"time": times, "sv": ["G01", "E11"]},
    )
    reader = RinexReader()
    samples = reader.to_signal_samples(obs, signal="S1C")
    assert len(samples) == 3
    assert all(s.cn0_dbhz is not None for s in samples)
