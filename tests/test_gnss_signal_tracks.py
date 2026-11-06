"""Testes da estimativa de sinal (C/N0 → PSD) e da construção de tracks."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from gnsstools4astro.application.services.signal_estimator import SignalEstimator
from gnsstools4astro.application.services.track_builder import TrackBuilder
from gnsstools4astro.domain.constellations import get_constellation_signals
from gnsstools4astro.domain.signal import (
    SignalSample,
    power_spectral_density,
    received_power_watts,
)
from gnsstools4astro.domain.station import GNSSStation
from gnsstools4astro.infrastructure.ephemeris.broadcast import propagate
from gnsstools4astro.infrastructure.ephemeris.skyfield import SkyfieldTracks
from gnsstools4astro.infrastructure.rinex.reader import RinexReader

# Referência do "Previous discussion": C/N0 = 45 dB-Hz → P_rec ≈ 1.27e-16 W;
# PSD ≈ 6.3e-23 W/Hz para B = 2 MHz.
CN0 = 45.0
EXPECTED_P_REC = 1.27e-16
EXPECTED_PSD = 6.3e-23


def test_received_power_matches_reference() -> None:
    p = received_power_watts(CN0)
    assert p == pytest.approx(EXPECTED_P_REC, rel=0.02)


def test_psd_matches_reference() -> None:
    psd = power_spectral_density(CN0, bandwidth_hz=2.0e6)
    assert psd == pytest.approx(EXPECTED_PSD, rel=0.02)


def test_psd_scales_with_bandwidth() -> None:
    psd_2mhz = power_spectral_density(CN0, bandwidth_hz=2.0e6)
    psd_4mhz = power_spectral_density(CN0, bandwidth_hz=4.0e6)
    assert psd_4mhz == pytest.approx(psd_2mhz / 2.0)


def test_signal_estimator_summary() -> None:
    estimator = SignalEstimator(get_constellation_signals())
    samples = [
        SignalSample(time=datetime(2024, 6, 10, 12, 0, tzinfo=UTC), sv="G01", cn0_dbhz=45.0),
        SignalSample(time=datetime(2024, 6, 10, 12, 0, tzinfo=UTC), sv="E11", cn0_dbhz=42.0),
    ]
    estimates = estimator.estimate(samples)
    assert len(estimates) == 2
    g01 = estimates[0]
    assert g01.constellation.value == "GPS"
    assert g01.eirp_dbw == pytest.approx(24.6)
    assert g01.psd_w_hz == pytest.approx(EXPECTED_PSD, rel=0.02)

    summaries = estimator.summarize(estimates)
    assert {s.sv for s in summaries} == {"E11", "G01"}


def test_track_builder_returns_dataset(gps_nav, glonass_nav, galileo_nav) -> None:
    station = GNSSStation(name="braz", lat_deg=-7.5, lon_deg=-35.0, height_m=10.0)
    builder = TrackBuilder(
        parser=RinexReader(),
        propagator=propagate,
        skyfactory=SkyfieldTracks,
        min_alt_deg=5.0,
    )
    times = [
        datetime(2024, 6, 10, 12, 0, tzinfo=UTC),
        datetime(2024, 6, 10, 13, 0, tzinfo=UTC),
    ]
    result = builder.build([gps_nav, glonass_nav, galileo_nav], station, times=times)

    assert len(result.satellites) > 0
    ds = result.dataset
    assert "alt_deg" in ds
    assert "az_deg" in ds
    assert "ra_deg" in ds
    assert "dec_deg" in ds
    assert ds.sizes["time"] > 0
    alts = np.asarray(ds["alt_deg"].values)
    azs = np.asarray(ds["az_deg"].values)
    valid_alt = alts[~np.isnan(alts)]
    valid_az = azs[~np.isnan(azs)]
    assert valid_alt.size > 0
    assert np.all(valid_alt >= 5.0)
    assert np.all((valid_az >= 0.0) & (valid_az <= 360.0))

    # Estrutura 2D indexável por SV.
    first_sv = str(ds["sv"].values[0])
    assert ds.sel(sv=first_sv).sizes["time"] == ds.sizes["time"]
