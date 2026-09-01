"""\
Módulo: tests/test_link_budget.py

Descrição:
    Testes do link budget GNSS (``EstimateLinkBudgetUseCase``): estimativa de
    EIRP a partir de C/N0 (obs) + range (nav) e geração de referência
    ``signals.json``.

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pytest

from gnsstools4astro.application.use_cases import (
    EstimateLinkBudgetUseCase,
    range_means_from_tracks,
)
from gnsstools4astro.domain.constellations import get_constellation_signals
from gnsstools4astro.domain.signal import BOLTZMANN, T_NOISE_K

SPEED_OF_LIGHT = 2.99792458e8
_REF = get_constellation_signals()


def _cn0_for_eirp(eirp_dbw: float, freq_hz: float, range_km: float) -> float:
    """C/N0 (dB-Hz) que um receptor de 0 dBi observaria para dada EIRP."""
    lam = SPEED_OF_LIGHT / freq_hz
    path_loss_lin = (4.0 * np.pi * range_km * 1e3 / lam) ** 2
    p_rec = (10.0 ** (eirp_dbw / 10.0)) / path_loss_lin
    n0 = BOLTZMANN * T_NOISE_K
    return 10.0 * np.log10(p_rec / n0)


@pytest.fixture
def use_case():
    return EstimateLinkBudgetUseCase(signals_by_constellation=_REF)


def test_eirp_recovered_from_cn0(use_case):
    """Dada uma EIRP conhecida, o enlace deve recuperá-la (inversa de Friis)."""
    freq_hz = 1575.42e6  # L1
    range_km = 20200.0
    target_eirp_dbw = 45.0
    cn0 = _cn0_for_eirp(target_eirp_dbw, freq_hz, range_km)

    eirp, p_rec = use_case.eirp_from_cn0(cn0, freq_hz, range_km * 1e3)
    assert eirp == pytest.approx(target_eirp_dbw, rel=1e-6)
    assert p_rec > 0.0


def test_estimate_per_sv_band(use_case):
    """C/N0 por SV → estimativa por banda da constelação."""
    cn0_per_sv = [("G01", 45.0), ("G02", 43.0)]
    range_km = {"G01": 20180.0, "G02": 22100.0}
    estimates = use_case.estimate(cn0_per_sv, range_km)

    # GPS tem 4 bandas (L5, L2C, L1 C/A, L1C) × 2 SV
    assert len(estimates) == 8
    bands = {e.band_name for e in estimates}
    assert {"L5", "L2C", "L1 C/A", "L1C"} <= bands
    assert all(e.constellation == "GPS" for e in estimates)
    assert all(np.isfinite(e.eirp_dbw) for e in estimates)


def test_estimate_skips_sv_without_range(use_case):
    cn0_per_sv = [("G01", 45.0), ("G02", 43.0)]
    estimates = use_case.estimate(cn0_per_sv, {"G01": 20180.0})
    svs = {e.sv for e in estimates}
    assert svs == {"G01"}


def test_summarize(use_case):
    cn0_per_sv = [("G01", 45.0), ("G02", 43.0)]
    range_km = {"G01": 20180.0, "G02": 22100.0}
    estimates = use_case.estimate(cn0_per_sv, range_km)
    summaries = use_case.summarize(estimates)

    by_band = {s.band_name: s for s in summaries}
    assert by_band["L5"].n_satellites == 2
    assert by_band["L5"].eirp_dbw_mean == pytest.approx(
        np.mean([s.eirp_dbw for s in estimates if s.band_name == "L5"])
    )


def test_to_signals_reference_updates_eirp(use_case):
    cn0_per_sv = [("G01", 45.0)]
    range_km = {"G01": 20180.0}
    estimates = use_case.estimate(cn0_per_sv, range_km)

    base = [
        {
            "constellation": "GPS",
            "eirp_dbw": 24.6,
            "rhcp": True,
            "signals": [
                {"name": "L5", "frequency": "1176.45 MHz", "modulation": "BPSK(10)"}
            ],
        }
    ]
    out = use_case.to_signals_reference(estimates, base=base)
    gps = [e for e in out if e["constellation"] == "GPS"][0]
    assert gps["eirp_dbw"] != 24.6  # atualizada pelo link budget
    assert isinstance(gps["eirp_dbw"], float)  # serializável (json.dumps)
    assert gps["signals"][0]["modulation"] == "BPSK(10)"  # preservada da base


def test_range_means_from_tracks():
    import xarray as xr

    tracks = xr.Dataset(
        {"distance_km": (("sv", "time"), np.array([[1.0, 3.0], [np.nan, 20.0]]))},
        coords={"sv": ["G01", "G02"], "time": [0, 1]},
    )
    means = range_means_from_tracks(tracks)
    assert means == {"G01": 2.0, "G02": 20.0}
