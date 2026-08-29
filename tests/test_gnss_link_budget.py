"""Testes do orçamento de enlace (FSPL e EIRP estimado)."""

from __future__ import annotations

import pytest
from gnsstools4astro.domain.link_budget import (
    eirp_estimate_dbw,
    free_space_path_loss_db,
    received_power_dbw,
    reference_frequency_mhz,
)
from gnsstools4astro.domain.satellite import Constellation


def test_free_space_path_loss_reference() -> None:
    # d = 20.000 km, f = 1575,42 MHz (L1 GPS).
    fspl = free_space_path_loss_db(20000.0, 1575.42)
    assert fspl == pytest.approx(182.42, rel=0.01)


def test_received_power_dbw() -> None:
    # P_rec ≈ 1.27e-16 W (C/N0 = 45 dB-Hz) → ≈ -158,96 dBW.
    assert received_power_dbw(1.27e-16) == pytest.approx(-158.96, rel=0.02)


def test_eirp_estimate_dbw() -> None:
    # EIRP_est = P_rec(dBW) + FSPL; com P_rec de 45 dB-Hz a 20.000 km em L1.
    eirp = eirp_estimate_dbw(1.27e-16, 20000.0, 1575.42)
    assert eirp == pytest.approx(23.46, abs=0.5)
    # Cresce com a potência e com a distância.
    assert eirp_estimate_dbw(2.0e-16, 20000.0, 1575.42) > eirp
    assert eirp_estimate_dbw(1.27e-16, 25000.0, 1575.42) > eirp


def test_reference_frequency_mhz() -> None:
    assert reference_frequency_mhz(Constellation.GPS) == pytest.approx(1575.42)
    assert reference_frequency_mhz(Constellation.GALILEO) == pytest.approx(1575.42)
    assert reference_frequency_mhz(Constellation.BEIDOU) == pytest.approx(1561.098)
    assert reference_frequency_mhz(Constellation.GLONASS) == pytest.approx(1602.0)
