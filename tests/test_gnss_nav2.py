"""Testes do parser NAV RINEX 2.11 (fix de tempos duplicados)."""

from __future__ import annotations

import numpy as np
from gnsstools4astro.infrastructure.ephemeris.broadcast import propagate_kepler
from gnsstools4astro.infrastructure.rinex.reader import RinexReader


def test_nav2_gps_parses_all_satellites(gps_nav) -> None:
    reader = RinexReader()
    nav = reader.load_nav(gps_nav)
    svs = {str(s) for s in nav["sv"].values}
    # O arquivo .24n de exemplo tem 32 satélites GPS.
    assert len(svs) == 32
    assert all(s.startswith("G") for s in svs)
    ephs = reader.to_ephemerides(nav)
    assert {e.sv for e in ephs} >= {"G01", "G02"}


def test_nav2_recovers_duplicate_epoch_satellite(galileo_nav) -> None:
    """SV com épocas duplicadas NÃO deve ser descartado.

    O georinex.nav2 emitia "duplicate times detected, skipping SV E02" e
    perdia o satélite; o parser próprio mantém o último registro da época.
    """
    reader = RinexReader()
    nav = reader.load_nav(galileo_nav)
    svs = {str(s) for s in nav["sv"].values}
    assert "E02" in svs  # antes: descartado pelo georinex
    ephs = reader.to_ephemerides(nav)
    e02 = [e for e in ephs if e.sv == "E02"]
    assert e02, "E02 deveria ter efeméride válida"
    pos = propagate_kepler(e02[0], e02[0].toe)
    radius_km = float(np.linalg.norm(pos)) / 1000.0
    # Raio orbital Galileo (~29.600 km).
    assert 29000.0 <= radius_km <= 30300.0


def test_nav2_glonass_in_meters(glonass_nav) -> None:
    reader = RinexReader()
    nav = reader.load_nav(glonass_nav)
    svs = {str(s) for s in nav["sv"].values}
    assert len(svs) == 24
    ephs = reader.to_glonass_ephemerides(nav)
    assert len(ephs) > 0
    # Posição no próprio tb tem raio GLONASS (~25.500 km) em metros.
    eph = ephs[0]
    r = float(np.linalg.norm(np.array([eph.x, eph.y, eph.z]))) / 1000.0
    assert 24500.0 <= r <= 26500.0
