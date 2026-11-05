"""Testes do serviço de tracks por TLE (catálogo, matching, dataset)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from gnsstools4astro.domain.satellite import Constellation
from gnsstools4astro.domain.station import GNSSStation
from gnsstools4astro.infrastructure.ephemeris.tle_tracks import (
    GnssCatalog,
    TleSatellite,
    TleTracksBuilder,
    _tles_from_json,
    _tles_to_json,
    match_tle_svs,
)

# TLE real (GPS BIIR-5, PRN 22) — época 2024-06-14.
_TLE1 = "1 26407U 00040A   24166.38776425 -.00000020  00000-0  00000-0 0  9997"
_TLE2 = "2 26407  55.0670 244.2966 0144891 294.5968  66.7095  2.00560368175250"


def _gps_tle() -> TleSatellite:
    return TleSatellite(
        norad_id=26407,
        name="GPS BIIR-5 (PRN 22)",
        constellation=Constellation.GPS,
        epoch=datetime(2024, 6, 14, 9, 18, tzinfo=UTC),
        tle_line1=_TLE1,
        tle_line2=_TLE2,
    )


def test_catalog_resource_has_gnss() -> None:
    catalog = GnssCatalog.from_resource()
    assert catalog is not None
    for const in Constellation:
        assert len(catalog[const]) >= 20, const


def test_tle_json_roundtrip() -> None:
    sat = _gps_tle()
    raw = _tles_to_json([sat])
    back = _tles_from_json(raw)
    assert len(back) == 1
    assert back[0].tle_id == sat.tle_id
    assert back[0].tle_line1 == _TLE1
    assert back[0].epoch.tzinfo is not None


def test_tle_tracks_dataset() -> None:
    station = GNSSStation(name="pbcg", lat_deg=-7.21368, lon_deg=-35.90714, height_m=534.1)
    times = [
        datetime(2024, 6, 10, 0, 0, tzinfo=UTC),
        datetime(2024, 6, 10, 1, 0, tzinfo=UTC),
    ]
    ds = TleTracksBuilder(station, min_alt_deg=-90.0).build([_gps_tle()], times=times)
    assert ds.sizes["time"] == 2
    assert ds.sizes["sv"] == 1
    assert {"alt_deg", "az_deg", "ra_deg", "dec_deg", "distance_km", "ecef_x_m"} <= set(
        ds.data_vars
    )
    # Raio orbital GPS ~26.560 km.
    r = np.hypot(
        ds["ecef_x_m"].values, np.hypot(ds["ecef_y_m"].values, ds["ecef_z_m"].values)
    ).mean()
    assert 26000.0 <= r / 1000.0 <= 27200.0


def test_match_tle_svs_by_position() -> None:
    # Posições sintéticas separadas por constelação: o casamento por vizinho
    # mais próximo deve acertar mesmo sem nomes "padrão".
    bcef = {"G01": np.array([2.6e7, 0.0, 0.0]), "G02": np.array([-2.6e7, 1e6, 0.0])}
    tecef = {
        "N11111": np.array([2.5999e7, 1000.0, 0.0]),
        "N22222": np.array([-2.6001e7, 1.0005e6, 0.0]),
    }
    tconst = {"N11111": Constellation.GPS, "N22222": Constellation.GPS}
    mapping = match_tle_svs(bcef, tecef, tconst, max_distance_km=100.0)
    assert mapping == {"N11111": "G01", "N22222": "G02"}


def test_match_tle_svs_respects_threshold() -> None:
    bcef = {"G01": np.array([2.6e7, 0.0, 0.0])}
    # TLE a ~500 km de distância → fora do limiar de 100 km.
    tecef = {"N11111": np.array([2.6e7, 5.0e5, 0.0])}
    tconst = {"N11111": Constellation.GPS}
    assert match_tle_svs(bcef, tecef, tconst, max_distance_km=100.0) == {}
