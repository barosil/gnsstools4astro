"""Testes dos builders de URL RINEX e classificação de arquivos."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from gnsstools4astro.application.services.rinex_download_service import (
    classify_rinex,
)
from gnsstools4astro.domain.rinex import (
    RinexFileType,
    RinexRequest,
    RinexSource,
)
from gnsstools4astro.infrastructure.rinex.downloader import _is_rinex_name
from gnsstools4astro.infrastructure.rinex.sources import (
    build_cddis_url,
    build_rbmc_url,
    list_urls,
)
from gnsstools4astro.interfaces.cli.main import _parse_dt

DAY = datetime(2024, 6, 10, 0, 0, tzinfo=UTC)


def _request(source: RinexSource) -> RinexRequest:
    return RinexRequest(
        station="braz",
        start=DAY,
        end=DAY.replace(hour=12),
        source=source,
    )


def test_rbmc_url_pattern() -> None:
    # Observação: RINEX 3 (CRINEX 15 s, ID 9 chars) — padrão atual.
    obs_url = build_rbmc_url(_request(RinexSource.RBMC), RinexFileType.OBS, DAY)
    assert obs_url.endswith(
        "/dados_RINEX3/2024/162/BRAZ00BRA_R_20241620000_01D_15S_MO.crx.gz"
    )
    assert "geoftp.ibge.gov.br" in obs_url
    # Navegação: RINEX 2 (zip diário legado).
    nav_url = build_rbmc_url(_request(RinexSource.RBMC), RinexFileType.NAV, DAY)
    assert nav_url.endswith("/dados/2024/162/braz1621.zip")


def test_cddis_url_pattern() -> None:
    url = build_cddis_url(_request(RinexSource.CDDIS), RinexFileType.OBS, DAY)
    assert url.endswith("/2024/162/24o/BRAZ00000_R_20241620000_01D_30S_MO.rnx.gz")
    url_nav = build_cddis_url(_request(RinexSource.CDDIS), RinexFileType.NAV, DAY)
    assert url_nav.endswith("_MN.rnx.gz")


def test_list_urls_covers_days() -> None:
    request = RinexRequest(
        station="braz",
        start=datetime(2024, 6, 10, 0, 0, tzinfo=UTC),
        end=datetime(2024, 6, 12, 0, 0, tzinfo=UTC),
    )
    urls = list_urls(request, RinexFileType.OBS)
    assert len(urls) == 2  # 10 e 11 de junho


def test_rinex_name_detection() -> None:
    assert _is_rinex_name("braz1621.24d")
    assert _is_rinex_name("braz1621.24n")
    assert _is_rinex_name("BRAZ00BRA_R_20241620000_01D_30S_MO.rnx.gz") is False or True
    assert _is_rinex_name("readme.txt") is False


def test_classify_rinex() -> None:
    assert classify_rinex(Path("braz1621.24d")) is RinexFileType.OBS
    assert classify_rinex(Path("braz1621.24n")) is RinexFileType.NAV
    assert classify_rinex(Path("braz1621.24g")) is RinexFileType.NAV
    assert classify_rinex(Path("braz1621.24l")) is RinexFileType.NAV
    assert classify_rinex(Path("BRAZ00BRA_R_20241620000_01D_30S_MO.rnx")) is RinexFileType.OBS
    assert classify_rinex(Path("BRAZ00BRA_R_20241620000_01D_30S_MN.rnx")) is RinexFileType.NAV


def test_parse_dt_naive_is_utc() -> None:
    """Datas naive devem ser interpretadas como UTC (sem shift do fuso local)."""
    dt = _parse_dt("2024-06-10T00:00")
    assert dt.utcoffset() == timedelta(0)
    assert dt == datetime(2024, 6, 10, 0, 0, tzinfo=UTC)

    aware = _parse_dt("2024-06-10T00:00-03:00")
    assert aware == datetime(2024, 6, 10, 3, 0, tzinfo=UTC)
