"""
Conftest do pacote gnsstools4astro.

Reproduz as fixtures de amostras RINEX do conftest raiz do monólito
(tests/conftest.py) para o ambiente isolado do pacote.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE = (
    PROJECT_ROOT
    / "src"
    / "GNSStools4Astro"
    / "RinexDownloader"
    / "data"
    / "braz_20240610"
)


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    """Diretório com os arquivos RINEX de exemplo (estação braz, 2024-06-10)."""
    if not _SAMPLE.exists():
        pytest.skip("dados de exemplo não encontrados")
    return _SAMPLE


@pytest.fixture(scope="session")
def gps_nav(sample_dir: Path) -> Path:
    return sample_dir / "braz1621.24n"


@pytest.fixture(scope="session")
def galileo_nav(sample_dir: Path) -> Path:
    return sample_dir / "braz1621.24l"


@pytest.fixture(scope="session")
def glonass_nav(sample_dir: Path) -> Path:
    return sample_dir / "braz1621.24g"


@pytest.fixture(scope="session")
def obs_file(sample_dir: Path) -> Path:
    return sample_dir / "braz1621.24d"


def pytest_configure(config):
    # Recursos não fechados de libs de terceiros geram ResourceWarning no __del__
    warnings.filterwarnings(
        "ignore",
        category=pytest.PytestUnraisableExceptionWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"numpy\.core\.",
        category=DeprecationWarning,
    )
