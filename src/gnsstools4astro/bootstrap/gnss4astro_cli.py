"""Entrypoint da CLI ``gnss4astro``."""

from __future__ import annotations

from collections.abc import Sequence

from gnsstools4astro.bootstrap.composition import (
    build_download_use_case,
    build_signal_use_case,
    build_tracks_use_case,
)
from gnsstools4astro.interfaces.cli.main import GNSS4AstroCLIUseCases, run


def build_cli_use_cases() -> GNSS4AstroCLIUseCases:
    """Constrói os casos de uso da CLI com infraestrutura concreta."""
    return GNSS4AstroCLIUseCases(
        download=build_download_use_case("data/rinex"),
        tracks=build_tracks_use_case(),
        signal=build_signal_use_case(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a CLI."""
    return run(argv, use_cases=build_cli_use_cases())


if __name__ == "__main__":
    raise SystemExit(main())
