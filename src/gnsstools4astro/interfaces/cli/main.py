"""Interface de linha de comando do GNSStools4Astro.

Subcomandos:

* ``download`` — baixa RINEX (obs+nav) de uma estação num intervalo;
* ``tracks`` — constrói tracks (alt/az/RA/Dec/ECEF) a partir de NAV;
* ``signal`` — estima PSD/potência e associa EIRP a partir de obs;
* ``sync-signals`` — sincroniza ``signals.json`` para um destino.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gnsstools4astro.application.use_cases import (
    BuildTracksUseCase,
    DownloadRinexUseCase,
    EstimateSignalUseCase,
)
from gnsstools4astro.bootstrap.composition import build_download_use_case
from gnsstools4astro.domain.constellations import sync_signals_reference
from gnsstools4astro.domain.rinex import RinexRequest, RinexSource
from gnsstools4astro.domain.station import (
    DEFAULT_STATION,
    DEFAULT_STATION_POSITION,
    GNSSStation,
)
from gnsstools4astro.infrastructure.logging_config import configure_logging
from gnsstools4astro.infrastructure.rinex.reader import RinexReader


class GNSS4AstroCLIUseCases:
    """Dependências da CLI (padrão radioskysims)."""

    def __init__(
        self,
        download: DownloadRinexUseCase,
        tracks: BuildTracksUseCase,
        signal: EstimateSignalUseCase,
    ) -> None:
        self.download = download
        self.tracks = tracks
        self.signal = signal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gnss4astro",
        description="Ferramentas GNSS/RINEX aplicadas à astronomia.",
    )
    parser.add_argument("--log-level", default="INFO", help="Nível de log (DEBUG/INFO/...)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Baixa RINEX (obs+nav) de uma estação num intervalo")
    p_dl.add_argument(
        "--station", default=DEFAULT_STATION, help=f"Estação (padrão: {DEFAULT_STATION})"
    )
    p_dl.add_argument("--start", required=True, help="Início UTC (YYYY-MM-DDTHH:MM)")
    p_dl.add_argument("--end", required=True, help="Fim UTC (YYYY-MM-DDTHH:MM)")
    p_dl.add_argument("--source", default="rbmc", choices=[s.value for s in RinexSource])
    p_dl.add_argument("--outdir", default="data/rinex", help="Diretório de saída")

    p_tr = sub.add_parser("tracks", help="Constrói tracks de satélites a partir de NAV")
    p_tr.add_argument("--nav", nargs="+", required=True, help="Arquivo(s) NAV (.n/.g/.l/.rnx)")
    p_tr.add_argument(
        "--station", default=DEFAULT_STATION, help=f"Nome da estação (padrão: {DEFAULT_STATION})"
    )
    p_tr.add_argument(
        "--lat", type=float, default=DEFAULT_STATION_POSITION.lat_deg, help="Latitude (graus)"
    )
    p_tr.add_argument(
        "--lon", type=float, default=DEFAULT_STATION_POSITION.lon_deg, help="Longitude (graus)"
    )
    p_tr.add_argument(
        "--height", type=float, default=DEFAULT_STATION_POSITION.height_m, help="Altitude (m)"
    )
    p_tr.add_argument("--start", default=None, help="Início UTC (opcional)")
    p_tr.add_argument("--end", default=None, help="Fim UTC (opcional)")
    p_tr.add_argument("--step", type=float, default=600.0, help="Passo da grade (s)")
    p_tr.add_argument("--min-alt", type=float, default=0.0, help="Elevação mínima (graus)")
    p_tr.add_argument("--out", default="tracks.nc", help="Saída NetCDF")

    p_sg = sub.add_parser("signal", help="Estima PSD/potência e associa EIRP a partir de obs")
    p_sg.add_argument("--obs", nargs="+", required=True, help="Arquivo(s) de observação")
    p_sg.add_argument("--signal", default="S1C", help="Variável de C/N0 (ex.: S1C)")
    p_sg.add_argument("--bandwidth-mhz", type=float, default=2.0, help="Largura de banda (MHz)")
    p_sg.add_argument("--json", action="store_true", help="Saída em JSON")

    p_sync = sub.add_parser("sync-signals", help="Sincroniza signals.json para um destino")
    p_sync.add_argument(
        "--target",
        required=True,
        help="Caminho de destino (ex.: src/radioskysims/resources/signals.json)",
    )

    return parser


def _parse_dt(value: str) -> datetime:
    """Interpreta datas ``naive`` como UTC (sem deslocamento de fuso local)."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def run(argv, use_cases: GNSS4AstroCLIUseCases) -> int:
    """Executa a CLI com os casos de uso fornecidos."""
    parser = _parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    if args.command == "download":
        request = RinexRequest(
            station=args.station,
            start=_parse_dt(args.start),
            end=_parse_dt(args.end),
            source=RinexSource(args.source),
        )
        # O caso de uso é construído com o outdir real para que o downloader
        # grave os arquivos no diretório pedido pelo usuário.
        result = build_download_use_case(args.outdir).run(request, args.outdir)
        print(f"Baixados: {len(result.all_files())} arquivos em {result.out_dir}")
        for path in result.all_files():
            print(f"  - {path}")

    elif args.command == "tracks":
        station = GNSSStation(
            name=args.station,
            lat_deg=args.lat,
            lon_deg=args.lon,
            height_m=args.height,
        )
        start = _parse_dt(args.start) if args.start else None
        end = _parse_dt(args.end) if args.end else None
        tracks = use_cases.tracks.run(
            [Path(p) for p in args.nav],
            station,
            start=start,
            end=end,
            step_s=args.step,
            min_alt_deg=args.min_alt,
        )
        out = tracks.to_netcdf(args.out)
        print(f"Tracks de {len(tracks.satellites)} satélites salvos em {out}")

    elif args.command == "signal":
        # Resumo vetorizado: média de C/N0 por satélite (rápido p/ obs 15 s).
        reader = RinexReader()
        sv_means: list[tuple[str, float]] = []
        for obs_path in args.obs:
            obs = reader.load_obs(obs_path)
            sv_means.extend(reader.mean_cn0_per_sv(obs, signal=args.signal))
        summaries = use_cases.signal.summarize_cn0(sv_means)

        if args.json:
            payload = [
                {
                    "sv": s.sv,
                    "constellation": s.constellation.value,
                    "cn0_mean_dbhz": round(s.cn0_mean_dbhz, 2),
                    "psd_mean_w_hz": s.psd_mean_w_hz,
                    "received_power_mean_w": s.received_power_mean_w,
                    "eirp_dbw": s.eirp_dbw,
                }
                for s in summaries
            ]
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"{'sv':<5} {'const':<9} {'C/N0(dB-Hz)':>12} {'P_rec(W)':>12} {'PSD(W/Hz)':>14} {'EIRP(dBW)':>10}"
            )
            for s in summaries:
                print(
                    f"{s.sv:<5} {s.constellation.value:<9} "
                    f"{s.cn0_mean_dbhz:>12.1f} {s.received_power_mean_w:>12.2e} "
                    f"{s.psd_mean_w_hz:>14.2e} "
                    f"{s.eirp_dbw if s.eirp_dbw is not None else '-':>10}"
                )

    elif args.command == "sync-signals":
        out = sync_signals_reference(args.target)
        print(f"signals.json sincronizado em {out}")

    return 0
