"""Serviço de download de RINEX por intervalo e fonte."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gnsstools4astro.application.ports import RinexDownloaderPort, RinexSourcePort
from gnsstools4astro.domain.rinex import RinexFileType, RinexRequest
from gnsstools4astro.infrastructure.logging_config import get_logger
from gnsstools4astro.infrastructure.rinex.sources import build_cddis_brdc_url

logger = get_logger("gnsstools4astro.application.download")

# RINEX 2.x: extensão ".{ano}{tipo}" — ex.: braz1621.24d (obs), .24n/.24g/.24l (nav).
_RINEX2_RE = re.compile(r"\.[0-9]{2}[odnglp]$", re.IGNORECASE)


@dataclass
class DownloadResult:
    """Resultado de um download de RINEX por intervalo."""

    request: RinexRequest
    out_dir: Path
    files: dict[RinexFileType, list[Path]] = field(default_factory=dict)

    def all_files(self) -> list[Path]:
        """Todos os arquivos baixados, na ordem obs + nav."""
        out: list[Path] = []
        for file_type in (RinexFileType.OBS, RinexFileType.NAV):
            out.extend(self.files.get(file_type, []))
        return out


def classify_rinex(path: Path) -> RinexFileType:
    """Classifica um arquivo RINEX como observação (obs) ou navegação (nav).

    Reconhece a nomenclatura RINEX 2.x (``.o``/``.d`` obs; ``.n``/``.g``/
    ``.l``/``.p`` nav; e ``.{ano}{tipo}``), CRINEX (``.crx`` obs) e RINEX 3.x
    (``_MO.rnx`` obs; ``_MN.rnx`` nav).
    """
    name = path.name.lower()
    if name.endswith("_mo.rnx"):
        return RinexFileType.OBS
    if name.endswith("_mn.rnx"):
        return RinexFileType.NAV
    for suffix, file_type in (
        (".crx", RinexFileType.OBS),
        (".o", RinexFileType.OBS),
        (".d", RinexFileType.OBS),
        (".n", RinexFileType.NAV),
        (".g", RinexFileType.NAV),
        (".l", RinexFileType.NAV),
        (".p", RinexFileType.NAV),
    ):
        if name.endswith(suffix):
            return file_type
    match = _RINEX2_RE.search(name)
    if match:
        last = name[-1]
        return RinexFileType.OBS if last in "od" else RinexFileType.NAV
    if name.endswith((".rnx", ".obs")):
        return RinexFileType.OBS
    if name.endswith(".nav"):
        return RinexFileType.NAV
    msg = f"não foi possível classificar arquivo RINEX: {path}"
    raise ValueError(msg)


class RinexDownloadService:
    """Baixa os arquivos RINEX de um intervalo usando uma fonte e um downloader.

    A RBMC/IBGE serve um **zip diário** com obs + nav; o CDDIS serve um
    arquivo por tipo. O serviço normaliza ambos para um resultado com listas
    de arquivos de observação e navegação.
    """

    def __init__(
        self,
        downloader: RinexDownloaderPort,
        source: RinexSourcePort,
    ) -> None:
        self._downloader = downloader
        self._source = source

    def download(
        self,
        request: RinexRequest,
        out_dir: str | Path,
    ) -> DownloadResult:
        """Baixa obs/nav de todos os dias do intervalo para ``out_dir``.

        Cada tipo de arquivo é buscado separadamente (padrão RINEX 3 na RBMC:
        obs em CRINEX ``.crx.gz``; nav no zip RINEX 2). Arquivos de outro tipo
        extraídos de um zip são descartados (ex.: obs dentro do zip de nav).

        Falhas pontuais (um dia/fonte indisponível) são registradas e não
        abortam o restante do intervalo.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = DownloadResult(request=request, out_dir=out_dir)
        obs_files: list[Path] = []
        nav_files: list[Path] = []

        for day in request.days():
            for file_type in request.file_types:
                # Navegação: prefere o NAV RINEX 3 combinado do IGS (CDDIS);
                # se indisponível para a data, cai para o NAV da fonte
                # (RINEX 2 na RBMC, RINEX 3 por estação no CDDIS).
                if file_type is RinexFileType.NAV:
                    candidates = [
                        ("rinex3", build_cddis_brdc_url(day)),
                        ("fallback", self._source.build_url(request, file_type, day)),
                    ]
                else:
                    candidates = [("default", self._source.build_url(request, file_type, day))]

                downloaded: list[Path] = []
                for kind, url in candidates:
                    dest = (
                        None
                        if file_type is RinexFileType.NAV
                        else _dest_path(request, day, file_type, out_dir)
                    )
                    try:
                        downloaded = self._downloader.fetch(url, dest=dest)
                    except Exception as exc:  # noqa: BLE001 - download multi-fonte best-effort
                        logger.warning(
                            "rinex_download_failed",
                            file_type=file_type.value,
                            day=day.isoformat(),
                            url=url,
                            source=kind,
                            error=str(exc),
                        )
                        continue
                    if kind == "rinex3":
                        logger.info(
                            "rinex3_nav_ok",
                            day=day.isoformat(),
                            url=url,
                        )
                    break

                for path in downloaded:
                    try:
                        found_type = classify_rinex(path)
                    except ValueError:
                        logger.warning("rinex_unclassified", path=str(path))
                        continue
                    # Descarta arquivos de outro tipo (ex.: obs do zip de nav).
                    if found_type is not file_type:
                        logger.debug(
                            "rinex_type_mismatch",
                            path=str(path),
                            expected=file_type.value,
                            found=found_type.value,
                        )
                        continue
                    bucket = obs_files if found_type is RinexFileType.OBS else nav_files
                    bucket.append(path)

        result.files[RinexFileType.OBS] = _dedupe(obs_files)
        result.files[RinexFileType.NAV] = _dedupe(nav_files)
        return result


def _dest_path(
    request: RinexRequest,
    day: datetime,
    file_type: RinexFileType,
    out_dir: Path,
) -> Path:
    """Nome de destino com extensão adequada (para o downloader descomprimir)."""
    if file_type is RinexFileType.OBS:
        return out_dir / f"{request.station}_{day:%Y%m%d}_MO.crx.gz"
    return out_dir / f"{request.station}_{day:%Y%m%d}_nav.zip"


def _dedupe(paths: list[Path]) -> list[Path]:
    """Remove duplicatas preservando a ordem."""
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out
