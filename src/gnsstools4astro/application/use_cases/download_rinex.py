"""Caso de uso: baixar RINEX de uma estação num intervalo."""

from __future__ import annotations

from pathlib import Path

from gnsstools4astro.application.services.rinex_download_service import (
    DownloadResult,
    RinexDownloadService,
)
from gnsstools4astro.domain.rinex import RinexRequest


class DownloadRinexUseCase:
    """Baixa obs/nav de um intervalo e devolve os arquivos locais."""

    def __init__(self, download_service: RinexDownloadService) -> None:
        self._service = download_service

    def run(self, request: RinexRequest, out_dir: str | Path) -> DownloadResult:
        """Executa o download do intervalo pedido."""
        return self._service.download(request, out_dir)
