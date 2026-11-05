"""Download de arquivos RINEX via httpx, com cache, retry e descompressão.

A RBMC/IBGE serve um **zip diário** com todos os arquivos (obs + nav);
o CDDIS serve arquivos individuais (RINEX 3) e exige autenticação Earthdata
(``CDDIS_USERNAME``/``CDDIS_PASSWORD`` no ``credentials.env``). O downloader
extrai todos os arquivos RINEX encontrados e devolve a lista de caminhos locais.
"""

from __future__ import annotations

import base64
import gzip
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Self

import httpx
from dotenv import find_dotenv, load_dotenv
from gnsstools4astro.infrastructure.logging_config import (
    get_logger,
    log_http_event,
    log_io_event,
)

logger = get_logger("gnsstools4astro.rinex.downloader")

# Carrega credenciais locais (CDDIS/Earthdata) quando presentes.
load_dotenv(find_dotenv("credentials.env"), override=False)

_RETRIES = 3
_BACKOFF_S = 1.0

# Autenticação Earthdata: o CDDIS redireciona para urs.earthdata.nasa.gov, que
# aceita HTTP Basic; com a sessão (cookie) estabelecida o arquivo é liberado.
_URSA_HOST = "urs.earthdata.nasa.gov"
_CDDIS_HOST = "cddis.nasa.gov"

# Extensões reconhecidas como "arquivo RINEX" após descompressão.
_RINEX_SUFFIXES = (
    ".rnx",
    ".obs",
    ".nav",
    ".o",
    ".n",
    ".g",
    ".l",
    ".p",
    ".d",
    ".21o",
    ".21n",
    ".22o",
    ".22n",
    ".23o",
    ".23n",
)

# RINEX 2.x: extensão ".{ano}{tipo}" — ex.: braz1621.24d (obs CRINEX),
# braz1621.24n (GPS nav), braz1621.24g (GLONASS nav), braz1621.24l (Galileo).
_RINEX2_RE = re.compile(r"\.[0-9]{2}[odnglp]$", re.IGNORECASE)


class RinexDownloadError(RuntimeError):
    """Falha ao baixar ou validar um arquivo RINEX."""


class RinexDownloader:
    """Baixa arquivos RINEX com httpx, valida a resposta e descomprime.

    Attributes:
        cache_dir: Diretório local onde os arquivos são armazenados.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        timeout_s: float = 90.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
        )

    def close(self) -> None:
        """Fecha o cliente HTTP subjacente."""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, url: str, *, dest: str | Path | None = None) -> list[Path]:
        """Baixa ``url`` (com retry) e retorna os caminhos dos RINEX.

        Para zips, extrai **todos** os arquivos RINEX contidos.

        Returns
        -------
        list[Path]
            Caminhos locais dos arquivos RINEX (obs/nav).
        """
        raw = self._download(url, dest)
        return self._extract(raw)

    def _download(self, url: str, dest: str | Path | None) -> Path:
        name = Path(dest) if dest else Path(url).name or "rinex.bin"
        out = self.cache_dir / name

        # Cache: não baixa de novo se o arquivo já existe e não está vazio.
        if out.exists() and out.stat().st_size > 0:
            log_io_event(logger, "rinex_cache_hit", "read", str(out))
            return out

        last_error: Exception | None = None
        for attempt in range(1, _RETRIES + 1):
            start = time.perf_counter()
            try:
                if _CDDIS_HOST in url:
                    response = self._get_authenticated(url)
                else:
                    response = self._client.get(url)
                duration_ms = (time.perf_counter() - start) * 1000
                log_http_event(
                    logger,
                    "rinex_download",
                    method="GET",
                    url=url,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    attempt=attempt,
                )
                if response.status_code != 200 or self._looks_like_login_page(url, response):
                    msg = (
                        f"HTTP {response.status_code} ao baixar {url} "
                        f"(tentativa {attempt}/{_RETRIES})"
                    )
                    last_error = RinexDownloadError(msg)
                else:
                    out.write_bytes(response.content)
                    log_io_event(
                        logger,
                        "rinex_saved",
                        "write",
                        str(out),
                        bytes_count=len(response.content),
                    )
                    return out
            except httpx.HTTPError as exc:
                last_error = exc

            if attempt < _RETRIES:
                time.sleep(_BACKOFF_S * attempt)

        assert last_error is not None
        msg = f"download falhou para {url}: {last_error}"
        raise RinexDownloadError(msg) from last_error

    def _get_authenticated(self, url: str) -> httpx.Response:
        """Baixa do CDDIS autenticando no Earthdata (Basic no urs).

        Fluxo equivalente ao ``curl -n`` com ``.netrc``: segue o redirect para
        ``urs.earthdata.nasa.gov`` enviando ``Authorization: Basic``; a sessão
        (cookie) é mantida pelo cliente e o arquivo é liberado.
        """
        user = os.environ.get("CDDIS_USERNAME")
        pwd = os.environ.get("CDDIS_PASSWORD")
        if not (user and pwd):
            return self._client.get(url)
        token = base64.b64encode(f"{user}:{pwd}".encode()).decode()
        auth_header = {"Authorization": f"Basic {token}"}

        response = self._client.get(url, follow_redirects=False)
        for _ in range(10):
            location = response.headers.get("location")
            if response.status_code not in (301, 302, 303, 307, 308) or not location:
                break
            next_url = (
                location if location.startswith("http") else str(httpx.URL(location).join(url))
            )
            headers = auth_header if _URSA_HOST in next_url else {}
            response = self._client.get(next_url, follow_redirects=False, headers=headers)
        return response

    @staticmethod
    def _looks_like_login_page(url: str, response: httpx.Response) -> bool:
        """Detecta a página de login Earthdata devolvida no lugar do arquivo."""
        if not url.lower().endswith((".gz", ".zip")):
            return False
        content = response.content
        if not content:
            return True
        if url.lower().endswith(".gz"):
            return content[:2] != b"\x1f\x8b"
        return content[:2] != b"PK"

    def _extract(self, raw: Path) -> list[Path]:
        """Descomprime gz/zip conforme necessário e devolve os RINEX."""
        suffix = raw.suffix.lower()
        if suffix == ".gz":
            target = raw.with_suffix("")
            self._gunzip(raw, target)
            raw.unlink(missing_ok=True)
            return self._extract(target)
        if suffix == ".zip":
            return self._unzip(raw)
        # Arquivo RINEX puro (ou CRINEX .d).
        return [raw]

    @staticmethod
    def _gunzip(src: Path, dst: Path) -> None:
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        log_io_event(logger, "rinex_gunzip", "write", str(dst))

    def _unzip(self, raw: Path) -> list[Path]:
        with zipfile.ZipFile(raw) as zf:
            names = zf.namelist()
            rinex_names = [n for n in names if _is_rinex_name(n)]
            if not rinex_names:
                msg = f"nenhum arquivo RINEX encontrado em {raw}"
                raise RinexDownloadError(msg)

            extracted: list[Path] = []
            for member in rinex_names:
                out = self.cache_dir / Path(member).name
                with zf.open(member) as f_in, out.open("wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                extracted.append(out)
            log_io_event(
                logger,
                "rinex_unzip",
                "write",
                str(raw),
                bytes_count=len(extracted),
            )
        raw.unlink(missing_ok=True)
        return extracted


def _is_rinex_name(name: str) -> bool:
    """True se o nome do arquivo parece ser RINEX."""
    lower = name.lower()
    if lower.endswith(_RINEX_SUFFIXES):
        return True
    return bool(_RINEX2_RE.search(lower))
