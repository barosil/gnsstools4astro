"""Leitor de NAV RINEX 2.11 próprio.

Contorna a limitação do ``georinex.nav2`` que **descarta o satélite inteiro**
quando o mesmo SV aparece com épocas (ToC) duplicadas num arquivo NAV RINEX 2
(ex.: Galileo ``.24l`` da RBMC). Aqui, quando há épocas duplicadas, **mantemos
o último registro** (mensagem de transmissão mais recente) em vez de descartar
o SV — o que também elimina os avisos "duplicate times detected".

O dataset de saída tem a mesma estrutura do ``georinex`` (coords ``time``/``sv``
e as mesmas variáveis de campo), para que ``RinexReader.to_ephemerides`` e
``to_glonass_ephemerides`` funcionem sem alteração. GLONASS é convertido para
metros (consistente com o georinex NAV3).
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# Número de linhas de continuação por tipo de arquivo.
# GPS ('N') e Galileo ('E') têm 8 linhas por registro; GLONASS ('G') tem 4.
_N_EXTRA: dict[str, int] = {"N": 7, "G": 3, "E": 7}
_STARTCOL2 = 3  # coluna (0-indexada) onde começam os dados numéricos
_LF = 19  # largura de cada campo numérico

# Campos por tipo de arquivo (mesmos nomes do georinex).
_FIELDS: dict[str, list[str]] = {
    "N": [
        "SVclockBias",
        "SVclockDrift",
        "SVclockDriftRate",
        "IODE",
        "Crs",
        "DeltaN",
        "M0",
        "Cuc",
        "Eccentricity",
        "Cus",
        "sqrtA",
        "Toe",
        "Cic",
        "Omega0",
        "Cis",
        "Io",
        "Crc",
        "omega",
        "OmegaDot",
        "IDOT",
        "CodesL2",
        "GPSWeek",
        "L2Pflag",
        "SVacc",
        "health",
        "TGD",
        "IODC",
        "TransTime",
        "FitIntvl",
    ],
    "E": [
        "SVclockBias",
        "SVclockDrift",
        "SVclockDriftRate",
        "IODnav",
        "Crs",
        "DeltaN",
        "M0",
        "Cuc",
        "Eccentricity",
        "Cus",
        "sqrtA",
        "Toe",
        "Cic",
        "Omega0",
        "Cis",
        "Io",
        "Crc",
        "omega",
        "OmegaDot",
        "IDOT",
        "DataSrc",
        "GALWeek",
        "SISA",
        "health",
        "BGDe5a",
        "BGDe5b",
        "TransTime",
    ],
    "G": [
        "SVclockBias",
        "SVrelFreqBias",
        "MessageFrameTime",
        "X",
        "dX",
        "dX2",
        "health",
        "Y",
        "dY",
        "dY2",
        "FreqNum",
        "Z",
        "dZ",
        "dZ2",
        "AgeOpInfo",
    ],
}

# Prefixo do SV por tipo de arquivo.
_SV_PREFIX = {"N": "G", "G": "R", "E": "E"}

# Variáveis GLONASS reportadas em km pelo RINEX 2 (georinex converte para m).
_GLONASS_KM_VARS = {"X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2"}


@dataclass(frozen=True)
class NavHeader:
    """Cabeçalho mínimo de um arquivo NAV."""

    version: float
    filetype: str  # 'N' (GPS), 'G' (GLONASS) ou 'E' (Galileo)
    systems: str


def _open_text(path: str | Path):
    """Abre o arquivo como texto, suportando gzip."""
    p = Path(path)
    if str(p).lower().endswith(".gz"):
        return gzip.open(p, "rt", encoding="ascii", errors="replace")
    return p.open("r", encoding="ascii", errors="replace")


def read_nav_header(path: str | Path) -> NavHeader | None:
    """Lê a primeira linha do cabeçalho e extrai versão/tipo.

    Retorna ``None`` se não parecer um NAV RINEX (ou se o arquivo estiver vazio).
    """
    with _open_text(path) as f:
        for _ in range(60):
            line = f.readline()
            if not line:
                return None
            if "RINEX VERSION / TYPE" in line:
                try:
                    version = float(line[:9])
                except ValueError:
                    return None
                filetype = line[20] if len(line) > 20 else ""
                systems = line[40] if len(line) > 40 else ""
                if filetype in _FIELDS:
                    return NavHeader(
                        version=version,
                        filetype=filetype,
                        systems=systems or filetype,
                    )
                return None
    return None


def is_rinex2_nav(path: str | Path) -> bool:
    """True se o arquivo for um NAV RINEX 2.11 (version < 3)."""
    header = read_nav_header(path)
    return header is not None and header.version < 3.0


def _epoch_from_line(ln: str) -> datetime | None:
    """Interpreta a época (ToC) da primeira linha do registro (colunas 4-22)."""
    try:
        year = int(ln[3:5])
        if 80 <= year <= 99:
            year += 1900
        else:
            year += 2000
        second = float(ln[17:22])
        sec_int = int(second)
        return datetime(
            year=year,
            month=int(ln[6:8]),
            day=int(ln[9:11]),
            hour=int(ln[12:14]),
            minute=int(ln[15:17]),
            second=sec_int,
            microsecond=round((second - sec_int) * 1e6),
        )
    except ValueError:
        return None


def _chunk_fields(raw: str, fields: list[str]) -> list[float]:
    """Divide a string acumulada de campos em floats (largura 19)."""
    n = min(len(fields), len(raw) // _LF)
    out: list[float] = []
    for k in range(n):
        token = raw[k * _LF : (k + 1) * _LF].replace("D", "E").strip()
        try:
            out.append(float(token))
        except ValueError:
            out.append(np.nan)
    return out


def read_nav2(path: str | Path) -> xr.Dataset:
    """Lê um arquivo NAV RINEX 2.11 para um dataset xarray.

    Em caso de épocas duplicadas para um mesmo SV, o **último** registro (mais
    recente no arquivo) prevalece — não descarta o satélite.
    """
    header = read_nav_header(path)
    if header is None:
        return xr.Dataset()

    fields = _FIELDS[header.filetype]
    svtype = _SV_PREFIX[header.filetype]
    n_extra = _N_EXTRA[header.filetype]

    # Acumula registros: (sv, epoch) -> vetor de campos (último vence).
    records: dict[tuple[str, datetime], list[float]] = {}

    with _open_text(path) as f:
        # Consome o cabeçalho.
        for line in f:
            if "END OF HEADER" in line:
                break

        for ln in f:
            if len(ln) < 22:
                continue
            epoch = _epoch_from_line(ln)
            if epoch is None:
                continue
            sv = f"{svtype}{int(ln[:2]):02d}"

            raw = ln[22:79]
            for _ in range(n_extra):
                cont = f.readline()
                if not cont:
                    break
                raw += cont[_STARTCOL2:79]

            records[(sv, epoch)] = _chunk_fields(raw, fields)

    if not records:
        return xr.Dataset()

    svs = sorted({sv for sv, _ in records})
    epochs = sorted({ep for _, ep in records})
    sv_idx = {sv: j for j, sv in enumerate(svs)}
    time_idx = {ep: i for i, ep in enumerate(epochs)}

    data = np.full((len(fields), len(epochs), len(svs)), np.nan)
    for (sv, epoch), dvec in records.items():
        i, j = time_idx[epoch], sv_idx[sv]
        data[: len(dvec), i, j] = dvec

    nav = xr.Dataset(
        coords={
            "time": np.asarray(epochs, dtype="datetime64[ns]"),
            "sv": np.asarray(svs),
        }
    )
    for fi, name in enumerate(fields):
        nav[name] = (("time", "sv"), data[fi, :, :])

    if header.filetype == "G":
        for name in _GLONASS_KM_VARS:
            nav[name] *= 1e3  # km → m (consistente com NAV3/georinex)

    nav.attrs["version"] = header.version
    nav.attrs["svtype"] = [svtype]
    nav.attrs["rinextype"] = "nav"
    nav.attrs["filename"] = Path(path).name
    return nav


# Expressão para validar se o arquivo parece NAV (reexportação conveniente).
_NAV_NAME_RE = re.compile(r"\.(?:rnx|nav|[0-9]{2}[nlg])(?:\.gz)?$", re.IGNORECASE)


def looks_like_nav(path: str | Path) -> bool:
    """Heurística por nome de arquivo (para fins de log/classificação)."""
    return bool(_NAV_NAME_RE.search(str(path)))
