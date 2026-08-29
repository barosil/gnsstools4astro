"""Carregamento dos dados de referência por constelação (signals.json).

A aplicação também pode sincronizar esta referência para o repositório
``radioskysims`` (``src/radioskysims/resources/signals.json``) via
:func:`sync_signals_reference`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .satellite import Constellation
from .signal import ConstellationSignals, FrequencyBand

_DEFAULT_RESOURCE = (
    Path(__file__).resolve().parents[1] / "resources" / "signals.json"
)

_FREQ_SCALE = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


def _to_hz(freq_str: str) -> float:
    """Converte ``'1575.42 MHz'`` para hertz."""
    parts = freq_str.strip().split()
    if len(parts) != 2:
        msg = f"frequência mal formatada: {freq_str!r}"
        raise ValueError(msg)
    value, unit = parts
    try:
        return float(value) * _FREQ_SCALE[unit]
    except KeyError:
        msg = f"unidade de frequência desconhecida: {unit!r}"
        raise ValueError(msg) from None


def _parse_entry(entry: dict[str, Any]) -> ConstellationSignals:
    constellation = Constellation(str(entry["constellation"]).upper())
    bands = [
        FrequencyBand(
            name=str(s["name"]),
            frequency_hz=_to_hz(str(s["frequency"])),
            modulation=s.get("modulation"),
        )
        for s in entry.get("signals", [])
    ]
    return ConstellationSignals(
        constellation=constellation,
        eirp_dbw=float(entry["eirp_dbw"]),
        rhcp=bool(entry.get("rhcp", True)),
        signals=bands,
    )


def load_constellation_signals(
    path: str | Path | None = None,
) -> list[ConstellationSignals]:
    """Carrega o arquivo de referência de sinais (JSON).

    Parameters
    ----------
    path : str or Path, optional
        Caminho alternativo do ``signals.json``. Padrão: recurso do pacote.
    """
    p = Path(path) if path else _DEFAULT_RESOURCE
    if not p.exists():
        msg = f"arquivo de referência não encontrado: {p}"
        raise FileNotFoundError(msg)
    with p.open(encoding="utf-8") as f:
        data: list[dict[str, Any]] = json.load(f)
    return [_parse_entry(e) for e in data]


def get_constellation_signals(
    path: str | Path | None = None,
) -> dict[Constellation, ConstellationSignals]:
    """Retorna mapeamento ``{constelação: ConstellationSignals}``."""
    return {cs.constellation: cs for cs in load_constellation_signals(path)}


def sync_signals_reference(
    target: str | Path,
    path: str | Path | None = None,
) -> Path:
    """Copia/atualiza o arquivo de referência para um destino.

    Útil para sincronizar ``src/radioskysims/resources/signals.json`` com a
    referência canônica deste pacote.

    Parameters
    ----------
    target : str or Path
        Caminho de destino do arquivo ``signals.json``.
    path : str or Path, optional
        Origem (padrão: recurso do pacote).

    Returns
    -------
    Path
        Caminho do arquivo de destino escrito.
    """
    src = Path(path) if path else _DEFAULT_RESOURCE
    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst
