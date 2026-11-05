"""Caso de uso: link budget GNSS — C/N0 (obs RINEX) → EIRP por banda.

O RINEX de observação fornece C/N0 por satélite; com a distância (range) das
efemérides NAV e um modelo de enlace (Friis), estimamos a **EIRP transmitida**
por banda. Os valores agregados por constelação/banda alimentam a referência
``signals.json`` (estimativa dos parâmetros de transmissão).

Modelo (link budget):
    P_rec = EIRP · G_rx · (λ / (4πR))²
    EIRP  = P_rec / (G_rx · (λ / (4πR))²)
    P_rec = k·T·10^(C/N0/10)      (k = Boltzmann, T = temperatura de ruído)

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
from numpy.typing import NDArray

from gnsstools4astro.domain.satellite import Constellation, constellation_from_sv
from gnsstools4astro.domain.signal import (
    DEFAULT_BANDWIDTH_HZ,
    ConstellationSignals,
    received_power_watts,
)

SPEED_OF_LIGHT = 2.99792458e8  # m/s


@dataclass
class LinkBudgetEstimate:
    """Estimativa de EIRP de uma banda de um satélite."""

    sv: str
    constellation: str
    band_name: str
    frequency_mhz: float
    range_km: float
    cn0_dbhz: float
    received_power_w: float
    eirp_dbw: float


@dataclass
class BandSummary:
    """Agregação das estimativas por (constelação, banda)."""

    constellation: str
    band_name: str
    frequency_mhz: float
    n_satellites: int
    eirp_dbw_mean: float
    eirp_dbw_std: float | None


class EstimateLinkBudgetUseCase:
    """Estima EIRP transmitida a partir de C/N0 (obs) e range (nav)."""

    def __init__(
        self,
        signals_by_constellation: dict[Constellation, ConstellationSignals],
        *,
        receiver_gain_dbi: float = 0.0,
        t_noise_k: float = 290.0,
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
    ) -> None:
        self._ref = signals_by_constellation
        self._gain_lin = 10.0 ** (receiver_gain_dbi / 10.0)
        self._t_noise_k = t_noise_k
        self._bandwidth_hz = bandwidth_hz

    # ------------------------------------------------------------------
    # Modelo de enlace
    # ------------------------------------------------------------------
    def eirp_from_cn0(
        self,
        cn0_dbhz: float,
        frequency_hz: float,
        range_m: float,
    ) -> tuple[float, float]:
        """EIRP (dBW) e potência recebida (W) a partir de C/N0 e range."""
        p_rec_w = received_power_watts(cn0_dbhz, self._t_noise_k)
        wavelength_m = SPEED_OF_LIGHT / frequency_hz
        path_loss_lin = (4.0 * np.pi * range_m / wavelength_m) ** 2
        eirp_lin = p_rec_w * path_loss_lin / self._gain_lin
        return 10.0 * np.log10(eirp_lin), p_rec_w

    # ------------------------------------------------------------------
    # Estimativa
    # ------------------------------------------------------------------
    def estimate(
        self,
        cn0_per_sv: list[tuple[str, float]],
        range_km_per_sv: dict[str, float] | NDArray,
    ) -> list[LinkBudgetEstimate]:
        """Estima EIRP por (satélite × banda) usando range e C/N0.

        Parameters
        ----------
        cn0_per_sv : list[(sv, cn0_mean_dbhz)]
            C/N0 médio por satélite (ex.: ``RinexReader.mean_cn0_per_sv``).
        range_km_per_sv : dict[str, float] | array-like
            Distância (km) por satélite (ex.: média de ``distance_km`` dos
            tracks). Pode ser um dict ``{sv: km}``.

        Returns
        -------
        list[LinkBudgetEstimate]
        """
        estimates: list[LinkBudgetEstimate] = []
        for sv, cn0 in sorted(cn0_per_sv, key=lambda item: item[0]):
            constellation = constellation_from_sv(sv)
            ref = self._ref.get(constellation)
            if ref is None:
                continue
            range_km = _lookup_range(range_km_per_sv, sv)
            if range_km is None or not np.isfinite(range_km) or range_km <= 0.0:
                continue
            for band in ref.signals:
                eirp_dbw, p_rec_w = self.eirp_from_cn0(cn0, band.frequency_hz, range_km * 1e3)
                estimates.append(
                    LinkBudgetEstimate(
                        sv=sv,
                        constellation=constellation.value,
                        band_name=band.name,
                        frequency_mhz=band.frequency_hz / 1e6,
                        range_km=float(range_km),
                        cn0_dbhz=float(cn0),
                        received_power_w=p_rec_w,
                        eirp_dbw=eirp_dbw,
                    )
                )
        return estimates

    def summarize(self, estimates: list[LinkBudgetEstimate]) -> list[BandSummary]:
        """Agrega EIRP por (constelação, banda): média e desvio."""
        groups: dict[tuple[str, str], list[float]] = {}
        meta: dict[tuple[str, str], float] = {}
        for est in estimates:
            key = (est.constellation, est.band_name)
            groups.setdefault(key, []).append(est.eirp_dbw)
            meta.setdefault(key, est.frequency_mhz)
        summaries: list[BandSummary] = []
        for (cons, band), values in sorted(groups.items()):
            summaries.append(
                BandSummary(
                    constellation=cons,
                    band_name=band,
                    frequency_mhz=meta[(cons, band)],
                    n_satellites=len(values),
                    eirp_dbw_mean=mean(values),
                    eirp_dbw_std=stdev(values) if len(values) > 1 else None,
                )
            )
        return summaries

    def to_signals_reference(
        self,
        estimates: list[LinkBudgetEstimate],
        base: list[dict] | dict | None = None,
    ) -> list[dict]:
        """Gera um fragmento no formato ``signals.json`` com EIRP estimada.

        A EIRP de cada constelação é a média das estimativas por banda (todas
        as bandas observadas). ``base`` pode ser:

        * lista de dicts no formato ``signals.json`` (preserva
          ``rhcp``/``modulation`` e só atualiza ``eirp_dbw``); ou
        * o dicionário ``{Constellation: ConstellationSignals}`` de
          :func:`gnsstools4astro.domain.constellations.get_constellation_signals`.

        Sem ``base``, monta a partir das bandas estimadas.

        Returns
        -------
        list[dict]
            Ex.: ``[{"constellation": "GPS", "eirp_dbw": 41.2, "rhcp": true,
            "signals": [{"name": "L5", "frequency": "1176.45 MHz",
            "modulation": "BPSK(10)"}]}]``
        """
        base_map: dict[str, dict] = {}
        if base is not None:
            if isinstance(base, dict):
                # dict {Constellation: ConstellationSignals} (get_constellation_signals)
                for cons, cs in base.items():
                    entry = {
                        "constellation": cons.value,
                        "eirp_dbw": cs.eirp_dbw,
                        "rhcp": cs.rhcp,
                        "signals": [
                            {
                                "name": b.name,
                                "frequency": f"{b.frequency_hz / 1e6:.6f} MHz",
                                "modulation": b.modulation,
                            }
                            for b in cs.signals
                        ],
                    }
                    base_map[str(cons.value).upper()] = entry
            else:
                for entry in base:
                    base_map[str(entry.get("constellation")).upper()] = entry

        by_cons: dict[str, list[LinkBudgetEstimate]] = {}
        for est in estimates:
            by_cons.setdefault(est.constellation, []).append(est)

        out: list[dict] = []
        for cons in sorted(by_cons):
            group = by_cons[cons]
            eirp_dbw = mean(e.eirp_dbw for e in group)
            entry = base_map.get(cons.upper())
            if entry:
                entry = dict(entry)
                entry["eirp_dbw"] = round(float(eirp_dbw), 2)
            else:
                bands = {(e.frequency_mhz, e.band_name) for e in group}
                entry = {
                    "constellation": cons,
                    "eirp_dbw": round(float(eirp_dbw), 2),
                    "rhcp": True,
                    "signals": [
                        {"name": name, "frequency": f"{freq:.6f} MHz"}
                        for freq, name in sorted(bands)
                    ],
                }
            out.append(entry)
        return out


def range_means_from_tracks(tracks) -> dict[str, float]:
    """Range médio (km) por satélite a partir do dataset de tracks."""
    if "distance_km" not in tracks:
        return {}
    distance = tracks["distance_km"]
    out: dict[str, float] = {}
    for sv in distance["sv"].values:
        sv = str(sv)
        values = np.asarray(distance.sel(sv=sv).values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            out[sv] = float(np.mean(finite))
    return out


def _lookup_range(range_km_per_sv, sv: str) -> float | None:
    if isinstance(range_km_per_sv, dict):
        return range_km_per_sv.get(sv)
    # array-like indexado por posição? converte para dict indexado por sv.
    try:
        idx = list(range_km_per_sv["sv"].values).index(sv)
        return float(np.asarray(range_km_per_sv["distance_km"].values)[idx])
    except (KeyError, TypeError, ValueError):
        return None
