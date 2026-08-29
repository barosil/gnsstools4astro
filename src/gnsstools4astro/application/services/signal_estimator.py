"""Serviço de estimativa de sinal transmitido (C/N0 → PSD/EIRP)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from gnsstools4astro.domain.satellite import Constellation, constellation_from_sv
from gnsstools4astro.domain.signal import (
    DEFAULT_BANDWIDTH_HZ,
    ConstellationSignals,
    SignalEstimate,
    SignalSample,
    power_spectral_density,
    received_power_watts,
)


@dataclass
class SignalSummary:
    """Resumo por satélite/constelação."""

    sv: str
    constellation: Constellation
    n_samples: int
    cn0_mean_dbhz: float
    psd_mean_w_hz: float
    received_power_mean_w: float
    eirp_dbw: float | None = None


class SignalEstimator:
    """Estima potência recebida, PSD e associa EIRP de referência por satélite."""

    def __init__(
        self,
        signals_by_constellation: dict[Constellation, ConstellationSignals],
        bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
    ) -> None:
        self._ref = signals_by_constellation
        self._bandwidth_hz = bandwidth_hz

    def estimate(self, samples: list[SignalSample]) -> list[SignalEstimate]:
        """Converte amostras C/N0 em estimativas de sinal."""
        estimates: list[SignalEstimate] = []
        for sample in samples:
            if not sample.valid:
                continue
            cn0 = float(sample.cn0_dbhz)
            constellation = constellation_from_sv(sample.sv)
            ref = self._ref.get(constellation)
            estimates.append(
                SignalEstimate(
                    time=sample.time,
                    sv=sample.sv,
                    constellation=constellation,
                    cn0_dbhz=cn0,
                    bandwidth_hz=self._bandwidth_hz,
                    received_power_w=received_power_watts(cn0),
                    psd_w_hz=power_spectral_density(cn0, self._bandwidth_hz),
                    eirp_dbw=ref.eirp_dbw if ref else None,
                    frequency_mhz=(
                        ref.signals[0].frequency_hz / 1e6 if ref and ref.signals else None
                    ),
                )
            )
        return estimates

    def summarize(self, estimates: list[SignalEstimate]) -> list[SignalSummary]:
        """Agrega as estimativas por satélite (média)."""
        by_sv: dict[str, list[SignalEstimate]] = {}
        for est in estimates:
            by_sv.setdefault(est.sv, []).append(est)

        summaries: list[SignalSummary] = []
        for sv in sorted(by_sv):
            group = by_sv[sv]
            first = group[0]
            summaries.append(
                SignalSummary(
                    sv=sv,
                    constellation=first.constellation,
                    n_samples=len(group),
                    cn0_mean_dbhz=mean(e.cn0_dbhz for e in group),
                    psd_mean_w_hz=mean(e.psd_w_hz for e in group),
                    received_power_mean_w=mean(e.received_power_w for e in group),
                    eirp_dbw=first.eirp_dbw,
                )
            )
        return summaries

    def summarize_cn0(
        self,
        sv_means: list[tuple[str, float]],
    ) -> list[SignalSummary]:
        """Resumo por satélite a partir de C/N0 médio (caminho vetorizado).

        Parameters
        ----------
        sv_means : list[(sv, cn0_mean_dbhz)]
            Médias de C/N0 por satélite (ex.: :meth:`RinexReader.mean_cn0_per_sv`).
        """
        summaries: list[SignalSummary] = []
        for sv, cn0_mean in sorted(sv_means, key=lambda item: item[0]):
            constellation = constellation_from_sv(sv)
            ref = self._ref.get(constellation)
            summaries.append(
                SignalSummary(
                    sv=sv,
                    constellation=constellation,
                    n_samples=1,
                    cn0_mean_dbhz=cn0_mean,
                    psd_mean_w_hz=power_spectral_density(cn0_mean, self._bandwidth_hz),
                    received_power_mean_w=received_power_watts(cn0_mean),
                    eirp_dbw=ref.eirp_dbw if ref else None,
                )
            )
        return summaries
