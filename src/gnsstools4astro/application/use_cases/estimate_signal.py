"""Caso de uso: estimar sinal transmitido (PSD/EIRP) a partir de obs."""

from __future__ import annotations

from gnsstools4astro.application.services.signal_estimator import (
    SignalEstimator,
    SignalSummary,
)
from gnsstools4astro.domain.satellite import Constellation
from gnsstools4astro.domain.signal import (
    ConstellationSignals,
    SignalEstimate,
    SignalSample,
)


class EstimateSignalUseCase:
    """Estima PSD/potência e associa EIRP de referência por satélite."""

    def __init__(
        self,
        estimator: SignalEstimator,
        signals_by_constellation: dict[Constellation, ConstellationSignals],
    ) -> None:
        self._estimator = estimator
        self._signals = signals_by_constellation

    def run(self, samples: list[SignalSample]) -> list[SignalEstimate]:
        """Converte amostras C/N0 em estimativas de sinal."""
        return self._estimator.estimate(samples)

    def summarize(self, estimates: list[SignalEstimate]) -> list[SignalSummary]:
        """Agrega as estimativas por satélite."""
        return self._estimator.summarize(estimates)

    def summarize_cn0(
        self,
        sv_means: list[tuple[str, float]],
    ) -> list[SignalSummary]:
        """Resumo por satélite a partir de C/N0 médio (caminho vetorizado)."""
        return self._estimator.summarize_cn0(sv_means)
