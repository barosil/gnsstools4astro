"""\
Módulo: tests/test_analytic_psd.py

Descrição:
    Testes das PSDs analíticas de modulações GNSS (``analytic_psd``).

Copyright BINGO Collaboration
Last modified: 2026-08-31
"""

from __future__ import annotations

import numpy as np
import pytest
from gnsstools4astro.domain.analytic_psd import (
    LOW_FLOOR_DB,
    analytic_psd,
    boc_sin_psd,
    bpsk_r_psd,
    parse_modulation,
    psd_db,
    psd_w_per_hz,
)

_FC = 1575.42e6


def _dense_grid(half_mhz: float = 40.0, step_hz: float = 1e4) -> np.ndarray:
    """Grid denso centrado na portadora para verificar integrais."""
    n = int(2 * half_mhz * 1e6 / step_hz) + 1
    return _FC + np.linspace(-half_mhz * 1e6, half_mhz * 1e6, n)


class TestBpsk:
    def test_peak_at_carrier(self):
        f = _dense_grid()
        g = bpsk_r_psd(f - _FC, 1.023e6)
        assert g.max() == pytest.approx(1.0 / 1.023e6, rel=1e-3)

    def test_unit_integral(self):
        f = _dense_grid()
        g = bpsk_r_psd(f - _FC, 1.023e6)
        integral = np.trapezoid(g, f)
        assert integral == pytest.approx(1.0, rel=2e-2)


class TestBoc:
    def test_null_at_carrier(self):
        f = _dense_grid()
        g = boc_sin_psd(f - _FC, 1.023e6, 1.023e6)
        assert g[np.argmin(np.abs(f - _FC))] == pytest.approx(0.0, abs=1e-20)

    def test_unit_integral(self):
        f = _dense_grid()
        g = boc_sin_psd(f - _FC, 1.023e6, 1.023e6)
        integral = np.trapezoid(g, f)
        assert integral == pytest.approx(1.0, rel=2e-2)

    def test_no_nan(self):
        f = _dense_grid()
        g = boc_sin_psd(f - _FC, 1.023e6, 1.023e6)
        assert np.isfinite(g).all()


class TestDispatch:
    @pytest.mark.parametrize(
        "mod",
        ["BPSK(1)", "BPSK(10)", "BPSK(0.5)", "QPSK(0.5)", "BOC(1,1)", "BOC(6,1)"],
    )
    def test_known_modulations(self, mod):
        f = _dense_grid()
        g = analytic_psd(mod, f, _FC)
        assert g.shape == f.shape
        assert np.isfinite(g).all()
        assert g.min() >= 0.0

    def test_cboc(self):
        f = _dense_grid()
        g = analytic_psd("CBOC(6,1,1/11)", f, _FC)
        assert g.shape == f.shape
        assert np.isfinite(g).all()
        integral = np.trapezoid(g, f)
        # O filtro de transmissão remove ~4% (cauda além de ~12 MHz).
        assert integral == pytest.approx(1.0, rel=6e-2)

    def test_unsupported_raises(self):
        with pytest.raises(NotImplementedError, match="não suportada"):
            analytic_psd("UNKNOWN", np.array([_FC]), _FC)

    def test_parse_bpsk(self):
        assert parse_modulation("BPSK(10)") == {"kind": "bpsk", "n": 10.0}

    def test_parse_cboc(self):
        p = parse_modulation("CBOC(6,1,1/11)")
        assert p["kind"] == "cboc"
        assert p["weights"] == pytest.approx([10 / 11, 1 / 11])


class TestScaling:
    def test_psd_w_per_hz(self):
        g = bpsk_r_psd(np.array([0.0]), 1.023e6)
        assert psd_w_per_hz(g, 288.0)[0] == pytest.approx(288.0 / 1.023e6)

    def test_psd_db_floor(self):
        db = psd_db(np.array([1e-300, 1e3]))
        assert db[0] == pytest.approx(LOW_FLOOR_DB)
        assert db[1] == pytest.approx(30.0)

    def test_no_high_floor_far_from_carrier(self):
        """Longe da portadora a PSD decai (sem piso de ruído de Welch)."""
        f = np.array([_FC - 227.0e6, _FC + 227.0e6])  # ~227 MHz de afastamento
        g = bpsk_r_psd(f - _FC, 1.023e6)
        db = psd_db(g)
        assert (db < -100.0).all()


class TestWideBandExtension:
    """PSD estendida realisticamente por toda a banda da simulação.

    Regressão para a correção que evita "artefatos de linha" ao longo de toda
    a banda de frequência durante o trânsito: a PSD é avaliada pela cauda
    analítica (``sinc²``/BOC) em TODO o grid, sem ser truncada a uma janela de
    ~20 MHz nem substituída por um piso constante.
    """

    def _bingo_band(self) -> np.ndarray:
        """Grid estilo BINGO: 980–1260 MHz em passos de 1 MHz."""
        return np.arange(980.0, 1260.0 + 1e-9, 1.0) * 1e6

    def test_psd_independent_of_grid_extent(self) -> None:
        """A PSD não depende da largura da banda (sem truncagem/windowing).

        Se a PSD fosse truncada a uma janela de ~20 MHz, os pontos comuns
        entre um grid estreito e um grid largo difeririam nas bordas.
        """
        fc = 1176.45e6
        wide = self._bingo_band()
        narrow = np.arange(1000.0, 1200.0 + 1e-9, 1.0) * 1e6

        d_wide = psd_db(analytic_psd("BPSK(10)", wide, fc))
        d_narrow = psd_db(analytic_psd("BPSK(10)", narrow, fc))

        common = (wide >= 1000e6) & (wide <= 1200e6)
        np.testing.assert_allclose(d_wide[common], d_narrow, rtol=0.0, atol=1e-9)

    def test_not_truncated_to_floor_across_band(self) -> None:
        fc = 1176.45e6
        f = self._bingo_band()
        db = psd_db(analytic_psd("BPSK(10)", f, fc))
        # Truncar em ±10 MHz deixaria ~90% da banda no piso numérico. Aqui só
        # eventuais nulos ESTRUTURAIS (zeros da sinc²) tocam o piso.
        n_floor = int((db <= LOW_FLOOR_DB + 1.0).sum())
        assert n_floor <= 5
        # Extremos da banda muito acima do piso (cauda real, não truncada).
        assert db[0] > -200.0
        assert db[-1] > -200.0


class TestTxFilterFloor:
    """Filtro de transmissão (~24 MHz) + piso fora de banda de -60 dBc/370 MHz."""

    def test_far_band_sits_on_flat_floor(self) -> None:
        f = np.arange(980.0, 1260.0 + 1e-9, 1.0) * 1e6
        g = analytic_psd("BPSK(10)", f, 1176.45e6)
        far = np.abs(f - 1176.45e6) > 40e6
        floor = 10 ** (-6.0) / 370e6
        np.testing.assert_allclose(g[far], floor, rtol=1e-6)

    def test_in_band_peak_preserved(self) -> None:
        f = np.array([_FC])
        assert analytic_psd("BPSK(1)", f, _FC)[0] == pytest.approx(1 / 1.023e6, rel=1e-3)
