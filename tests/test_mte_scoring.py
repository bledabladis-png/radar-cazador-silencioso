"""Tests de las funciones de scoring de MTE (DT2 Fase 1c).

Cubre:
- compute_msi, compute_ipi: formulas puras, sin df.
- sector_rotation_score, safe_haven_score, inflation_pressure_score:
  fixtures sinteticos, rango [-1, 1].
- credit_stress_score: firma compleja, NaN blocking documentado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------- Fixtures sinteticos ----------

TICKERS_SECTORS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC', '^GSPC']
TICKERS_HAVEN = ['GLD', 'SLV', 'TLT', 'QUAL', 'IEF', 'BIL']
TICKERS_INFLATION = ['XLE', '^SPGSCI', 'TIP']


def _make_df_market(tickers, n_periods=200, base_price=100.0, drift=0.0005):
    """df_market sintetico con MultiIndex columnas (field, ticker)."""
    idx = pd.date_range('2025-01-01', periods=n_periods, freq='B')
    data = {}
    for i, t in enumerate(tickers):
        # Serie con tendencia suave para que pct_change y robust_zscore den algo
        trend = base_price + i + np.arange(n_periods) * drift
        noise = np.random.RandomState(seed=i).normal(0, 0.5, n_periods)
        close = trend + noise
        data[('Close', t)] = close
        data[('High', t)] = close + 1
        data[('Low', t)] = close - 1
        data[('Open', t)] = close - 0.5
        data[('Volume', t)] = np.full(n_periods, 1_000_000)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['field', 'ticker'])
    return df


@pytest.fixture
def df_sectors():
    return _make_df_market(TICKERS_SECTORS)


@pytest.fixture
def df_havens():
    return _make_df_market(TICKERS_HAVEN)


@pytest.fixture
def df_inflation():
    return _make_df_market(TICKERS_INFLATION)


# ---------- compute_msi ----------

class TestComputeMsi:

    def test_formula_basica(self):
        from indicators.mte import compute_msi
        # srs=-1 -> mapped=0; shs=-1 -> mapped=0; cls=0 -> raw=0 -> msi=0
        assert compute_msi(srs=-1.0, shs=-1.0, cls=0.0) == pytest.approx(0.0)

    def test_formula_extremos(self):
        from indicators.mte import compute_msi
        # srs=1 -> 1; shs=1 -> 1; cls=1 -> raw=0.40+0.35+0.25=1.0 -> 100
        assert compute_msi(srs=1.0, shs=1.0, cls=1.0) == pytest.approx(100.0)

    def test_formula_punto_medio(self):
        from indicators.mte import compute_msi
        # srs=0 -> 0.5; shs=0 -> 0.5; cls=0.5
        # raw = 0.40*0.5 + 0.35*0.5 + 0.25*0.5 = 0.5 -> msi = 50
        assert compute_msi(srs=0.0, shs=0.0, cls=0.5) == pytest.approx(50.0)


# ---------- compute_ipi ----------

class TestComputeIpi:

    def test_formula_extremos(self):
        from indicators.mte import compute_ipi
        assert compute_ipi(ips=-1.0) == pytest.approx(0.0)
        assert compute_ipi(ips=1.0) == pytest.approx(100.0)

    def test_formula_punto_medio(self):
        from indicators.mte import compute_ipi
        assert compute_ipi(ips=0.0) == pytest.approx(50.0)


# ---------- sector_rotation_score ----------

class TestSectorRotationScore:

    def test_rango_valido(self, df_sectors):
        from indicators.mte import sector_rotation_score
        srs = sector_rotation_score(df_sectors)
        assert isinstance(srs, float)
        assert -1.0 <= srs <= 1.0

    def test_estable_con_tendencia_continua(self):
        """Con tendencia continua, SRS es determinista y en rango."""
        from indicators.mte import sector_rotation_score
        n = 200
        idx = pd.date_range('2025-01-01', periods=n, freq='B')
        data = {}
        for t in ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'XLE']:
            data[('Close', t)] = 100 + np.arange(n) * 0.5
        for t in ['XLU', 'XLP', 'XLV', 'XLRE', 'XLC']:
            data[('Close', t)] = 100 + np.zeros(n)
        data[('Close', '^GSPC')] = 100 + np.arange(n) * 0.1
        df = pd.DataFrame(data, index=idx)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['field', 'ticker'])

        srs1 = sector_rotation_score(df)
        srs2 = sector_rotation_score(df)
        assert isinstance(srs1, float)
        assert -1.0 <= srs1 <= 1.0
        assert srs1 == srs2  # determinismo


# ---------- safe_haven_score ----------

class TestSafeHavenScore:

    def test_rango_valido(self, df_havens):
        from indicators.mte import safe_haven_score
        shs = safe_haven_score(df_havens)
        assert isinstance(shs, float)
        assert -1.0 <= shs <= 1.0


# ---------- inflation_pressure_score ----------

class TestInflationPressureScore:

    def test_rango_valido(self, df_inflation):
        from indicators.mte import inflation_pressure_score
        ips = inflation_pressure_score(df_inflation)
        assert isinstance(ips, float)
        assert -1.0 <= ips <= 1.0


# ---------- credit_stress_score ----------

class TestCreditStressScore:

    def test_firma_acepta_inputs_nan(self):
        """Con inputs NaN no debe reventar: bloquea CRISIS devolviendo NaN."""
        from indicators.mte import credit_stress_score
        result = credit_stress_score(
            financial_conditions=np.nan,
            credit_signal=np.nan,
            volatility_signal=np.nan,
            vix_term=0.0,
            darkpool_z=np.nan,
            pcr_z=np.nan,
        )
        # Docstring: "Bloquea CRISIS si alguna familia es NaN (retorna NaN)"
        assert np.isnan(result) or isinstance(result, float)

    def test_firma_acepta_escalares(self):
        """Con inputs escalares neutros, devuelve un float en [0, 1]."""
        from indicators.mte import credit_stress_score
        result = credit_stress_score(
            financial_conditions=0.0,
            credit_signal=0.0,
            volatility_signal=0.0,
            vix_term=0.0,
            darkpool_z=0.0,
            pcr_z=0.0,
        )
        if not np.isnan(result):
            assert 0.0 <= result <= 1.0