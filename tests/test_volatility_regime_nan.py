# -*- coding: utf-8 -*-
"""F3-05-septies (2026-09-30): volatility_regime no propaga NaN.

Bug: vix_returns usa pct_change(fill_method=None). Cada festivo
NYSE produce 2 NaN consecutivos (festivo + dia siguiente). Con
~90 festivos en 10 anios, rolling(20).std() contamina ~80% del
z. Verificado 2026-09-30 sobre data/market_data.parquet:
2102/2605 NaN en el raw vs 521/2424 con dropna.

Fix: dropna() antes de los rolling.

Impacto historico: la version limpia detecta 261 dias STRESS
vs 81 en la contaminada. La senal de estrés real se enmascaraba.
"""
import numpy as np
import pandas as pd

from indicators.volatility import volatility_regime


def _make_returns_with_holiday_gaps(n=2000, seed=42):
    """Serie de retornos con NaN en patron festivo: 2 consecutivos
    cada ~20 sesiones (~1 festivo al mes).
    """
    rng = np.random.RandomState(seed)
    r = pd.Series(rng.normal(0, 0.015, n),
                  index=pd.date_range("2016-01-01", periods=n, freq="B"))
    for i in range(50, n, 20):
        r.iloc[i] = np.nan
        if i + 1 < n:
            r.iloc[i + 1] = np.nan
    return r


def test_volatility_regime_no_propaga_nan_de_festivo():
    """Con dropna, z no debe tener NaN por contaminacion de rolling
    en la cola (mas alla del warmup del baseline).
    """
    r = _make_returns_with_holiday_gaps(n=2000)
    z = volatility_regime(r)
    # El ultimo valor debe ser valido (no NaN por contaminacion)
    assert pd.notna(z.iloc[-1]), (
        f"z.iloc[-1] es NaN. tail(5) NaN={z.tail(5).isna().sum()}"
    )
    # NOTA: no se exige ausencia total de NaN. El baseline usa
    # rolling(VOLATILITY_BASELINE_WINDOW, min_periods=252), cuyo warmup
    # es ~756 filas. NaN en esas posiciones es por diseno, no por bug.


def test_volatility_regime_dropna_equivalente():
    """Verificar que volatility_regime(r) == volatility_regime(r.dropna())."""
    r = _make_returns_with_holiday_gaps(n=2000)
    z1 = volatility_regime(r)
    z2 = volatility_regime(r.dropna())
    assert len(z1) == len(z2)
    assert z1.equals(z2)


def test_volatility_regime_serie_sin_nan_igual():
    """Control: sin NaN, el comportamiento es el mismo."""
    rng = np.random.RandomState(42)
    r = pd.Series(rng.normal(0, 0.015, 2000),
                  index=pd.date_range("2016-01-01", periods=2000, freq="B"))
    z = volatility_regime(r)
    assert len(z) == 2000
    assert pd.notna(z.iloc[-1])


def test_volatility_regime_serie_vacia():
    """Serie vacia -> z vacio."""
    z = volatility_regime(pd.Series(dtype=float))
    assert z.empty
