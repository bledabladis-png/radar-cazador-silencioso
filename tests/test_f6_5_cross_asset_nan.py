"""F6-5 (2026-09-28): ratio y z-score coherentes cuando el ultimo Close
es NaN.

F6-23: antes ratio_series conservaba NaN final y robust_zscore (con
ffill interno) podia dar z-score valido sobre el mismo periodo. Fix:
dropna() antes de leer iloc[-1] y de calcular z-score.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.cross_asset import compute_cross_asset_ratios


def _make_market(n=80, last_hg_nan=False, last_gc_nan=False):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["HG=F", "GC=F"]])
    df = pd.DataFrame(index=dates, columns=cols, dtype=float)
    df[("Close", "HG=F")] = np.linspace(4.0, 5.0, n)
    df[("Close", "GC=F")] = np.linspace(1900.0, 2000.0, n)
    if last_hg_nan:
        df.iloc[-1, df.columns.get_loc(("Close", "HG=F"))] = np.nan
    if last_gc_nan:
        df.iloc[-1, df.columns.get_loc(("Close", "GC=F"))] = np.nan
    return df


def test_copper_gold_sin_nan_ratio_valido():
    df = _make_market()
    out = compute_cross_asset_ratios(df)
    assert out["copper_gold"] is not None
    assert np.isfinite(out["copper_gold"])
    assert out["copper_gold"] > 0


def test_copper_gold_con_ultimo_nan_usa_ultimo_valido():
    """F6-23: ultimo Close NaN -> ratio salta al ultimo valido."""
    df = _make_market(last_hg_nan=True)
    out = compute_cross_asset_ratios(df)
    assert out["copper_gold"] is not None
    assert np.isfinite(out["copper_gold"])
    # Ratio debe coincidir con el penultimo (no NaN)
    expected = df[("Close", "HG=F")].iloc[-2] / df[("Close", "GC=F")].iloc[-2]
    assert abs(out["copper_gold"] - expected) < 1e-9


def test_copper_gold_ratio_y_zscore_coherentes():
    """F6-23: si ratio es valido, z-score tambien lo es (misma serie)."""
    df = _make_market(last_hg_nan=True)
    out = compute_cross_asset_ratios(df)
    assert out["copper_gold"] is not None
    assert out["copper_gold_zscore"] is not None
    assert np.isfinite(out["copper_gold_zscore"])


def test_serie_vacia_devuelve_none():
    dates = pd.date_range("2026-01-01", periods=5, freq="D")
    cols = pd.MultiIndex.from_product([["Close"], ["HG=F", "GC=F"]])
    df = pd.DataFrame(index=dates, columns=cols, dtype=float)
    df[("Close", "HG=F")] = np.nan
    df[("Close", "GC=F")] = np.nan
    out = compute_cross_asset_ratios(df)
    assert out["copper_gold"] is None
    assert out["copper_gold_zscore"] is None