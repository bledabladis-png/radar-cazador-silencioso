# -*- coding: utf-8 -*-
"""Centinela H1: compute_macro_regime tolerante a input vacio o parcial.

Cubre B6-H1 (2026-10-31). Tres casos:
  1. df_market vacio -> NO_DATA + conf 0.0 + score vacia.
  2. df_market con estructura pero NaN -> regimen calculado (MIXED) sin crash.
  3. df_market con datos validos -> regimen concreto + conf numerica.

Y source-level: los 3 guards presentes (upstream, downstream, tolerancia).
"""

import pathlib

import numpy as np
import pandas as pd

from regimes.macro_regime import compute_macro_regime


ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_df_market_vacio_devuelve_no_data():
    score, regime, conf, _ = compute_macro_regime(
        df_market=pd.DataFrame(), df_macro_manual=pd.DataFrame(),
        liquidity_score=0.0, vol_score=0.0,
    )
    assert regime == "NO_DATA"
    assert conf == 0.0
    assert isinstance(score, pd.Series)
    assert len(score) == 0


def test_inputs_none_no_crashea():
    score, regime, conf, _ = compute_macro_regime(
        df_market=None, df_macro_manual=None,
        liquidity_score=0.0, vol_score=0.0,
    )
    assert regime == "NO_DATA"
    assert conf == 0.0
    assert len(score) == 0


def test_df_market_nan_sin_columnas_opcionales_no_crashea():
    idx = pd.date_range("2026-10-01", periods=5)
    cols = pd.MultiIndex.from_tuples([("Close", t) for t in
        ["XLK","XLF","XLV","XLE","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]],
        names=["field","ticker"])
    df_market = pd.DataFrame(np.nan, index=idx, columns=cols)
    df_macro = pd.DataFrame({"date": pd.to_datetime([]), "value": pd.Series([], dtype=float)})
    score, regime, conf, _ = compute_macro_regime(df_market, df_macro, 0.0, 0.0)
    assert regime == "MIXED"
    assert len(score) == 5
    assert isinstance(conf, float)


def test_df_market_con_datos_validos_calcula_regimen():
    idx = pd.date_range("2026-06-01", periods=120)
    rng = np.random.default_rng(0)
    tickers = ["^GSPC","^NDX","^RUT","^STOXX50E","EEM","EWJ","BIL","IEF","TLT","HYG","LQD","^VIX","^VIX3M","DX-Y.NYB","GC=F","HG=F","CL=F","NG=F","XLK","XLF","XLV","XLE","XLY","XLP","XLI","XLB","XLU","XLRE","XLC"]
    cols = pd.MultiIndex.from_tuples([("Close", t) for t in tickers], names=["field","ticker"])
    base = {t: 100.0 for t in tickers}
    base.update({"^VIX": 15.0, "^VIX3M": 16.0, "BIL": 91.0, "IEF": 95.0, "TLT": 100.0})
    data = np.array([base[t] + np.cumsum(rng.standard_normal(120)*0.3) for t in tickers]).T
    df_market = pd.DataFrame(data, index=idx, columns=cols)
    df_macro = pd.DataFrame({"date": pd.to_datetime(["2026-06-01","2026-07-01","2026-08-01"]), "value": [0.1, 0.15, 0.2]})
    score, regime, conf, _ = compute_macro_regime(df_market, df_macro, 0.1, 0.1)
    assert regime != "NO_DATA"
    assert 0.0 <= conf <= 1.0
    assert len(score) == 120


def test_guards_presentes_en_source():
    src = (ROOT / "regimes" / "macro_regime.py").read_text(encoding="utf-8")
    assert "if df_market is None:" in src
    assert "len(all_signals) == 0" in src
    assert "len(macro_score) == 0" in src
    assert "def _last_sig" in src
    assert "NO_DATA" in src
