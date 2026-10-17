# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa4-commitB: contrato del schema de
macro_regime.csv.

Tras el renombrado:
- financial_regime: canonica, regimen financiero.
- real_liq_regime: canonica, liquidez real FRED.
- liquidity_regime: alias legacy de financial_regime (mismo valor).
"""
from unittest.mock import patch

import pandas as pd

from src.pipeline import finalize as fin


def _setup_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True, exist_ok=True)


def _call(tmp_path, monkeypatch, **kwargs):
    _setup_tmp(tmp_path, monkeypatch)
    defaults = {
        "macro_score": pd.Series([0.0, -0.1]),
        "macro_regime": "MIXED",
        "macro_conf": 0.5,
        "financial_regime": "HIGH_STRESS",
        "vol_regime": "NORMAL",
        "sector_results": {"regime": "NARROW RALLY", "ranking": []},
        "df_macro_manual": pd.DataFrame({"date": [pd.Timestamp("2026-10-16")]}),
    }
    defaults.update(kwargs)
    with patch("src.pipeline.finalize.is_market_day", return_value=True):
        fin.save_regime_history(**defaults)
    p = tmp_path / "outputs" / "history" / "macro_regime.csv"
    return pd.read_csv(p) if p.exists() else None


def test_schema_tiene_3_columnas_de_regimen(tmp_path, monkeypatch):
    df = _call(tmp_path, monkeypatch)
    assert df is not None
    for col in ("financial_regime", "real_liq_regime", "liquidity_regime"):
        assert col in df.columns, "Falta columna {}".format(col)


def test_financial_regime_canonica(tmp_path, monkeypatch):
    df = _call(tmp_path, monkeypatch, financial_regime="EXTREME_STRESS")
    assert df.iloc[0]["financial_regime"] == "EXTREME_STRESS"


def test_liquidity_regime_es_alias_de_financial(tmp_path, monkeypatch):
    df = _call(tmp_path, monkeypatch, financial_regime="ESTRECHA")
    assert df.iloc[0]["liquidity_regime"] == "ESTRECHA"
    assert df.iloc[0]["financial_regime"] == "ESTRECHA"


def test_real_liq_regime_se_persiste_si_se_pasa(tmp_path, monkeypatch):
    df = _call(tmp_path, monkeypatch, real_liq_regime="ABUNDANTE")
    assert df.iloc[0]["real_liq_regime"] == "ABUNDANTE"


def test_real_liq_regime_nan_si_no_se_pasa(tmp_path, monkeypatch):
    df = _call(tmp_path, monkeypatch)
    assert pd.isna(df.iloc[0]["real_liq_regime"])


def test_financial_y_real_liq_pueden_diferir(tmp_path, monkeypatch):
    df = _call(
        tmp_path, monkeypatch,
        financial_regime="HIGH_STRESS", real_liq_regime="ABUNDANTE",
    )
    assert df.iloc[0]["financial_regime"] == "HIGH_STRESS"
    assert df.iloc[0]["real_liq_regime"] == "ABUNDANTE"
    assert df.iloc[0]["liquidity_regime"] == "HIGH_STRESS"
