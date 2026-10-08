"""Contrato: save_regime_history acepta effective_date explicito.

Regresion 2026-10-08: cuando FRED publica con fecha natural y el
pipeline resuelve effective_date a una sesion NYSE anterior, la
fila del historico de regimenes debe llevar la fecha efectiva, no
la fecha natural. Con effective_date=None mantiene el
comportamiento legacy (df_macro_manual.date) por retrocompatibilidad.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.pipeline import finalize as fin


def _setup_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "history").mkdir(parents=True, exist_ok=True)


def _run(tmp_path, monkeypatch, df_macro, effective_date):
    _setup_tmp(tmp_path, monkeypatch)
    with patch("src.pipeline.finalize.is_market_day", return_value=True):
        fin.save_regime_history(
            macro_score=pd.Series([0.0, -0.1]),
            macro_regime="MIXED",
            macro_conf=0.5,
            liquidity_regime="ESTRECHA",
            vol_regime="NORMAL",
            sector_results={"regime": "NARROW RALLY", "ranking": []},
            df_macro_manual=df_macro,
            effective_date=effective_date,
        )
    p = tmp_path / "outputs" / "history" / "macro_regime.csv"
    return pd.read_csv(p) if p.exists() else None


def test_effective_date_prevalece_sobre_macro(tmp_path, monkeypatch):
    df_macro = pd.DataFrame({"date": [pd.Timestamp("2026-10-08")]})
    df = _run(tmp_path, monkeypatch, df_macro, effective_date="2026-10-07")
    assert df is not None
    assert pd.Timestamp(df.iloc[0]["date"]) == pd.Timestamp("2026-10-07")


def test_effective_date_none_usa_macro(tmp_path, monkeypatch):
    df_macro = pd.DataFrame({"date": [pd.Timestamp("2026-09-11")]})
    df = _run(tmp_path, monkeypatch, df_macro, effective_date=None)
    assert df is not None
    assert pd.Timestamp(df.iloc[0]["date"]) == pd.Timestamp("2026-09-11")


def test_effective_date_no_bursatil_skip(tmp_path, monkeypatch):
    _setup_tmp(tmp_path, monkeypatch)
    df_macro = pd.DataFrame({"date": [pd.Timestamp("2026-09-11")]})
    with patch("src.pipeline.finalize.is_market_day", return_value=False):
        fin.save_regime_history(
            macro_score=pd.Series([0.0, -0.1]),
            macro_regime="MIXED",
            macro_conf=0.5,
            liquidity_regime="ESTRECHA",
            vol_regime="NORMAL",
            sector_results={"regime": "NARROW RALLY", "ranking": []},
            df_macro_manual=df_macro,
            effective_date="2026-10-10",  # sabado
        )
    p = tmp_path / "outputs" / "history" / "macro_regime.csv"
    assert not p.exists()
