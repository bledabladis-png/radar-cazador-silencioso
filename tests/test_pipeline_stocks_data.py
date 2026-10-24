# -*- coding: utf-8 -*-
"""Tests de src.pipeline.stocks_data."""
import pandas as pd
from unittest.mock import patch

from src.pipeline import stocks_data as sd


def _make_df(n=5, n_valid_close=2):
    idx = pd.date_range("2026-09-20", periods=n, freq="D")
    cols = []
    for i in range(n_valid_close + 2):
        for campo in ["Open", "High", "Low", "Close", "Volume"]:
            cols.append((campo, "T{}".format(i)))
    df = pd.DataFrame([[1.0] * len(cols)] * n, index=idx)
    df.columns = pd.MultiIndex.from_tuples(cols)
    return df


def test_load_stocks_data_sin_datos(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with patch("src.pipeline.stocks_data.download_stock_prices", return_value=None):
        out = sd.load_stocks_data()
    assert out["df_stocks"] is None
    assert out["df_stocks_effective_meta"] is None
    assert out["holdings_df"] is None


def test_load_stocks_data_carga_holdings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,AAA,10.0\n", encoding="utf-8")
    fake_eff = {
        "status": "INSUFFICIENT_COVERAGE", "date": None,
        "requested_date": pd.Timestamp("2026-09-25"),
        "lag_days": 0, "coverage": 0.5,
        "n_observed": 1, "n_eligible": 2,
    }
    df = _make_df()
    with patch("src.pipeline.stocks_data.download_stock_prices", return_value=df), \
         patch("src.pipeline.stocks_data.resolve_effective_date", return_value=fake_eff):
        out = sd.load_stocks_data()
    assert out["df_stocks"] is None
    assert out["holdings_df"] is not None
    assert not out["holdings_df"].empty


def test_load_stocks_data_holiday_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,AAA,10.0\n", encoding="utf-8")
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    cols = [("Close", "AAA"), ("Close", "BBB"), ("Close", "CCC"), ("Close", "DDD")]
    df = pd.DataFrame([[1.0]*4]*5, index=idx)
    df.columns = pd.MultiIndex.from_tuples(cols)
    df.iloc[-1, :] = float("nan")
    fake_eff = {
        "status": "OK", "date": idx[-1],
        "requested_date": idx[-1],
        "lag_days": 0, "coverage": 1.0,
        "n_observed": 4, "n_eligible": 4,
    }
    with patch("src.pipeline.stocks_data.download_stock_prices", return_value=df), \
         patch("src.pipeline.stocks_data.resolve_effective_date", return_value=fake_eff):
        out = sd.load_stocks_data()
    assert out["df_stocks"] is None
    assert out["holdings_df"] is not None


def test_load_stocks_data_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,AAA,10.0\n", encoding="utf-8")
    idx = pd.date_range("2026-09-20", periods=5, freq="D")
    cols = [("Close", "AAA"), ("Close", "BBB")]
    df = pd.DataFrame([[1.0]*2]*5, index=idx)
    df.columns = pd.MultiIndex.from_tuples(cols)
    fake_eff = {
        "status": "OK", "date": idx[-1],
        "requested_date": idx[-1],
        "lag_days": 0, "coverage": 1.0,
        "n_observed": 2, "n_eligible": 2,
    }
    with patch("src.pipeline.stocks_data.download_stock_prices", return_value=df), \
         patch("src.pipeline.stocks_data.resolve_effective_date", return_value=fake_eff):
        out = sd.load_stocks_data()
    assert out["df_stocks"] is not None
    assert out["df_stocks_effective_meta"] is not None
    assert out["holdings_df"] is not None


def test_load_stocks_data_insufficient_coverage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "etf_holdings.csv").write_text(
        "etf,ticker,weight\nXLK,AAA,10.0\n", encoding="utf-8")
    df = _make_df()
    fake_eff = {
        "status": "INSUFFICIENT_COVERAGE", "date": None,
        "requested_date": pd.Timestamp("2026-09-25"),
        "lag_days": 0, "coverage": 0.5,
        "n_observed": 1, "n_eligible": 2,
    }
    with patch("src.pipeline.stocks_data.download_stock_prices", return_value=df), \
         patch("src.pipeline.stocks_data.resolve_effective_date", return_value=fake_eff):
        out = sd.load_stocks_data()
    assert out["df_stocks"] is None
    assert out["holdings_df"] is not None
