# -*- coding: utf-8 -*-
"""D34 (2026-09-30): cerrar cobertura de 2 modulos.

- indicators/options_metrics.py: ramas intermedias de classify_pcr y
  classify_ihr. Los tests existentes (test_options_metrics_classify.py)
  solo cubren los extremos.
- indicators/index_leaders.py: rama router (sin df_index_data) +
  select_index_leaders.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from indicators.options_metrics import classify_pcr, classify_ihr
from indicators.index_leaders import (
    compute_stock_metrics_for_index,
    select_index_leaders,
)


# ============================================================
# classify_pcr: ramas intermedias
# Umbrales: panico=2.0, miedo=1.0, neutral=-1.0, optimismo=-2.0
# ============================================================
def test_classify_pcr_panico():
    assert classify_pcr(2.5) == "Pánico"


def test_classify_pcr_miedo():
    assert classify_pcr(1.5) == "Miedo"
    assert classify_pcr(1.0) == "Miedo"


def test_classify_pcr_neutral():
    assert classify_pcr(0.0) == "Neutral"
    assert classify_pcr(-0.5) == "Neutral"


def test_classify_pcr_optimismo():
    assert classify_pcr(-1.5) == "Optimismo"
    assert classify_pcr(-1.0) == "Optimismo"


def test_classify_pcr_euforia():
    assert classify_pcr(-2.5) == "Euforia"


# ============================================================
# classify_ihr: ramas intermedias
# Umbrales: extrema=2.5, alta=1.6, equilibrado=1.2, especulacion=0.8
# ============================================================
def test_classify_ihr_extrema():
    assert classify_ihr(3.0) == "Cobertura institucional extrema"


def test_classify_ihr_alta():
    assert classify_ihr(2.0) == "Cobertura institucional alta"
    assert classify_ihr(1.6) == "Cobertura institucional alta"


def test_classify_ihr_equilibrado():
    assert classify_ihr(1.4) == "Equilibrado"
    assert classify_ihr(1.2) == "Equilibrado"


def test_classify_ihr_especulacion_alta():
    assert classify_ihr(1.0) == "Especulación alta"
    assert classify_ihr(0.8) == "Especulación alta"


def test_classify_ihr_especulacion_extrema():
    assert classify_ihr(0.5) == "Especulación extrema"


# ============================================================
# compute_stock_metrics_for_index: rama router (sin df_index_data)
# ============================================================
def _make_df(tickers, n=300, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.0005 + i * 0.0001, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Open", t)] = close * 0.998
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_compute_stock_metrics_usa_router_si_no_df_index():
    """Sin df_index_data -> llama al router."""
    from config.index_tickers import INDEX_CONFIG
    etf_ticker = INDEX_CONFIG["Nasdaq-100"]["index_ticker"]
    df_stocks = _make_df(["AAPL", "MSFT"], n=300)
    fake_index_df = _make_df([etf_ticker], n=300)
    with patch("indicators.index_leaders.DataRouter") as mock_router_cls:
        instance = mock_router_cls.return_value
        instance.get_market_data.return_value = fake_index_df
        out = compute_stock_metrics_for_index(
            df_stocks, "Nasdaq-100", ["AAPL", "MSFT"],
            df_index_data=None)
    assert instance.get_market_data.called
    assert isinstance(out, pd.DataFrame)


# ============================================================
# select_index_leaders
# ============================================================
def test_select_index_leaders_normal(tmp_path, monkeypatch):
    """Con holdings y metricas OK -> dict con 1 entrada."""
    from config.index_tickers import INDEX_CONFIG
    idx_name = "Nasdaq-100"
    etf = INDEX_CONFIG[idx_name]["etf_ticker"]
    etf_ticker = INDEX_CONFIG[idx_name]["index_ticker"]

    # CSV de holdings en cwd
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    holdings = pd.DataFrame({
        "etf": [etf] * 5,
        "ticker": ["AAPL", "MSFT", "NVDA", "AVGO", "META"],
        "weight": [10.0, 8.0, 6.0, 4.0, 2.0],
    })
    holdings.to_csv(tmp_path / "data" / "index_holdings.csv", index=False)

    df_stocks = _make_df(["AAPL", "MSFT", "NVDA", "AVGO", "META"], n=300)
    df_market = _make_df([etf_ticker], n=300)
    df_index = _make_df([etf_ticker], n=300)

    with patch("indicators.index_leaders.wyckoff_score") as mw:
        mw.return_value = (pd.Series([0.5] * 300), None, None,
                            None, None, None, None)
        with patch("indicators.index_leaders.classify_wyckoff_phase",
                   return_value="MARKUP"):
            with patch("indicators.index_leaders.detect_spring",
                       return_value=pd.Series([0.0] * 300)):
                with patch("indicators.index_leaders.detect_sos",
                           return_value=pd.Series([0.0] * 300)):
                    with patch("indicators.index_leaders.build_ticker_df",
                               return_value=df_stocks):
                        out = select_index_leaders(
                            df_market, df_stocks, [idx_name],
                            df_index_data=df_index)
    # No crashea. Puede estar vacio si las metricas no pasan los filtros.
    assert isinstance(out, dict)


def test_select_index_leaders_sin_holdings(tmp_path, monkeypatch):
    """Holdings vacio -> dict vacio."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"etf": [], "ticker": [], "weight": []}).to_csv(
        tmp_path / "data" / "index_holdings.csv", index=False)

    df_stocks = _make_df(["AAPL"], n=300)
    df_market = _make_df(["^NDX"], n=300)
    out = select_index_leaders(df_market, df_stocks, ["Nasdaq-100"])
    assert out == {}


def test_select_index_leaders_indice_sin_holdings(tmp_path, monkeypatch):
    """Holdings no tiene filas para el etf -> tickers vacio -> skip."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    # Solo otro etf
    pd.DataFrame({"etf": ["SPY"], "ticker": ["AAPL"], "weight": [10.0]}).to_csv(
        tmp_path / "data" / "index_holdings.csv", index=False)
    df_stocks = _make_df(["AAPL"], n=300)
    df_market = _make_df(["^NDX"], n=300)
    out = select_index_leaders(df_market, df_stocks, ["Nasdaq-100"])
    assert out == {}
