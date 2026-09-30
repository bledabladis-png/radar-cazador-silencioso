# -*- coding: utf-8 -*-
"""Tests de regresion para D1: script QQQ returns generalizado a N tickers.

Contexto: el script original calculaba solo QQQ. Ahora calcula QQQ + SPY
para dar contexto de benchmark en el reporte. El render ya itera sobre
el CSV, asi que el fix es agnostico al numero de filas.
"""

import pandas as pd
import pytest

from scripts import qqq_returns_yahoo as mod


def _make_prices(days=2600, start=100.0, growth=0.0003):
    """Serie sintetica de precios con tendencia positiva."""
    idx = pd.date_range("2015-01-01", periods=days, freq="B")
    prices = pd.Series([start * (1 + growth) ** i for i in range(days)], index=idx)
    return prices


def test_d1_tickers_lista_2():
    """TICKERS debe tener QQQ y SPY."""
    assert len(mod.TICKERS) >= 2
    tickers = [t for t, _ in mod.TICKERS]
    assert "QQQ" in tickers
    assert "SPY" in tickers


def test_d1_calculate_returns_acepta_display_label():
    """displayLabel debe ser el parametro, no hardcoded."""
    prices = _make_prices()
    r = mod.calculate_returns(prices, "TEST LABEL")
    assert r["displayLabel"] == "TEST LABEL"


def test_d1_save_csv_con_dos_filas(tmp_path, monkeypatch):
    """save_csv debe escribir N filas."""
    out = tmp_path / "test.csv"
    monkeypatch.setattr(mod, "OUTPUT", out)
    rows = [
        {"ytd": 20.0, "y1": 25.0, "y3": 100.0, "y5": 100.0, "y10": 500.0,
         "inception": 1500.0, "label": "marketPrice",
         "displayLabel": "QQQ (Yahoo Finance)", "effectiveDate": "2026-09-24",
         "as_of_date": "2026-09-24 14:47:22", "performancePeriod": "daily"},
        {"ytd": 15.0, "y1": 20.0, "y3": 80.0, "y5": 80.0, "y10": 400.0,
         "inception": 1200.0, "label": "marketPrice",
         "displayLabel": "SPY (Yahoo Finance)", "effectiveDate": "2026-09-24",
         "as_of_date": "2026-09-24 14:47:22", "performancePeriod": "daily"},
    ]
    mod.save_csv(rows)
    df = pd.read_csv(out)
    assert len(df) == 2
    assert set(df["displayLabel"]) == {"QQQ (Yahoo Finance)", "SPY (Yahoo Finance)"}


def test_d1_calculate_returns_estructura():
    """El dict devuelto debe tener las 11 claves esperadas."""
    prices = _make_prices()
    r = mod.calculate_returns(prices, "X")
    keys = {"ytd", "y1", "y3", "y5", "y10", "inception", "label",
            "displayLabel", "effectiveDate", "as_of_date", "performancePeriod"}
    assert keys.issubset(r.keys())


# --- get_adjusted_prices (huecos 29-36) ---

def test_get_adjusted_prices_vacio(monkeypatch):
    def fake_download(*a, **k):
        return pd.DataFrame()
    monkeypatch.setattr(mod.yf, "download", fake_download)
    with pytest.raises(RuntimeError, match="No se pudieron descargar"):
        mod.get_adjusted_prices("QQQ")


def test_get_adjusted_prices_historial_corto(monkeypatch):
    def fake_download(*a, **k):
        idx = pd.date_range("2026-01-01", periods=100, freq="B")
        return pd.DataFrame({"Close": [100.0] * 100}, index=idx)
    monkeypatch.setattr(mod.yf, "download", fake_download)
    with pytest.raises(RuntimeError, match="Historial insuficiente"):
        mod.get_adjusted_prices("QQQ")


def test_get_adjusted_prices_happy(monkeypatch):
    idx = pd.date_range("2015-01-01", periods=2600, freq="B")
    def fake_download(*a, **k):
        return pd.DataFrame({"Close": [100.0 + i * 0.1 for i in range(2600)]}, index=idx)
    monkeypatch.setattr(mod.yf, "download", fake_download)
    s = mod.get_adjusted_prices("QQQ")
    assert len(s) == 2600
    assert isinstance(s, pd.Series)

# --- calculate_returns (huecos 47, 53) ---

def test_calculate_returns_ytd_nan_sin_prev_year():
    """Serie confinada al año actual -> prev_year_prices vacio -> ytd nan."""
    idx = pd.date_range("2026-01-02", "2026-12-31", freq="B")
    prices = pd.Series([100.0 + i * 0.1 for i in range(len(idx))], index=idx)
    r = mod.calculate_returns(prices, "X")
    import math
    assert math.isnan(r["ytd"])


def test_calculate_returns_period_return_nan_serie_corta():
    """Serie de 300 filas: y1 (252) OK, y3/y5/y10 nan."""
    idx = pd.date_range("2026-01-02", periods=300, freq="B")
    prices = pd.Series([100.0 + i * 0.1 for i in range(300)], index=idx)
    r = mod.calculate_returns(prices, "X")
    import math
    assert not math.isnan(r["y1"])
    assert math.isnan(r["y3"])
    assert math.isnan(r["y5"])
    assert math.isnan(r["y10"])

# --- main (huecos 84-94) ---

def test_main_happy(monkeypatch, tmp_path):
    idx = pd.date_range("2015-01-01", periods=2600, freq="B")
    prices = pd.Series([100.0 + i * 0.1 for i in range(2600)], index=idx)
    monkeypatch.setattr(mod, "get_adjusted_prices", lambda t, reference_date=None: prices)
    out = tmp_path / "test_main.csv"
    monkeypatch.setattr(mod, "OUTPUT", out)
    mod.main()
    assert out.exists()
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(df) == len(mod.TICKERS)
    assert "effectiveDate" in df.columns