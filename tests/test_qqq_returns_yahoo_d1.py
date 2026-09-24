# -*- coding: utf-8 -*-
"""Tests de regresion para D1: script QQQ returns generalizado a N tickers.

Contexto: el script original calculaba solo QQQ. Ahora calcula QQQ + SPY
para dar contexto de benchmark en el reporte. El render ya itera sobre
el CSV, asi que el fix es agnostico al numero de filas.
"""

import sys
from pathlib import Path

import pandas as pd

# Importar el script como modulo (vive en scripts/, no en src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "qqq_returns_yahoo",
    str(Path(__file__).resolve().parents[1] / "scripts" / "qqq_returns_yahoo.py"),
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


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