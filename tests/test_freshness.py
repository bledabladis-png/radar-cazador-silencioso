# -*- coding: utf-8 -*-
"""
Tests de frescura de datos.

Capas:
  1. Unit tests de clasificadores (siempre corren, sin I/O).
  2. Integracion sobre datos reales (skipif si el fichero no existe).
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.report.helpers import (
    _classify_freshness,
    _classify_finra_freshness,
    _classify_fred_freshness,
)
from indicators.data_quality import classify_freshness
from src.stock_data_loader import _classify_ticker

BASE = Path(__file__).resolve().parents[1]


# ============================================================
# CAPA 1 - Unit tests de clasificadores
# ============================================================

def test_classify_freshness_boundaries():
    """daily: 3/7/14 dias."""
    assert _classify_freshness(0) == "CURRENT"
    assert _classify_freshness(3) == "CURRENT"
    assert _classify_freshness(4) == "RECENT"
    assert _classify_freshness(7) == "RECENT"
    assert _classify_freshness(8) == "STALE"
    assert _classify_freshness(14) == "STALE"
    assert _classify_freshness(15) == "ARCHIVAL"
    assert _classify_freshness(100) == "ARCHIVAL"


def test_classify_finra_boundaries():
    """finra: 30/45/60 dias (retraso regulatorio)."""
    assert _classify_finra_freshness(0) == "CURRENT"
    assert _classify_finra_freshness(30) == "CURRENT"
    assert _classify_finra_freshness(31) == "RECENT"
    assert _classify_finra_freshness(45) == "RECENT"
    assert _classify_finra_freshness(46) == "STALE"
    assert _classify_finra_freshness(60) == "STALE"
    assert _classify_finra_freshness(61) == "ARCHIVAL"


def test_classify_fred_boundaries():
    """fred: 30/60/90 dias."""
    assert _classify_fred_freshness(0) == "CURRENT"
    assert _classify_fred_freshness(30) == "CURRENT"
    assert _classify_fred_freshness(31) == "RECENT"
    assert _classify_fred_freshness(60) == "RECENT"
    assert _classify_fred_freshness(61) == "STALE"
    assert _classify_fred_freshness(90) == "STALE"
    assert _classify_fred_freshness(91) == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_cftc():
    """CFTC: 45/90/120 dias."""
    assert classify_freshness(0, "cftc") == "CURRENT"
    assert classify_freshness(45, "cftc") == "CURRENT"
    assert classify_freshness(46, "cftc") == "RECENT"
    assert classify_freshness(90, "cftc") == "RECENT"
    assert classify_freshness(91, "cftc") == "STALE"
    assert classify_freshness(120, "cftc") == "STALE"
    assert classify_freshness(121, "cftc") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_sec():
    """SEC: 45/90/120 dias."""
    assert classify_freshness(45, "sec") == "CURRENT"
    assert classify_freshness(90, "sec") == "RECENT"
    assert classify_freshness(120, "sec") == "STALE"
    assert classify_freshness(121, "sec") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_finra():
    """FINRA: 30/45/60 dias."""
    assert classify_freshness(30, "finra") == "CURRENT"
    assert classify_freshness(45, "finra") == "RECENT"
    assert classify_freshness(60, "finra") == "STALE"
    assert classify_freshness(61, "finra") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_fred():
    """FRED: 30/60/90 dias."""
    assert classify_freshness(30, "fred") == "CURRENT"
    assert classify_freshness(60, "fred") == "RECENT"
    assert classify_freshness(90, "fred") == "STALE"
    assert classify_freshness(91, "fred") == "ARCHIVAL"


def test_classify_freshness_por_frecuencia_daily():
    """daily: 3/7/14 dias."""
    assert classify_freshness(3, "daily") == "CURRENT"
    assert classify_freshness(7, "daily") == "RECENT"
    assert classify_freshness(14, "daily") == "STALE"
    assert classify_freshness(15, "daily") == "ARCHIVAL"


def test_classify_freshness_nan():
    """age=NaN -> N/D."""
    assert classify_freshness(float("nan"), "daily") == "N/D"
    assert classify_freshness(pd.NA, "daily") == "N/D"


def test_classify_ticker_failed_casos_vacios():
    """_classify_ticker: df None/vacio, sin Close, todo NaN."""
    assert _classify_ticker("AAPL", None) == "FAILED"
    assert _classify_ticker("AAPL", pd.DataFrame()) == "FAILED"
    assert _classify_ticker("AAPL", pd.DataFrame({"Open": [1, 2]})) == "FAILED"
    empty_close = pd.DataFrame({("Close", "AAPL"): [float("nan"), float("nan")]})
    assert _classify_ticker("AAPL", empty_close) == "FAILED"


def test_classify_ticker_ok_reciente():
    """_classify_ticker: dato de hoy -> OK."""
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=1), hoy],
    )
    assert _classify_ticker("AAPL", df) == "OK"


def test_classify_ticker_partial():
    """_classify_ticker: 10 dias -> PARTIAL."""
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=11), hoy - pd.Timedelta(days=10)],
    )
    assert _classify_ticker("AAPL", df) == "PARTIAL"


def test_classify_ticker_stale():
    """_classify_ticker: 30 dias -> STALE."""
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, 101.0]},
        index=[hoy - pd.Timedelta(days=31), hoy - pd.Timedelta(days=30)],
    )
    assert _classify_ticker("AAPL", df) == "STALE"


def test_classify_ticker_ultimo_nan_failed():
    """_classify_ticker: ultimo valor NaN -> FAILED (hueco >3d tras ffill)."""
    hoy = pd.Timestamp.now().normalize()
    df = pd.DataFrame(
        {("Close", "AAPL"): [100.0, float("nan")]},
        index=[hoy - pd.Timedelta(days=1), hoy],
    )
    assert _classify_ticker("AAPL", df) == "FAILED"


# ============================================================
# CAPA 2 - Integracion sobre datos reales (skipif si no existen)
# ============================================================

MAX_DAILY_AGE = 4   # tolera fin de semana + 1 festivo
MAX_DAILY_AGE_EU = 5
MAX_DQ_AGE = 3


def _age_days(ts):
    return (pd.Timestamp.now() - pd.Timestamp(ts)).days


@pytest.mark.skipif(
    not (BASE / "data" / "market_data.parquet").exists(),
    reason="market_data.parquet no existe (CI fresco)",
)
def test_market_data_fresh():
    df = pd.read_parquet(BASE / "data" / "market_data.parquet")
    assert len(df) > 0, "market_data.parquet vacio"
    last = df.index[-1]
    age = _age_days(last)
    assert age <= MAX_DAILY_AGE, f"market_data stale: {age} dias (max {MAX_DAILY_AGE})"


@pytest.mark.skipif(
    not (BASE / "data" / "stock_prices.parquet").exists(),
    reason="stock_prices.parquet no existe (CI fresco)",
)
def test_stock_prices_fresh():
    df = pd.read_parquet(BASE / "data" / "stock_prices.parquet")
    assert len(df) > 0, "stock_prices.parquet vacio"
    last = df.index[-1]
    age = _age_days(last)
    assert age <= MAX_DAILY_AGE, f"stock_prices stale: {age} dias (max {MAX_DAILY_AGE})"


@pytest.mark.skipif(
    not (BASE / "data" / "stock_prices.parquet").exists(),
    reason="stock_prices.parquet no existe (CI fresco)",
)
def test_european_tickers_recent():
    """Al menos el 80% de los tickers europeos tiene datos recientes."""
    import re
    df = pd.read_parquet(BASE / "data" / "stock_prices.parquet")
    tickers = set(df.columns.get_level_values(1))
    eu_pat = re.compile(r"\.(PA|AS|MI|DE|MC)$")
    eu = [t for t in tickers if eu_pat.search(t)]
    if not eu:
        pytest.skip("sin tickers europeos en stock_prices.parquet")

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=MAX_DAILY_AGE_EU)
    recientes = 0
    for t in eu:
        try:
            series = df[("Close", t)].dropna()
            if len(series) > 0 and series.index[-1] >= cutoff:
                recientes += 1
        except KeyError:
            continue

    ratio = recientes / len(eu)
    assert ratio >= 0.80, f"solo {recientes}/{len(eu)} ({ratio:.0%}) europeos recientes"


@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "data_quality.csv").exists(),
    reason="data_quality.csv no existe (CI fresco)",
)
def test_data_quality_recent():
    """La ultima ejecucion de data_quality debe ser reciente."""
    df = pd.read_csv(BASE / "outputs" / "history" / "data_quality.csv", parse_dates=["date"])
    last = df["date"].max()
    age = _age_days(last)
    assert age <= MAX_DQ_AGE, f"data_quality.csv stale: {age} dias (max {MAX_DQ_AGE})"


@pytest.mark.skipif(
    not (BASE / "outputs" / "history" / "data_quality.csv").exists(),
    reason="data_quality.csv no existe (CI fresco)",
)
def test_data_quality_sin_archival():
    """Ninguna fuente debe estar en ARCHIVAL en la ultima ejecucion."""
    df = pd.read_csv(BASE / "outputs" / "history" / "data_quality.csv", parse_dates=["date"])
    last_date = df["date"].max()
    ultima = df[df["date"] == last_date]
    archival = ultima[ultima["freshness"] == "ARCHIVAL"]
    assert archival.empty, (
        f"fuentes ARCHIVAL en {last_date.date()}: {archival['source'].tolist()}"
    )