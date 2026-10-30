# -*- coding: utf-8 -*-
"""Tests de check_ticker_health (F4).

Contrato: detecta tickers ACTIVE con datos stale no clasificados en
ticker_health.yaml. Los clasificados (WBD DELISTED) se ignoran. WARN,
no FAIL, para no bloquear el pipeline: el canal al operador es issue
automatico via _update_issue -> clasificar en yaml -> siguiente run OK.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.health_check import (
    check_ticker_health, OK, WARN, SKIP, FAIL, HEALTH_STALE_SESSIONS,
)
from src import ticker_health as th


@pytest.fixture(autouse=True)
def _reset_cache():
    th._reset_cache()
    yield
    th._reset_cache()


def _df(tickers_last_dates):
    """Construye un df sintetico con index de HEALTH_STALE_SESSIONS+5 sesiones."""
    n = HEALTH_STALE_SESSIONS + 5
    idx = pd.date_range("2026-10-01", periods=n, freq="B")
    data = {}
    for t, last in tickers_last_dates.items():
        col = [float("nan")] * n
        # llenar desde 0 hasta el indice de 'last' (inclusive)
        pos = idx.get_indexer([pd.Timestamp(last)])[0]
        assert pos >= 0, f"{t}: {last} fuera del index"
        for i in range(pos + 1):
            col[i] = 100.0 + i
        data[("Close", t)] = col
    df = pd.DataFrame(data, index=idx)
    return df


def test_ok_si_todos_recientes():
    idx = pd.date_range("2026-10-01", periods=HEALTH_STALE_SESSIONS + 5, freq="B")
    last = idx[-1]
    df = _df({"AAPL": last, "MSFT": last})
    res = check_ticker_health(df)
    assert len(res) == 1
    assert res[0].status == OK


def test_warn_si_ticker_stale_no_clasificado():
    idx = pd.date_range("2026-10-01", periods=HEALTH_STALE_SESSIONS + 5, freq="B")
    # AAPL reciente, MSFT stale hace 20 sesiones (fuera del cutoff idx[-10])
    last_aapl = idx[-1]
    last_msft = idx[-HEALTH_STALE_SESSIONS - 1] if HEALTH_STALE_SESSIONS + 1 <= len(idx) else idx[0]
    df = _df({"AAPL": last_aapl, "MSFT": last_msft})
    res = check_ticker_health(df)
    assert len(res) == 1
    assert res[0].status == WARN
    assert "MSFT" in res[0].message
    assert "ticker_health.yaml" in res[0].message


def test_stale_clasificado_no_alerta():
    """WBD (DELISTED) ignorado aunque este stale en el df."""
    idx = pd.date_range("2026-10-01", periods=HEALTH_STALE_SESSIONS + 5, freq="B")
    last_aapl = idx[-1]
    last_wbd = idx[0]
    df = _df({"AAPL": last_aapl, "WBD": last_wbd})
    res = check_ticker_health(df)
    assert len(res) == 1
    assert res[0].status == OK


def test_skip_df_vacio():
    res = check_ticker_health(pd.DataFrame())
    assert len(res) == 1
    assert res[0].status == SKIP


def test_skip_pocas_sesiones():
    df = _df.__wrapped__({"AAPL": "2026-10-03"}) if hasattr(_df, "__wrapped__") else None
    idx = pd.date_range("2026-10-01", periods=3, freq="B")
    df = pd.DataFrame({("Close", "AAPL"): [1.0, 2.0, 3.0]}, index=idx)
    res = check_ticker_health(df)
    assert len(res) == 1
    assert res[0].status == SKIP
    assert "pocas sesiones" in res[0].message


def test_fail_sin_columnas_close():
    idx = pd.date_range("2026-10-01", periods=HEALTH_STALE_SESSIONS + 5, freq="B")
    df = pd.DataFrame({"foo": [1.0] * len(idx)}, index=idx)
    res = check_ticker_health(df)
    assert len(res) == 1
    assert res[0].status == FAIL
    assert "Close" in res[0].message


def test_warn_incluye_hasta_10_y_contador():
    """Con 12 tickers stale, el mensaje lista 10 y anade +2 mas."""
    idx = pd.date_range("2026-10-01", periods=HEALTH_STALE_SESSIONS + 5, freq="B")
    last_ok = idx[-1]
    last_stale = idx[0]
    tickers = {f"T{i:02d}": last_stale for i in range(12)}
    tickers["OK"] = last_ok
    df = _df(tickers)
    res = check_ticker_health(df)
    assert res[0].status == WARN
    assert "+2 mas" in res[0].message
    # 12 stale + "OK" reciente -> 12 stale
    assert "12 tickers ACTIVE stale" in res[0].message