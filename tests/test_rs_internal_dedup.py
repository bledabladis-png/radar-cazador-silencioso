# -*- coding: utf-8 -*-
"""Test de regresion O2 (2026-09-18).

Bug: src/pipeline/sector_metrics.py:74 usaba
    append_dedup(hist, new, ["date","sector"])
sobre rs_internal, cuyo writer emite 1 fila por (date, sector, ticker).
El subset colapsaba todas las filas de tickers de un sector en una sola.

Fix: subset -> ["date","sector","ticker"].

Este test verifica el contrato funcional del fix, no la linea concreta.
"""
import pandas as pd
from src.utils import append_dedup


def _df(rows):
    return pd.DataFrame(rows, columns=[
        "date", "sector", "ticker", "price_ret_20d",
    ])


def test_subset_sin_ticker_colapsa_por_sector():
    """Comportamiento del helper: subset sin ticker colapsa filas."""
    hist = pd.DataFrame()
    new = _df([
        ["2026-09-17", "XLK", "AAPL", 0.01],
        ["2026-09-17", "XLK", "MSFT", 0.02],
        ["2026-09-17", "XLK", "NVDA", 0.03],
    ])
    out = append_dedup(hist, new, ["date", "sector"])
    # El bug: 3 tickers colapsan a 1 (el ultimo, keep='last')
    assert len(out) == 1, (
        f"append_dedup con subset [date,sector] deberia colapsar "
        f"3 filas a 1. Resultado: {len(out)}"
    )
    assert out.iloc[0]["ticker"] == "NVDA"


def test_subset_con_ticker_preserva_todos():
    """Fix: subset con ticker preserva los 3 tickers del sector."""
    hist = pd.DataFrame()
    new = _df([
        ["2026-09-17", "XLK", "AAPL", 0.01],
        ["2026-09-17", "XLK", "MSFT", 0.02],
        ["2026-09-17", "XLK", "NVDA", 0.03],
    ])
    out = append_dedup(hist, new, ["date", "sector", "ticker"])
    assert len(out) == 3, (
        f"append_dedup con subset [date,sector,ticker] debe preservar "
        f"3 filas. Resultado: {len(out)}"
    )
    assert set(out["ticker"].tolist()) == {"AAPL", "MSFT", "NVDA"}


def test_subset_con_ticker_dedup_entre_runs():
    """Fix: si dos runs escriben el mismo (date,sector,ticker), queda 1."""
    hist = _df([
        ["2026-09-17", "XLK", "AAPL", 0.01],
        ["2026-09-17", "XLK", "MSFT", 0.02],
    ])
    new = _df([
        ["2026-09-17", "XLK", "AAPL", 0.99],  # revision del mismo dia
        ["2026-09-17", "XLK", "NVDA", 0.03],
    ])
    out = append_dedup(hist, new, ["date", "sector", "ticker"])
    assert len(out) == 3, f"Esperado 3 filas, got {len(out)}"
    aapl = out[out["ticker"] == "AAPL"].iloc[0]
    assert aapl["price_ret_20d"] == 0.99, "keep=last debe preservar el valor nuevo"