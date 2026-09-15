# -*- coding: utf-8 -*-
"""Tests FU-020 Fase 1: integracion de resolve_effective_date en breadth_equity.

En el flujo real, leaders.py recorta df_stocks a la fecha efectiva antes
de pasarlo a los motores. Estos tests simulan ese flujo.
"""
import numpy as np
import pandas as pd

from indicators.breadth_equity import compute_advance_decline
from src.effective_date import resolve_effective_date


def _make_stocks(dates, tickers, presence):
    """DataFrame MultiIndex (field, ticker) con Close valores variables por fecha.

    Los valores varian linealmente para que diff() != 0 (necesario para A/D).
    """
    data = {}
    for t in tickers:
        closes = []
        base = 100.0 + hash(t) % 50
        for i, p in enumerate(presence[t]):
            closes.append(base + i * 0.5 if p else np.nan)
        for field in ("Open", "High", "Low", "Close", "Volume"):
            if field == "Close":
                data[(field, t)] = closes
            else:
                data[(field, t)] = [v + 1.0 if not np.isnan(v) else np.nan
                                    for v in closes]
    df = pd.DataFrame(data, index=pd.to_datetime(dates))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_breadth_declares_effective_meta_when_passed():
    """A/D sobre df recortado a effective_date -> incluye metadata."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    tickers = [f"T{i}" for i in range(25)]
    presence = {t: [True, True, True] for t in tickers}
    df = _make_stocks(dates, tickers, presence)

    meta = resolve_effective_date(df, tickers, min_coverage=0.90)
    assert meta["status"] == "OK"
    assert pd.Timestamp(meta["date"]) == pd.Timestamp("2026-09-15")

    # Simular leaders.py: recortar a effective_date
    df_trimmed = df.loc[:meta["date"]]

    ad = compute_advance_decline(df_trimmed, effective_meta=meta)
    assert ad is not None
    assert ad["effective_date"] == "2026-09-15"
    assert ad["coverage"] == 1.0
    assert ad["n_observed"] == 25
    assert ad["n_eligible"] == 25
    assert ad["lag_days"] == 0


def test_breadth_backdates_when_last_row_partial():
    """Fila superior parcial -> meta backdates y df se recorta a esa fecha."""
    dates = ["2026-09-11", "2026-09-14", "2026-09-15"]
    tickers = [f"T{i}" for i in range(25)]
    presence = {t: [True, True, True] for t in tickers}
    for t in tickers[5:]:
        presence[t][-1] = False
    df = _make_stocks(dates, tickers, presence)

    meta = resolve_effective_date(df, tickers, min_coverage=0.90)
    assert meta["status"] == "OK"
    assert pd.Timestamp(meta["date"]) == pd.Timestamp("2026-09-14")
    assert meta["lag_days"] == 1

    df_trimmed = df.loc[:meta["date"]]

    ad = compute_advance_decline(df_trimmed, effective_meta=meta)
    assert ad is not None
    assert ad["effective_date"] == "2026-09-14"
    assert ad["lag_days"] == 1
    assert ad["n_eligible"] == 25
    assert ad["n_observed"] == 25


def test_breadth_no_meta_when_not_passed():
    """Sin effective_meta -> dict no incluye effective_date."""
    dates = ["2026-09-11", "2026-09-14"]
    tickers = [f"T{i}" for i in range(25)]
    presence = {t: [True, True] for t in tickers}
    df = _make_stocks(dates, tickers, presence)

    ad = compute_advance_decline(df)
    assert ad is not None
    assert "effective_date" not in ad
    assert "coverage" not in ad


def test_breadth_insufficient_meta_does_not_add_effective_date():
    """effective_meta INSUFFICIENT -> compute_advance_decline devuelve None."""
    dates = ["2026-09-14", "2026-09-15"]
    tickers = [f"T{i}" for i in range(25)]
    presence = {t: [False, False] for t in tickers}
    df = _make_stocks(dates, tickers, presence)

    meta = resolve_effective_date(df, tickers, min_coverage=0.90)
    assert meta["status"] == "INSUFFICIENT_COVERAGE"

    ad = compute_advance_decline(df, effective_meta=meta)
    assert ad is None
