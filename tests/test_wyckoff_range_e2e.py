from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_distribution import (
    WyckoffEvent, classify_distribution, classify_distribution_at,
)


def _make_df(n=60, ticker="TEST"):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 110.0, n)
    df = pd.DataFrame(
        {
            ("Open", ticker): close,
            ("High", ticker): close + 0.5,
            ("Low", ticker): close - 0.5,
            ("Close", ticker): close,
            ("Volume", ticker): np.full(n, 1e6),
        },
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def _fake_bc(idx, pos=10, high=120.0):
    return WyckoffEvent(
        ticker="TEST", event_type="BC",
        event_date=idx[pos], available_at=idx[pos],
        status="CONFIRMED", evidence={"high": high},
    )


def _fake_ar(idx, pos=20, avail_pos=22, support=100.0):
    return WyckoffEvent(
        ticker="TEST", event_type="AR",
        event_date=idx[pos], available_at=idx[avail_pos],
        status="CONFIRMED", evidence={"support": support},
    )


def _fake_sow(idx, status="CONFIRMED", pos=40):
    return WyckoffEvent(
        ticker="TEST", event_type="SOW",
        event_date=idx[pos], available_at=idx[pos],
        status=status, evidence={},
    )


def _fake_trend(direction):
    return {"direction": direction, "available_at": None, "evidence": {}}


def test_classify_distribution_wrapper_uses_last_bar():
    df = _make_df(30)
    result = classify_distribution(df, "TEST")
    assert result.as_of == df.index[-1]


def test_composition_returns_not_evaluable_without_bc(monkeypatch):
    df = _make_df(30)
    bad_bc = WyckoffEvent(
        ticker="TEST", event_type="BC",
        event_date=None, available_at=None, status="NOT_OBSERVED",
    )
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_bc", lambda *a, **k: bad_bc)
    result = classify_distribution_at(df, "TEST", df.index[-1])
    assert result.classification == "NOT_EVALUABLE"
    assert result.evidence["reason"] == "bc_not_confirmed"


def test_composition_distribution_confirmed(monkeypatch):
    df = _make_df(60)
    idx = df.index
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_bc",
                        lambda *a, **k: _fake_bc(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_ar",
                        lambda *a, **k: _fake_ar(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_trend_context",
                        lambda *a, **k: _fake_trend("UP"))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_sow_in_range",
                        lambda *a, **k: _fake_sow(idx, "CONFIRMED"))

    result = classify_distribution_at(df, "TEST", idx[-1])
    assert result.classification == "DISTRIBUTION_CONFIRMED"
    assert result.structural_state == "RANGE_BROKEN"
    assert result.break_direction == "DOWN"
    assert result.range_type == "DISTRIBUTION_CANDIDATE"
    assert result.trend_direction == "UP"
    assert result.coherence_error is None


def test_composition_break_without_context_on_trend_mismatch(monkeypatch):
    df = _make_df(60)
    idx = df.index
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_bc",
                        lambda *a, **k: _fake_bc(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_ar",
                        lambda *a, **k: _fake_ar(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_trend_context",
                        lambda *a, **k: _fake_trend("DOWN"))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_sow_in_range",
                        lambda *a, **k: _fake_sow(idx, "CONFIRMED"))

    # Forzamos range_type DISTRIBUTION_CANDIDATE con trend DOWN -> incoherencia
    result = classify_distribution_at(
        df, "TEST", idx[-1],
        range_kwargs={"range_type": "DISTRIBUTION_CANDIDATE"},
    )
    assert result.classification == "BREAK_WITHOUT_DISTRIBUTION_CONTEXT"
    assert result.coherence_error is not None
    assert "DISTRIBUTION_CANDIDATE" in result.coherence_error


def test_composition_break_pending_on_sow_candidate(monkeypatch):
    df = _make_df(60)
    idx = df.index
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_bc",
                        lambda *a, **k: _fake_bc(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_ar",
                        lambda *a, **k: _fake_ar(idx))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_trend_context",
                        lambda *a, **k: _fake_trend("UP"))
    monkeypatch.setattr("indicators.wyckoff_distribution.detect_sow_in_range",
                        lambda *a, **k: _fake_sow(idx, "CANDIDATE"))

    result = classify_distribution_at(df, "TEST", idx[-1])
    assert result.classification == "BREAK_PENDING"
    assert result.structural_state == "RANGE_ACTIVE"
    assert result.sow_status == "CANDIDATE"


@pytest.mark.integration_real_data
def test_classify_distribution_real_acs_mc():
    p = Path("data/stock_prices_extended.parquet")
    if not p.exists():
        pytest.skip("parquet no disponible")
    df = pd.read_parquet(p)
    ticker = "ACS.MC"
    if ticker not in df.columns.get_level_values(1):
        pytest.skip(f"{ticker} no en parquet")
    close = df[("Close", ticker)].dropna()
    as_of = close.index[-1]
    result = classify_distribution_at(df, ticker, as_of)
    assert result.classification == "DISTRIBUTION_CONFIRMED"
    assert result.coherence_error is None
    assert result.structural_state == "RANGE_BROKEN"
    assert result.break_direction == "DOWN"