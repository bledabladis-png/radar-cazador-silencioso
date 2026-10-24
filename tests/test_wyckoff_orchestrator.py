"""Tests del orquestador de fases Wyckoff."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_event import ClassifiedBreak
from indicators.wyckoff_orchestrator import (
    classify_phase_at, classify_phase, FinalPhase, _is_data_insufficient,
)


def _make_df(n=30, ticker="TEST"):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, 100.0)
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


def _cb(ticker, as_of, classification, reason=None):
    return ClassifiedBreak(
        ticker=ticker, as_of=as_of, range_id=None,
        classification=classification,
        structural_state="NOT_EVALUABLE",
        break_direction=None, range_type=None,
        trend_direction=None, sow_status=None,
        coherence_error=None,
        evidence={"reason": reason} if reason else {},
    )


def _patch_both(monkeypatch, dist_cb, acc_cb):
    monkeypatch.setattr(
        "indicators.wyckoff_orchestrator.classify_distribution_at",
        lambda *a, **k: dist_cb,
    )
    monkeypatch.setattr(
        "indicators.wyckoff_orchestrator.classify_accumulation_at",
        lambda *a, **k: acc_cb,
    )


def test_conflict_when_both_confirmed(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED"),
                _cb("TEST", as_of, "ACCUMULATION_CONFIRMED"))
    r = classify_phase_at(df, "TEST", as_of)
    assert isinstance(r, FinalPhase)
    assert r.phase == "CONFLICT"
    assert r.conflict_reason is not None
    assert "simultaneamente" in r.conflict_reason


def test_distribution_when_only_dist_confirmed(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED"),
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"))
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "DISTRIBUTION"
    assert r.conflict_reason is None


def test_accumulation_when_only_acc_confirmed(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
                _cb("TEST", as_of, "ACCUMULATION_CONFIRMED"))
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "ACCUMULATION"


def test_range_when_neither_confirmed(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
                _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"))
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "RANGE"


def test_range_when_both_no_bc_no_sc(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"))
    r = classify_phase_at(df, "TEST", as_of)
    # "no BC / no SC" no es falta de datos, es ausencia de setup -> RANGE
    assert r.phase == "RANGE"


def test_insufficient_data_when_both_data_missing(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="nan_in_features"),
                _cb("TEST", as_of, "NOT_EVALUABLE", reason="insufficient_data"))
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "INSUFFICIENT_DATA"


def test_is_data_insufficient_classification():
    as_of = pd.Timestamp("2020-01-01")
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "nan_in_features")) is True
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "insufficient_data")) is True
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "empty_df_up_to_as_of")) is True
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "score_error: algo")) is True
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "bc_not_confirmed")) is False
    assert _is_data_insufficient(_cb("T", as_of, "NOT_EVALUABLE", "sc_not_confirmed")) is False
    assert _is_data_insufficient(_cb("T", as_of, "RANGE_ACTIVE_NO_BREAK")) is False


def test_wrapper_uses_last_bar(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(monkeypatch,
                _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
                _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"))
    r = classify_phase(df, "TEST")
    assert r.as_of == as_of