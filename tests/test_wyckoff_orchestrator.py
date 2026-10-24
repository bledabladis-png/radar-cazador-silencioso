"""Tests del orquestador de fases Wyckoff (regla simetrica)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.wyckoff_event import ClassifiedBreak
from indicators.wyckoff_orchestrator import (
    classify_phase_at, classify_phase, _is_data_insufficient,
    _is_dist_current, _is_acc_current,
)


def _make_df(n=30, ticker="TEST", final_price=100.0):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.full(n, final_price)
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


def _cb(ticker, as_of, classification, reason=None,
        structural_state="NOT_EVALUABLE",
        range_active_at=None, resolution_event_date=None,
        range_support=None, range_resistance=None, range_tol_abs=None,
        current_price=None):
    ev = {}
    if reason:
        ev["reason"] = reason
    if current_price is not None:
        ev["current_price"] = current_price
    return ClassifiedBreak(
        ticker=ticker, as_of=as_of, range_id=None,
        classification=classification,
        structural_state=structural_state,
        break_direction=None, range_type=None,
        trend_direction=None, sow_status=None,
        coherence_error=None,
        evidence=ev,
        range_active_at=range_active_at,
        resolution_event_date=resolution_event_date,
        range_support=range_support,
        range_resistance=range_resistance,
        range_tol_abs=range_tol_abs,
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


# --- Helpers de vigencia (simetrica) ---

def test_is_dist_current_inside_range():
    cb = _cb("T", pd.Timestamp("2026-01-01"), "DISTRIBUTION_CONFIRMED",
             range_support=80.0, range_resistance=100.0, range_tol_abs=2.0)
    assert _is_dist_current(cb, 80.0) is True     # dentro
    assert _is_dist_current(cb, 90.0) is True     # dentro
    assert _is_dist_current(cb, 78.0) is True     # support-tol = 78, limite inferior
    assert _is_dist_current(cb, 102.0) is True    # resistance+tol = 102, limite superior
    assert _is_dist_current(cb, 77.0) is False    # < support-tol (rotura bajista)
    assert _is_dist_current(cb, 103.0) is False   # > resistance+tol (rotura alcista)
    assert _is_dist_current(cb, None) is False


def test_is_dist_current_missing_levels():
    cb = _cb("T", pd.Timestamp("2026-01-01"), "DISTRIBUTION_CONFIRMED")
    assert _is_dist_current(cb, 100.0) is False


def test_is_acc_current_inside_range():
    cb = _cb("T", pd.Timestamp("2026-01-01"), "ACCUMULATION_CONFIRMED",
             range_support=80.0, range_resistance=100.0, range_tol_abs=2.0)
    assert _is_acc_current(cb, 80.0) is True
    assert _is_acc_current(cb, 90.0) is True
    assert _is_acc_current(cb, 78.0) is True
    assert _is_acc_current(cb, 102.0) is True
    assert _is_acc_current(cb, 77.0) is False
    assert _is_acc_current(cb, 103.0) is False


# --- CONFLICT (ambas vigentes, precio dentro de ambos rangos) ---

def test_conflict_when_both_ranges_contain_price(monkeypatch):
    # dist rango [95, 105], acc rango [98, 102], precio 100 dentro de ambos
    df = _make_df(final_price=100.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2026-01-01"),
            range_support=95.0, range_resistance=105.0, range_tol_abs=1.0),
        _cb("TEST", as_of, "ACCUMULATION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2025-06-01"),
            range_support=98.0, range_resistance=102.0, range_tol_abs=1.0),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "CONFLICT"
    assert r.event_phase == "DISTRIBUTION"  # dist mas reciente


# --- Distribucion vigente ---

def test_distribution_when_price_inside_dist_range(monkeypatch):
    df = _make_df(final_price=100.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2026-01-01"),
            range_support=95.0, range_resistance=105.0, range_tol_abs=1.0),
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "DISTRIBUTION"
    assert r.event_phase == "DISTRIBUTION"
    assert "dist vigente" in r.current_phase_basis


# --- Acumulacion vigente ---

def test_accumulation_when_price_inside_acc_range(monkeypatch):
    df = _make_df(final_price=100.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
        _cb("TEST", as_of, "ACCUMULATION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2026-05-01"),
            range_support=95.0, range_resistance=105.0, range_tol_abs=1.0),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "ACCUMULATION"
    assert "acc vigente" in r.current_phase_basis


# --- Rotura alcista de dist (caso CVX) ---

def test_distribution_failed_by_upward_break(monkeypatch):
    """CVX: precio rompio resistance al alza -> dist fallida -> RANGE."""
    df = _make_df(final_price=226.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2026-06-25"),
            range_support=160.0, range_resistance=175.0, range_tol_abs=8.0),
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "RANGE"
    assert r.event_phase == "DISTRIBUTION"
    assert "dist fallida" in r.current_phase_basis


# --- Rotura bajista de dist (caso RHM.DE) ---

def test_distribution_consummated_by_downward_break(monkeypatch):
    """RHM.DE: precio rompio support a la baja -> MARKDOWN (dist consumada)."""
    df = _make_df(final_price=1015.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "DISTRIBUTION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2025-08-12"),
            range_support=1649.0, range_resistance=1944.0, range_tol_abs=15.0),
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "MARKDOWN"
    assert r.event_phase == "DISTRIBUTION"
    assert "markdown" in r.current_phase_basis


def test_accumulation_consummated_by_upward_break(monkeypatch):
    """AMD: precio rompio resistance al alza -> MARKUP (acc consumada)."""
    df = _make_df(final_price=507.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
        _cb("TEST", as_of, "ACCUMULATION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2025-04-01"),
            range_support=76.0, range_resistance=122.0, range_tol_abs=2.0),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "MARKUP"
    assert r.event_phase == "ACCUMULATION"
    assert "markup" in r.current_phase_basis


def test_accumulation_failed_by_downward_break(monkeypatch):
    """Acc confirmada pero precio cayo bajo support -> RANGE (acc fallida)."""
    df = _make_df(final_price=70.0)
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
        _cb("TEST", as_of, "ACCUMULATION_CONFIRMED",
            structural_state="RANGE_BROKEN",
            resolution_event_date=pd.Timestamp("2025-06-01"),
            range_support=95.0, range_resistance=120.0, range_tol_abs=2.0),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "RANGE"
    assert r.event_phase == "ACCUMULATION"
    assert "acc fallida" in r.current_phase_basis



# --- Range / insufficient ---

def test_range_when_neither_confirmed(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
        _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "RANGE"
    assert r.event_phase == "RANGE"


def test_range_when_both_no_bc_no_sc(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="bc_not_confirmed"),
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="sc_not_confirmed"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "RANGE"


def test_insufficient_data_when_both_data_missing(monkeypatch):
    df = _make_df()
    as_of = df.index[-1]
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="nan_in_features"),
        _cb("TEST", as_of, "NOT_EVALUABLE", reason="insufficient_data"),
    )
    r = classify_phase_at(df, "TEST", as_of)
    assert r.phase == "INSUFFICIENT_DATA"


# --- Utilidades ---

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
    _patch_both(
        monkeypatch,
        _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
        _cb("TEST", as_of, "RANGE_ACTIVE_NO_BREAK"),
    )
    r = classify_phase(df, "TEST")
    assert r.as_of == as_of