from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_range import (
    Range, WyckoffEvent, build_range_from_bc_ar, RANGE_TOLERANCE_FRAC,
)


def _mk_event(ticker, etype, event_date, available_at, status="CONFIRMED",
              evidence=None):
    return WyckoffEvent(
        ticker=ticker, event_type=etype,
        event_date=event_date, available_at=available_at,
        status=status, evidence=evidence or {},
    )


def _idx(n=10):
    return pd.date_range("2020-01-01", periods=n, freq="B")


def _dummy_df():
    idx = _idx()
    df = pd.DataFrame(
        {("Close", "TEST"): np.full(len(idx), 100.0)},
        index=idx,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["field", "ticker"])
    return df


def test_build_range_from_bc_ar_active():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   evidence={"high": 120.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   evidence={"support": 100.0})
    r = build_range_from_bc_ar(_dummy_df(), "TEST", bc, ar,
                               range_type="DISTRIBUTION_CANDIDATE")
    assert isinstance(r, Range)
    assert r.state == "RANGE_ACTIVE"
    assert r.formation_path == "BC_AR"
    assert r.range_type == "DISTRIBUTION_CANDIDATE"
    assert r.support == 100.0
    assert r.resistance == 120.0
    assert r.candidate_start_date == idx[2]
    assert r.start_available_at == idx[7]
    assert r.range_active_at == idx[7]
    assert r.tolerance_frac == RANGE_TOLERANCE_FRAC
    assert abs(r.tol_abs - RANGE_TOLERANCE_FRAC * 20.0) < 1e-9
    assert r.end is None
    assert r.closed_at is None
    assert r.break_direction is None
    assert r.evidence["bc_high"] == 120.0
    assert r.evidence["ar_support"] == 100.0


def test_build_range_raises_if_bc_not_confirmed():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   status="NOT_OBSERVED", evidence={"high": 120.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   evidence={"support": 100.0})
    with pytest.raises(ValueError, match="bc_event no CONFIRMED"):
        build_range_from_bc_ar(_dummy_df(), "TEST", bc, ar)


def test_build_range_raises_if_ar_not_confirmed():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   evidence={"high": 120.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   status="CANDIDATE", evidence={"support": 100.0})
    with pytest.raises(ValueError, match="ar_event no CONFIRMED"):
        build_range_from_bc_ar(_dummy_df(), "TEST", bc, ar)


def test_build_range_raises_if_support_above_resistance():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   evidence={"high": 100.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   evidence={"support": 105.0})
    with pytest.raises(ValueError, match="support"):
        build_range_from_bc_ar(_dummy_df(), "TEST", bc, ar)


def test_build_range_defaults_range_type_unclassified():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   evidence={"high": 120.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   evidence={"support": 100.0})
    r = build_range_from_bc_ar(_dummy_df(), "TEST", bc, ar)
    assert r.range_type == "UNCLASSIFIED"


def test_build_range_custom_tolerance_and_trend_context():
    idx = _idx()
    bc = _mk_event("TEST", "BC", idx[2], idx[2],
                   evidence={"high": 200.0})
    ar = _mk_event("TEST", "AR", idx[5], idx[7],
                   evidence={"support": 100.0})
    tc = {"trend_direction": "UP", "trend_available_at": str(idx[0])}
    r = build_range_from_bc_ar(
        _dummy_df(), "TEST", bc, ar,
        tolerance_frac=0.02, trend_context=tc,
    )
    assert r.tolerance_frac == 0.02
    assert abs(r.tol_abs - 0.02 * 100.0) < 1e-9
    assert r.trend_context["trend_direction"] == "UP"