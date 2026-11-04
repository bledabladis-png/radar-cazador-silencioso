# -*- coding: utf-8 -*-
# Tests del contrato SSGA_PRIMARY_FLOW (auditoria 11-03, hallazgo 10.1).
from datetime import date

import pandas as pd

from src.temporal_contracts.ssga_primary_flow import (
    SsgaPrimaryFlow,
    _expected_close,
)
from src.temporal_contracts.base import (
    STATUS_INSUFFICIENT,
    STATUS_PENDING,
    STATUS_STALE,
)


def test_max_lag_5():
    assert SsgaPrimaryFlow.max_lag_days == 5


def test_expected_close_04_11_2026():
    # 2026-11-04 es miercoles, ultimo cierre completo 2026-11-03
    assert _expected_close(date(2026, 11, 4)) == date(2026, 11, 3)


def test_expected_close_lunes_va_al_viernes():
    # 2026-11-02 es lunes, ultimo cierre 2026-10-30 (viernes)
    assert _expected_close(date(2026, 11, 2)) == date(2026, 10, 30)


def test_expected_close_none():
    assert _expected_close(None) is None


def test_resolve_inexistente_pending(tmp_path):
    c = SsgaPrimaryFlow(history_path=tmp_path / "no_existe.csv")
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING
    assert res.effective_date is None
    assert res.expected_date == date(2026, 11, 3)
    assert res.coverage == 0.0


def test_resolve_vacio_pending(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["Date"]).to_csv(p, index=False)
    c = SsgaPrimaryFlow(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING


def test_resolve_stale_lag_1(tmp_path):
    # effective=2026-11-02, expected=2026-11-03, lag=1 -> STALE (normal)
    p = tmp_path / "h.csv"
    pd.DataFrame([
        {"Date": "2026-10-30"},
        {"Date": "2026-11-02"},
    ]).to_csv(p, index=False)
    c = SsgaPrimaryFlow(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.effective_date == date(2026, 11, 2)
    assert res.expected_date == date(2026, 11, 3)
    assert res.lag_days == 1
    assert res.status == STATUS_STALE


def test_resolve_insufficient_lag_7(tmp_path):
    # effective=2026-10-27, expected=2026-11-03, lag=7 > 5 -> INSUFFICIENT
    p = tmp_path / "h.csv"
    pd.DataFrame([{"Date": "2026-10-27"}]).to_csv(p, index=False)
    c = SsgaPrimaryFlow(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.lag_days == 7
    assert res.status == STATUS_INSUFFICIENT
