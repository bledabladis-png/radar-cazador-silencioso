# -*- coding: utf-8 -*-
# Tests del contrato FRED_MACRO (auditoria 11-03, hallazgo 10.1).
import json
from datetime import date

from src.temporal_contracts.fred_macro import (
    FredMacro,
    _expected_close,
)
from src.temporal_contracts.base import (
    STATUS_INSUFFICIENT,
    STATUS_PENDING,
    STATUS_STALE,
)


def test_max_lag_90():
    assert FredMacro.max_lag_days == 90


def test_expected_close_04_11_2026():
    assert _expected_close(date(2026, 11, 4)) == date(2026, 11, 3)


def test_expected_close_none():
    assert _expected_close(None) is None


def test_resolve_inexistente_pending(tmp_path):
    c = FredMacro(state_path=tmp_path / "no_existe.json")
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING
    assert res.effective_date is None
    assert res.expected_date == date(2026, 11, 3)
    assert res.coverage == 0.0


def test_resolve_json_invalido_pending(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("no soy json", encoding="utf-8")
    c = FredMacro(state_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING


def test_resolve_sin_campo_date_pending(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"score": 0.5}), encoding="utf-8")
    c = FredMacro(state_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING


def test_resolve_stale_normal(tmp_path):
    # effective=2026-11-02, expected=2026-11-03, lag=1 -> STALE (normal FRED)
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"date": "2026-11-02", "score": -0.23}), encoding="utf-8")
    c = FredMacro(state_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.effective_date == date(2026, 11, 2)
    assert res.expected_date == date(2026, 11, 3)
    assert res.lag_days == 1
    assert res.status == STATUS_STALE


def test_resolve_insufficient_lag_100(tmp_path):
    # effective=2026-07-26, expected=2026-11-03, lag=100 > 90 -> INSUFFICIENT
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"date": "2026-07-26"}), encoding="utf-8")
    c = FredMacro(state_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.lag_days == 100
    assert res.status == STATUS_INSUFFICIENT
