# -*- coding: utf-8 -*-
# Tests del contrato FINRA_DARKPOOL (auditoria 11-03, hallazgo 10.1).
from datetime import date

import pandas as pd

from src.temporal_contracts.finra_darkpool import (
    FINRA_LAG_MIN_DAYS,
    FinraDarkpool,
    _expected_week,
)
from src.temporal_contracts.base import (
    STATUS_INSUFFICIENT,
    STATUS_PENDING,
    STATUS_STALE,
)


def test_lag_min_14():
    assert FINRA_LAG_MIN_DAYS == 14


def test_expected_semana_04_11_2026():
    # 2026-11-04 - 14d = 2026-10-21, lunes = 2026-10-19
    assert _expected_week(date(2026, 11, 4)) == date(2026, 10, 19)


def test_expected_semana_exacta_lunes():
    # 2026-11-02 (lunes) - 14d = 2026-10-19 (lunes)
    assert _expected_week(date(2026, 11, 2)) == date(2026, 10, 19)


def test_expected_none():
    assert _expected_week(None) is None


def test_resolve_history_inexistente(tmp_path):
    c = FinraDarkpool(history_path=tmp_path / "no_existe.csv")
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING
    assert res.effective_date is None
    assert res.expected_date == date(2026, 10, 19)
    assert res.coverage == 0.0


def test_resolve_history_vacia(tmp_path):
    p = tmp_path / "empty.csv"
    pd.DataFrame(columns=["week", "ratio"]).to_csv(p, index=False)
    c = FinraDarkpool(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.status == STATUS_PENDING


def test_resolve_stale_normal(tmp_path):
    # effective=2026-10-12, expected=2026-10-19, lag=7 -> STALE (normal FINRA)
    p = tmp_path / "h.csv"
    pd.DataFrame([
        {"week": "2026-09-28", "ratio": 0.20},
        {"week": "2026-10-05", "ratio": 0.21},
        {"week": "2026-10-12", "ratio": 0.22},
    ]).to_csv(p, index=False)
    c = FinraDarkpool(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.effective_date == date(2026, 10, 12)
    assert res.expected_date == date(2026, 10, 19)
    assert res.lag_days == 7
    assert res.status == STATUS_STALE


def test_resolve_insufficient_lag_excesivo(tmp_path):
    # effective=2026-08-03, expected=2026-10-19, lag=77 > 45 -> INSUFFICIENT
    p = tmp_path / "h.csv"
    pd.DataFrame([{"week": "2026-08-03", "ratio": 0.20}]).to_csv(p, index=False)
    c = FinraDarkpool(history_path=p)
    res = c.resolve(None, date(2026, 11, 4))
    assert res.lag_days == 77
    assert res.status == STATUS_INSUFFICIENT
