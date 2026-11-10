# -*- coding: utf-8 -*-
"""Tests del contrato SEC_NPORT."""
from datetime import date

import pandas as pd

from src.temporal_contracts.sec_nport import SecNport


def _make_csv(tmp_path, rows):
    p = tmp_path / "nport.csv"
    pd.DataFrame(rows, columns=["REPORT_DATE", "FILING_DATE"]).to_csv(
        p, index=False
    )
    return p


def test_sin_csv_devuelve_pending(tmp_path):
    c = SecNport(history_path=tmp_path / "no.csv")
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.status == "PENDING"


def test_effective_es_max_report_date(tmp_path):
    p = _make_csv(tmp_path, [["2026-06-30", "2026-08-28"]])
    c = SecNport(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.effective_date == date(2026, 6, 30)


def test_esperado_coincide_status_ok(tmp_path):
    # reference=10-nov: Q3-2026 deadline 29-nov (aun no pasado) -> expected=30-jun.
    # effective=30-jun. lag=0. OK.
    p = _make_csv(tmp_path, [["2026-06-30", "2026-08-28"]])
    c = SecNport(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.status == "OK"
    assert r.lag_days == 0


def test_tras_deadline_q3_esperado_q3(tmp_path):
    # reference=01-dic-2026: Q3 deadline 29-nov (pasado) -> expected=30-sep.
    # effective=30-jun. lag=92. <=180 -> STALE.
    p = _make_csv(tmp_path, [["2026-06-30", "2026-08-28"]])
    c = SecNport(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-12-01"))
    assert r.status == "STALE"
    assert r.lag_days == 92


def test_muy_retrasado_insufficient(tmp_path):
    # reference=2027-10-01: expected=30-jun-2027. effective=30-jun-2026. lag=365 > 180.
    p = _make_csv(tmp_path, [["2026-06-30", "2026-08-28"]])
    c = SecNport(history_path=p)
    r = c.resolve(None, pd.Timestamp("2027-10-01"))
    assert r.status == "INSUFFICIENT"


def test_atributos_clase():
    c = SecNport(history_path="/tmp/x")
    assert c.name == "SEC_NPORT"
    assert c.family == "SEC"
    assert c.max_lag_days == 180
    assert c.critical is False
