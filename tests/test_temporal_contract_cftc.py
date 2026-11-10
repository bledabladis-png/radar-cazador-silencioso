# -*- coding: utf-8 -*-
"""Tests del contrato CFTC_POSITION_FLOW."""
from datetime import date

import pandas as pd

from src.temporal_contracts.cftc_position_flow import CftcPositionFlow


def _make_csv(tmp_path, dates):
    p = tmp_path / "cftc.csv"
    rows = [[d, "X", "Y", 1, 2, 3] for d in dates]
    pd.DataFrame(
        rows,
        columns=["date", "contract", "participant",
                 "net_position", "position_change", "flow_z"],
    ).to_csv(p, index=False)
    return p


def test_sin_csv_devuelve_pending(tmp_path):
    c = CftcPositionFlow(history_path=tmp_path / "no.csv")
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.status == "PENDING"


def test_effective_es_max_date(tmp_path):
    p = _make_csv(tmp_path, ["2026-10-27", "2026-11-03"])
    c = CftcPositionFlow(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.effective_date == date(2026, 11, 3)


def test_csv_por_delante_margen_ok_lag_cero(tmp_path):
    # reference=10-nov, margen 10d, expected=27-oct, effective=3-nov > expected
    p = _make_csv(tmp_path, ["2026-11-03"])
    c = CftcPositionFlow(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.status == "OK"
    assert r.lag_days == 0


def test_csv_retrasado_una_semana_stale(tmp_path):
    # reference=24-nov (martes), margen 10d, expected=10-nov, effective=3-nov -> lag=7d
    p = _make_csv(tmp_path, ["2026-11-03"])
    c = CftcPositionFlow(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-24"))
    assert r.status == "STALE"
    assert r.lag_days == 7


def test_csv_muy_retrasado_insufficient(tmp_path):
    # reference=2027-01-15, margen 10d, expected~05-ene, effective=03-nov -> lag 63d > 45
    p = _make_csv(tmp_path, ["2026-11-03"])
    c = CftcPositionFlow(history_path=p)
    r = c.resolve(None, pd.Timestamp("2027-01-15"))
    assert r.status == "INSUFFICIENT"


def test_date_string_parseable(tmp_path):
    p = _make_csv(tmp_path, ["2026-11-03"])
    c = CftcPositionFlow(history_path=p)
    r = c.resolve(None, pd.Timestamp("2026-11-10"))
    assert r.effective_date == date(2026, 11, 3)


def test_atributos_clase():
    c = CftcPositionFlow(history_path="/tmp/x")
    assert c.name == "CFTC_POSITION_FLOW"
    assert c.family == "CFTC"
    assert c.max_lag_days == 45
    assert c.critical is False
