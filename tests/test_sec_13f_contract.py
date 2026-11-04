# -*- coding: utf-8 -*-
# Tests del contrato SEC_13F (auditoria 11-03, hallazgo 10.1).
from datetime import date

import pandas as pd

from src.temporal_contracts.sec_13f import (
    SEC_13F_DEADLINE_DAYS,
    Sec13F,
    _expected_quarter_end,
)
from src.temporal_contracts.base import (
    STATUS_INSUFFICIENT,
    STATUS_OK,
    STATUS_PENDING,
)


def test_deadline_es_45_dias():
    assert SEC_13F_DEADLINE_DAYS == 45


def test_expected_q2_antes_deadline_q3():
    # 2026-11-04: Q3 ends 2026-09-30, due 2026-11-14 -> no due aun.
    # Esperado Q2 (2026-06-30).
    assert _expected_quarter_end(date(2026, 11, 4)) == date(2026, 6, 30)


def test_expected_q3_tras_deadline():
    # 2026-11-20: Q3 due 2026-11-14 -> Q3 esperado.
    assert _expected_quarter_end(date(2026, 11, 20)) == date(2026, 9, 30)


def test_expected_q4_tras_deadline():
    # 2026-02-20: Q4 2025 due 2026-02-14 -> Q4 2025 esperado.
    assert _expected_quarter_end(date(2026, 2, 20)) == date(2025, 12, 31)


def test_expected_none_sin_referencia():
    assert _expected_quarter_end(None) is None


def test_resolve_data_dir_vacio(tmp_path):
    c = Sec13F(data_dir=tmp_path)
    res = c.resolve(df_market=None, reference_date=date(2026, 11, 4))
    assert res.status == STATUS_PENDING
    assert res.effective_date is None
    assert res.expected_date == date(2026, 6, 30)
    assert res.coverage == 0.0


def test_resolve_data_dir_inexistente(tmp_path):
    c = Sec13F(data_dir=tmp_path / "no_existe")
    res = c.resolve(df_market=None, reference_date=date(2026, 11, 4))
    assert res.status == STATUS_PENDING


def test_resolve_con_datos_ok(tmp_path):
    q = tmp_path / "2026Q2"
    q.mkdir()
    pd.DataFrame({"PERIODOFREPORT": ["2026-06-30"] * 10}).to_parquet(
        q / "SUBMISSION.parquet"
    )
    c = Sec13F(data_dir=tmp_path)
    res = c.resolve(df_market=None, reference_date=date(2026, 11, 4))
    assert res.effective_date == date(2026, 6, 30)
    assert res.expected_date == date(2026, 6, 30)
    assert res.lag_days == 0
    assert res.coverage == 1.0
    assert res.status == STATUS_OK


def test_resolve_un_quarter_detras_insufficient(tmp_path):
    # Solo Q1 presente, Q2 esperado. lag = 91 dias > 90 -> INSUFFICIENT.
    q = tmp_path / "2026Q1"
    q.mkdir()
    pd.DataFrame({"PERIODOFREPORT": ["2026-03-31"] * 10}).to_parquet(
        q / "SUBMISSION.parquet"
    )
    c = Sec13F(data_dir=tmp_path)
    res = c.resolve(df_market=None, reference_date=date(2026, 11, 4))
    assert res.effective_date == date(2026, 3, 31)
    assert res.expected_date == date(2026, 6, 30)
    assert res.lag_days == 91
    assert res.status == STATUS_INSUFFICIENT
