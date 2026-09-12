# tests/test_b1_session_integrity.py
"""Tests B1 (2026-09-12): integridad de sesion esperada en el loader.

Verifican que un NaN en una sesion NYSE no es imputado silenciosamente
por ffill, y que la clasificacion lo marca como DATA_ISSUE.

Cubren T-INT-1, T-INT-2, T-INT-3, T-INT-3b, T-INT-3c del informe v2.
"""
import numpy as np
import pandas as pd
from datetime import datetime, date

from src.stock_data_loader import (
    _fill_holes_respecting_sessions,
    _classify_ticker,
)


# Sabado 2026-09-12 10:00 -> expected_session = viernes 2026-09-11
REFERENCE_DATE = datetime(2026, 9, 12, 10, 0)
EXPECTED_SESSION = date(2026, 9, 11)


def _mk_df(dates, closes, ticker='TEST'):
    # DataFrame MultiIndex (Close, ticker) con fechas y valores dados.
    idx = pd.DatetimeIndex(dates)
    cols = pd.MultiIndex.from_tuples([('Close', ticker)])
    arr = np.array(closes, dtype=float).reshape(-1, 1)
    return pd.DataFrame(arr, index=idx, columns=cols)


# ---- T-INT-1: NaN en sesion esperada NO se rellena ----

def test_t_int_1_nan_en_sesion_no_se_rellena():
    df = _mk_df(['2026-09-10', '2026-09-11'], [100.0, np.nan])
    df_filled, diag = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[('Close', 'TEST')]
    assert pd.isna(s.loc['2026-09-11']), "NaN en sesion NYSE debe preservarse"
    assert diag['n_nan_preserved'] >= 1

    status, reason = _classify_ticker('TEST', df_filled, EXPECTED_SESSION)
    assert status == 'DATA_ISSUE'
    assert reason == 'MISSING_CLOSE_EXPECTED_SESSION'


# ---- T-INT-2: NaN en no-sesion SI se rellena, saltando NaN protegido ----

def test_t_int_2_nan_en_no_sesion_se_rellena():
    df = _mk_df(
        ['2026-09-10', '2026-09-11', '2026-09-12'],
        [100.0, np.nan, np.nan],
    )
    df_filled, _ = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    s = df_filled[('Close', 'TEST')]
    assert pd.isna(s.loc['2026-09-11']), "09-11 (viernes) sigue NaN"
    assert s.loc['2026-09-12'] == 100.0, "09-12 (sabado) recibe ffill desde 09-10"


# ---- T-INT-3: fila esperada ausente ----

def test_t_int_3_fila_esperada_ausente():
    df = _mk_df(['2026-09-08', '2026-09-09', '2026-09-10'],
                [100.0, 101.0, 102.0])
    status, reason = _classify_ticker('TEST', df, EXPECTED_SESSION)
    assert status == 'DATA_ISSUE'
    assert reason == 'EXPECTED_SESSION_ABSENT'


# ---- T-INT-3b: repro del incidente (LIN 461.62 + NaN) ----

def test_t_int_3b_repro_incidente_lin():
    df = _mk_df(['2026-09-10', '2026-09-11'], [461.62, np.nan])
    df_filled, _ = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    val = df_filled[('Close', 'TEST')].loc['2026-09-11']
    assert not (val == 461.62), "no debe duplicar el valor del 09-10"
    assert pd.isna(val)
    status, _ = _classify_ticker('TEST', df_filled, EXPECTED_SESSION)
    assert status == 'DATA_ISSUE'


# ---- T-INT-3c: fill no anade filas ----

def test_t_int_3c_fill_no_anade_filas():
    df = _mk_df(['2026-09-08', '2026-09-09', '2026-09-10'],
                [100.0, 101.0, 102.0])
    df_filled, _ = _fill_holes_respecting_sessions(df, REFERENCE_DATE)
    assert len(df_filled) == 3
    fechas = [d.date() for d in df_filled.index]
    assert date(2026, 9, 11) not in fechas