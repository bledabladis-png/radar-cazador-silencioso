# tests/test_c2f_breadth_fallback.py
"""Tests C2-followup (2026-09-12): preservacion de la ultima observacion
valida de Sector Breadth cuando no hay nueva observacion de mercado.

Regresion corregida: C2 omitia la seccion completa del reporte en dia
no bursatil. El fallback devuelve el ultimo snapshot VALIDO con
is_stale=True, sin modificar el CSV.
"""
import numpy as np
import pandas as pd
from datetime import datetime

from src.pipeline.breadth_metrics import (
    _compute_sector_breadth_health,
    _load_latest_valid_breadth_snapshot,
)
from src.report.sectorial import render_sector_breadth


SATURDAY = datetime(2026, 9, 12, 10, 0)
FRIDAY = datetime(2026, 9, 11, 23, 30)

SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']


def _mk_breadth_csv(path, dates, per_date_rows=11, invalid_date=None):
    """Genera un CSV con N fechas x 11 sectores. Si invalid_date se
    especifica, inserta una fila con esa fecha y valores 0/0 (corrupta)."""
    rows = []
    for d in dates:
        for i, s in enumerate(SECTORS[:per_date_rows]):
            rows.append({
                'date': d,
                'sector': s,
                'n_total': 27,
                'n_valid_ema200': 20,
                'pct_above_ema20': 15.0 + i,
                'pct_above_ema50': 30.0,
                'pct_above_ema200': 60.0,
                'pct_rs_positive': 35.0,
                'pct_momentum_positive': 20.0,
                'count_accumulation': 4,
                'count_markup': 8,
                'count_distribution': 4,
                'count_markdown': 0,
                'new_highs': 1,
                'new_lows': 3,
                'advances': 13,
                'declines': 7,
                'unchanged': 0,
                'ad_net': 6,
                'advance_pct': 65.0,
            })
    if invalid_date is not None:
        for s in SECTORS:
            rows.append({
                'date': invalid_date,
                'sector': s,
                'n_total': 27,
                'n_valid_ema200': 20,
                'pct_above_ema20': 0.0, 'pct_above_ema50': 0.0, 'pct_above_ema200': 0.0,
                'pct_rs_positive': 0.0, 'pct_momentum_positive': 0.0,
                'count_accumulation': 0, 'count_markup': 0, 'count_distribution': 0, 'count_markdown': 0,
                'new_highs': 0, 'new_lows': 0,
                'advances': 0, 'declines': 0, 'unchanged': 20, 'ad_net': 0,
                'advance_pct': np.nan,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _mk_stocks(end_date='2026-09-11', n=300):
    idx = pd.date_range(end=end_date, periods=n, freq='D')
    data = {}
    for t in ['AAA', 'BBB']:
        data[('Close', t)] = [100.0 + i * 0.1 for i in range(n)]
        data[('High', t)] = [101.0] * n
        data[('Low', t)] = [99.0] * n
        data[('Open', t)] = [100.0] * n
        data[('Volume', t)] = [1000.0] * n
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _mk_market(end_date='2026-09-11', n=300):
    idx = pd.date_range(end=end_date, periods=n, freq='D')
    mkt = {('XLK', 'Close'): [100.0 + i * 0.05 for i in range(n)]}
    df = pd.DataFrame(mkt, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ============================================================
# T-C2F-1: sabado -> fallback + is_stale=True + CSV real intacto
# ============================================================
def test_t_c2f_1_sabado_devuelve_fallback(tmp_path):
    csv = tmp_path / "sb.csv"
    _mk_breadth_csv(csv, dates=['2026-09-10', '2026-09-11'])
    df_market = _mk_market()
    df_stocks = _mk_stocks()
    holdings = pd.DataFrame({'etf': ['XLK'] * 11, 'ticker': SECTORS})

    mtime_before = csv.stat().st_mtime
    size_before = csv.stat().st_size

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=csv)

    assert result is not None, "debe devolver snapshot historico"
    assert is_stale is True
    assert len(result) == 11
    assert result['date'].max() == pd.Timestamp('2026-09-11')
    # CSV real no modificado
    assert csv.stat().st_mtime == mtime_before
    assert csv.stat().st_size == size_before


# ============================================================
# T-C2F-2: viernes sesion -> nueva observacion, is_stale=False
# ============================================================
def test_t_c2f_2_viernes_genera_nueva(tmp_path):
    csv = tmp_path / "sb.csv"
    _mk_breadth_csv(csv, dates=['2026-09-10'])
    df_market = _mk_market()
    df_stocks = _mk_stocks(end_date='2026-09-11')
    holdings = pd.DataFrame({'etf': ['XLK'] * 11, 'ticker': SECTORS})

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=FRIDAY, output_path=csv)

    assert result is not None
    assert is_stale is False
    # append_dedup normaliza date a string YYYY-MM-DD; comparamos tras parsear.
    assert pd.to_datetime(result['date']).max() == pd.Timestamp('2026-09-11')


# ============================================================
# T-C2F-3: render con is_stale=True incluye aviso + fecha real
# ============================================================
def test_t_c2f_3_render_stale():
    snap = pd.DataFrame([{
        'date': pd.Timestamp('2026-09-11'),
        'sector': s,
        'n_total': 27, 'n_valid_ema200': 20,
        'pct_above_ema20': 15.0, 'pct_above_ema50': 30.0, 'pct_above_ema200': 60.0,
        'pct_rs_positive': 35.0, 'pct_momentum_positive': 20.0,
        'count_accumulation': 4, 'count_markup': 8, 'count_distribution': 4, 'count_markdown': 0,
        'new_highs': 1, 'new_lows': 3,
        'advances': 13, 'declines': 7, 'unchanged': 0, 'ad_net': 6, 'advance_pct': 65.0,
    } for s in SECTORS])

    lines = render_sector_breadth(snap, is_stale=True)
    body = "\n".join(lines)

    assert "## Sector Breadth & Health" in body
    assert "Sin actualizacion - mercado cerrado" in body
    assert "2026-09-11" in body
    # La fecha de las filas debe seguir siendo 09-11, no 09-12
    assert "2026-09-12" not in body


# ============================================================
# T-C2F-4: render sin stale -> no aviso
# ============================================================
def test_t_c2f_4_render_no_stale():
    snap = pd.DataFrame([{
        'date': pd.Timestamp('2026-09-11'),
        'sector': s,
        'n_total': 27, 'n_valid_ema200': 20,
        'pct_above_ema20': 15.0, 'pct_above_ema50': 30.0, 'pct_above_ema200': 60.0,
        'pct_rs_positive': 35.0, 'pct_momentum_positive': 20.0,
        'count_accumulation': 4, 'count_markup': 8, 'count_distribution': 4, 'count_markdown': 0,
        'new_highs': 1, 'new_lows': 3,
        'advances': 13, 'declines': 7, 'unchanged': 0, 'ad_net': 6, 'advance_pct': 65.0,
    } for s in SECTORS])

    lines = render_sector_breadth(snap, is_stale=False)
    body = "\n".join(lines)

    assert "## Sector Breadth & Health" in body
    assert "Sin actualizacion" not in body


# ============================================================
# T-C2F-5: fallback ignora fila con fecha no-bursatil
# ============================================================
def test_t_c2f_5_fallback_ignora_sabado(tmp_path):
    csv = tmp_path / "sb.csv"
    # Inserta una fila 2026-09-12 (sabado) DESPUES de las validas
    _mk_breadth_csv(csv, dates=['2026-09-10', '2026-09-11'], invalid_date='2026-09-12')

    snap = _load_latest_valid_breadth_snapshot(csv)

    assert snap is not None
    assert snap['date'].max() == pd.Timestamp('2026-09-11'), \
        "debe ignorar la fila del sabado 09-12"


# ============================================================
# T-C2F-6: fallback con CSV sin fecha valida -> None
# ============================================================
def test_t_c2f_6_fallback_sin_fechas_validas(tmp_path):
    csv = tmp_path / "sb.csv"
    # Solo sabados
    _mk_breadth_csv(csv, dates=['2026-09-12', '2026-09-13'])
    snap = _load_latest_valid_breadth_snapshot(csv)
    assert snap is None


# ============================================================
# T-INT-13: integracion - reporte con is_stale conserva la seccion
# ============================================================
def test_t_int_13_seccion_presente_en_stale():
    snap = pd.DataFrame([{
        'date': pd.Timestamp('2026-09-11'),
        'sector': s,
        'n_total': 27, 'n_valid_ema200': 20,
        'pct_above_ema20': 15.0, 'pct_above_ema50': 30.0, 'pct_above_ema200': 60.0,
        'pct_rs_positive': 35.0, 'pct_momentum_positive': 20.0,
        'count_accumulation': 4, 'count_markup': 8, 'count_distribution': 4, 'count_markdown': 0,
        'new_highs': 1, 'new_lows': 3,
        'advances': 13, 'declines': 7, 'unchanged': 0, 'ad_net': 6, 'advance_pct': 65.0,
    } for s in SECTORS])

    lines = render_sector_breadth(snap, is_stale=True)
    body = "\n".join(lines)

    assert "## Sector Breadth & Health" in body
    assert "2026-09-11" in body
    assert "2026-09-12" not in body
    assert "+6" in body  # A/D de la fila valida
    assert "Sin actualizacion" in body