# tests/test_c2_temporalidad.py
"""Tests C2 (2026-09-12): temporalidad de Sector Breadth.

Verifican:
- T-INT-4:  dia no bursatil -> no se genera observacion
- T-INT-8:  as_of_date explicito sesion -> date == esa sesion
- T-INT-9:  as_of_date explicito no-sesion -> ValueError
- T-INT-10: as_of_date=None -> usa df_stocks.index[-1] (ultima observada)
- T-INT-11: reference_date sesion -> fecha correcta en CSV temporal
- T-INT-12: reference_date sabado + datos hasta viernes -> no observacion

Aislamiento: los tests usan output_path=tmp_path para no tocar el CSV real.
"""
import pandas as pd
import pytest
from datetime import datetime, date
from pathlib import Path

from indicators.sector_breadth import compute_sector_breadth
from src.pipeline.breadth_metrics import _compute_sector_breadth_health


# Viernes 2026-09-11 23:30: sesion NYSE, DESPUES de PUBLISH_HOUR (23h).
# A esta hora Yahoo ya publico el cierre del dia -> expected_session = 09-11.
FRIDAY = datetime(2026, 9, 11, 23, 30)
# Viernes 2026-09-11 18:00: sesion NYSE, ANTES de PUBLISH_HOUR.
# A esta hora el cierre aun no esta publicado -> expected_session = 09-10.
FRIDAY_PRE_PUBLISH = datetime(2026, 9, 11, 18, 0)
# Sabado 2026-09-12: no sesion.
SATURDAY = datetime(2026, 9, 12, 10, 0)


def _mk_market_and_stocks(end_date='2026-09-11', n=300):
    """Genera df_market y df_stocks con datos diarios hasta end_date."""
    idx = pd.date_range(end=end_date, periods=n, freq='D')
    stocks_data = {}
    for t in ['AAA', 'BBB']:
        stocks_data[('Close', t)] = [100.0 + i * 0.1 for i in range(n)]
        stocks_data[('High', t)] = [101.0] * n
        stocks_data[('Low', t)] = [99.0] * n
        stocks_data[('Open', t)] = [100.0] * n
        stocks_data[('Volume', t)] = [1000.0] * n
    df_stocks = pd.DataFrame(stocks_data, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)

    mkt_data = {('XLK', 'Close'): [100.0 + i * 0.05 for i in range(n)]}
    df_market = pd.DataFrame(mkt_data, index=idx)
    df_market.columns = pd.MultiIndex.from_tuples(df_market.columns)

    holdings = pd.DataFrame({'etf': ['XLK', 'XLK'], 'ticker': ['AAA', 'BBB']})
    return df_market, df_stocks, holdings


# ============================================================
# T-INT-4 (revisado 2026-10-10): sabado con datos hasta viernes ->
# escribe fila del viernes. reference_date es la fecha del run, no
# la sesion objetivo. Coherente con pipeline_gate.py (target_session
# != dia calendario del slot).
# ============================================================
def test_t_int_4_sabado_produce_sesion_viernes(tmp_path):
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    assert result is not None
    assert is_stale is False
    assert out.exists(), "debe escribir la fila del viernes"
    assert pd.to_datetime(result.iloc[0]['date']).date() == date(2026, 9, 11)


# ============================================================
# T-INT-4b (nuevo 2026-10-10): sabado con datos stale (jueves) ->
# DATA_PENDING. El candado correcto es expected_session vs
# observed_last, no is_market_day(reference_date).
# ============================================================
def test_t_int_4b_sabado_con_datos_stale_devuelve_stale(tmp_path):
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-10')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    # Sin CSV previo y df_stocks stale -> fallback None, stale=True
    assert result is None
    assert is_stale is True


# ============================================================
# T-INT-8: as_of_date explicito sesion -> date == esa sesion
# ============================================================
def test_t_int_8_as_of_date_sesion_explicito():
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    session = date(2026, 9, 11)

    result = compute_sector_breadth(
        df_market, df_stocks, holdings, as_of_date=session)

    assert len(result) > 0
    assert result.iloc[0]['date'] == pd.Timestamp(session)


# ============================================================
# T-INT-9: as_of_date explicito no-sesion -> ValueError
# ============================================================
def test_t_int_9_as_of_date_no_sesion_valueerror():
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    sabado = date(2026, 9, 12)

    with pytest.raises(ValueError, match='no es sesion NYSE'):
        compute_sector_breadth(
            df_market, df_stocks, holdings, as_of_date=sabado)


# ============================================================
# T-INT-10: as_of_date=None -> df_stocks.index[-1]
# ============================================================
def test_t_int_10_as_of_date_none_usa_index():
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')

    result = compute_sector_breadth(df_market, df_stocks, holdings)

    assert len(result) > 0
    expected = pd.Timestamp(df_stocks.index[-1]).normalize()
    assert result.iloc[0]['date'] == expected


# ============================================================
# T-INT-10b: as_of_date=None + df_stocks vacio -> ValueError
# ============================================================
def test_t_int_10b_as_of_date_none_df_vacio_valueerror():
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    df_stocks_empty = df_stocks.iloc[0:0]

    with pytest.raises(ValueError, match='df_stocks vacio'):
        compute_sector_breadth(df_market, df_stocks_empty, holdings)


# ============================================================
# T-INT-11: reference_date sesion -> date correcta en CSV temporal
# ============================================================
def test_t_int_11_reference_date_sesion(tmp_path):
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=FRIDAY, output_path=out)

    assert result is not None
    assert is_stale is False
    assert len(result) > 0
    expected = pd.Timestamp(date(2026, 9, 11))
    # append_dedup normaliza date a string YYYY-MM-DD; parseamos antes de comparar.
    assert pd.to_datetime(result.iloc[0]['date']) == expected
    assert out.exists()
    csv = pd.read_csv(out)
    assert pd.Timestamp(csv.iloc[0]['date']) == expected


# ============================================================
# T-INT-12 (revisado 2026-10-10): sabado + datos hasta viernes ->
# escribe fila del viernes. Es el caso normal del slot 03:17 UTC del
# sabado: produce el cierre del viernes.
# ============================================================
def test_t_int_12_sabado_con_datos_hasta_viernes(tmp_path):
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    assert result is not None
    assert is_stale is False
    assert out.exists()
    assert pd.to_datetime(result.iloc[0]['date']).date() == date(2026, 9, 11)


# ============================================================
# Auditor: el CSV real no debe cambiar durante los tests
# ============================================================
def test_t_int_11b_reference_date_pre_publish(tmp_path):
    """reference_date antes de PUBLISH_HOUR -> expected_session = dia anterior.

    Documenta el comportamiento de market_calendar.PUBLISH_HOUR: a las 18:00
    el cierre del mismo dia aun no esta publicado, por lo que la sesion
    esperada es la del dia bursatil anterior.
    """
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=FRIDAY_PRE_PUBLISH, output_path=out)

    assert result is not None
    assert is_stale is False
    expected = pd.Timestamp(date(2026, 9, 10))  # jueves, dia bursatil anterior
    assert pd.to_datetime(result.iloc[0]['date']) == expected


def test_t_int_4c_sabado_no_toca_csv_real(tmp_path):
    """2026-10-10: la funcion escribe cuando el sabado produce el
    viernes, pero SIEMPRE debe respetar output_path. Nunca debe
    escribir al CSV real si se le pasa una ruta temporal.
    """
    real_csv = Path('outputs/history/sector_breadth.csv')
    if not real_csv.exists():
        pytest.skip("CSV real no existe")
    mtime_before = real_csv.stat().st_mtime
    size_before = real_csv.stat().st_size

    out = tmp_path / "sb.csv"
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    _ = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    # El CSV real no debe cambiar (aunque la funcion escriba al output_path)
    assert real_csv.stat().st_mtime == mtime_before, "el CSV real no debe cambiar"
    assert real_csv.stat().st_size == size_before

    # El output_path si debe haberse escrito
    assert out.exists()
    df_out = pd.read_csv(out)
    assert pd.to_datetime(df_out['date']).max().date() == date(2026, 9, 11)