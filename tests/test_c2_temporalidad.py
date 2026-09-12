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
# T-INT-4: dia no bursatil -> no se genera observacion
# ============================================================
def test_t_int_4_dia_no_bursatil_no_genera(tmp_path):
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    # C2F: sin CSV historico, fallback devuelve (None, True).
    assert result is None
    assert is_stale is True
    assert not out.exists(), "no debe crear CSV en dia no bursatil"


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
# T-INT-12: reference_date sabado + datos hasta viernes -> no observacion
# ============================================================
def test_t_int_12_sabado_con_datos_hasta_viernes(tmp_path):
    # Caso critico: df_stocks.index[-1] = viernes (sesion), reference_date = sabado.
    # El comportamiento correcto es NO generar, aunque la ultima fecha del df
    # sea una sesion valida. Sin reference_date no seria detectable.
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    out = tmp_path / "sb.csv"

    result, is_stale = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY, output_path=out)

    # C2F: sin CSV historico, fallback devuelve (None, True).
    assert result is None
    assert is_stale is True
    assert not out.exists()


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


def test_csv_real_no_modificado():
    real_csv = Path('outputs/history/sector_breadth.csv')
    if not real_csv.exists():
        pytest.skip("CSV real no existe")
    mtime_before = real_csv.stat().st_mtime
    size_before = real_csv.stat().st_size
    # Ejecutar una operacion que escribia al CSV real sin output_path
    df_market, df_stocks, holdings = _mk_market_and_stocks(end_date='2026-09-11')
    _ = _compute_sector_breadth_health(
        df_stocks, df_market, holdings,
        reference_date=SATURDAY)  # sabado -> no escribe
    mtime_after = real_csv.stat().st_mtime
    size_after = real_csv.stat().st_size
    assert mtime_before == mtime_after, "el CSV real no debe cambiar"
    assert size_before == size_after