# -*- coding: utf-8 -*-
"""FU-020/R2: compute_sector_scores debe resolver effective_date.

Contrato: cuando el df tiene un festivo (o dia no bursatil) al final
con close NaN, el score publicado NO debe ser el del festivo. Debe
ser el de la ultima fecha con cobertura suficiente del universo
sectorial.

Aplicable a: 2018-12-05 (Bush), 2025-01-09 (Carter), y cualquier
festivo NYSE con fila vacia en el parquet.
"""
import numpy as np
import pandas as pd

from config.tickers import MARKET_TICKERS
from regimes.sector_regime import compute_sector_scores


def _make_df_with_trailing_holiday():
    """300 dias bursatiles + 1 festivo NYSE al final con close NaN."""
    sectors = MARKET_TICKERS["sectors"]
    bdays = pd.date_range(end="2025-08-29", periods=300, freq="B")
    holiday = pd.Timestamp("2025-09-01")
    all_dates = bdays.append(pd.DatetimeIndex([holiday]))

    rng = np.random.RandomState(42)
    data = {}
    data[("Close", "^GSPC")] = list(
        100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
    ) + [np.nan]
    sorted_sectors = sorted(sectors)
    for s in sorted_sectors:
        k = sorted_sectors.index(s)
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, 300))
        data[("Close", s)] = list(close) + [np.nan]
        data[("Open", s)] = list(close * 0.998) + [np.nan]
        data[("High", s)] = list(close * 1.01) + [np.nan]
        data[("Low", s)] = list(close * 0.99) + [np.nan]
        data[("Volume", s)] = list(
            rng.randint(1000000, 5000000, 300).astype(float)
        ) + [np.nan]
    df = pd.DataFrame(data, index=pd.DatetimeIndex(all_dates))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df, holiday


def test_scores_index_no_incluye_festivo_final():
    """El ultimo indice de scores no puede ser el festivo."""
    df, holiday = _make_df_with_trailing_holiday()
    res = compute_sector_scores(df)
    assert res is not None
    last_idx = res["scores"].index[-1]
    assert last_idx != holiday, (
        f"score final no puede ser el festivo {holiday}, es {last_idx}"
    )


def test_last_scores_corresponde_a_dia_con_datos():
    """last_scores debe tener valores validos (no todo NaN)."""
    df, holiday = _make_df_with_trailing_holiday()
    res = compute_sector_scores(df)
    assert res is not None
    ls = res["last_scores"]
    assert ls.notna().all(), (
        f"last_scores tiene NaN: {ls[ls.isna()].to_dict()}"
    )


def test_sin_festivo_final_no_hay_efecto():
    """Sin festivo al final, el comportamiento es el de siempre."""
    sectors = MARKET_TICKERS["sectors"]
    bdays = pd.date_range(end="2025-08-29", periods=300, freq="B")
    rng = np.random.RandomState(42)
    data = {}
    data[("Close", "^GSPC")] = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300))
    sorted_sectors = sorted(sectors)
    for s in sorted_sectors:
        k = sorted_sectors.index(s)
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, 300))
        data[("Close", s)] = close
        data[("Open", s)] = close * 0.998
        data[("High", s)] = close * 1.01
        data[("Low", s)] = close * 0.99
        data[("Volume", s)] = rng.randint(1000000, 5000000, 300).astype(float)
    df = pd.DataFrame(data, index=bdays)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    res = compute_sector_scores(df)
    assert res is not None
    assert res["scores"].index[-1] == pd.Timestamp("2025-08-29")