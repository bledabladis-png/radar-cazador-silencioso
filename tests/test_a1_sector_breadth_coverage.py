# -*- coding: utf-8 -*-
"""Tests de regresion A1: cobertura contra top-20 componentes, no ETF completo.

Contexto: compute_sector_breadth contaba n_total como todos los componentes
del ETF (78 para XLF). Pero solo se descargan top-20 por weight. El resultado
era una cobertura artificialmente baja (26%) con marca [BAJA].

Fix: replicar el head(TOP_N_SECTOR_COMPONENTS) que aplica get_stock_list().
"""

import pandas as pd

from config.settings import TOP_N_SECTOR_COMPONENTS
from indicators.sector_breadth import compute_sector_breadth


def _make_df_market():
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    return pd.DataFrame({("Close", "XLF"): range(300)}, index=idx)


def _make_df_stocks(tickers, n_rows=300):
    idx = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    data = {}
    for t in tickers:
        data[("Close", t)] = [100 + i * 0.1 for i in range(n_rows)]
    return pd.DataFrame(data, index=idx)


def test_a1_n_total_respeta_cap():
    """n_total nunca debe exceder TOP_N_SECTOR_COMPONENTS."""
    # 50 componentes en el ETF, pero solo 20 se descargan
    tickers_etf = [f"T{i:02d}" for i in range(50)]
    tickers_descargados = tickers_etf[:20]

    holdings = pd.DataFrame({
        "etf": ["XLF"] * 50,
        "ticker": tickers_etf,
        "weight": [1.0 - i * 0.01 for i in range(50)],
    })

    df_stocks = _make_df_stocks(tickers_descargados)
    df_market = _make_df_market()

    result = compute_sector_breadth(
        df_market, df_stocks, holdings,
        as_of_date=df_stocks.index[-1],
    )
    assert len(result) == 1
    assert result.iloc[0]["n_total"] == TOP_N_SECTOR_COMPONENTS


def test_a1_n_total_menor_si_etf_tiene_menos():
    """Si el ETF tiene < TOP_N componentes, n_total = ese numero."""
    tickers = [f"T{i}" for i in range(8)]
    holdings = pd.DataFrame({
        "etf": ["XLB"] * 8,
        "ticker": tickers,
        "weight": [1.0] * 8,
    })
    df_stocks = _make_df_stocks(tickers)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df_market = pd.DataFrame({("Close", "XLB"): range(300)}, index=idx)

    result = compute_sector_breadth(
        df_market, df_stocks, holdings,
        as_of_date=df_stocks.index[-1],
    )
    assert result.iloc[0]["n_total"] == 8


def test_a1_orden_por_weight_respetado():
    """Los top-20 deben ser los de mayor weight, no los primeros del CSV."""
    # 30 componentes; los 20 descargados son los de MAYOR weight (que NO son
    # los primeros del CSV)
    tickers_todos = [f"T{i:02d}" for i in range(30)]
    weights_todos = [0.01 * (30 - i) for i in range(30)]  # descendente: T00 mayor
    # En el CSV los pongo en orden inverso (peor primero)
    orden_csv = list(range(30))[::-1]
    holdings = pd.DataFrame({
        "etf": ["XLF"] * 30,
        "ticker": [tickers_todos[i] for i in orden_csv],
        "weight": [weights_todos[i] for i in orden_csv],
    })

    # Descargados: los 20 de mayor weight (T00..T19)
    tickers_descargados = tickers_todos[:20]
    df_stocks = _make_df_stocks(tickers_descargados)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    df_market = pd.DataFrame({("Close", "XLF"): range(300)}, index=idx)

    result = compute_sector_breadth(
        df_market, df_stocks, holdings,
        as_of_date=df_stocks.index[-1],
    )
    # Los 20 iterados deben ser T00..T19 -> 20 validos
    assert result.iloc[0]["n_total"] == TOP_N_SECTOR_COMPONENTS
    assert result.iloc[0]["n_valid_ema20"] == TOP_N_SECTOR_COMPONENTS


def test_a1_constante_existe():
    """La constante debe existir y valer 20."""
    assert TOP_N_SECTOR_COMPONENTS == 20