# -*- coding: utf-8 -*-
"""FASE-A (2026-11-01): cobertura dual derivada de datos."""
import pandas as pd
import pytest

from indicators.sector_concentration import compute_sector_concentration


def _make_df_stocks(tickers, n=60):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    data = {}
    for i, t in enumerate(tickers):
        data[(t, 'Close')] = [10 + i + 0.1 * j for j in range(n)]
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_cob_sobre_descarga_100_si_todos_descargados():
    tickers = ['A', 'B', 'C']
    df_stocks = _make_df_stocks(tickers)
    metrics = pd.DataFrame({
        'ticker': tickers, 'sector': ['XLK'] * 3,
        'rs': [1.0, 0.9, 1.1], 'rs_mom': [0.01, -0.02, 0.03],
        'flow_proxy_z': [0.1, 0.2, 0.3],
        'wyckoff_score': [0.5, 0.6, 0.7],
        'wls': [1.0, 0.7, 1.2],
    })
    holdings = pd.DataFrame({
        'etf': ['XLK'] * 3, 'ticker': tickers,
        'weight': [0.4, 0.35, 0.25],
    })
    result = compute_sector_concentration(
        df_stocks, holdings, metrics, reference_date=df_stocks.index[-1])
    r = result.iloc[0]
    assert r['n_descargados'] == 3
    assert r['n_total'] == 3
    assert r['coverage_rs_sobre_descarga'] == pytest.approx(100.0)
    assert r['peso_total_etf'] == pytest.approx(1.0)
    assert r['coverage_peso'] == pytest.approx(100.0)


def test_cob_peso_parcial():
    tickers_desc = ['A', 'B', 'C']
    df_stocks = _make_df_stocks(tickers_desc)
    metrics = pd.DataFrame({
        'ticker': tickers_desc, 'sector': ['XLK'] * 3,
        'rs': [1.0, 0.9, 1.1], 'rs_mom': [0.01, -0.02, 0.03],
        'flow_proxy_z': [0.1, 0.2, 0.3],
        'wyckoff_score': [0.5, 0.6, 0.7],
        'wls': [1.0, 0.7, 1.2],
    })
    holdings = pd.DataFrame({
        'etf': ['XLK'] * 5,
        'ticker': ['A', 'B', 'C', 'D', 'E'],
        'weight': [0.2, 0.15, 0.15, 0.3, 0.2],
    })
    result = compute_sector_concentration(
        df_stocks, holdings, metrics, reference_date=df_stocks.index[-1])
    r = result.iloc[0]
    assert r['n_descargados'] == 3
    assert r['n_total'] == 5
    assert r['coverage_rs_sobre_descarga'] == pytest.approx(100.0)
    assert r['coverage_peso'] == pytest.approx((0.2 + 0.15 + 0.15) / 1.0 * 100)
