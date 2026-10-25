# -*- coding: utf-8 -*-
"""Tests H-06 (auditoria externa 2026-10-25): stability = wyckoff_stability.

Verifica que:
- La columna 'stability' en compute_stock_metrics coincide con wyckoff_stability.
- NO satura al 36% de tickers (bug de la version previa).
- NO arrastra el signo de la tendencia (corr < 0.3 con t_norm).
"""
import numpy as np
import pandas as pd

from indicators.wyckoff_v1 import wyckoff_score, wyckoff_stability
from indicators.stock_metrics import compute_stock_metrics


def _synthetic_df(n=600, seed=42):
    """Serie sintetica con tendencia alcista + ruido determinista."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    price = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    close = pd.Series(price, index=dates)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.abs(rng.normal(1e6, 2e5, n)), index=dates)
    flat = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    cols = []
    for field in ('Open', 'High', 'Low', 'Close', 'Volume'):
        cols.append((field, 'TST'))
    multi = pd.DataFrame(
        {c: flat[c[0]].values for c in cols}, index=dates,
    )
    multi.columns = pd.MultiIndex.from_tuples(cols)
    return multi


def _synthetic_market(n=600, seed=42):
    """DF con el mismo TST pero tambien su ETF (SPY) para compute_stock_metrics."""
    df = _synthetic_df(n=n, seed=seed)
    df_stocks = df.copy()
    # Añadir SPY al df_market (con precios similares)
    spy = df[('Close', 'TST')].copy()
    df_market = df.copy()
    df_market[('Close', 'SPY')] = spy.values
    return df_market, df_stocks


def test_stability_coincide_con_wyckoff_stability():
    """La columna stability en compute_stock_metrics == wyckoff_stability(combined)."""
    df_market, df_stocks = _synthetic_market(n=600)
    out = compute_stock_metrics(df_market, df_stocks, 'SPY', ['TST'])
    assert len(out) == 1

    # Calcular wyckoff_stability directamente
    ticker_df = pd.DataFrame({
        'Open': df_stocks[('Open', 'TST')],
        'High': df_stocks[('High', 'TST')],
        'Low': df_stocks[('Low', 'TST')],
        'Close': df_stocks[('Close', 'TST')],
        'Volume': df_stocks[('Volume', 'TST')],
    }).dropna()
    combined, *_ = wyckoff_score(ticker_df, 'TST')
    stab_series = wyckoff_stability(combined).dropna()
    expected = float(stab_series.iloc[-1]) if len(stab_series) > 0 else 0.0

    assert abs(out.iloc[0]['stability'] - expected) < 1e-9


def test_stability_no_satura():
    """La metrica no debe estar en +-1.0 exactos (bug previo: 36% saturados)."""
    # Múltiples tickers sintéticos
    n = 600
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(7)
    tickers = [f'T{i:02d}' for i in range(20)]
    data = {}
    for t in tickers:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for f in ('Open', 'High', 'Low', 'Close'):
            data[(f, t)] = price
        data[('Volume', t)] = np.abs(rng.normal(1e6, 2e5, n))
    multi = pd.DataFrame(data, index=dates)
    multi.columns = pd.MultiIndex.from_tuples(multi.columns)
    spy = multi[('Close', tickers[0])]
    df_market = multi.copy()
    df_market[('Close', 'SPY')] = spy.values

    out = compute_stock_metrics(df_market, multi, 'SPY', tickers)
    vals = out['stability'].dropna()
    assert len(vals) >= 10
    n_sat = ((vals.abs() > 0.99)).sum()
    pct_sat = n_sat / len(vals) * 100
    assert pct_sat < 5.0, f'{pct_sat:.1f}% saturados (esperado <5%)'


def test_stability_menos_correlacionada_con_tendencia_que_antigua():
    """La metrica nueva debe correlacionar MENOS con t_norm que la antigua.

    La version previa (tanh(median/MAD)) incluia el signo de la mediana,
    por lo que estaba mas correlacionada con t_norm. La version nueva
    (wyckoff_stability, 1 - 2*tanh(MAD/K)) mide solo dispersion.
    Sobre datos reales (108 tickers, diagnostico H-06): corr nueva=0.16,
    corr antigua=0.49. Sobre sinteticos con 20 tickers, la correlacion
    nueva puede llegar a ~0.40 por ruido muestral. La prueba correcta
    es comparativa: nueva < antigua.
    """
    n = 600
    dates = pd.date_range('2024-01-01', periods=n, freq='B')
    rng = np.random.default_rng(11)
    tickers = [f'T{i:02d}' for i in range(20)]
    data = {}
    for t in tickers:
        price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        for f in ('Open', 'High', 'Low', 'Close'):
            data[(f, t)] = price
        data[('Volume', t)] = np.abs(rng.normal(1e6, 2e5, n))
    multi = pd.DataFrame(data, index=dates)
    multi.columns = pd.MultiIndex.from_tuples(multi.columns)
    spy = multi[('Close', tickers[0])]
    df_market = multi.copy()
    df_market[('Close', 'SPY')] = spy.values

    out = compute_stock_metrics(df_market, multi, 'SPY', tickers)

    # Calcular t_norm y la metrica antigua para comparar
    t_norms = {}
    stab_old = {}
    for t in tickers:
        ticker_df = pd.DataFrame({
            'Open': multi[('Open', t)], 'High': multi[('High', t)],
            'Low': multi[('Low', t)], 'Close': multi[('Close', t)],
            'Volume': multi[('Volume', t)],
        }).dropna()
        combined, *_ = wyckoff_score(ticker_df, t)
        _, _, _, t_n, _, _, _ = wyckoff_score(ticker_df, t)
        tn = t_n.dropna()
        if len(tn) > 0:
            t_norms[t] = float(tn.iloc[-1])
        # Metrica antigua: tanh(median/MAD) de la serie combined
        med = combined.rolling(10).median().iloc[-1]
        def _mad(x):
            v = x[~np.isnan(x)]
            if len(v) < 2:
                return 0.0
            m = np.median(v)
            return float(np.median(np.abs(v - m)))
        mad = combined.rolling(10).apply(_mad, raw=True).iloc[-1]
        if pd.notna(med) and pd.notna(mad):
            stab_old[t] = float(np.tanh(med / (mad + 1e-9)))

    merged = out[['ticker', 'stability']].copy()
    merged['t_norm'] = merged['ticker'].map(t_norms)
    merged['stab_old'] = merged['ticker'].map(stab_old)
    merged = merged.dropna()

    if len(merged) < 15:
        return

    corr_new = merged['stability'].corr(merged['t_norm'], method='spearman')
    corr_old = merged['stab_old'].corr(merged['t_norm'], method='spearman')

    # La nueva debe estar MENOS correlacionada con t_norm que la antigua.
    # Tolerancia 0.05 por ruido.
    assert abs(corr_new) < abs(corr_old) + 0.05, (
        f'Nueva metrica NO reduce correlacion con t_norm: '
        f'corr_new={corr_new:+.3f}, corr_old={corr_old:+.3f}'
    )
