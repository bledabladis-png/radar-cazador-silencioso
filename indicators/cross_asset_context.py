# -*- coding: utf-8 -*-
"""
Contexto Cross-Asset v1.1
Describe la relación de cada sector con activos transversales agrupados por familia.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np

from src.utils import _observation_date_from_df
from src.utils import get_col
from config.tickers import MARKET_TICKERS

SECTORS = MARKET_TICKERS['sectors']
CROSS_ASSET_FAMILIES = {
    'equity': ['SPY', 'QQQ', 'IWM'],
    'rates': ['TLT', 'IEF'],
    'credit': ['HYG', 'LQD'],
    'commodities': ['USO', 'GLD', 'DBC'],
    'fx': ['DX-Y.NYB', 'EURUSD=X'],
    'volatility': ['^VIX'],
}

def _get_returns_series(df, ticker):
    """Devuelve serie de retornos diarios. VIX usa diferencia de nivel."""
    if ticker == '^VIX':
        try:
            close = get_col(df, ticker, 'Close')
        except (KeyError, TypeError):
            if ticker in df.columns:
                close = df[ticker].astype(float)
            else:
                return pd.Series(dtype=float)
        return close.diff()
    else:
        try:
            close = get_col(df, ticker, 'Close')
        except (KeyError, TypeError):
            if ticker in df.columns:
                close = df[ticker].astype(float)
            else:
                return pd.Series(dtype=float)
        return close.pct_change(fill_method=None)

def compute_cross_asset_context(df_market, windows=(20, 60), min_obs_ratio=0.75):
    """
    Calcula correlaciones sector-activo transversal por familia.
    Devuelve (detalle_df, resumen_df).
    """
    if df_market is None:
        return pd.DataFrame(), pd.DataFrame()

    all_tickers = list(SECTORS) + [a for assets in CROSS_ASSET_FAMILIES.values() for a in assets]
    returns_dict = {}
    for t in all_tickers:
        r = _get_returns_series(df_market, t)
        if not r.empty:
            returns_dict[t] = r

    if len(returns_dict) < 8:
        return pd.DataFrame(), pd.DataFrame()

    returns_df = pd.DataFrame(returns_dict).sort_index().dropna(how='all')
    detail_rows = []
    summary_rows = []

    for window in windows:
        min_obs = max(int(window * min_obs_ratio), 5)
        panel = returns_df.iloc[-window:]

        for sector in SECTORS:
            if sector not in panel.columns:
                continue
            for family, assets in CROSS_ASSET_FAMILIES.items():
                fam_corrs = []
                for asset in assets:
                    if asset not in panel.columns:
                        continue
                    pair = panel[[sector, asset]].dropna()
                    n_obs = len(pair)
                    if n_obs < min_obs:
                        corr = np.nan
                    else:
                        corr = pair[sector].corr(pair[asset])
                    detail_rows.append({
                        'date': _observation_date_from_df(returns_df),
                        'window': window,
                        'sector': sector,
                        'asset_class': family,
                        'asset': asset,
                        'correlation': corr,
                        'n_obs_pair': n_obs,
                    })
                    fam_corrs.append(corr)

                valid = [c for c in fam_corrs if pd.notna(c)]
                n_assets_valid = len(valid)
                if n_assets_valid > 0:
                    s = pd.Series(valid)
                    mean_corr = s.mean()
                    median_corr = s.median()
                    min_corr = s.min()
                    max_corr = s.max()
                else:
                    mean_corr = median_corr = min_corr = max_corr = np.nan

                summary_rows.append({
                    'date': _observation_date_from_df(returns_df),
                    'window': window,
                    'sector': sector,
                    'asset_class': family,
                    'n_assets_valid': n_assets_valid,
                    'mean_corr': mean_corr,
                    'median_corr': median_corr,
                    'min_corr': min_corr,
                    'max_corr': max_corr,
                })

    detail_df = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)
    return detail_df, summary_df