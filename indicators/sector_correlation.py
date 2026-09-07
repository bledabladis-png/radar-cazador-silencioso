# -*- coding: utf-8 -*-
"""
Correlación entre sectores v1.0
Mide co-movimiento de retornos diarios de 11 sectores SPDR.
No alimenta motores, scores, pesos ni State Machine.
Consume retornos oficiales desde df_market.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']

def _classify_corr(mean_corr):
    if pd.isna(mean_corr):
        return 'N/D'
    if mean_corr >= 0.8:
        return 'Co-movimiento muy alto'
    elif mean_corr >= 0.6:
        return 'Co-movimiento alto'
    elif mean_corr >= 0.4:
        return 'Co-movimiento moderado'
    elif mean_corr >= 0.2:
        return 'Co-movimiento bajo'
    else:
        return 'Diferenciación elevada'

def compute_sector_correlation(df_market, windows=(20, 60), min_obs_ratio=0.75):
    """
    Calcula matrices de correlación de Pearson para retornos diarios de sectores.
    Devuelve (matrix_df, summary_df).
    """
    if df_market is None:
        return pd.DataFrame(), pd.DataFrame()

    # Obtener precios de cierre alineados por fecha
    closes = {}
    for ticker in SECTORS:
        try:
            # Formato MultiIndex del sistema
            closes[ticker] = get_col(df_market, ticker, 'Close')
        except (KeyError, TypeError):
            # Formato simple: columnas ticker
            if ticker in df_market.columns:
                closes[ticker] = df_market[ticker].astype(float)
            else:
                continue

    if len(closes) < 8:
        return pd.DataFrame(), pd.DataFrame()

    prices_df = pd.DataFrame(closes).sort_index()
    returns_df = prices_df.pct_change(fill_method=None).dropna(how='all')

    matrix_rows = []
    summary_rows = []

    for window in windows:
        if len(returns_df) < window:
            continue
        # Ventana común: últimas 'window' filas completas
        panel = returns_df.iloc[-window:]
        # Eliminar columnas con demasiados NaN dentro de la ventana
        min_obs = int(window * min_obs_ratio)
        panel_clean = panel.dropna(axis=1, thresh=min_obs)

        n_sectors = panel_clean.shape[1]
        if n_sectors < 8:
            summary_rows.append({
                'date': pd.Timestamp.now().normalize(),
                'window': window,
                'n_sectors': n_sectors,
                'n_valid_pairs': 0,
                'corr_mean': np.nan,
                'corr_median': np.nan,
                'corr_p25': np.nan,
                'corr_p75': np.nan,
                'corr_min': np.nan,
                'corr_max': np.nan,
                'correlation_reading': 'N/D',
            })
            continue

        # Para cada par, exigir min_obs observaciones comunes
        corr_matrix = panel_clean.corr(min_periods=min_obs)

        pairs = []
        for i, s1 in enumerate(panel_clean.columns):
            for s2 in panel_clean.columns[i+1:]:
                corr_val = corr_matrix.loc[s1, s2]
                n_obs = panel[[s1, s2]].dropna().shape[0]
                pairs.append({
                    'date': pd.Timestamp.now().normalize(),
                    'window': window,
                    'sector1': s1,
                    'sector2': s2,
                    'correlation': corr_val,
                    'n_obs_pair': n_obs,
                })

        # Resumen
        pair_corrs = [p['correlation'] for p in pairs if pd.notna(p['correlation'])]
        n_valid_pairs = len(pair_corrs)
        if n_valid_pairs == 0:
            mean_corr = np.nan
            median_corr = np.nan
            p25 = np.nan
            p75 = np.nan
            min_corr = np.nan
            max_corr = np.nan
        else:
            s = pd.Series(pair_corrs)
            mean_corr = s.mean()
            median_corr = s.median()
            p25 = s.quantile(0.25)
            p75 = s.quantile(0.75)
            min_corr = s.min()
            max_corr = s.max()

        summary_rows.append({
            'date': pd.Timestamp.now().normalize(),
            'window': window,
            'n_sectors': n_sectors,
            'n_valid_pairs': n_valid_pairs,
            'corr_mean': mean_corr,
            'corr_median': median_corr,
            'corr_p25': p25,
            'corr_p75': p75,
            'corr_min': min_corr,
            'corr_max': max_corr,
            'correlation_reading': _classify_corr(mean_corr),
        })

        matrix_rows.extend(pairs)

    matrix_df = pd.DataFrame(matrix_rows) if matrix_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    return matrix_df, summary_df