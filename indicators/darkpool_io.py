"""DT3 Fase 3: IO de darkpool (tickers + volumenes).

Funciones con acceso a CSV/config. Re-exportadas por
indicators/darkpool.py para preservar API interna.
"""
import re

import pandas as pd

from config.tickers import MARKET_TICKERS


def _get_all_tickers():
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)
    try:
        holdings = pd.read_csv('data/etf_holdings.csv')
        if 'ticker' in holdings.columns:
            tickers.extend(holdings['ticker'].tolist())
    except Exception as e:
        print(f'  [WARN] darkpool: etf_holdings.csv no disponible: {e}')
    # Filtro adicional: solo tickers con formato razonable de acción
    tickers = [t for t in tickers if not t.startswith('^')]
    valid = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        # Ticker típico: 1-6 caracteres, letras, punto o guion, no empezar por número
        if re.fullmatch(r'[A-Z][A-Z0-9.-]{0,5}', t) and not t.startswith('-'):
            valid.append(t)
    return list(set(valid))


def _get_volume_from_df(df, week_start, end_date_str):
    volumes = {}
    try:
        week_data = df.loc[week_start:end_date_str]
        for col in week_data.columns:
            if col[0] == 'Volume':
                ticker = col[1]
                vol = week_data[col].sum()
                if pd.notna(vol) and vol > 0:
                    volumes[ticker] = vol
    except Exception as e:
        print(f'  [WARN] darkpool: error extrayendo volumenes: {e}')
    return volumes
