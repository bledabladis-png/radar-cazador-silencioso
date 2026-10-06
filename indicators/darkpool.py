# -*- coding: utf-8 -*-
"""Dark Pools (FINRA ATS Transparency Data).

Rediseno K-DT3-AUDIT-01 (2026-10-06):
- Se elimina `momentum` del dict: era copia exacta de z_score.
- Se unifica `n_tickers_ats` / `n_tickers_total` en `n_tickers`.
- Se elimina `ratio_ewm` calculada en darkpool.py (dead code, se
  recalculaba dentro del scoring con otro span).
- `end_date` corregido: yfinance `end` es exclusivo, se necesita
  +5 dias para cubrir el viernes de la semana FINRA.
- `compute_window_stats` reemplaza a `_compute_z_for_window`.
"""
import os

import pandas as pd
from datetime import timedelta

from data.providers.finra import FinraProvider
from indicators.darkpool_scoring import (  # noqa: F401 (re-export)
    robust_zscore,
    rolling_percentile,
    classify_darkpool,
    compute_window_stats,
)
from indicators.darkpool_io import (  # noqa: F401 (re-export)
    _get_all_tickers,
    _get_volume_from_df,
)
from indicators.darkpool_history import (  # noqa: F401 (re-export)
    _backfill_history,
)

__all__ = [
    'compute_darkpool_signals',
    'robust_zscore',
    'rolling_percentile',
    'classify_darkpool',
    'compute_window_stats',
    '_get_all_tickers',
    '_get_volume_from_df',
    '_backfill_history',
]

# Ventanas progresivas. El z oficial usa la mas larga con datos.
_WINDOWS = (13, 26, 52, 104)
_HISTORY_PATH = 'outputs/history/darkpool_history.csv'
_HISTORY_MAX = 104


def compute_darkpool_signals(df_market=None, df_stocks=None):
    finra = FinraProvider()
    week_start = finra.get_latest_week()
    if not week_start:
        return None

    ats_data = finra.get_all_tiers(week_start)
    if ats_data.empty:
        return None
    if 'issueSymbolIdentifier' not in ats_data.columns:
        return None
    if 'totalWeeklyShareQuantity' not in ats_data.columns:
        return None
    ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
    ats_volume_dict = ats_volume.to_dict()

    # K-DT3-AUDIT-01: end_date +5 dias. yfinance `end` es exclusivo;
    # +4 dias solo cubre hasta el jueves, perdia el viernes.
    end_date = pd.to_datetime(week_start) + timedelta(days=5)
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Volumen total por ticker: primero market_data, luego stock_prices.
    # Si un ticker esta en ambos, gana stock_prices (universo mas amplio
    # de acciones individuales; market_data es principalmente ETFs).
    volumes = {}
    _df_market = df_market
    if _df_market is None:
        try:
            _df_market = pd.read_parquet('data/market_data.parquet')
        except (OSError, ValueError, pd.errors.ParserError) as e:
            print(f'  [WARN] darkpool: market_data.parquet no disponible: {e}')
    if _df_market is not None:
        volumes.update(_get_volume_from_df(_df_market, week_start, end_date_str))

    _df_stocks = df_stocks
    if _df_stocks is None:
        try:
            _df_stocks = pd.read_parquet('data/stock_prices.parquet')
        except (OSError, ValueError, pd.errors.ParserError) as e:
            print(f'  [WARN] darkpool: stock_prices.parquet no disponible: {e}')
    if _df_stocks is not None:
        volumes.update(_get_volume_from_df(_df_stocks, week_start, end_date_str))

    if not volumes:
        return None

    # Solo tickers con ATS>0. Dark pool pct = ATS / total. Se descarta
    # >100% (inconsistencia de calendario FINRA vs Yahoo).
    resultados = []
    for t, vol_total in volumes.items():
        vol_ats = ats_volume_dict.get(t, 0)
        if vol_ats <= 0:
            continue
        dark_pool_pct = (vol_ats / vol_total) * 100
        if dark_pool_pct <= 100:
            resultados.append({
                'ticker': t,
                'ats_volume': vol_ats,
                'total_volume': vol_total,
                'dark_pool_pct': dark_pool_pct,
            })

    if not resultados:
        return None

    df_res = pd.DataFrame(resultados)
    media_dp = float(df_res['dark_pool_pct'].mean())

    # Historico: leer, anexar la semana actual si no existe, backfill si corto.
    try:
        hist = pd.read_csv(_HISTORY_PATH, parse_dates=['week'])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist = pd.DataFrame(columns=['week', 'ratio'])

    current_week_date = pd.to_datetime(week_start)
    if current_week_date not in hist['week'].values:
        new_row = pd.DataFrame([{'week': current_week_date, 'ratio': media_dp / 100}])
        hist = pd.concat([hist, new_row], ignore_index=True)

    if len(hist) < _HISTORY_MAX:
        hist = _backfill_history(hist, finra)

    hist = hist.drop_duplicates(subset='week', keep='last')
    hist.sort_values('week', inplace=True)
    hist.reset_index(drop=True, inplace=True)

    # Stats por ventana. Cada ventana calcula z + percentile + state
    # sobre la misma serie suavizada.
    windows = {f'{w}w': compute_window_stats(hist, w) for w in _WINDOWS}

    # El resumen oficial es la ventana mas larga con z finito.
    summary = None
    for w in reversed(_WINDOWS):
        stats = windows[f'{w}w']
        if pd.notna(stats['z']):
            summary = stats
            break
    if summary is None:
        summary = {'z': float('nan'), 'percentile': float('nan'),
                   'state': 'Sin historial suficiente'}

    # Escritura atomica. lineterminator para no generar churn EOL.
    _tmp_path = _HISTORY_PATH + '.tmp'
    hist.to_csv(_tmp_path, index=False, lineterminator=chr(10))
    os.replace(_tmp_path, _HISTORY_PATH)

    return {
        'status': 'OK',
        'week': week_start,
        'fecha': week_start,  # alias historico de week
        'media_dark_pool': media_dp,
        'z_score': summary['z'],
        'percentile': summary['percentile'],
        'state': summary['state'],
        'n_tickers': len(df_res),
        'datos': df_res,
        'z_windows': {
            f'{w}w': windows[f'{w}w'] if pd.notna(windows[f'{w}w']['z']) else None
            for w in _WINDOWS
        },
    }
