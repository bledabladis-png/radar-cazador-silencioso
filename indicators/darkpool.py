# -*- coding: utf-8 -*-
import pandas as pd
from datetime import timedelta
from data.providers.finra import FinraProvider
from indicators.darkpool_scoring import (  # noqa: F401 (re-export)
    robust_zscore,
    rolling_percentile,
    classify_darkpool,
    _compute_z_for_window,
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
    # Re-exports internos (preservan la API historica del modulo).
    'robust_zscore',
    'rolling_percentile',
    'classify_darkpool',
    '_compute_z_for_window',
    '_get_all_tickers',
    '_get_volume_from_df',
    '_backfill_history',
]

def compute_darkpool_signals(df_market=None, df_stocks=None):
    finra = FinraProvider()
    week_start = finra.get_latest_week()
    if not week_start:
        return None

    ats_data = finra.get_all_tiers(week_start)
    if ats_data.empty:
        return None
    if 'issueSymbolIdentifier' in ats_data.columns and 'totalWeeklyShareQuantity' in ats_data.columns:
        ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
        ats_volume_dict = ats_volume.to_dict()
    else:
        return None

    end_date = pd.to_datetime(week_start) + timedelta(days=4)
    end_date_str = end_date.strftime('%Y-%m-%d')

    volumes = {}
    # FU-021-5 Fase 7 (Q-P.7): fuente canonica en memoria.
    # Fallback a parquet solo si el caller no propaga df_market/df_stocks.
    _df_market = df_market
    if _df_market is None:
        try:
            _df_market = pd.read_parquet('data/market_data.parquet')
        except Exception as e:
            print(f'  [WARN] darkpool: market_data.parquet no disponible: {e}')
    if _df_market is not None:
        volumes.update(_get_volume_from_df(_df_market, week_start, end_date_str))

    _df_stocks = df_stocks
    if _df_stocks is None:
        try:
            _df_stocks = pd.read_parquet('data/stock_prices.parquet')
        except Exception as e:
            print(f'  [WARN] darkpool: stock_prices.parquet no disponible: {e}')
    if _df_stocks is not None:
        volumes.update(_get_volume_from_df(_df_stocks, week_start, end_date_str))

    if not volumes:
        return None

    resultados = []
    for t, vol_total in volumes.items():
        vol_ats = ats_volume_dict.get(t, 0)
        if vol_ats > 0:
            dark_pool_pct = (vol_ats / vol_total) * 100
            if dark_pool_pct <= 100:
                resultados.append({
                    'ticker': t,
                    'ats_volume': vol_ats,
                    'total_volume': vol_total,
                    'dark_pool_pct': dark_pool_pct
                })

    if not resultados:
        return None

    df_res = pd.DataFrame(resultados)
    media_dp = df_res['dark_pool_pct'].mean()

    try:
        hist = pd.read_csv('outputs/history/darkpool_history.csv', parse_dates=['week'])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist = pd.DataFrame(columns=['week', 'ratio'])

    current_week_date = pd.to_datetime(week_start)
    if current_week_date not in hist['week'].values:
        new_row = pd.DataFrame([{'week': current_week_date, 'ratio': media_dp / 100}])
        hist = pd.concat([hist, new_row], ignore_index=True)

    if len(hist) < 104:
        hist = _backfill_history(hist, finra)

    hist = hist.drop_duplicates(subset='week', keep='last')
    hist.sort_values('week', inplace=True)
    hist.reset_index(drop=True, inplace=True)

    if len(hist) >= 4:
        hist['ratio_ewm'] = hist['ratio'].ewm(span=4).mean()
    else:
        hist['ratio_ewm'] = hist['ratio']

    # Ventanas progresivas
    z_13, mom_13, pct_13, state_13 = _compute_z_for_window(hist, 13)
    z_26, mom_26, pct_26, state_26 = _compute_z_for_window(hist, 26)
    z_52, mom_52, pct_52, state_52 = _compute_z_for_window(hist, 52)
    z_104, mom_104, pct_104, state_104 = _compute_z_for_window(hist, 104)

    # El Z-Score oficial es el de 104 semanas (o el mas largo disponible)
    z = z_104 if pd.notna(z_104) else (z_52 if pd.notna(z_52) else (z_26 if pd.notna(z_26) else z_13))
    momentum = mom_104 if pd.notna(mom_104) else (mom_52 if pd.notna(mom_52) else (mom_26 if pd.notna(mom_26) else mom_13))
    percentile = pct_104 if pd.notna(pct_104) else (pct_52 if pd.notna(pct_52) else (pct_26 if pd.notna(pct_26) else pct_13))
    state = state_104 if state_104 != "Sin historial suficiente" else (state_52 if state_52 != "Sin historial suficiente" else (state_26 if state_26 != "Sin historial suficiente" else state_13))

    hist.to_csv('outputs/history/darkpool_history.csv', index=False)

    return {
        'status': 'OK',
        'week': week_start,
        'fecha': week_start,
        'media_dark_pool': media_dp,
        'z_score': z,
        'momentum': momentum,
        'percentile': percentile,
        'state': state,
        'n_tickers_ats': len(df_res[df_res['ats_volume'] > 0]),
        'n_tickers_total': len(df_res),
        'datos': df_res,
        'z_windows': {
            '13w': {'z': z_13, 'state': state_13} if pd.notna(z_13) else None,
            '26w': {'z': z_26, 'state': state_26} if pd.notna(z_26) else None,
            '52w': {'z': z_52, 'state': state_52} if pd.notna(z_52) else None,
            '104w': {'z': z_104, 'state': state_104} if pd.notna(z_104) else None,
        }
    }

