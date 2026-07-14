import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.providers.finra import FinraProvider
from config.tickers import MARKET_TICKERS

def robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))
    return (series - median) / (1.4826 * mad)

def rolling_percentile(series):
    last = series.iloc[-1]
    return (series < last).mean() * 100

def classify_darkpool(z):
    if z >= 2.5:
        return "Acumulacion extrema"
    elif z >= 1.5:
        return "Acumulacion fuerte"
    elif z >= 0.5:
        return "Acumulacion moderada"
    elif z > -0.5:
        return "Neutral"
    elif z > -1.5:
        return "Distribucion moderada"
    elif z > -2.5:
        return "Distribucion fuerte"
    else:
        return "Distribucion extrema"

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
    except:
        pass
    return list(set([t for t in tickers if not t.startswith('^')]))

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
    except:
        pass
    return volumes

def compute_darkpool_signals():
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
    try:
        df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
        volumes.update(_get_volume_from_df(df_market, week_start, end_date_str))
    except:
        pass
    try:
        df_stocks = pd.read_csv('data/stock_prices.csv', header=[0,1], index_col=0, parse_dates=True)
        volumes.update(_get_volume_from_df(df_stocks, week_start, end_date_str))
    except:
        pass

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

    # --- Pipeline profesional: Historial de ratio agregado ---
    try:
        hist = pd.read_csv('outputs/darkpool_history.csv', parse_dates=['week'])
        hist = pd.concat([hist, pd.DataFrame([{'week': pd.to_datetime(week_start), 'ratio': media_dp / 100}])], ignore_index=True)
    except:
        hist = pd.DataFrame([{'week': pd.to_datetime(week_start), 'ratio': media_dp / 100}])

    if len(hist) >= 4:
        hist['ratio_ewm'] = hist['ratio'].ewm(span=4).mean()
    else:
        hist['ratio_ewm'] = hist['ratio']

    z = np.nan
    percentile = np.nan
    momentum = np.nan
    state = "Sin historial suficiente"

    if len(hist) >= 104:
        z_series = hist['ratio_ewm'].rolling(104).apply(lambda x: robust_zscore(pd.Series(x)).iloc[-1], raw=False)
        z = z_series.iloc[-1]
        percentile = rolling_percentile(hist['ratio_ewm'].iloc[-104:])
        momentum = z_series.ewm(span=4).mean().iloc[-1]
        state = classify_darkpool(z)

    hist.to_csv('outputs/darkpool_history.csv', index=False)

    return {
        'status': 'OK',
        'week': week_start,
        'fecha': datetime.now().strftime('%Y-%m-%d'),
        'media_dark_pool': media_dp,
        'z_score': z,
        'momentum': momentum,
        'percentile': percentile,
        'state': state,
        'n_tickers_ats': len(df_res[df_res['ats_volume'] > 0]),
        'n_tickers_total': len(df_res),
        'datos': df_res
    }
