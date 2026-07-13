import pandas as pd
from datetime import datetime, timedelta
from data.providers.finra import FinraProvider
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
    except:
        pass
    return list(set([t for t in tickers if not t.startswith('^')]))

def _get_volume_from_df(df, week_start, end_date_str):
    """Extrae un diccionario {ticker: volumen_total} de un DataFrame con MultiIndex."""
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

    # Datos ATS de FINRA
    ats_data = finra.get_all_tiers(week_start)
    if ats_data.empty:
        return None
    if 'issueSymbolIdentifier' in ats_data.columns and 'totalWeeklyShareQuantity' in ats_data.columns:
        ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
        ats_volume = ats_volume.to_dict()
    else:
        return None

    end_date = pd.to_datetime(week_start) + timedelta(days=4)
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Volúmenes de mercado (market_data.csv)
    volumes = {}
    try:
        df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
        volumes.update(_get_volume_from_df(df_market, week_start, end_date_str))
    except:
        pass

    # Volúmenes de acciones líderes (stock_prices.csv)
    try:
        df_stocks = pd.read_csv('data/stock_prices.csv', header=[0,1], index_col=0, parse_dates=True)
        volumes.update(_get_volume_from_df(df_stocks, week_start, end_date_str))
    except:
        pass

    if not volumes:
        return None

    # Calcular Dark Pool % para todos los tickers disponibles
    resultados = []
    for t, vol_total in volumes.items():
        vol_ats = ats_volume.get(t, 0)
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
    max_row = df_res.loc[df_res['dark_pool_pct'].idxmax()]

    return {
        'status': 'OK',
        'week': week_start,
        'fecha': datetime.now().strftime('%Y-%m-%d'),
        'media_dark_pool': media_dp,
        'ticker_max': max_row['ticker'],
        'max_dark_pool': max_row['dark_pool_pct'],
        'n_tickers_ats': len(df_res[df_res['ats_volume'] > 0]),
        'n_tickers_total': len(df_res),
        'datos': df_res
    }
