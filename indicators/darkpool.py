import pandas as pd
from datetime import datetime, timedelta
from data.providers.finra import FinraProvider

def compute_darkpool_signals():
    finra = FinraProvider()
    # Forzar descarga sin verificar is_available()
    week_start = finra.get_latest_week()
    if not week_start:
        return None

    # Descargar datos de los tres mercados (T1, T2, OTCE)
    ats_data = finra.get_all_tiers(week_start)
    if ats_data.empty:
        return None

    # Agrupar por símbolo y sumar volumen ATS
    if 'issueSymbolIdentifier' in ats_data.columns and 'totalWeeklyShareQuantity' in ats_data.columns:
        ats_volume = ats_data.groupby('issueSymbolIdentifier')['totalWeeklyShareQuantity'].sum()
        ats_volume = ats_volume.reset_index(name='ats_volume')
    else:
        return None

    # Cargar volumen total de Yahoo Finance para la misma semana
    try:
        df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
    except FileNotFoundError:
        return None

    # La semana de FINRA empieza en week_start (lunes) y termina 4 días después (viernes)
    end_date = pd.to_datetime(week_start) + timedelta(days=4)
    week_data = df_market.loc[week_start:end_date.strftime('%Y-%m-%d')]

    if week_data.empty:
        return None

    # Tickers de interés (sin GLD, SLV, USO, UNG que no estan en Yahoo)
    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'TLT', 'HYG', 'LQD', 'EEM']
    resultados = []
    for t in tickers:
        try:
            vol_total = week_data[('Volume', t)].sum()
            row = ats_volume.loc[ats_volume['issueSymbolIdentifier'] == t, 'ats_volume']
            vol_ats = row.values[0] if len(row) > 0 else 0
            if vol_total > 0:
                dark_pool_pct = (vol_ats / vol_total) * 100
                resultados.append({
                    'ticker': t,
                    'ats_volume': vol_ats,
                    'total_volume': vol_total,
                    'dark_pool_pct': dark_pool_pct
                })
        except:
            pass

    if not resultados:
        return None

    df_res = pd.DataFrame(resultados)
    # Filtrar valores anómalos (mayores al 100% indican desalineacion de fechas)
    df_res = df_res[df_res['dark_pool_pct'] <= 100]
    if df_res.empty:
        return None

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
