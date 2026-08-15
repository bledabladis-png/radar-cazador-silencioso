"""
Flujo de posicionamiento institucional desde CFTC TFF.
Calcula CFTC_POSITION_FLOW = Change_in_Net_Position por participante.
Fuente: https://publicreporting.cftc.gov/api/v3/views/gpe5-46if/export.csv
"""
import pandas as pd
from io import StringIO
import requests
from pathlib import Path
from datetime import datetime, timedelta

CFTC_TFF_URL = "https://publicreporting.cftc.gov/api/v3/views/gpe5-46if/export.csv"
CACHE_PATH = Path('data/cache/cftc_tff.csv')
HISTORY_PATH = Path('outputs/history/cftc_position_flow.csv')

TARGET_CONTRACTS = [
    'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE',
    'NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE',
    'RUSSELL E-MINI - CHICAGO MERCANTILE EXCHANGE',
    'DJIA Consolidated - CHICAGO BOARD OF TRADE',
    'VIX FUTURES - CBOE FUTURES EXCHANGE',
    'UST 10Y NOTE - CHICAGO BOARD OF TRADE',
]

def _download_and_cache():
    """Descarga CSV de CFTC y lo guarda en caché si no existe o si han pasado >23h."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    use_cache = CACHE_PATH.exists()
    if use_cache:
        mtime = datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
        if datetime.now() - mtime > timedelta(hours=23):
            use_cache = False

    if use_cache:
        print('  Usando caché CFTC TFF')
        return pd.read_csv(CACHE_PATH, low_memory=False)

    print('  Descargando CFTC TFF (Futures Only)...')
    r = requests.post(
        CFTC_TFF_URL,
        params={'accessType': 'DOWNLOAD'},
        headers={'User-Agent': 'Mozilla/5.0'},
        timeout=120
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), low_memory=False)
    df.to_csv(CACHE_PATH, index=False)
    print(f'  CFTC TFF guardado en caché: {len(df)} filas')
    return df

def _calculate_position_flow(df):
    """Calcula flujo de posicionamiento por contrato y participante."""
    if 'Market_and_Exchange_Names' not in df.columns:
        raise ValueError('No se encontró Market_and_Exchange_Names')

    # Seleccionar contratos objetivo
    mask = df['Market_and_Exchange_Names'].isin(TARGET_CONTRACTS)
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    # Parsear fecha
    df['date'] = pd.to_datetime(df['Report_Date_as_YYYY_MM_DD'], format='%Y %b %d %I:%M:%S %p', errors='coerce')
    df = df.dropna(subset=['date'])

    # Convertir columnas numéricas relevantes
    numeric_cols = [
        'Asset_Mgr_Positions_Long_All','Asset_Mgr_Positions_Short_All',
        'Lev_Money_Positions_Long_All','Lev_Money_Positions_Short_All',
        'Dealer_Positions_Long_All','Dealer_Positions_Short_All',
        'Change_in_Asset_Mgr_Long_All','Change_in_Asset_Mgr_Short_All',
        'Change_in_Lev_Money_Long_All','Change_in_Lev_Money_Short_All',
        'Change_in_Dealer_Long_All','Change_in_Dealer_Short_All',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    rows = []
    participants = ['asset_mgr', 'lev_money', 'dealer']

    for contract, group in df.groupby('Market_and_Exchange_Names'):
        group = group.sort_values('date')
        for participant in participants:
            if participant == 'asset_mgr':
                long_col = 'Asset_Mgr_Positions_Long_All'
                short_col = 'Asset_Mgr_Positions_Short_All'
                chg_long = 'Change_in_Asset_Mgr_Long_All'
                chg_short = 'Change_in_Asset_Mgr_Short_All'
            elif participant == 'lev_money':
                long_col = 'Lev_Money_Positions_Long_All'
                short_col = 'Lev_Money_Positions_Short_All'
                chg_long = 'Change_in_Lev_Money_Long_All'
                chg_short = 'Change_in_Lev_Money_Short_All'
            else:
                long_col = 'Dealer_Positions_Long_All'
                short_col = 'Dealer_Positions_Short_All'
                chg_long = 'Change_in_Dealer_Long_All'
                chg_short = 'Change_in_Dealer_Short_All'

            if long_col not in group.columns or short_col not in group.columns:
                continue

            net_position = group[long_col] - group[short_col]
            if chg_long in group.columns and chg_short in group.columns:
                position_change = group[chg_long] - group[chg_short]
            else:
                position_change = net_position.diff()

            temp = pd.DataFrame({
                'date': group['date'].values,
                'contract': contract,
                'participant': participant,
                'net_position': net_position.values,
                'position_change': position_change.values,
            })

            # Z-score rodante de 52 semanas
            temp['flow_z'] = (
                (temp['position_change'] - temp['position_change'].rolling(52, min_periods=10).mean())
                / (temp['position_change'].rolling(52, min_periods=10).std() + 1e-9)
            )
            rows.append(temp)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result = result.dropna(subset=['position_change'])
    return result

def get_cftc_position_flow_data() -> pd.DataFrame:
    """Descarga, procesa y devuelve los últimos flujos CFTC por contrato y participante."""
    try:
        raw = _download_and_cache()
        result = _calculate_position_flow(raw)
        if result.empty:
            return result

        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(HISTORY_PATH, index=False)
        print(f'  Histórico CFTC guardado: {HISTORY_PATH}')

        # Último dato por contrato y participante
        last = result.sort_values('date').groupby(['contract','participant']).tail(1)
        return last.reset_index(drop=True)
    except Exception as e:
        print(f'  Error en CFTC Position Flow: {e}')
        return pd.DataFrame()


