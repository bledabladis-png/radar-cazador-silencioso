
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
    'S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE',
    'NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE',
    'RUSSELL 2000 MINI INDEX FUTURE - ICE FUTURES U.S.',
    'DJIA Consolidated - CHICAGO BOARD OF TRADE',
    'VIX FUTURES - CBOE FUTURES EXCHANGE',
    '10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE',
]

PARTICIPANT_COLS = {
    'asset_mgr': ('Asset_Mgr_Positions_Long_All', 'Asset_Mgr_Positions_Short_All',
                  'Change_in_Asset_Mgr_Long_All', 'Change_in_Asset_Mgr_Short_All'),
    'lev_money': ('Lev_Money_Positions_Long_All', 'Lev_Money_Positions_Short_All',
                  'Change_in_Lev_Money_Long_All', 'Change_in_Lev_Money_Short_All'),
    'dealer': ('Dealer_Positions_Long_All', 'Dealer_Positions_Short_All',
               'Change_in_Dealer_Long_All', 'Change_in_Dealer_Short_All'),
}

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

    required_cols = ['Market_and_Exchange_Names', 'Report_Date_as_YYYY_MM_DD']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'CFTC CSV no tiene columnas requeridas: {missing}')
    if df.empty:
        raise ValueError('CFTC CSV vacío')

    df.to_csv(CACHE_PATH, index=False)
    print(f'  CFTC TFF guardado en caché: {len(df)} filas')
    return df

def _calculate_position_flow(df):
    """Calcula flujo de posicionamiento por contrato y participante."""
    if 'Market_and_Exchange_Names' not in df.columns:
        raise ValueError('No se encontró Market_and_Exchange_Names')

    mask = df['Market_and_Exchange_Names'].isin(TARGET_CONTRACTS)
    df = df[mask].copy()
    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(
        df['Report_Date_as_YYYY_MM_DD'],
        errors='coerce',
        format='mixed'
    )
    df = df.dropna(subset=['date'])

    # Convertir columnas numéricas
    all_numeric = []
    for cols in PARTICIPANT_COLS.values():
        all_numeric.extend(cols)
    for col in all_numeric:
        if col in df.columns:
            # Limpiar comas y espacios antes de convertir a numérico
            df[col] = (
                df[col].astype(str)
                .str.replace(',', '', regex=False)
                .str.replace(' ', '', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    result = pd.DataFrame()
    for participant, (long_col, short_col, chg_long, chg_short) in PARTICIPANT_COLS.items():
        if long_col not in df.columns or short_col not in df.columns:
            continue

        temp = pd.DataFrame({
            'date': df['date'].values,
            'contract': df['Market_and_Exchange_Names'].values,
            'participant': participant,
            'net_position': df[long_col] - df[short_col],
        })

        if chg_long in df.columns and chg_short in df.columns:
            pos_change = df[chg_long] - df[chg_short]
            # Rellenar NaN con diff por si los Change_in_* no vienen completos
            for contract, group in temp.groupby('contract'):
                idx = group.index
                diff = temp.loc[idx, 'net_position'].diff()
                pos_change.loc[idx] = pos_change.loc[idx].fillna(diff)
            temp['position_change'] = pos_change.values
        else:
            temp['position_change'] = temp.groupby('contract')['net_position'].diff()

        # Calcular flow_z
        temp = temp.sort_values(['contract', 'date'])
        rolling_mean = temp.groupby('contract')['position_change'].transform(
            lambda x: x.rolling(52, min_periods=10).mean()
        )
        rolling_std = temp.groupby('contract')['position_change'].transform(
            lambda x: x.rolling(52, min_periods=10).std()
        )
        temp['flow_z'] = ((temp['position_change'] - rolling_mean) / (rolling_std + 1e-9)).fillna(0.0)

        result = pd.concat([result, temp], ignore_index=True)

    if result.empty:
        return result

    result = result.dropna(subset=['position_change', 'net_position'])
    # Para cada contrato/participante, conservar la fila más reciente
    result = result.sort_values('date', ascending=False)
    result = result.drop_duplicates(subset=['contract', 'participant'], keep='first')
    return result

def get_cftc_position_flow_data() -> pd.DataFrame:
    """Descarga, procesa y devuelve los últimos flujos CFTC por contrato y participante."""
    try:
        raw = _download_and_cache()
        result = _calculate_position_flow(raw)
        if result.empty:
            return result

        # Filtrar a los últimos 365 días para mostrar solo lo reciente
        max_date = result['date'].max()
        recent = result[result['date'] >= max_date - pd.Timedelta(days=365)]

        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        recent.to_csv(HISTORY_PATH, index=False)
        print(f'  Histórico CFTC guardado (últimos 365 días): {HISTORY_PATH}')

        return recent.sort_values('date', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f'  Error en CFTC Position Flow: {e}')
        return pd.DataFrame()
