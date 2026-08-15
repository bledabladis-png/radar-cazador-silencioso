"""
Flujo primario de ETFs SPDR desde State Street (SSGA).
Calcula ETF Primary Flow = (SharesOutstanding_t - SharesOutstanding_{t-1}) * NAV_t
"""
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

SECTOR_TICKERS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLRE','XLU','XLC']
CACHE_DIR = Path('data/cache/ssga_navhist')
HISTORY_PATH = Path('outputs/history/etf_primary_flow.csv')

def _download_single(ticker: str) -> pd.DataFrame:
    """Descarga y parsea el histórico de NAV/Shares para un ETF SPDR."""
    url = f'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/navhist-us-en-{ticker.lower()}.xlsx'
    print(f'  Descargando {ticker} desde SSGA...')
    r = requests.get(url, headers={'User-Agent':'Mozilla/5.0'}, timeout=30)
    r.raise_for_status()

    df_raw = pd.read_excel(BytesIO(r.content), header=None)

    # Localizar fila de cabecera que contenga 'Date'
    header_row = None
    for i, row in df_raw.iterrows():
        if any(str(cell).strip() == 'Date' for cell in row):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f'No se encontró cabecera Date para {ticker}')

    # Construir DataFrame con las columnas correctas
    headers = [str(cell).strip() for cell in df_raw.iloc[header_row].tolist()]
    df = df_raw.iloc[header_row+1:].copy()
    df.columns = headers

    # Renombrar columnas necesarias
    rename = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'nav':
            rename[col] = 'nav'
        elif 'shares' in col_lower:
            rename[col] = 'shares_outstanding'
        elif 'total net assets' in col_lower:
            rename[col] = 'total_net_assets'

    df = df.rename(columns=rename)

    required = ['Date', 'nav', 'shares_outstanding', 'total_net_assets']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas {missing} en {ticker}')

    df = df[required]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df['shares_outstanding'] = pd.to_numeric(df['shares_outstanding'], errors='coerce')
    df['total_net_assets'] = pd.to_numeric(df['total_net_assets'], errors='coerce')
    df = df.dropna(subset=['Date', 'nav', 'shares_outstanding'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def _compute_primary_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas de flujo primario y z-score."""
    df = df.copy()
    df['primary_flow_usd'] = df['shares_outstanding'].diff() * df['nav']
    df['primary_flow_pct'] = (df['primary_flow_usd'] / df['total_net_assets']) * 100.0

    def robust_z(series):
        if len(series) < 20:
            return 0.0
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return 0.0
        return (series.iloc[-1] - median) / (1.4826 * mad + 1e-9)

    df['primary_flow_z'] = df['primary_flow_pct'].rolling(120).apply(robust_z, raw=False)
    return df

def get_etf_primary_flow_data(force_download: bool = False) -> pd.DataFrame:
    """
    Descarga/lee caché, calcula flujo primario y guarda histórico consolidado.
    Devuelve DataFrame con columnas:
    ticker, nav, shares_outstanding, total_net_assets, primary_flow_usd, primary_flow_pct, primary_flow_z
    para la última fecha disponible de cada ticker.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for ticker in SECTOR_TICKERS:
        cache_file = CACHE_DIR / f'{ticker}.csv'
        use_cache = (not force_download) and cache_file.exists()
        if use_cache:
            # Comprobar si el archivo es de hace menos de 24h
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime > timedelta(hours=23):
                use_cache = False

        if use_cache:
            print(f'  Usando caché para {ticker}')
            df = pd.read_csv(cache_file)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df = _download_single(ticker)
            df.to_csv(cache_file, index=False)

        df = _compute_primary_flow(df)
        df['ticker'] = ticker
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    # Guardar histórico completo
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(HISTORY_PATH, index=False)
    print(f'  Histórico guardado: {HISTORY_PATH}')

    # Obtener último registro por ticker
    last_df = full_df.dropna(subset=['primary_flow_pct']).groupby('ticker').tail(1)
    return last_df[['ticker','nav','shares_outstanding','total_net_assets',
                    'primary_flow_usd','primary_flow_pct','primary_flow_z']].reset_index(drop=True)
