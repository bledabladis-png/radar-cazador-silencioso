"""
Flujo primario de ETFs SPDR desde State Street (SSGA).
Calcula ETF Primary Flow = (SharesOutstanding_t - SharesOutstanding_{t-1}) * NAV_t
"""
import pandas as pd
import requests
from io import BytesIO
from datetime import datetime, timedelta
from pathlib import Path

from src.utils import append_dedup

from ._fund_flow_utils import fund_flow_robust_zscore_with_regime

SECTOR_TICKERS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLRE','XLU','XLC','FEZ']
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

    headers = [str(cell).strip() for cell in df_raw.iloc[header_row].tolist()]
    df = df_raw.iloc[header_row+1:].copy()
    df.columns = headers

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
    """Añade columnas de flujo primario y z-score.

    A2.4-01 (2026-09-28): migrado de implementacion local (rolling.apply
    con robust_z inline, sin regimen) al contrato comun
    _fund_flow_utils.fund_flow_robust_zscore_with_regime. Coherente con
    DAXEX (blackrock_fund_data), ISF.L (blackrock_isf_fund_data),
    LYXI (amundi_fund_data) e IWM (blackrock_iwm_fund_data).
    Publica dos columnas: primary_flow_z y primary_flow_z_regime.
    """
    df = df.copy()
    df['primary_flow_usd'] = df['shares_outstanding'].diff() * df['nav']
    df['primary_flow_pct'] = (df['primary_flow_usd'] / df['total_net_assets']) * 100.0

    df['primary_flow_z'], df['primary_flow_z_regime'] = fund_flow_robust_zscore_with_regime(
        df['primary_flow_pct'], window=120, min_periods=20,
    )
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
    errors = []
    for ticker in SECTOR_TICKERS:
        try:
            cache_file = CACHE_DIR / f'{ticker}.csv'
            use_cache = (not force_download) and cache_file.exists()
            if use_cache:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - mtime > timedelta(hours=23):
                    use_cache = False

            if use_cache:
                print(f'  Usando caché para {ticker}')
                df = pd.read_csv(cache_file)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            else:
                df = _download_single(ticker)
                # A5-79 (2026-09-28): escritura atomica tmp + replace.
                _tmp = cache_file.with_suffix(cache_file.suffix + '.tmp')
                df.to_csv(_tmp, index=False)
                _tmp.replace(cache_file)

            df = _compute_primary_flow(df)
            df['ticker'] = ticker
            all_frames.append(df)
        except Exception as e:
            print(f'  Error procesando {ticker}: {e}')
            errors.append(ticker)
            continue

    if not all_frames:
        return pd.DataFrame()

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df = full_df.sort_values(['ticker', 'Date']).reset_index(drop=True)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Fix D (2026-09-30): preservar historico con append_dedup.
    # Antes: full_df.to_csv sobreescribia el CSV. Si el proveedor o su
    # cache devolvia menos filas (delay, cache stale, glitch), el CSV
    # perdia filas permanentemente. Verificado con test_flows_primary_
    # preserve_history.py.
    if HISTORY_PATH.exists():
        try:
            _hist_existing = pd.read_csv(HISTORY_PATH)
            # append_dedup normaliza solo 'date' (minuscula). SSGA usa
            # 'Date' (mayuscula). Normalizar ambos a YYYY-MM-DD antes
            # del dedup, o drop_duplicates no detecta coincidencias
            # (string '2026-09-25' vs '2026-09-25 00:00:00') y duplica
            # todo el historico. Detectado en run E2E 2026-09-30.
            _hist_existing['Date'] = pd.to_datetime(
                _hist_existing['Date'], errors='coerce'
            ).dt.strftime('%Y-%m-%d')
            full_df['Date'] = pd.to_datetime(
                full_df['Date'], errors='coerce'
            ).dt.strftime('%Y-%m-%d')
            full_df = append_dedup(_hist_existing, full_df, ["ticker", "Date"])
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as _e:
            print(f'  [WARN] etf_primary_flow existente ilegible: {_e}')
    # A5-79 (2026-09-28): escritura atomica del historico consolidado.
    _tmp_hist = HISTORY_PATH.with_suffix(HISTORY_PATH.suffix + '.tmp')
    full_df.to_csv(_tmp_hist, index=False)
    _tmp_hist.replace(HISTORY_PATH)
    print(f'  Histórico guardado: {HISTORY_PATH}')
    if errors:
        print(f'  Tickers con error: {errors}')

    last_df = full_df.dropna(subset=['primary_flow_pct']).groupby('ticker').tail(1)
    return last_df[['ticker','nav','shares_outstanding','total_net_assets',
                    'primary_flow_usd','primary_flow_pct','primary_flow_z',
                    'primary_flow_z_regime']].reset_index(drop=True)

