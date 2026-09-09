import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from data.providers.backup_providers import BackupProvider
import os
import time

CACHE_HOURS = 23

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
}

def normalize_yahoo_ticker(t):
    """Convierte tickers problemáticos al formato que acepta Yahoo Finance."""
    return YAHOO_TICKER_MAP.get(t, t)

def get_usa_tickers():
    """Obtiene solo tickers de sectores USA (data/etf_holdings.csv)."""
    tickers = []
    try:
        import pandas as pd
        df_sect = pd.read_csv('data/etf_holdings.csv')
        if 'weight' in df_sect.columns:
            df_sect = df_sect.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_sect.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass
    # Eliminar duplicados
    seen = set()
    result = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def get_stock_list():
    """Obtiene lista de tickers para descargar: top 20 de cada sector y de cada índice."""
    tickers = []

    # 1) Sectores USA (data/etf_holdings.csv)
    try:
        df_sect = pd.read_csv('data/etf_holdings.csv')
        if 'weight' in df_sect.columns:
            df_sect = df_sect.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_sect.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # 2) Índices americanos/europeos (data/index_holdings.csv)
    try:
        df_idx = pd.read_csv('data/index_holdings.csv')
        if 'weight' in df_idx.columns:
            df_idx = df_idx.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_idx.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # Eliminar duplicados preservando orden
    seen = set()
    result = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def _get_yf_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None

def _classify_ticker(ticker, df):
    """Clasifica un ticker según la disponibilidad y frescura de sus datos.

    Devuelve: 'OK', 'PARTIAL', 'STALE', 'FAILED'.
    """
    if df is None or df.empty:
        return 'FAILED'
    # Buscar columna Close para el ticker
    close_col = None
    if isinstance(df.columns, pd.MultiIndex):
        # Intentar ('Close', ticker)
        if ('Close', ticker) in df.columns:
            close_col = ('Close', ticker)
        else:
            # Quizás solo hay una columna sin MultiIndex (caso individual)
            pass
    else:
        if 'Close' in df.columns:
            close_col = 'Close'
    if close_col is None:
        return 'FAILED'

    series = df[close_col].dropna()
    if len(series) == 0:
        return 'FAILED'

    last_date = series.index[-1]
    days_since = (pd.Timestamp.now() - last_date).days
    if days_since <= 5:
        return 'OK'
    elif days_since <= 15:
        return 'PARTIAL'
    else:
        return 'STALE'

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_stock_prices():
    cache_path = 'data/stock_prices.csv'
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            return pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)

    tickers = get_stock_list()
    backup = BackupProvider()
    if not tickers:
        return None

    print(f"Descargando precios para {len(tickers)} tickers...")

    # Una única sesión para todas las descargas
    session = _get_yf_session()
    all_data = []
    classification = {'OK': [], 'PARTIAL': [], 'STALE': [], 'FAILED': []}
    failed_tickers = []

    batch_size = 10  # lote mayor para eficiencia
    delay = 2

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Lote {i//batch_size + 1}: {batch}")
        try:
            if session:
                data_batch = yf.download(batch, period='5y', auto_adjust=True, session=session)
            else:
                data_batch = yf.download(batch, period='5y', auto_adjust=True)
            if not data_batch.empty:
                all_data.append(data_batch)
                # Clasificar cada ticker del lote
                for ticker in batch:
                    status = _classify_ticker(ticker, data_batch)
                    classification[status].append(ticker)
                    if status == 'FAILED':
                        failed_tickers.append(ticker)
            else:
                # lote vacío: todos fallidos
                classification['FAILED'].extend(batch)
                failed_tickers.extend(batch)
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
            classification['FAILED'].extend(batch)
            failed_tickers.extend(batch)

        if i + batch_size < len(tickers):
            time.sleep(delay)

    # Reintentar fallidos individualmente (misma sesión)
    if failed_tickers:
        print(f"Reintentando {len(failed_tickers)} tickers fallidos individualmente...")
        for ticker in failed_tickers[:]:  # copia para iterar y modificar listas
            try:
                if session:
                    data_single = yf.download(ticker, period='5y', auto_adjust=True, session=session)
                else:
                    data_single = yf.download(ticker, period='5y', auto_adjust=True)
                if not data_single.empty:
                    all_data.append(data_single)
                    status = _classify_ticker(ticker, data_single)
                    # actualizar clasificación
                    if status != 'FAILED':
                        classification['FAILED'].remove(ticker)
                        classification[status].append(ticker)
                        if ticker in failed_tickers:
                            failed_tickers.remove(ticker)
            except Exception:
                pass

        # Para los que aún fallan, usar BackupProvider
        if failed_tickers:
            print(f"Intentando BackupProvider para {len(failed_tickers)} tickers...")
            backup_data = backup.get_prices(failed_tickers, period='5y')
            if backup_data is not None and not backup_data.empty:
                all_data.append(backup_data)
                for ticker in failed_tickers[:]:
                    status = _classify_ticker(ticker, backup_data)
                    if status != 'FAILED':
                        classification['FAILED'].remove(ticker)
                        classification[status].append(ticker)
                        failed_tickers.remove(ticker)

    # Imprimir resumen de clasificación
    print("=== Clasificación de tickers ===")
    print(f"OK ({len(classification['OK'])}): {classification['OK']}")
    print(f"PARTIAL ({len(classification['PARTIAL'])}): {classification['PARTIAL']}")
    print(f"STALE ({len(classification['STALE'])}): {classification['STALE']}")
    print(f"FAILED ({len(classification['FAILED'])}): {classification['FAILED']}")

    if not all_data:
        return None

    data = pd.concat(all_data, axis=1)
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    data.to_csv(cache_path)
    return data
