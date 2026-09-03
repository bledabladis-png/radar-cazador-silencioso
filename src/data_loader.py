import pandas as pd
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import time
from config.tickers import MARKET_TICKERS
from config.settings import CACHE_HOURS
from data.providers.router import DataRouter
from data.providers.backup_providers import BackupProvider

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
}

def normalize_yahoo_ticker(t):
    """Convierte tickers problemáticos al formato que acepta Yahoo Finance."""
    return YAHOO_TICKER_MAP.get(t, t)

def _ticker_list():
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)

    # Añadir tickers de los líderes sectoriales (etf_holdings.csv)
    try:
        import pandas as _pd
        holdings = _pd.read_csv('data/etf_holdings.csv')
        if 'ticker' in holdings.columns:
            # Lista negra de tickers inválidos detectados en holdings (futuros, CUSIP, efectivo)
            INVALID_TICKERS = {'XARU6','IXDU6','IXIU6','IXAU6','IXTU6','IXRU6',
                               'IXPU6','IXCU6','IXYU6','IXSU6','XASU6',
                               '2602335D','-'}
            raw_tickers = holdings['ticker'].tolist()
            for t in raw_tickers:
                if isinstance(t, str) and t not in INVALID_TICKERS:
                    tickers.append(normalize_yahoo_ticker(t))
    except:
        pass

    return list(set(tickers))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_market_data():
    cache_path = 'data/market_data.csv'
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            return pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)

    tickers = _ticker_list()
    router = DataRouter()
    backup = BackupProvider()

    # Descarga por lotes para evitar bloqueos
    batch_size = 5
    delay = 2
    all_data = []
    batches_failed = []   # lotes completos que fallaron
    batch_errors = []     # mensajes de error asociados a cada lote fallido

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando lote {i//batch_size + 1}: {batch}")
        try:
            data_batch = router.get_market_data(batch, period="10y")
            if data_batch is not None and not data_batch.empty:
                all_data.append(data_batch)
            else:
                # Si el lote devuelve vacío, registrar tickers
                batches_failed.append(batch)
                batch_errors.append('Datos vacíos')
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
            try:
                backup_data = backup.get_prices(batch, period="10y")
                if backup_data is not None and not backup_data.empty:
                    all_data.append(backup_data)
                    print(f"  Respaldo obtuvo datos para {batch}")
                else:
                    batches_failed.append(batch)
                    batch_errors.append(str(e))
            except Exception as be:
                print(f"  Error en respaldo: {be}")
                batches_failed.append(batch)
                batch_errors.append(str(e))
        if i + batch_size < len(tickers):
            time.sleep(delay)

    if not all_data:
        raise RuntimeError("No se pudo descargar ningún ticker.")

    # --- Registrar tickers con fallos de descarga ---
    failed = []
    for batch in batches_failed:
        failed.extend(batch)
    if failed:
        # Crear directorio si no existe
        os.makedirs('outputs/audit', exist_ok=True)
        with open('outputs/audit/download_failures.md', 'w', encoding='utf-8') as f:
            f.write('# Fallos de descarga de tickers\n\n')
            f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            f.write('| Ticker | Lote | Error |\n')
            f.write('|--------|------|-------|\n')
            for i, batch in enumerate(batches_failed):
                for t in batch:
                    err_msg = batch_errors[i] if i < len(batch_errors) else 'Error desconocido'
                    f.write(f'| {t} | {i+1} | {err_msg} |\n')
        print(f'  Se registraron {len(failed)} tickers con fallos en outputs/audit/download_failures.md')
    else:
        # Si no hay fallos, borrar archivo anterior
        if os.path.exists('outputs/audit/download_failures.md'):
            os.remove('outputs/audit/download_failures.md')
    # ----------------------------------------------------

    data = pd.concat(all_data, axis=1)

    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    from src.utils import clean_oil_prices
    data = clean_oil_prices(data)

    data.to_csv(cache_path)
    return data
