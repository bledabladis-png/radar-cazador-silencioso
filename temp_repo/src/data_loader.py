import pandas as pd
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import time
from config.tickers import MARKET_TICKERS
from config.settings import CACHE_HOURS
from data.providers.router import DataRouter

def _ticker_list():
    tickers = []
    for group in MARKET_TICKERS.values():
        if isinstance(group, dict):
            tickers.extend(group.values())
        elif isinstance(group, list):
            tickers.extend(group)
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
    
    # Descarga por lotes para evitar bloqueos
    batch_size = 5
    delay = 2
    all_data = []
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando lote {i//batch_size + 1}: {batch}")
        try:
            data_batch = router.get_market_data(batch, period="10y")
            if not data_batch.empty:
                all_data.append(data_batch)
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
        if i + batch_size < len(tickers):
            time.sleep(delay)
    
    if not all_data:
        raise RuntimeError("No se pudo descargar ningún ticker.")
    
    data = pd.concat(all_data, axis=1)
    
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)
    
    from src.utils import clean_oil_prices
    data = clean_oil_prices(data)
    
    data.to_csv(cache_path)
    return data
