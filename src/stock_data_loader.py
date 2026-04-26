"""
stock_data_loader.py – Descarga y caché de precios de acciones líderes (OHLC).
No genera señales de trading; solo proporciona datos para el análisis de líderes.
"""

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta
import os
import time

CACHE_FILE = "data/stock_prices.csv"
MAX_TICKERS_PER_BATCH = 50   # yfinance puede manejar unos 50-100, pero por seguridad usamos 50
DELAY_BETWEEN_BATCHES = 1    # segundo

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_stock_batch(tickers, start, end):
    """Descarga un lote de tickers con reintentos."""
    return yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by='ticker'
    )

def fetch_stooq_single(ticker, start, end):
    """Descarga un ticker individual desde Stooq (fallback), incluyendo OHLC."""
    import requests
    from io import StringIO

    # Mapeo de tickers comunes con guiones o símbolos especiales
    mapping = {
        'BRK-B': 'brk-b.us',
        'BRK-A': 'brk-a.us',
        'UBER': 'uber.us',
        # añadir más si se detectan problemas
    }
    symbol = mapping.get(ticker, ticker.lower().replace("^", "") + ".us")
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df.loc[start:end]
        if df.empty:
            return None
        # Renombrar columnas
        close = df['Close'].rename(f"{ticker}_close")
        volume = df['Volume'].rename(f"{ticker}_volume")
        open_ = df['Open'].rename(f"{ticker}_open")
        high = df['High'].rename(f"{ticker}_high")
        low = df['Low'].rename(f"{ticker}_low")
        return pd.DataFrame({
            f"{ticker}_close": close,
            f"{ticker}_volume": volume,
            f"{ticker}_open": open_,
            f"{ticker}_high": high,
            f"{ticker}_low": low
        })
    except Exception as e:
        print(f"[Stooq] Error con {ticker}: {e}")
        return None

def fetch_stock_prices(tickers_list, force=False):
    """
    Descarga precios OHLC y volúmenes para una lista de tickers.
    Utiliza caché local (24h) y descarga por lotes.
    Retorna DataFrame con columnas:
        TICKER_close, TICKER_volume, TICKER_open, TICKER_high, TICKER_low
    """
    if not tickers_list:
        print("[StockDataLoader] Lista de tickers vacía. No se descarga nada.")
        return pd.DataFrame()

    # Verificar caché
    if not force and os.path.exists(CACHE_FILE):
        file_time = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE))
        if datetime.now() - file_time < timedelta(hours=23):
            print("[StockDataLoader] Usando caché de precios de acciones.")
            try:
                return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
            except Exception as e:
                print(f"[StockDataLoader] Error al leer caché: {e}. Se descargarán datos nuevos.")
                # si falla la lectura, se procede a descargar

    # Si no hay caché o está expirada, descargar
    end = datetime.now()
    start = end - timedelta(days=3650)  # 10 años (mismo horizonte que market_data)
    
    all_data = pd.DataFrame()
    # Dividir tickers en lotes
    for i in range(0, len(tickers_list), MAX_TICKERS_PER_BATCH):
        batch = tickers_list[i:i+MAX_TICKERS_PER_BATCH]
        print(f"[StockDataLoader] Descargando lote {i//MAX_TICKERS_PER_BATCH + 1}: {batch}")
        try:
            data = download_stock_batch(batch, start, end)
            if data is not None and not data.empty:
                # Procesar cada ticker del lote
                for ticker in batch:
                    # Verificar si el ticker está en los datos descargados
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker not in data.columns.levels[0]:
                            continue
                        df_ticker = data[ticker]
                    else:
                        # DataFrame plano (un solo ticker)
                        if ticker not in data.columns and 'Close' not in data.columns:
                            continue
                        df_ticker = data
                    # Extraer columnas OHLCV
                    close_col = 'Close' if 'Close' in df_ticker.columns else 'close' if 'close' in df_ticker.columns else None
                    vol_col = 'Volume' if 'Volume' in df_ticker.columns else 'volume' if 'volume' in df_ticker.columns else None
                    open_col = 'Open' if 'Open' in df_ticker.columns else 'open' if 'open' in df_ticker.columns else None
                    high_col = 'High' if 'High' in df_ticker.columns else 'high' if 'high' in df_ticker.columns else None
                    low_col = 'Low' if 'Low' in df_ticker.columns else 'low' if 'low' in df_ticker.columns else None
                    if close_col is None or vol_col is None:
                        continue
                    # Construir DataFrame temporal con todas las columnas
                    temp_dict = {
                        f"{ticker}_close": df_ticker[close_col],
                        f"{ticker}_volume": df_ticker[vol_col]
                    }
                    if open_col is not None:
                        temp_dict[f"{ticker}_open"] = df_ticker[open_col]
                    if high_col is not None:
                        temp_dict[f"{ticker}_high"] = df_ticker[high_col]
                    if low_col is not None:
                        temp_dict[f"{ticker}_low"] = df_ticker[low_col]
                    temp = pd.DataFrame(temp_dict)
                    all_data = pd.concat([all_data, temp], axis=1)
            else:
                print(f"[StockDataLoader] Lote vacío.")
        except Exception as e:
            print(f"[StockDataLoader] Error en lote {batch}: {e}")
        # Pausa para no saturar
        time.sleep(DELAY_BETWEEN_BATCHES)

    # Verificar qué tickers de la lista original no están en all_data
    missing = [t for t in tickers_list if f"{t}_close" not in all_data.columns]
    if missing:
        print(f"[StockDataLoader] Descargando {len(missing)} tickers desde Stooq (fallback)...")
        for t in missing:
            df_stooq = fetch_stooq_single(t, start, end)
            if df_stooq is not None and not df_stooq.empty:
                all_data = pd.concat([all_data, df_stooq], axis=1)
                print(f"[Stooq] {t} descargado correctamente.")
            else:
                print(f"[Stooq] No se pudo obtener {t}. Se omitirá del análisis.")
            time.sleep(0.5)  # pausa entre peticiones individuales

    if all_data.empty:
        raise Exception("No se pudieron descargar datos de acciones líderes.")

    # Guardar caché
    all_data.to_csv(CACHE_FILE)
    print(f"[StockDataLoader] Caché guardada en {CACHE_FILE}")
    return all_data