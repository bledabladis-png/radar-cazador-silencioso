# etl.py - Módulo de extracción, transformación y carga de datos
# Descarga datos históricos de Tiingo y los prepara para el radar

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import pickle

# Configuración
DATA_PATH = os.path.dirname(os.path.abspath(__file__))  # carpeta actual
TICKERS_FILE = os.path.join(DATA_PATH, 'tickers.csv')
RAW_DATA_FILE = os.path.join(DATA_PATH, 'data_raw.pkl')
CLEAN_DATA_FILE = os.path.join(DATA_PATH, 'data_clean.pkl')

# Lista de tickers (ampliable)
TICKERS = [
    'SPY', 'QQQ', 'IWM', 'DIA',  # índices
    'XLF', 'XLK', 'XLV', 'XLI', 'XLB', 'XLE', 'XLU', 'XLP', 'XLY', 'XLRE',  # sectores
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'JNK',  # bonos
    'EFA', 'EEM', 'VWO', 'EWJ', 'EWG', 'EWU', 'EWQ',  # internacional
    'ARKK', 'ICLN', 'TAN', 'PBW', 'ROBO', 'BOTZ',  # temáticos
]

# Función para obtener API key de Tiingo de forma segura
def get_tiingo_key():
    """Intenta leer la clave de Tiingo desde:
       1. Variable de entorno TIINGO_API_KEY (producción)
       2. Archivo local tiingo_key.txt (desarrollo local)
    """
    # 1. Variable de entorno
    key = os.environ.get('TIINGO_API_KEY')
    if key:
        return key
    
    # 2. Archivo local (solo para pruebas, NO subir a GitHub)
    key_file = os.path.join(DATA_PATH, 'tiingo_key.txt')
    if os.path.exists(key_file):
        with open(key_file, 'r') as f:
            key = f.read().strip()
        if key:
            return key
    
    # 3. Si no se encuentra, error
    raise ValueError(
        "No se encontró la clave de Tiingo.\n"
        "Para desarrollo local: crea un archivo 'tiingo_key.txt' con tu clave.\n"
        "Para producción: define la variable de entorno TIINGO_API_KEY."
    )

# Descarga datos de un ticker desde Tiingo
def download_ticker(ticker, api_key, start_date='2000-01-01', end_date=None):
    """Descarga datos diarios de un ticker desde Tiingo."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Token {api_key}'}
    params = {'startDate': start_date, 'endDate': end_date, 'format': 'json'}
    for intento in range(3):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                print(f"   ⏳ Límite de tasa alcanzado. Esperando 60 segundos...")
                time.sleep(60)
                continue
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text[:100]}")
            data = response.json()
            if not data:
                raise Exception("No data")
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            # Renombrar columnas al formato estándar
            df.rename(columns={
                'adjOpen': 'open',
                'adjHigh': 'high',
                'adjLow': 'low',
                'adjClose': 'close',
                'adjVolume': 'volume'
            }, inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            # Añadir columna de retorno diario
            df['return'] = df['close'].pct_change()
            return df
        except Exception as e:
            print(f"   Error con {ticker} (intento {intento+1}): {e}")
            time.sleep(5)
    return None

# Validación básica de datos
def validate_data(df):
    """Comprueba que los datos son mínimamente consistentes."""
    if df is None or df.empty:
        return False
    if df.isnull().any().any():
        print("   ⚠️ Datos con nulos")
        return False
    if (df[['open','high','low','close']] <= 0).any().any():
        print("   ⚠️ Precios no positivos")
        return False
    if (df['volume'] < 0).any():
        print("   ⚠️ Volumen negativo")
        return False
    return True

# Función principal de ETL
def run_etl(force_download=False):
    """Ejecuta el proceso de extracción y limpieza de datos."""
    print("🚀 Iniciando ETL...")
    api_key = get_tiingo_key()
    data_raw = {}

    # Verificar si ya existen datos descargados
    if not force_download and os.path.exists(RAW_DATA_FILE):
        print("📦 Cargando datos desde caché (RAW)...")
        with open(RAW_DATA_FILE, 'rb') as f:
            data_raw = pickle.load(f)
    else:
        print("🌐 Descargando datos de Tiingo...")
        for i, ticker in enumerate(TICKERS):
            if i % 5 == 0:
                print(f"   Procesando {i+1}/{len(TICKERS)}...")
            df = download_ticker(ticker, api_key)
            if df is not None:
                data_raw[ticker] = df
            # Pausa para respetar límites
            time.sleep(1)
        # Guardar datos crudos
        with open(RAW_DATA_FILE, 'wb') as f:
            pickle.dump(data_raw, f)

    # Limpiar y validar
    data_clean = {}
    for ticker, df in data_raw.items():
        if validate_data(df):
            # Añadir metadatos útiles (fecha inicio, fin)
            df.attrs['ticker'] = ticker
            df.attrs['start'] = df.index[0].strftime('%Y-%m-%d')
            df.attrs['end'] = df.index[-1].strftime('%Y-%m-%d')
            data_clean[ticker] = df
        else:
            print(f"   ❌ {ticker}: datos inválidos")

    # Guardar datos limpios
    with open(CLEAN_DATA_FILE, 'wb') as f:
        pickle.dump(data_clean, f)

    print(f"✅ ETL completado. {len(data_clean)} tickers válidos.")
    return data_clean

# Si se ejecuta como script
if __name__ == "__main__":
    run_etl(force_download=True)