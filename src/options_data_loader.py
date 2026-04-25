"""
options_data_loader.py – Gestión de datos de volumen de opciones de CBOE con caché local.
"""

import os
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta

DATA_OPTIONS_DIR = "data/options_volume"

def get_local_csv_path(year):
    os.makedirs(DATA_OPTIONS_DIR, exist_ok=True)
    return os.path.join(DATA_OPTIONS_DIR, f"daily_volume_{year}.csv")

def download_options_data_for_year(year, force=False):
    """Descarga el archivo de un año si no existe localmente o si force=True."""
    local_path = get_local_csv_path(year)
    
    if not force and os.path.exists(local_path):
        print(f"[OptionsDataLoader] Usando caché local para {year}")
        return pd.read_csv(local_path, parse_dates=['Trade Date'])
    
    print(f"[OptionsDataLoader] Descargando datos de opciones para {year} desde CBOE...")
    base_url = "https://www.cboe.com/us/options/market_statistics/historical_data/download/all_symbols/"
    params = {
        'reportType': 'volume',
        'month': '',
        'year': year,
        'volumeType': 'sum',
        'volumeAggType': 'daily',
        'exchanges': 'CBOE'
    }
    try:
        response = requests.get(base_url, params=params, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        os.makedirs(DATA_OPTIONS_DIR, exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"[OptionsDataLoader] Datos guardados en {local_path}")
        return df
    except Exception as e:
        print(f"[OptionsDataLoader] Error descargando {year}: {e}")
        return pd.DataFrame()

def get_historical_options_data(years, max_age_hours=23):
    """
    Obtiene datos históricos de opciones para una lista de años.
    Para el año actual, respeta max_age_hours (similar al caché de market_data.csv).
    Para años anteriores, solo descarga si no existen localmente.
    """
    current_year = datetime.now().year
    all_dfs = []
    for year in years:
        if year == current_year:
            local_path = get_local_csv_path(year)
            force_download = False
            if os.path.exists(local_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(local_path))
                if datetime.now() - file_time > timedelta(hours=max_age_hours):
                    print(f"[OptionsDataLoader] El archivo de {year} tiene más de {max_age_hours} horas, se descargará de nuevo.")
                    force_download = True
            df = download_options_data_for_year(year, force=force_download)
        else:
            df = download_options_data_for_year(year, force=False)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()
