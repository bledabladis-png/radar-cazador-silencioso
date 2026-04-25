"""
hkex_pcr.py – Descarga del ratio Put/Call desde HKEX (Bolsa de Hong Kong).
Datos oficiales, gratuitos, con formato CSV estable.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO

def download_hkex_pcr(start_date, end_date):
    """
    Descarga el ratio Put/Call de HKEX para el rango de fechas especificado.
    start_date, end_date: datetime o string 'YYYY-MM-DD'
    Retorna DataFrame con columnas 'date' (datetime) y 'pcr' (Put/Call ratio).
    """
    # Convertir fechas al formato requerido por HKEX: AAAA/M/D (ej. 2026/4/16)
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)   
   
    date_from = f"{start_date.year}/{start_date.month}/{start_date.day}"
    date_to = f"{end_date.year}/{end_date.month}/{end_date.day}"

    url = "https://www.hkex.com.hk/eng/sorc/market_data/statistics_putcall_ratio.aspx"
    params = {
        'action': 'ajax',
        'type': 'getCSV',
        'ucode': 'All',
        'date_form': date_from,
        'date_to': date_to,
        'page': 1
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        
        # El CSV viene con dos líneas de encabezado; saltamos las dos primeras
        lines = response.text.strip().splitlines()
        if len(lines) < 3:
            return pd.DataFrame()
        
        # Unir las líneas desde la tercera (índice 2) en adelante
        csv_data = "\n".join(lines[2:])
        df = pd.read_csv(StringIO(csv_data))
        
        # Renombrar columnas (pueden tener nombres con espacios)
        df.columns = [c.strip() for c in df.columns]
        # La columna de fecha suele ser 'Date (D/M/Y)'
        date_col = [c for c in df.columns if 'Date' in c][0]
        df['date'] = pd.to_datetime(df[date_col], format='%d/%m/%Y')
        df['pcr'] = df['Put/Call Ratio']
        
        return df[['date', 'pcr']].sort_values('date')
    except Exception as e:
        print(f"[HKEX PCR] Error: {e}")
        return pd.DataFrame()

def get_pcr_for_period(end_date=None, days_back=730):
    """
    Descarga PCR para un periodo de 'days_back' días hasta end_date.
    Por defecto, 2 años (730 días) hasta hoy.
    """
    if end_date is None:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return download_hkex_pcr(start_date, end_date)

def get_hkex_pcr_series(days_back=730):
    """
    Obtiene la serie diaria del Put/Call Ratio de HKEX.
    """
    df = get_pcr_for_period(days_back=days_back)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.set_index('date')
    return df['pcr'].sort_index()