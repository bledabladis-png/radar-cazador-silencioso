# -*- coding: utf-8 -*-
# scripts/update_sector_holdings.py
# Actualiza data/etf_holdings.csv desde los archivos oficiales de State Street
import requests
import pandas as pd
from io import BytesIO
import os
import sys
from datetime import datetime

# ETFs sectoriales con sus URLs de State Street (formato US)
SECTOR_ETFS = {
    'XLK': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlk.xlsx',
    'XLF': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlf.xlsx',
    'XLV': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlv.xlsx',
    'XLE': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xle.xlsx',
    'XLY': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xly.xlsx',
    'XLP': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlp.xlsx',
    'XLI': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xli.xlsx',
    'XLB': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlb.xlsx',
    'XLU': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlu.xlsx',
    'XLRE': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlre.xlsx',
    'XLC': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlc.xlsx',
}

OUTPUT_FILE = 'data/etf_holdings.csv'
updated = []
failed = []

# Cargar holdings existentes (para fallback)
existing = {}
if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_csv(OUTPUT_FILE)
        for etf in df_existing['etf'].unique():
            existing[etf] = df_existing[df_existing['etf'] == etf]['ticker'].tolist()
    except:
        pass

all_data = []

for etf, url in SECTOR_ETFS.items():
    print(f'Procesando {etf}...')
    try:
        resp = requests.get(url, allow_redirects=True, timeout=30)
        if resp.status_code != 200:
            raise Exception(f'HTTP {resp.status_code}')
        
        # Leer Excel, los datos empiezan en fila 5 (índice 4 en 0-based)
        df = pd.read_excel(BytesIO(resp.content), header=None)
        
        # Buscar la fila de encabezados que contiene 'Ticker'
        ticker_col = None
        weight_col = None
        header_row = None
        for i, row in df.iterrows():
            for j, cell in row.items():
                if isinstance(cell, str) and cell.strip() == 'Ticker':
                    ticker_col = j
                    header_row = i
                if isinstance(cell, str) and 'weight' in cell.lower():
                    weight_col = j
            if ticker_col is not None:
                break
        
        if ticker_col is None:
            raise Exception('No se encontro columna Ticker en el Excel')
        
        # Extraer tickers y pesos desde la fila siguiente a los encabezados
        tickers = []
        weights = []
        for i in range(header_row + 1, len(df)):
            ticker = df.iloc[i, ticker_col]
            if isinstance(ticker, str) and ticker.strip():
                tickers.append(ticker.strip().upper())
                if weight_col is not None:
                    w = df.iloc[i, weight_col]
                    try:
                        weights.append(float(str(w).replace(',', '.')))
                    except:
                        weights.append(0.0)
                else:
                    weights.append(0.0)
        
        if tickers:
            all_data.append(pd.DataFrame({'etf': etf, 'ticker': tickers, 'weight': weights}))
            updated.append(etf)
            print(f'  {len(tickers)} tickers extraidos')
        else:
            raise Exception('No se encontraron tickers validos')
            
    except Exception as e:
        print(f'  ERROR: {e}')
        failed.append(etf)
        # Usar datos anteriores si existen
        if etf in existing:
            print(f'  Usando {len(existing[etf])} tickers del archivo anterior')
            all_data.append(pd.DataFrame({'etf': etf, 'ticker': existing[etf]}))

# Guardar CSV
if all_data:
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['etf','ticker'], keep='last')
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f'\nArchivo guardado: {OUTPUT_FILE}')
    print(f'Actualizados: {len(updated)} ETFs ({", ".join(updated)})')
    if failed:
        print(f'Fallidos (usan datos anteriores): {len(failed)} ETFs ({", ".join(failed)})')
else:
    print('\nERROR: No se pudo generar ningun dato.')
    sys.exit(1)
