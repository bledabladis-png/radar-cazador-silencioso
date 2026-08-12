# -*- coding: utf-8 -*-
# scripts/update_index_holdings.py
# Actualiza data/index_holdings.csv desde State Street (EE.UU.) y conserva Europa manual
import requests
import pandas as pd
from io import BytesIO
import os
import sys
from datetime import datetime

# ETFs de EE.UU. (State Street)
US_ETFS = {
    'SPY': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx',
    'DIA': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx',
    'QQQ': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-qqq.xlsx',
    'IWM': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-iwm.xlsx',
}

OUTPUT_FILE = 'data/index_holdings.csv'

# Cargar datos actuales (para conservar Europa)
existing = {}
if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_csv(OUTPUT_FILE)
        for etf in df_existing['etf'].unique():
            existing[etf] = df_existing[df_existing['etf'] == etf]['ticker'].tolist()
    except:
        pass

all_data = []
updated = []
failed = []

# Actualizar ETFs de EE.UU.
for etf, url in US_ETFS.items():
    print(f'Procesando {etf}...')
    try:
        resp = requests.get(url, allow_redirects=True, timeout=30)
        if resp.status_code != 200:
            raise Exception(f'HTTP {resp.status_code}')
        df = pd.read_excel(BytesIO(resp.content), header=None)
        
        # Buscar columna Ticker
        ticker_col = None
        header_row = None
        for i, row in df.iterrows():
            for j, cell in row.items():
                if isinstance(cell, str) and cell.strip() == 'Ticker':
                    ticker_col = j
                    header_row = i
                    break
            if ticker_col is not None:
                break
        
        if ticker_col is None:
            raise Exception('No se encontro columna Ticker')
        
        tickers = []
        for i in range(header_row + 1, len(df)):
            ticker = df.iloc[i, ticker_col]
            if isinstance(ticker, str) and ticker.strip():
                tickers.append(ticker.strip().upper())
        
        if tickers:
            all_data.append(pd.DataFrame({'etf': etf, 'ticker': tickers}))
            updated.append(etf)
            print(f'  {len(tickers)} tickers extraidos')
        else:
            raise Exception('No se encontraron tickers')
            
    except Exception as e:
        print(f'  ERROR: {e}')
        failed.append(etf)
        if etf in existing:
            print(f'  Usando {len(existing[etf])} tickers anteriores')
            all_data.append(pd.DataFrame({'etf': etf, 'ticker': existing[etf]}))

# Conservar ETFs europeos sin cambios
european_etfs = ['FEZ', 'LYXI', 'DAXEX', 'ISF.L']
for etf in european_etfs:
    if etf in existing:
        all_data.append(pd.DataFrame({'etf': etf, 'ticker': existing[etf]}))
        print(f'{etf}: conservando {len(existing[etf])} tickers manuales')
    else:
        print(f'{etf}: sin datos previos, omitiendo')

# Guardar
if all_data:
    df_final = pd.concat(all_data, ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f'\nArchivo guardado: {OUTPUT_FILE}')
    print(f'Actualizados: {len(updated)} ETFs ({", ".join(updated)})')
    if failed:
        print(f'Fallidos (usan datos anteriores): {len(failed)} ETFs ({", ".join(failed)})')
else:
    print('\nERROR: No se pudo generar ningun dato.')
    sys.exit(1)
