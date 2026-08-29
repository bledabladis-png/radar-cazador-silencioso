# -*- coding: utf-8 -*-
# scripts/update_index_holdings.py
# Actualiza data/index_holdings.csv:
#   - SPY, DIA desde State Street (Excel oficial)
#   - QQQ desde Invesco API (JSON)
#   - IWM desde BlackRock (CSV oficial)
#   - ETFs europeos conservados manualmente
import requests
import pandas as pd
from io import BytesIO, StringIO
import os
import sys

# ETFs de EE.UU. que se actualizan desde State Street (Excel)
US_ETFS = {
    'SPY': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx',
    'DIA': 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-dia.xlsx',
}

# CUSIP de QQQ (Invesco)
QQQ_CUSIP = '46090E103'
INVESCO_URL = (
    'https://dng-api.invesco.com/cache/v1/accounts/'
    'en_US/shareclasses/{shareclass}/holdings/fund'
)

# URL de BlackRock para IWM (iShares Russell 2000 ETF)
BLACKROCK_IWM_URL = (
    'https://www.blackrock.com/varnish-api/blk-one01-product-data/'
    'product-data/api/v1/get-fund-document?appType=PRODUCT_PAGE&'
    'appSubType=ISHARES&targetSite=us-ishares&locale=en_US&'
    'portfolioId=239710&userType=individual&component=holdings'
)

OUTPUT_FILE = 'data/index_holdings.csv'

def normalize_ticker(ticker):
    """Convierte tickers problemáticos a su formato canónico Yahoo."""
    mapping = {
        'BRK.B': 'BRK-B',
        'BF.B': 'BF-B',
        'MOGA': 'MOG-A',
    }
    return mapping.get(ticker, ticker)

def get_invesco_qqq_holdings():
    """Descarga todos los holdings de QQQ desde Invesco API (sin limitación top10)."""
    url = INVESCO_URL.format(shareclass=QQQ_CUSIP)
    params = {
        'idType': 'cusip',
        'productType': 'ETF',
    }
    print('Descargando QQQ desde Invesco API (todos los holdings)...')
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    holdings = data.get('holdings', [])
    if not holdings:
        raise ValueError('No se encontraron holdings en la respuesta de Invesco')

    rows = []
    for h in holdings:
        ticker = h.get('ticker')
        if not ticker:
            continue

        # Filtrar solo Common Stock (securityTypeCode == 'COM')
        sec_code = h.get('securityTypeCode', '')
        if sec_code and sec_code != 'COM':
            continue

        rows.append({
            'etf': 'QQQ',
            'ticker': normalize_ticker(ticker.strip().upper()),
            'name': h.get('issuerName', ''),
            'weight': h.get('percentageOfTotalNetAssets', 0.0),
        })

    if not rows:
        raise ValueError('No se obtuvieron holdings equity para QQQ')

    df = pd.DataFrame(rows)
    print(f'  {len(df)} tickers extraídos de QQQ')
    return df


def get_blackrock_iwm_holdings():
    """Descarga holdings de IWM desde BlackRock y devuelve DataFrame."""
    print('Descargando IWM desde BlackRock (CSV oficial)...')
    resp = requests.get(BLACKROCK_IWM_URL, timeout=30)
    resp.raise_for_status()
    csv_text = resp.text

    # Localizar la línea de cabecera
    lines = csv_text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('Ticker,'):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError('No se encontró cabecera Ticker en el CSV de BlackRock')

    # Leer CSV a partir de la cabecera
    df_raw = pd.read_csv(StringIO('\n'.join(lines[header_idx:])), dtype=str)

    # Filtrar solo Equity
    if 'Asset Class' in df_raw.columns:
        df_equity = df_raw[df_raw['Asset Class'].str.strip().str.lower() == 'equity']
    else:
        df_equity = df_raw

    rows = []
    for _, row in df_equity.iterrows():
        ticker = row.get('Ticker')
        if not ticker or not isinstance(ticker, str):
            continue
        name = row.get('Name', '')
        weight_str = row.get('Weight (%)', '0')
        try:
            weight = float(str(weight_str).replace(',', '.'))
        except:
            weight = 0.0

        rows.append({
            'etf': 'IWM',
            'ticker': normalize_ticker(ticker.strip().upper()),
            'name': name.strip() if isinstance(name, str) else '',
            'weight': weight,
        })

    if not rows:
        raise ValueError('No se obtuvieron holdings equity para IWM')

    df = pd.DataFrame(rows)
    print(f'  {len(df)} tickers extraídos de IWM')
    return df


def get_state_street_holdings(etf, url):
    """Descarga holdings de un ETF de State Street y devuelve DataFrame con weight."""
    print(f'Procesando {etf}...')
    resp = requests.get(url, allow_redirects=True, timeout=30)
    if resp.status_code != 200:
        raise Exception(f'HTTP {resp.status_code}')
    df_raw = pd.read_excel(BytesIO(resp.content), header=None)

    # Buscar fila de cabecera que contiene 'Ticker' y 'Weight'
    ticker_col = None
    weight_col = None
    name_col = None
    header_row = None
    for i, row in df_raw.iterrows():
        for j, cell in row.items():
            if isinstance(cell, str):
                cell_str = cell.strip().lower()
                if cell_str == 'ticker':
                    ticker_col = j
                    header_row = i
                if 'weight' in cell_str:
                    weight_col = j
                if 'name' in cell_str:
                    name_col = j
        if ticker_col is not None:
            break

    if ticker_col is None:
        raise Exception('No se encontró columna Ticker')

    tickers = []
    names = []
    weights = []
    for i in range(header_row + 1, len(df_raw)):
        ticker = df_raw.iloc[i, ticker_col]
        if isinstance(ticker, str) and ticker.strip():
            tickers.append(normalize_ticker(ticker.strip().upper()))
            name = df_raw.iloc[i, name_col] if name_col is not None else ''
            if not isinstance(name, str):
                name = ''
            names.append(name.strip())
            if weight_col is not None:
                w_raw = df_raw.iloc[i, weight_col]
                try:
                    w = float(str(w_raw).replace(',', '.'))
                except:
                    w = 0.0
                weights.append(w)
            else:
                weights.append(0.0)

    if not tickers:
        raise Exception('No se encontraron tickers')

    print(f'  {len(tickers)} tickers extraídos')
    return pd.DataFrame({'etf': etf, 'ticker': tickers, 'name': names, 'weight': weights})


def main():
    # Cargar datos existentes para fallback y conservar Europa
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

    # --- Actualizar SPY y DIA desde State Street ---
    for etf, url in US_ETFS.items():
        try:
            df_etf = get_state_street_holdings(etf, url)
            all_data.append(df_etf)
            updated.append(etf)
        except Exception as e:
            print(f'  ERROR: {e}')
            failed.append(etf)
            if etf in existing:
                print(f'  Usando {len(existing[etf])} tickers anteriores')
                all_data.append(pd.DataFrame({'etf': etf, 'ticker': existing[etf]}))

    # --- Actualizar QQQ desde Invesco API ---
    print('Procesando QQQ...')
    try:
        df_qqq = get_invesco_qqq_holdings()
        all_data.append(df_qqq)
        updated.append('QQQ')
    except Exception as e:
        print(f'  ERROR: {e}')
        failed.append('QQQ')
        if 'QQQ' in existing:
            print(f'  Usando {len(existing["QQQ"])} tickers anteriores')
            all_data.append(pd.DataFrame({'etf': 'QQQ', 'ticker': existing['QQQ']}))

    # --- Actualizar IWM desde BlackRock ---
    print('Procesando IWM...')
    try:
        df_iwm = get_blackrock_iwm_holdings()
        all_data.append(df_iwm)
        updated.append('IWM')
    except Exception as e:
        print(f'  ERROR: {e}')
        failed.append('IWM')
        if 'IWM' in existing:
            print(f'  Usando {len(existing["IWM"])} tickers anteriores')
            all_data.append(pd.DataFrame({'etf': 'IWM', 'ticker': existing['IWM']}))

    # --- Conservar ETFs europeos sin cambios ---
    european_etfs = ['FEZ', 'LYXI', 'DAXEX', 'ISF.L']
    for etf in european_etfs:
        if etf in existing:
            all_data.append(pd.DataFrame({'etf': etf, 'ticker': existing[etf]}))
            print(f'{etf}: conservando {len(existing[etf])} tickers manuales')
        else:
            print(f'{etf}: sin datos previos, omitiendo')

    # Guardar resultado final
    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f'\nArchivo guardado: {OUTPUT_FILE}')
        print(f'Actualizados: {len(updated)} ETFs ({", ".join(updated)})')
        if failed:
            print(f'Fallidos (usan datos anteriores): {len(failed)} ETFs ({", ".join(failed)})')
    else:
        print('\nERROR: No se pudo generar ningún dato.')
        sys.exit(1)


if __name__ == '__main__':
    main()
