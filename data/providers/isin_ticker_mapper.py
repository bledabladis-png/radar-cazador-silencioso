import pandas as pd
import requests
import time
from pathlib import Path

# Rutas
FLOWS_CSV = Path('outputs/report/sec_nport_international_leader_flows.csv')
MAPPING_DIR = Path('data/mappings')
MAPPING_CSV = MAPPING_DIR / 'isin_ticker_map.csv'

def fetch_yahoo_symbol(isin: str):
    """Consulta Yahoo Finance Search por ISIN y devuelve el símbolo."""
    url = 'https://query1.finance.yahoo.com/v1/finance/search'
    params = {
        'q': isin,
        'quotesCount': 10,
        'newsCount': 0,
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        quotes = data.get('quotes', [])
        if not quotes:
            return None
        # Tomar el primer quote de tipo EQUITY
        for q in quotes:
            if q.get('quoteType') == 'EQUITY':
                return q.get('symbol')
        return quotes[0].get('symbol')  # fallback
    except Exception as e:
        print(f'  Error consultando {isin}: {e}')
        return None

def build_mapping():
    """Genera o actualiza el mapeo ISIN->ticker a partir del CSV de flujos."""
    if not FLOWS_CSV.exists():
        print(f'No existe {FLOWS_CSV}. Ejecuta antes sec_nport_international_leader_flows.py')
        return

    print('Leyendo flujos internacionales...')
    df = pd.read_csv(FLOWS_CSV)
    if 'IDENTIFIER_ISIN' not in df.columns:
        raise ValueError('Falta IDENTIFIER_ISIN en el CSV')

    isins = df['IDENTIFIER_ISIN'].dropna().unique().tolist()
    print(f'ISIN únicos a mapear: {len(isins)}')

    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    mapping = {}
    for i, isin in enumerate(isins, 1):
        symbol = fetch_yahoo_symbol(isin)
        mapping[isin] = symbol if symbol else ''
        print(f'  [{i}/{len(isins)}] {isin} -> {symbol}')
        time.sleep(0.5)  # pausa para no saturar

    map_df = pd.DataFrame({'isin': list(mapping.keys()), 'ticker': list(mapping.values())})
    map_df.to_csv(MAPPING_CSV, index=False)
    print(f'Mapeo guardado en {MAPPING_CSV}')

if __name__ == '__main__':
    build_mapping()
