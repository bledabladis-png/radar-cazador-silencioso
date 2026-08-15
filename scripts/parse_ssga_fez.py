import requests
import pandas as pd
from io import BytesIO

URL = 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-fez.xlsx'

FEZ_TICKER_MAP = {
    'ASML': 'ASML.AS',
    'SIE': 'SIE.DE',
    'SAN': 'SAN.MC',
    'SU': 'SU.PA',
    'SAP': 'SAP.DE',
    'TTE': 'TTE.PA',
    'ALV': 'ALV.DE',
    'BBVA': 'BBVA.MC',
    'SAF': 'SAF.PA',
    'ENR': 'ENR.DE',
    'IBE': 'IBE.MC',
    'UCG': 'UCG.MI',
    'AIR': 'AIR.PA',
    'BNP': 'BNP.PA',
    'MC': 'MC.PA',
    'AI': 'AI.PA',
    'ISP': 'ISP.MI',
    'DTE': 'DTE.DE',
    'OR': 'OR.PA',
    'INGA': 'INGA.AS'
}

print('Descargando FEZ desde SSGA...')
resp = requests.get(URL, allow_redirects=True, timeout=30)
resp.raise_for_status()

df = pd.read_excel(BytesIO(resp.content), header=None)

# Buscar fila de cabecera con Ticker e Identifier
ticker_col = None
identifier_col = None
header_row = None
for i, row in df.iterrows():
    for j, cell in row.items():
        if isinstance(cell, str):
            cell_str = cell.strip()
            if cell_str == 'Ticker':
                ticker_col = j
                header_row = i
            elif cell_str.lower() in ('identifier', 'cusip', 'isin'):
                identifier_col = j
    if ticker_col is not None:
        break

if ticker_col is None:
    raise ValueError('No se encontró la columna Ticker')

# Extraer tickers, identifiers, name y weight
holdings = []
for i in range(header_row + 1, len(df)):
    ticker = df.iloc[i, ticker_col]
    if isinstance(ticker, str) and ticker.strip():
        name = df.iloc[i, 0]
        identifier = ''
        if identifier_col is not None:
            val = df.iloc[i, identifier_col]
            identifier = str(val).strip() if val is not None else ''
        weight = None
        for j in range(len(df.columns)):
            cell = df.iloc[i, j]
            if isinstance(cell, (int, float)) and cell > 0 and cell < 100:
                weight = cell
                break
        if weight is not None:
            holdings.append({
                'etf': 'FEZ',
                'ticker': FEZ_TICKER_MAP.get(ticker.strip(), ticker.strip()),
                'identifier': identifier,
                'name': name if isinstance(name, str) else '',
                'weight': weight
            })

holdings.sort(key=lambda x: x['weight'], reverse=True)
top20 = holdings[:20]

out = 'outputs/holdings/FEZ_final_holdings.csv'
pd.DataFrame(top20, columns=['etf','ticker','identifier','name','weight']).to_csv(out, index=False)
print(f'Guardado: {out}')
print(pd.DataFrame(top20).to_string(index=False))
