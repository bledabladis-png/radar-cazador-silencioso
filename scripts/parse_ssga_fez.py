import requests
import pandas as pd
from io import BytesIO

URL = 'https://www.ssga.com/us/en/intermediary/library-content/products/fund-data/etfs/us/holdings-daily-us-en-fez.xlsx'

print('Descargando FEZ desde SSGA...')
resp = requests.get(URL, allow_redirects=True, timeout=30)
resp.raise_for_status()

df = pd.read_excel(BytesIO(resp.content), header=None)

# Buscar la fila de cabecera que contiene 'Ticker'
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
    raise ValueError('No se encontró la columna Ticker')

# Extraer tickers y pesos desde la fila siguiente a la cabecera
holdings = []
for i in range(header_row + 1, len(df)):
    ticker = df.iloc[i, ticker_col]
    if isinstance(ticker, str) and ticker.strip():
        name = df.iloc[i, 0]  # columna Name suele ser la 0
        # Buscar Weight
        weight = None
        for j in range(len(df.columns)):
            cell = df.iloc[i, j]
            if isinstance(cell, (int, float)) and cell > 0 and cell < 100:
                weight = cell
                break
        if weight is not None:
            holdings.append({
                'etf': 'FEZ',
                'ticker': ticker.strip(),
                'name': name if isinstance(name, str) else '',
                'weight': weight
            })

# Ordenar por peso descendente y tomar top 10
holdings.sort(key=lambda x: x['weight'], reverse=True)
top10 = holdings[:10]

out = 'outputs/FEZ_final_holdings.csv'
pd.DataFrame(top10).to_csv(out, index=False)
print(f'Guardado: {out}')
print(pd.DataFrame(top10).to_string(index=False))
