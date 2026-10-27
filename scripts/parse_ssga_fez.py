import sys
from pathlib import Path

import requests
import pandas as pd
from io import BytesIO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import time

import yfinance as yf

from src.holdings_filter import is_valid_holding_ticker

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
    'INGA': 'INGA.AS',
    # 2026-10-27: 27 tickers FEZ sin sufijo de mercado. Verificados
    # contra Yahoo con period=5d. Sin sufijo, Yahoo los resolvia a
    # ADRs americanos erroneos (DG=Dollar General, EL=Estee Lauder,
    # BN=Brookfield, AD=Kroger, ABI/ARGX/RACE sin sufijo).
    'CS': 'CS.PA',
    'ENEL': 'ENEL.MI',
    'IFX': 'IFX.DE',
    'ITX': 'ITX.MC',
    'DHL': 'DHL.DE',
    'ENI': 'ENI.MI',
    'DBK': 'DBK.DE',
    # 'NDA FI': 'NDA-FI.HE',  # 2026-10-27: excluido. Yahoo lo resuelve
    # pero viola D11 (2 separadores: - y .). Sin alternativa valida
    # (NDA1V.HE, NDA.FI, NDAFI.HE, NDA.HE no resuelven). Nordea queda
    # fuera de FEZ (-1.20% de peso; FEZ 96.29 -> 95.09).
    'RHM': 'RHM.DE',
    'BAS': 'BAS.DE',
    'ENGI': 'ENGI.PA',
    'RMS': 'RMS.PA',
    'BAYN': 'BAYN.DE',
    'PRX': 'PRX.AS',
    'NOKIA': 'NOKIA.HE',
    'SGO': 'SGO.PA',
    'MBG': 'MBG.DE',
    'ADYEN': 'ADYEN.AS',
    'ADS': 'ADS.DE',
    'BMW': 'BMW.DE',
    'ABI': 'ABI.BR',
    'DG': 'DG.PA',
    'EL': 'EL.PA',
    'ARGX': 'ARGX.BR',
    'RACE': 'RACE.MI',
    'BN': 'BN.PA',
    'AD': 'AD.AS',
}

# Overrides por nombre cuando SSGA exporta tickers ambiguos
# (ej: Sanofi y Banco Santander comparten 'SAN' en el fichero SSGA).
FEZ_NAME_OVERRIDE = {
    'SANOFI': 'SAN.PA',           # Sanofi -> Euronext Paris
    'BANCO SANTANDER': 'SAN.MC',  # Santander -> BME Madrid
}

def validate_yahoo(ticker):
    """True si Yahoo resuelve el ticker con 5d de historico."""
    try:
        h = yf.Ticker(ticker).history(period='5d', auto_adjust=False)
        return not h.empty
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def resolve_fez_ticker(raw_ticker, name):
    name_upper = (name or '').upper()
    for needle, ticker in FEZ_NAME_OVERRIDE.items():
        if needle in name_upper:
            return ticker
    return FEZ_TICKER_MAP.get(raw_ticker.strip(), raw_ticker.strip())

def main():
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
        if isinstance(ticker, str) and is_valid_holding_ticker(ticker):
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
                yahoo_ticker = resolve_fez_ticker(ticker, name if isinstance(name, str) else '')
                # 2026-10-27: validar contra Yahoo (como BlackRock).
                # Sin validacion, tickers sin mapear salen crudos y
                # rompen la descarga. Los fallos se excluyen.
                if not validate_yahoo(yahoo_ticker):
                    print(f'FAIL: {ticker:8s} -> {yahoo_ticker} [excluido]')
                    time.sleep(0.1)
                    continue
                time.sleep(0.1)
                holdings.append({
                    'etf': 'FEZ',
                    'ticker': yahoo_ticker,
                    'identifier': identifier,
                    'name': name if isinstance(name, str) else '',
                    'weight': weight
                })

    holdings.sort(key=lambda x: x['weight'], reverse=True)

    out = 'outputs/holdings/FEZ_final_holdings.csv'
    pd.DataFrame(holdings, columns=['etf','ticker','identifier','name','weight']).to_csv(out, index=False)
    print(f'Guardado: {out} ({len(holdings)} holdings, sin truncar)')
    print(pd.DataFrame(holdings).to_string(index=False))
    return 0

if __name__ == '__main__':
    sys.exit(main())
