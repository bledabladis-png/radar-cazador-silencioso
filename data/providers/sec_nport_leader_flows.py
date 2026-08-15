import pandas as pd
from pathlib import Path

NportChange = Path('outputs/history/sec_nport_position_change_quarterly.csv')
Holdings = Path('data/etf_holdings.csv')
Leaders = Path('outputs/report/analisis_lideres.csv')
Output = Path('outputs/report/sec_nport_leader_flows.csv')

def main():
    print('Cargando cambios N-PORT...')
    change = pd.read_csv(NportChange)
    print(f'Filas en cambios: {len(change)}')

    print('Cargando holdings sectoriales...')
    holdings = pd.read_csv(Holdings)
    print(f'Columnas holdings: {holdings.columns.tolist()}')
    if 'identifier' not in holdings.columns:
        raise ValueError('Falta columna identifier en etf_holdings.csv. Regenerar con update_sector_holdings.py')

    print('Cargando líderes sectoriales...')
    leaders = pd.read_csv(Leaders)
    print(f'Filas líderes: {len(leaders)}')

    # Mapa CUSIP -> ticker
    cusip_ticker = holdings[['identifier','ticker']].drop_duplicates().dropna(subset=['identifier'])
    cusip_ticker['identifier'] = cusip_ticker['identifier'].str.upper().str.strip()
    change['ISSUER_CUSIP'] = change['ISSUER_CUSIP'].str.upper().str.strip()

    # Unir cambios con holdings por CUSIP
    merged = change.merge(cusip_ticker, left_on='ISSUER_CUSIP', right_on='identifier', how='left')
    # Filtrar solo acciones que son líderes actuales
    leader_tickers = set(leaders['ticker'].str.upper())
    merged = merged[merged['ticker'].isin(leader_tickers)]

    # Seleccionar columnas útiles
    result = merged[['REPORT_DATE','REGISTRANT_NAME','ISSUER_NAME','ISSUER_CUSIP','ticker',
                     'BALANCE','PREV_BALANCE','POSITION_CHANGE','POSITION_CHANGE_PCT']]

    result.to_csv(Output, index=False)
    print(f'Guardado en {Output}')
    print(f'Flujos reales de líderes: {len(result)}')
    print(result.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
