import pandas as pd
from pathlib import Path

NportChange = Path('outputs/history/sec_nport_position_change_quarterly.csv')
Output = Path('outputs/report/sec_nport_international_leader_flows.csv')
MappingCsv = Path('data/mappings/isin_ticker_map.csv')

def main():
    print('Cargando cambios N-PORT...')
    change = pd.read_csv(NportChange)
    change_fez = change[change['SERIES_NAME'].str.contains('EURO STOXX 50', case=False, na=False)].copy()
    print(f'Cambios FEZ: {len(change_fez)}')

    # Cargar mapeo ISIN->ticker si existe
    ticker_col = None
    if MappingCsv.exists():
        map_df = pd.read_csv(MappingCsv)
        map_df.columns = ['isin','ticker']
        change_fez = change_fez.merge(map_df, left_on='IDENTIFIER_ISIN', right_on='isin', how='left')
        ticker_col = 'ticker'
    else:
        change_fez['ticker'] = ''
        ticker_col = 'ticker'

    # Seleccionar columnas finales, incluyendo ticker si está disponible
    cols = ['REPORT_DATE','REGISTRANT_NAME','SERIES_NAME','ISSUER_NAME',
            'IDENTIFIER_ISIN','ISSUER_CUSIP','ticker',
            'BALANCE','PREV_BALANCE','POSITION_CHANGE','POSITION_CHANGE_PCT']
    result = change_fez[cols].copy()

    result.to_csv(Output, index=False)
    print(f'Guardado en {Output}')
    print(result.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
