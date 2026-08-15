import pandas as pd
from pathlib import Path

NportChange = Path('outputs/history/sec_nport_position_change_quarterly.csv')
Output = Path('outputs/report/sec_nport_international_leader_flows.csv')

def main():
    print('Cargando cambios N-PORT...')
    change = pd.read_csv(NportChange)
    change_fez = change[change['SERIES_NAME'].str.contains('EURO STOXX 50', case=False, na=False)].copy()
    print(f'Cambios FEZ: {len(change_fez)}')

    result = change_fez[['REPORT_DATE','REGISTRANT_NAME','SERIES_NAME','ISSUER_NAME',
                         'IDENTIFIER_ISIN','ISSUER_CUSIP','BALANCE','PREV_BALANCE',
                         'POSITION_CHANGE','POSITION_CHANGE_PCT']].copy()
    result.to_csv(Output, index=False)
    print(f'Guardado en {Output}')
    print(result.head(30).to_string(index=False))

if __name__ == '__main__':
    main()
