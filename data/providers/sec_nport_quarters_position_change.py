import sys
import pandas as pd
from pathlib import Path

BASE = Path('data/nport')


def discover_quarters(base=Path('data/nport')):
    """Descubre dinamicamente los trimestres disponibles en data/nport/.

    F5.7-19: antes la lista de trimestres estaba hardcoded a dos
    valores fijos. El workflow trimestral descarga uno nuevo cada
    3 meses, pero este script solo procesaba los originales. Los
    trimestres nuevos quedaban invisibles. Ademas, si un trimestre
    de la lista no existia en disco, load_quarter fallaba.
    """
    if not base.exists():
        return []
    names = sorted([
        d.name for d in base.iterdir()
        if d.is_dir() and d.name.lower().startswith('20') and 'q' in d.name.lower()
    ])
    return names
OUTPUT = Path('outputs/history/sec_nport_position_change_quarterly.csv')
TARGET_CIKS = {1064641, 884394, 1041130, 936958, 1168164}

def load_quarter(q):
    print(f'Cargando {q}...')
    sub = pd.read_csv(BASE / q / 'SUBMISSION.tsv', sep='\t', low_memory=False)
    sub = sub[['ACCESSION_NUMBER', 'REPORT_DATE']].drop_duplicates()
    sub['REPORT_DATE'] = pd.to_datetime(sub['REPORT_DATE'], format='%d-%b-%Y', errors='coerce')

    reg = pd.read_csv(BASE / q / 'REGISTRANT.tsv', sep='\t', low_memory=False)
    reg = reg[reg['CIK'].isin(TARGET_CIKS)][['ACCESSION_NUMBER','CIK','REGISTRANT_NAME']]

    info = pd.read_csv(BASE / q / 'FUND_REPORTED_INFO.tsv', sep='\t', low_memory=False)
    info = info[info['ACCESSION_NUMBER'].isin(reg['ACCESSION_NUMBER'])][['ACCESSION_NUMBER','SERIES_NAME','SERIES_ID']].drop_duplicates()

    hold_cols = ['ACCESSION_NUMBER','HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP','BALANCE','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE']
    hold = pd.read_csv(BASE / q / 'FUND_REPORTED_HOLDING.tsv', sep='\t', usecols=hold_cols, dtype=str, low_memory=False)
    hold = hold[hold['ACCESSION_NUMBER'].isin(reg['ACCESSION_NUMBER'])]
    hold = hold[hold['ASSET_CAT'] == 'EC']

    ids = pd.read_csv(BASE / q / 'IDENTIFIERS.tsv', sep='\t', usecols=['HOLDING_ID','IDENTIFIER_ISIN','IDENTIFIER_TICKER'], dtype=str, low_memory=False)
    ids = ids.dropna(subset=['IDENTIFIER_ISIN']).drop_duplicates(subset=['HOLDING_ID'], keep='first')

    # Merge
    hold = hold.merge(reg, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(info, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(sub, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(ids, on='HOLDING_ID', how='left')

    hold['BALANCE'] = pd.to_numeric(hold['BALANCE'], errors='coerce')
    hold['CURRENCY_VALUE'] = pd.to_numeric(hold['CURRENCY_VALUE'], errors='coerce')
    hold['PERCENTAGE'] = pd.to_numeric(hold['PERCENTAGE'], errors='coerce')
    return hold

def main():
    quarters = discover_quarters()
    if len(quarters) < 2:
        print(f'ERROR: data/nport/ contiene {len(quarters)} trimestre(s); '
              f'se necesitan >=2 para calcular cambios.')
        print('En CI, data/nport/ es efimero (gitignored): el step previo '
              'update_sec_nport_data.py debe ejecutarse con --backfill >=1.')
        sys.exit(1)
    print(f'Trimestres detectados: {quarters}')
    frames = [load_quarter(q) for q in quarters]
    all_data = pd.concat(frames, ignore_index=True)

    all_data['SECURITY_KEY'] = all_data['IDENTIFIER_ISIN'].fillna(all_data['ISSUER_CUSIP'])
    # Granularidad correcta: fondo (CIK+SERIES_ID) + seguridad + fecha
    all_data = all_data.sort_values(['CIK','SERIES_ID','SECURITY_KEY','REPORT_DATE'])
    all_data['PREV_BALANCE'] = all_data.groupby(['CIK','SERIES_ID','SECURITY_KEY'])['BALANCE'].shift(1)
    all_data['POSITION_CHANGE'] = all_data['BALANCE'] - all_data['PREV_BALANCE']
    all_data['POSITION_CHANGE_PCT'] = all_data['POSITION_CHANGE'] / all_data['PREV_BALANCE'].replace(0, pd.NA) * 100

    change = all_data.dropna(subset=['POSITION_CHANGE']).copy()
    cols = [
        'ACCESSION_NUMBER','REPORT_DATE','CIK','REGISTRANT_NAME','SERIES_NAME','SERIES_ID',
        'ISSUER_NAME','ISSUER_CUSIP','IDENTIFIER_ISIN','SECURITY_KEY','BALANCE','PREV_BALANCE',
        'POSITION_CHANGE','POSITION_CHANGE_PCT'
    ]
    change = change[cols]
    change.to_csv(OUTPUT, index=False)
    print(f'Guardado en {OUTPUT}')
    print(f'Cambios calculados: {len(change)}')
    print(change[change['SERIES_NAME'].str.contains('EURO STOXX', case=False, na=False)].head(20).to_string(index=False))

if __name__ == '__main__':
    main()
