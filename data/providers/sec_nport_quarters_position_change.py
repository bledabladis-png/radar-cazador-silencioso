import pandas as pd
from pathlib import Path

QUARTERS = ['2026q1', '2026q2']
BASE = Path('data/nport')
OUTPUT = Path('outputs/history/sec_nport_position_change_quarterly.csv')
TARGET_CIKS = {1064641, 884394, 1041130, 936958}

def load_quarter(q):
    print(f'Cargando {q}...')
    sub = pd.read_csv(BASE / q / 'SUBMISSION.tsv', sep='\t', low_memory=False)
    sub = sub[['ACCESSION_NUMBER', 'REPORT_DATE']].drop_duplicates()
    sub['REPORT_DATE'] = pd.to_datetime(sub['REPORT_DATE'], format='%d-%b-%Y', errors='coerce')

    reg = pd.read_csv(BASE / q / 'REGISTRANT.tsv', sep='\t', low_memory=False)
    reg = reg[reg['CIK'].isin(TARGET_CIKS)][['ACCESSION_NUMBER','CIK','REGISTRANT_NAME']]

    hold_cols = ['ACCESSION_NUMBER','HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP','BALANCE','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE']
    hold = pd.read_csv(BASE / q / 'FUND_REPORTED_HOLDING.tsv', sep='\t', usecols=hold_cols, dtype=str, low_memory=False)
    hold = hold[hold['ACCESSION_NUMBER'].isin(reg['ACCESSION_NUMBER'])]
    hold = hold[hold['ASSET_CAT'] == 'EC']

    ids = pd.read_csv(BASE / q / 'IDENTIFIERS.tsv', sep='\t', usecols=['HOLDING_ID','IDENTIFIER_ISIN','IDENTIFIER_TICKER'], dtype=str, low_memory=False)
    ids = ids.dropna(subset=['IDENTIFIER_ISIN']).drop_duplicates(subset=['HOLDING_ID'], keep='first')

    # Merge
    hold = hold.merge(reg, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(sub, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(ids, on='HOLDING_ID', how='left')

    hold['BALANCE'] = pd.to_numeric(hold['BALANCE'], errors='coerce')
    hold['CURRENCY_VALUE'] = pd.to_numeric(hold['CURRENCY_VALUE'], errors='coerce')
    hold['PERCENTAGE'] = pd.to_numeric(hold['PERCENTAGE'], errors='coerce')
    return hold

def main():
    frames = []
    for q in QUARTERS:
        df = load_quarter(q)
        frames.append(df)
    all_data = pd.concat(frames, ignore_index=True)

    # Ordenar y calcular cambio por fondo y cusip (o isin)
    all_data['SECURITY_KEY'] = all_data['IDENTIFIER_ISIN'].fillna(all_data['ISSUER_CUSIP'])
    all_data = all_data.sort_values(['CIK','SECURITY_KEY','REPORT_DATE'])
    all_data['PREV_BALANCE'] = all_data.groupby(['CIK','SECURITY_KEY'])['BALANCE'].shift(1)
    all_data['POSITION_CHANGE'] = all_data['BALANCE'] - all_data['PREV_BALANCE']
    all_data['POSITION_CHANGE_PCT'] = all_data['POSITION_CHANGE'] / all_data['PREV_BALANCE'].replace(0, pd.NA) * 100

    change = all_data.dropna(subset=['POSITION_CHANGE']).copy()
    change = change[['REPORT_DATE','REGISTRANT_NAME','ISSUER_NAME','ISSUER_CUSIP','IDENTIFIER_ISIN','SECURITY_KEY','BALANCE','PREV_BALANCE','POSITION_CHANGE','POSITION_CHANGE_PCT']]
    change.to_csv(OUTPUT, index=False)
    print(f'Guardado en {OUTPUT}')
    print(f'Cambios calculados: {len(change)}')
    print(change[['REPORT_DATE','REGISTRANT_NAME','ISSUER_NAME','SECURITY_KEY','BALANCE','PREV_BALANCE','POSITION_CHANGE','POSITION_CHANGE_PCT']].head(20).to_string(index=False))

if __name__ == '__main__':
    main()
