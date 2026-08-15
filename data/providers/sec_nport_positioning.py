"""
Extrae posiciones N-PORT de fondos seleccionados y genera un CSV limpio.
No calcula flow; solo posiciones institucionales con granularidad FUND+SECURITY+REPORT_DATE.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/nport')
OUTPUT_DIR = Path('outputs/history')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CIKS = {1064641, 884394, 1041130, 936958}

def load_table(name, usecols=None):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f'Falta {name} en {DATA_DIR}')
    return pd.read_csv(path, sep='\t', usecols=usecols, low_memory=False)

def first_non_null(series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else None

def main():
    print('Leyendo SUBMISSION.tsv ...')
    sub = pd.read_csv(DATA_DIR / 'SUBMISSION.tsv', sep='\t', low_memory=False)
    sub = sub[['ACCESSION_NUMBER','FILING_DATE','REPORT_DATE','SUB_TYPE','IS_LAST_FILING']]
    # Parsear fechas
    sub['FILING_DATE'] = pd.to_datetime(sub['FILING_DATE'], format='%d-%b-%Y', errors='coerce')
    sub['REPORT_DATE'] = pd.to_datetime(sub['REPORT_DATE'], format='%d-%b-%Y', errors='coerce')

    print('Leyendo REGISTRANT.tsv ...')
    reg = pd.read_csv(DATA_DIR / 'REGISTRANT.tsv', sep='\t', low_memory=False)
    reg = reg[['ACCESSION_NUMBER','CIK','REGISTRANT_NAME']]
    reg_sel = reg[reg['CIK'].isin(TARGET_CIKS)]
    print(f'Registros REGISTRANT seleccionados: {len(reg_sel)}')

    # Unir submission con registrant para obtener cik + report_date
    sub_sel = sub[sub['ACCESSION_NUMBER'].isin(reg_sel['ACCESSION_NUMBER'])]
    acc_meta = sub_sel.merge(reg_sel, on='ACCESSION_NUMBER', how='left')
    print(f'ACCESSIONs con metadata: {len(acc_meta)}')

    # Mapeo accession -> report_date/filing_date/cik/name
    acc_info = acc_meta.set_index('ACCESSION_NUMBER')[['REPORT_DATE','FILING_DATE','CIK','REGISTRANT_NAME']]

    print('Leyendo FUND_REPORTED_HOLDING.tsv por chunks ...')
    hold_cols = [
        'ACCESSION_NUMBER','HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP',
        'BALANCE','UNIT','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE'
    ]
    chunks = pd.read_csv(
        DATA_DIR / 'FUND_REPORTED_HOLDING.tsv',
        sep='\t', usecols=hold_cols, dtype=str,
        chunksize=500000, low_memory=False
    )
    holdings_list = []
    target_accessions = set(acc_info.index)
    for chunk in chunks:
        mask = chunk['ACCESSION_NUMBER'].isin(target_accessions)
        if mask.any():
            holdings_list.append(chunk[mask])
    holdings = pd.concat(holdings_list, ignore_index=True) if holdings_list else pd.DataFrame(columns=hold_cols)
    print(f'Holdings filtrados: {len(holdings)}')

    # Filtrar solo Equity (EC)
    holdings = holdings[holdings['ASSET_CAT'] == 'EC'].copy()
    print(f'Después de filtrar Equity: {len(holdings)}')

    # Convertir numéricos
    holdings['BALANCE'] = pd.to_numeric(holdings['BALANCE'], errors='coerce')
    holdings['CURRENCY_VALUE'] = pd.to_numeric(holdings['CURRENCY_VALUE'], errors='coerce')
    holdings['PERCENTAGE'] = pd.to_numeric(holdings['PERCENTAGE'], errors='coerce')

    holding_ids = set(holdings['HOLDING_ID'].dropna().unique())
    print(f'HOLDING_IDs: {len(holding_ids)}')

    print('Leyendo IDENTIFIERS.tsv por chunks ...')
    id_cols = ['HOLDING_ID','IDENTIFIER_ISIN','IDENTIFIER_TICKER']
    chunks_ids = pd.read_csv(
        DATA_DIR / 'IDENTIFIERS.tsv',
        sep='\t', usecols=id_cols, dtype=str,
        chunksize=500000, low_memory=False
    )
    ids_list = []
    for chunk in chunks_ids:
        mask = chunk['HOLDING_ID'].isin(holding_ids)
        if mask.any():
            ids_list.append(chunk[mask])
    ids = pd.concat(ids_list, ignore_index=True) if ids_list else pd.DataFrame(columns=id_cols)
    print(f'Identificadores crudos: {len(ids)}')

    print('Consolidando identificadores ...')
    ids_consolidated = ids.groupby('HOLDING_ID').agg({
        'IDENTIFIER_ISIN': first_non_null,
        'IDENTIFIER_TICKER': first_non_null,
    }).reset_index()
    print(f'Identificadores consolidados: {len(ids_consolidated)}')

    # Merge
    positions = holdings.merge(ids_consolidated, on='HOLDING_ID', how='left')
    positions = positions.drop_duplicates(subset=['ACCESSION_NUMBER','HOLDING_ID'])

    # Añadir metadata de fecha y fondo
    positions = positions.merge(acc_info, left_on='ACCESSION_NUMBER', right_index=True, how='left')

    # Seleccionar y ordenar columnas finales
    final_cols = [
        'ACCESSION_NUMBER','CIK','REGISTRANT_NAME','REPORT_DATE','FILING_DATE',
        'HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP','IDENTIFIER_ISIN','IDENTIFIER_TICKER',
        'BALANCE','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE'
    ]
    positions = positions[final_cols]
    positions = positions.sort_values(['CIK','REPORT_DATE','HOLDING_ID'])

    out_file = OUTPUT_DIR / 'sec_nport_positions.csv'
    positions.to_csv(out_file, index=False)
    print(f'Guardado en {out_file}')
    print(f'Total filas: {len(positions)}')
    print(positions.head(20).to_string(index=False))

if __name__ == '__main__':
    main()

