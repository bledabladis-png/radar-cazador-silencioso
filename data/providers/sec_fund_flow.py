"""
Extrae flujos de fondos N-PORT (Sales/Reinvestment/Redemption) con fechas reales.
No se integra en run.py diario. Capa SEC_FUND_FLOW separada.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/nport')
OUTPUT_DIR = Path('outputs/history')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CIKS = {1064641, 884394, 1041130, 936958}

def load_filtered_info():
    print('Leyendo REGISTRANT.tsv ...')
    reg = pd.read_csv(DATA_DIR / 'REGISTRANT.tsv', sep='\t', low_memory=False)
    reg_sel = reg[reg['CIK'].isin(TARGET_CIKS)][['ACCESSION_NUMBER','CIK','REGISTRANT_NAME']]
    accessions = set(reg_sel['ACCESSION_NUMBER'])
    print(f'Registros REGISTRANT seleccionados: {len(reg_sel)}')

    print('Leyendo SUBMISSION.tsv ...')
    sub = pd.read_csv(DATA_DIR / 'SUBMISSION.tsv', sep='\t', low_memory=False)
    sub = sub[['ACCESSION_NUMBER','REPORT_DATE']].drop_duplicates()
    sub['REPORT_DATE'] = pd.to_datetime(sub['REPORT_DATE'], format='%d-%b-%Y', errors='coerce')
    sub = sub.dropna(subset=['REPORT_DATE'])

    print('Leyendo FUND_REPORTED_INFO.tsv ...')
    info = pd.read_csv(DATA_DIR / 'FUND_REPORTED_INFO.tsv', sep='\t', low_memory=False)
    info_sel = info[info['ACCESSION_NUMBER'].isin(accessions)].copy()
    print(f'Filas FUND_REPORTED_INFO seleccionadas: {len(info_sel)}')

    # Unir con metadata de fondo
    info_sel = info_sel.merge(reg_sel, on='ACCESSION_NUMBER', how='left')
    # Unir con fecha de reporte
    info_sel = info_sel.merge(sub, on='ACCESSION_NUMBER', how='left')
    return info_sel

def main():
    df = load_filtered_info()
    if df.empty:
        print('No hay datos para los CIKs objetivo.')
        return

    flow_cols_mon1 = ['SALES_FLOW_MON1','REINVESTMENT_FLOW_MON1','REDEMPTION_FLOW_MON1']
    flow_cols_mon2 = ['SALES_FLOW_MON2','REINVESTMENT_FLOW_MON2','REDEMPTION_FLOW_MON2']
    flow_cols_mon3 = ['SALES_FLOW_MON3','REINVESTMENT_FLOW_MON3','REDEMPTION_FLOW_MON3']

    for col in flow_cols_mon1 + flow_cols_mon2 + flow_cols_mon3:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['NET_FUND_FLOW_MON1'] = df['SALES_FLOW_MON1'] + df['REINVESTMENT_FLOW_MON1'] - df['REDEMPTION_FLOW_MON1']
    df['NET_FUND_FLOW_MON2'] = df['SALES_FLOW_MON2'] + df['REINVESTMENT_FLOW_MON2'] - df['REDEMPTION_FLOW_MON2']
    df['NET_FUND_FLOW_MON3'] = df['SALES_FLOW_MON3'] + df['REINVESTMENT_FLOW_MON3'] - df['REDEMPTION_FLOW_MON3']

    df['NET_FUND_FLOW_PCT_MON1'] = df['NET_FUND_FLOW_MON1'] / df['NET_ASSETS'] * 100
    df['NET_FUND_FLOW_PCT_MON2'] = df['NET_FUND_FLOW_MON2'] / df['NET_ASSETS'] * 100
    df['NET_FUND_FLOW_PCT_MON3'] = df['NET_FUND_FLOW_MON3'] / df['NET_ASSETS'] * 100

    rows = []
    for _, row in df.iterrows():
        base = {
            'ACCESSION_NUMBER': row['ACCESSION_NUMBER'],
            'CIK': row['CIK'],
            'REGISTRANT_NAME': row['REGISTRANT_NAME'],
            'SERIES_NAME': row.get('SERIES_NAME',''),
            'SERIES_ID': row.get('SERIES_ID',''),
            'TOTAL_ASSETS': row.get('TOTAL_ASSETS'),
            'NET_ASSETS': row.get('NET_ASSETS'),
            'REPORT_DATE': row['REPORT_DATE'],
        }
        # Asignar fechas reales por mes
        report_date = row['REPORT_DATE']
        if pd.notna(report_date):
            mon_dates = {
                'MON1': report_date - pd.DateOffset(months=2),
                'MON2': report_date - pd.DateOffset(months=1),
                'MON3': report_date,
            }
        else:
            mon_dates = {'MON1': None, 'MON2': None, 'MON3': None}

        for month, date_val in mon_dates.items():
            rows.append({
                **base,
                'MONTH_LABEL': month,
                'MONTH_DATE': date_val,
                'SALES_FLOW': row.get(f'SALES_FLOW_{month}'),
                'REINVESTMENT_FLOW': row.get(f'REINVESTMENT_FLOW_{month}'),
                'REDEMPTION_FLOW': row.get(f'REDEMPTION_FLOW_{month}'),
                'NET_FUND_FLOW': row.get(f'NET_FUND_FLOW_{month}'),
                'NET_FUND_FLOW_PCT': row.get(f'NET_FUND_FLOW_PCT_{month}'),
            })

    result = pd.DataFrame(rows)
    out_file = OUTPUT_DIR / 'sec_fund_flow.csv'
    result.to_csv(out_file, index=False)
    print(f'Guardado en {out_file}')
    print(f'Total filas: {len(result)}')
    print(result.head(20).to_string(index=False))

if __name__ == '__main__':
    main()
