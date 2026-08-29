import argparse
import datetime
import requests
import zipfile
from pathlib import Path
from io import BytesIO
import pandas as pd

BASE_URL = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"
NEEDED_FILES = [
    "SUBMISSION.tsv",
    "REGISTRANT.tsv",
    "FUND_REPORTED_INFO.tsv",
    "FUND_REPORTED_HOLDING.tsv",
    "IDENTIFIERS.tsv",
]
TARGET_CIKS = {1064641, 884394, 1041130, 936958, 1168164}

def download_quarter(quarter):
    out_dir = Path(f"data/nport/{quarter}")
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{quarter}_nport.zip"
    print(f"Descargando {url} ...")
    r = requests.get(url, headers={"User-Agent": "Macro_Sectorial contacto@example.com"}, timeout=600)

    if r.status_code == 404:
        print("No disponible aún en EDGAR.")
        return False

    r.raise_for_status()
    print(f"Descargado: {len(r.content)/(1024*1024):.2f} MB")
    z = zipfile.ZipFile(BytesIO(r.content))
    for name in NEEDED_FILES:
        print(f"  Extrayendo {name} ...")
        z.extract(name, out_dir)

    return True

def process_quarter(quarter):
    base = Path(f"data/nport/{quarter}")
    sub = pd.read_csv(base / "SUBMISSION.tsv", sep='\t', low_memory=False)
    sub = sub[['ACCESSION_NUMBER', 'REPORT_DATE']].drop_duplicates()
    sub['REPORT_DATE'] = pd.to_datetime(sub['REPORT_DATE'], format='%d-%b-%Y', errors='coerce')

    reg = pd.read_csv(base / "REGISTRANT.tsv", sep='\t', low_memory=False)
    reg = reg[reg['CIK'].isin(TARGET_CIKS)][['ACCESSION_NUMBER', 'CIK', 'REGISTRANT_NAME']]

    hold_cols = ['ACCESSION_NUMBER','HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP','BALANCE','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE']
    hold = pd.read_csv(base / "FUND_REPORTED_HOLDING.tsv", sep='\t', usecols=hold_cols, dtype=str, low_memory=False)
    hold = hold[hold['ACCESSION_NUMBER'].isin(reg['ACCESSION_NUMBER'])]
    hold = hold[hold['ASSET_CAT'] == 'EC']

    ids = pd.read_csv(base / "IDENTIFIERS.tsv", sep='\t', usecols=['HOLDING_ID','IDENTIFIER_ISIN','IDENTIFIER_TICKER'], dtype=str, low_memory=False)
    ids = ids.dropna(subset=['IDENTIFIER_ISIN']).drop_duplicates(subset=['HOLDING_ID'], keep='first')

    # Merge
    hold = hold.merge(reg, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(sub, on='ACCESSION_NUMBER', how='left')
    hold = hold.merge(ids, on='HOLDING_ID', how='left')

    hold['BALANCE'] = pd.to_numeric(hold['BALANCE'], errors='coerce')
    hold['CURRENCY_VALUE'] = pd.to_numeric(hold['CURRENCY_VALUE'], errors='coerce')
    hold['PERCENTAGE'] = pd.to_numeric(hold['PERCENTAGE'], errors='coerce')

    # Guardar posiciones
    pos_cols = [
        'ACCESSION_NUMBER','REPORT_DATE','CIK','REGISTRANT_NAME',
        'HOLDING_ID','ISSUER_NAME','ISSUER_CUSIP','IDENTIFIER_ISIN','IDENTIFIER_TICKER',
        'BALANCE','CURRENCY_VALUE','PERCENTAGE','ASSET_CAT','ISSUER_TYPE'
    ]
    pos = hold[pos_cols]
    pos.to_csv(f"outputs/history/sec_nport_positions_{quarter}.csv", index=False)

    # Calcular cambios dentro del trimestre (por fondo y activo)
    hold['SECURITY_KEY'] = hold['IDENTIFIER_ISIN'].fillna(hold['ISSUER_CUSIP'])
    hold = hold.sort_values(['CIK','SECURITY_KEY','REPORT_DATE'])
    hold['PREV_BALANCE'] = hold.groupby(['CIK','SECURITY_KEY'])['BALANCE'].shift(1)
    hold['POSITION_CHANGE'] = hold['BALANCE'] - hold['PREV_BALANCE']
    hold['POSITION_CHANGE_PCT'] = hold['POSITION_CHANGE'] / hold['PREV_BALANCE'].replace(0, pd.NA) * 100

    change = hold.dropna(subset=['POSITION_CHANGE'])[
        ['REPORT_DATE','CIK','REGISTRANT_NAME','SECURITY_KEY','ISSUER_NAME','BALANCE','PREV_BALANCE','POSITION_CHANGE','POSITION_CHANGE_PCT']
    ]
    change.to_csv(f"outputs/history/sec_nport_position_change_{quarter}.csv", index=False)
    print(f"Posiciones guardadas: outputs/history/sec_nport_positions_{quarter}.csv")
    print(f"Cambios guardados: outputs/history/sec_nport_position_change_{quarter}.csv")

def get_last_closed_quarter() -> str:
    """Calcula el último trimestre cerrado según el calendario de publicación de la SEC."""
    today = datetime.date.today()
    year = today.year
    month = today.month

    if month in (1, 2, 3):
        return f"{year - 1}q4"
    if month in (4, 5, 6):
        return f"{year}q1"
    if month in (7, 8, 9):
        return f"{year}q2"
    return f"{year}q3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--quarter',
        required=False,
        help='Ej. 2026q2. Si se omite, se calcula el último trimestre cerrado.'
    )
    args = parser.parse_args()
    quarter = (args.quarter or get_last_closed_quarter()).lower()
    print(f"Trimestre objetivo: {quarter}")

    if not download_quarter(quarter):
        print("Trimestre no publicado. Finalizando sin error.")
        return

    process_quarter(quarter)

if __name__ == '__main__':
    main()
