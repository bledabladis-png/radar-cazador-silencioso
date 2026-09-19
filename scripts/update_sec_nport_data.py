import argparse
import datetime
import sys
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

# Universo objetivo: SERIES_IDs de los ETFs que cubren el radar USA.
# Filtrado por SERIES_ID (unidad atomica del ETF), no por CIK (un trust
# contiene 200+ ETFs distintos).
TARGET_SERIES = {
    # Invesco
    "S000101292",  # Invesco QQQ Trust, Series 1
    # iShares
    "S000004310",  # iShares Core S&P 500 ETF
    "S000004307",  # iShares Core S&P Mid-Cap ETF
    "S000004313",  # iShares Core S&P Small-Cap ETF
    "S000004344",  # iShares Russell 2000 ETF
    "S000004317",  # iShares Core S&P Total U.S. Stock Market ETF
    # Vanguard
    "S000002839",  # VANGUARD 500 INDEX FUND
    "S000002848",  # VANGUARD TOTAL STOCK MARKET INDEX FUND
    "S000002844",  # VANGUARD MID-CAP INDEX FUND
    "S000002845",  # VANGUARD SMALL-CAP INDEX FUND
    "S000002841",  # VANGUARD EXTENDED MARKET INDEX FUND
}

def download_quarter(quarter):
    out_dir = Path(f"data/nport/{quarter}")
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{quarter}_nport.zip"
    print(f"Descargando {url} ...")
    r = requests.get(url, headers={"User-Agent": "Macro_Sectorial contacto@example.com"}, timeout=600)

    if r.status_code == 404:
        print("No disponible aun en EDGAR.")
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

    # 1. SUBMISSION: ACCESSION -> REPORT_DATE
    sub = pd.read_csv(base / "SUBMISSION.tsv", sep="\t", low_memory=False)
    sub = sub[["ACCESSION_NUMBER", "REPORT_DATE"]].drop_duplicates()
    sub["REPORT_DATE"] = pd.to_datetime(sub["REPORT_DATE"], format="%d-%b-%Y", errors="coerce")

    # 2. FUND_REPORTED_INFO: filtrar por SERIES_ID -> ACCESSIONs relevantes
    info = pd.read_csv(base / "FUND_REPORTED_INFO.tsv", sep="\t", low_memory=False)
    info = info[info["SERIES_ID"].isin(TARGET_SERIES)][["ACCESSION_NUMBER", "SERIES_ID", "SERIES_NAME"]]
    accessions = set(info["ACCESSION_NUMBER"].unique())
    print(f"  ACCESSIONs seleccionados: {len(accessions)} ({len(TARGET_SERIES)} series objetivo)")

    # 3. REGISTRANT: metadata del registrant por ACCESSION
    reg = pd.read_csv(base / "REGISTRANT.tsv", sep="\t", low_memory=False)
    reg = reg[reg["ACCESSION_NUMBER"].isin(accessions)][["ACCESSION_NUMBER", "CIK", "REGISTRANT_NAME"]]

    # 4. FUND_REPORTED_HOLDING: solo holdings de accessions relevantes
    hold_cols = ["ACCESSION_NUMBER","HOLDING_ID","ISSUER_NAME","ISSUER_CUSIP","BALANCE","CURRENCY_VALUE","PERCENTAGE","ASSET_CAT","ISSUER_TYPE"]
    hold = pd.read_csv(base / "FUND_REPORTED_HOLDING.tsv", sep="\t", usecols=hold_cols, dtype=str, low_memory=False)
    hold = hold[hold["ACCESSION_NUMBER"].isin(accessions)]
    hold = hold[hold["ASSET_CAT"] == "EC"]

    # 5. IDENTIFIERS: pivot por HOLDING_ID (ISIN y TICKER en filas separadas)
    ids = pd.read_csv(base / "IDENTIFIERS.tsv", sep="\t",
                      usecols=["HOLDING_ID","IDENTIFIER_ISIN","IDENTIFIER_TICKER"], dtype=str, low_memory=False)
    ids_isin = ids[ids["IDENTIFIER_ISIN"].notna()][["HOLDING_ID","IDENTIFIER_ISIN"]].drop_duplicates(subset=["HOLDING_ID"], keep="first")
    ids_tick = ids[ids["IDENTIFIER_TICKER"].notna()][["HOLDING_ID","IDENTIFIER_TICKER"]].drop_duplicates(subset=["HOLDING_ID"], keep="first")
    ids = ids_isin.merge(ids_tick, on="HOLDING_ID", how="outer")

    # 6. Merge final
    hold = hold.merge(reg, on="ACCESSION_NUMBER", how="left")
    hold = hold.merge(info, on="ACCESSION_NUMBER", how="left")
    hold = hold.merge(sub, on="ACCESSION_NUMBER", how="left")
    hold = hold.merge(ids, on="HOLDING_ID", how="left")

    hold["BALANCE"] = pd.to_numeric(hold["BALANCE"], errors="coerce")
    hold["CURRENCY_VALUE"] = pd.to_numeric(hold["CURRENCY_VALUE"], errors="coerce")
    hold["PERCENTAGE"] = pd.to_numeric(hold["PERCENTAGE"], errors="coerce")

    pos_cols = [
        "ACCESSION_NUMBER","REPORT_DATE","CIK","REGISTRANT_NAME","SERIES_ID","SERIES_NAME",
        "HOLDING_ID","ISSUER_NAME","ISSUER_CUSIP","IDENTIFIER_ISIN","IDENTIFIER_TICKER",
        "BALANCE","CURRENCY_VALUE","PERCENTAGE","ASSET_CAT","ISSUER_TYPE",
    ]
    pos = hold[pos_cols]
    out_pos = f"outputs/history/sec_nport_positions_{quarter}.csv"
    pos.to_csv(out_pos, index=False)
    print(f"Posiciones guardadas: {out_pos}")

    # 7. Cambios intra-trimestre
    hold["SECURITY_KEY"] = hold["IDENTIFIER_ISIN"].fillna(hold["ISSUER_CUSIP"])
    hold = hold.sort_values(["CIK","SECURITY_KEY","REPORT_DATE"])
    hold["PREV_BALANCE"] = hold.groupby(["CIK","SECURITY_KEY"])["BALANCE"].shift(1)
    hold["POSITION_CHANGE"] = hold["BALANCE"] - hold["PREV_BALANCE"]
    hold["POSITION_CHANGE_PCT"] = hold["POSITION_CHANGE"] / hold["PREV_BALANCE"].replace(0, pd.NA) * 100

    change = hold.dropna(subset=["POSITION_CHANGE"])[
        ["REPORT_DATE","CIK","REGISTRANT_NAME","SERIES_ID","SERIES_NAME","SECURITY_KEY","ISSUER_NAME","BALANCE","PREV_BALANCE","POSITION_CHANGE","POSITION_CHANGE_PCT"]
    ]
    out_change = f"outputs/history/sec_nport_position_change_{quarter}.csv"
    change.to_csv(out_change, index=False)
    print(f"Cambios guardados: {out_change}")

def get_last_closed_quarter() -> str:
    today = datetime.date.today()
    year, month = today.year, today.month
    if month in (1, 2, 3):  return f"{year - 1}q4"
    if month in (4, 5, 6):  return f"{year}q1"
    if month in (7, 8, 9):  return f"{year}q2"
    return f"{year}q3"

def previous_quarter(q):
    year, qnum = int(q[:4]), int(q[-1])
    if qnum == 1:
        return f"{year - 1}q4"
    return f"{year}q{qnum - 1}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarter", required=False, help="Ej. 2026q2. Si se omite, se calcula el ultimo trimestre cerrado.")
    args = parser.parse_args()
    quarter = (args.quarter or get_last_closed_quarter()).lower()
    print(f"Trimestre objetivo inicial: {quarter}")

    for _ in range(4):
        if download_quarter(quarter):
            process_quarter(quarter)
            print(f"N-PORT actualizado correctamente: {quarter}")
            return
        print(f"  {quarter} no disponible en EDGAR, probando trimestre anterior...")
        quarter = previous_quarter(quarter)

    print("ERROR: ningun trimestre disponible en los ultimos 4 intentos.")
    sys.exit(1)

if __name__ == "__main__":
    main()
