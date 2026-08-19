# -*- coding: utf-8 -*-
"""
Actualizador oficial de flujo primario QQQ desde SEC.

Fuente:
    SEC EDGAR — Invesco QQQ Trust, Series 1
    CIK: 0001067839

Proceso:
    1. Ejecuta extract_sec_qqq_primary_flow_v4.py para N-30B-2 anual.
    2. Descarga y parsea N-CSRS semestral más reciente.
    3. Consolida en outputs/history/qqq_sec_primary_flow.csv.

Frecuencia:
    Semestral / manual.
"""

from __future__ import annotations

import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNUAL_CSV = PROJECT_ROOT / "data" / "invesco" / "QQQ" / "sec_primary_flow_history.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "history" / "qqq_sec_primary_flow.csv"

CIK = "1067839"
CIK_PADDED = f"{int(CIK):010}"
HEADERS = {"User-Agent": "Macro Sectorial Radar v4.3 contact@example.com"}

def get_latest_filing_info(form_types=("N-CSRS",)):
    """Obtiene dinámicamente el filing más reciente para los tipos solicitados."""
    url = f"https://data.sec.gov/submissions/CIK{CIK_PADDED}.json"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    data = r.json()
    filings = data.get("filings", {}).get("recent", {})

    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    docs = filings.get("primaryDocument", [])

    best = None
    for form, date, acc, doc in zip(forms, dates, accessions, docs):
        if form in form_types and doc.lower().endswith((".htm", ".html")):
            if best is None or date > best["filing_date"]:
                best = {
                    "form": form,
                    "filing_date": date,
                    "accession_number": acc,
                    "primary_document": doc,
                    "accession_clean": acc.replace("-", ""),
                }
    return best

# Los N-CSRS / N-30B-2 se obtienen dinámicamente desde EDGAR.


def run_annual_extractor() -> None:
    """Ejecuta el extractor anual ya existente."""
    script = PROJECT_ROOT / "scripts" / "extract_sec_qqq_primary_flow_v4.py"
    print("Ejecutando extractor anual N-30B-2...")
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def parse_number(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "—", "-", ""}:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    elif s.startswith("("):
        negative = True
        s = s[1:]
    elif s == ")":
        return None
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()
    try:
        return -float(s) if negative else float(s)
    except ValueError:
        return None


def normalize_label(raw):
    if raw is None:
        return ""
    s = str(raw).replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip().lower().replace("(", "").replace(")", "")
    return s


def find_label_row(table, label):
    target = normalize_label(label)
    for idx, row in table.iterrows():
        for cell in row:
            if normalize_label(cell) == target:
                return int(idx)
    return None


def get_ncsrs_semiannual() -> dict | None:
    """Descarga y parsea el N-CSRS más reciente."""
    info = get_latest_filing_info(("N-CSRS",))
    if not info:
        print("No se encontró N-CSRS reciente en EDGAR.")
        return None

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{CIK}/"
        f"{info['accession_clean']}/{info['primary_document']}"
    )

    print(f"Descargando {info['form']} {info['filing_date']} ...")
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for html_table in soup.find_all("table"):
        try:
            table = pd.read_html(StringIO(str(html_table)))[0]
        except Exception:
            continue

        flat = " ".join(table.astype(str).fillna("").values.ravel()).lower()
        if "six months ended" in flat and "shares sold" in flat and "shares repurchased" in flat:
            period_row = 1
            sem_col = None
            for col in range(table.shape[1]):
                cell = str(table.iloc[period_row, col])
                if "six months ended" in cell.lower():
                    sem_col = col
                    break
            if sem_col is None:
                continue

            r_sold = find_label_row(table, "Shares sold")
            r_rep = find_label_row(table, "Shares repurchased")
            r_beg = find_label_row(table, "Shares outstanding, beginning of period")
            r_end = find_label_row(table, "Shares outstanding, end of period")
            r_proc = find_label_row(table, "Proceeds from shares sold")
            r_val = find_label_row(table, "Value of shares repurchased")

            period_text = str(table.iloc[period_row, sem_col]).strip()
            m = re.search(r"March\s+31,\s+(\d{4})", period_text, re.IGNORECASE)
            if not m:
                continue
            year_sem = int(m.group(1))

            def get_val(row_idx, col):
                return parse_number(table.iloc[row_idx, col]) if row_idx is not None else None

            row = {
                "filing_date": info["filing_date"],
                "accession_number": info["accession_number"],
                "primary_document": info["primary_document"],
                "source_url": url,
                "period_type": "SEMIANNUAL",
                "period_end_date": f"{year_sem}-03-31",
                "year": year_sem,
                "shares_sold": get_val(r_sold, sem_col),
                "shares_repurchased": get_val(r_rep, sem_col),
                "net_shares_flow": None,
                "shares_beginning": get_val(r_beg, sem_col),
                "shares_end": get_val(r_end, sem_col),
                "proceeds_shares_sold": get_val(r_proc, sem_col),
                "value_shares_repurchased": get_val(r_val, sem_col),
                "primary_flow_usd": None,
                "flow_pct_assets": None,
                "flow_zscore": None,
            }

            if row["shares_sold"] is not None and row["shares_repurchased"] is not None:
                row["net_shares_flow"] = row["shares_sold"] + row["shares_repurchased"]
            if row["proceeds_shares_sold"] is not None and row["value_shares_repurchased"] is not None:
                row["primary_flow_usd"] = row["proceeds_shares_sold"] + row["value_shares_repurchased"]

            return row

    return None


def main() -> None:
    run_annual_extractor()

    if not ANNUAL_CSV.exists():
        raise FileNotFoundError(f"No se generó el CSV anual: {ANNUAL_CSV}")

    annual = pd.read_csv(ANNUAL_CSV)
    annual = annual.dropna(subset=["year"]).copy()
    annual["year"] = annual["year"].astype(int)
    annual["period_type"] = "ANNUAL"
    annual["period_end_date"] = annual["year"].astype(str) + "-09-30"
    annual["net_shares_flow"] = annual["shares_sold"] + annual["shares_repurchased"]
    annual["primary_flow_usd"] = annual["proceeds_shares_sold"] + annual["value_shares_repurchased"]
    annual["flow_pct_assets"] = None
    annual["flow_zscore"] = None

    sem_row = get_ncsrs_semiannual()
    if sem_row:
        sem_df = pd.DataFrame([sem_row])
    else:
        print("No se pudo extraer N-CSRS semestral.")
        sem_df = pd.DataFrame()

    cols = [
        "filing_date", "accession_number", "primary_document", "source_url",
        "period_type", "period_end_date", "year",
        "shares_sold", "shares_repurchased", "net_shares_flow",
        "shares_beginning", "shares_end",
        "proceeds_shares_sold", "value_shares_repurchased",
        "primary_flow_usd", "flow_pct_assets", "flow_zscore"
    ]

    combined = pd.concat([annual, sem_df], ignore_index=True)
    combined = combined[cols]
    combined = combined.sort_values("year", ascending=False).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Consolidado guardado en: {OUTPUT_CSV}")
    print(f"Registros totales: {len(combined)}")
    print(combined.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
