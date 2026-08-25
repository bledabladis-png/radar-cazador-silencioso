# -*- coding: utf-8 -*-
"""
Extractor definitivo de flujo primario QQQ desde SEC N-30B-2.

Versión 4:
    - Filtra tablas por contenido clave.
    - Localiza la fila de años dinámicamente.
    - Extrae valores por grupo de columnas.
    - Soporta informes modernos y antiguos.

Fuente:
    SEC EDGAR - Invesco QQQ Trust, Series 1
    CIK: 0001067839

Salida:
    data/invesco/QQQ/sec_primary_flow_history.csv
"""

from __future__ import annotations

import re
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

CIK = "1067839"
CIK_PADDED = f"{int(CIK):010}"
BASE_SUBMISSIONS = f"https://data.sec.gov/submissions/CIK{CIK_PADDED}.json"
HEADERS = {"User-Agent": "Macro Sectorial Radar v4.3 contact@example.com"}

def _get_with_retry(url, retries=3, backoff=2):
    """GET con reintentos para 429, 500 y 503."""
    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            if resp.status_code in (429, 500, 503):
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} {resp.reason}",
                    response=resp
                )
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            print(f"  Reintento {attempt+1}/{retries} para {url} ({exc})")
            time.sleep(backoff * (attempt + 1))
    raise last_exc


OUTPUT = Path("data/invesco/QQQ/sec_primary_flow_history.csv")

LABELS = [
    "Shares sold",
    "Shares repurchased",
    "Shares outstanding, beginning of year",
    "Shares outstanding, end of year",
    "Proceeds from shares sold",
    "Value of shares repurchased",
]


def parse_number(raw: Any) -> float | None:
    """Convierte texto contable de SEC en float."""
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
        # Formato antiguo: el paréntesis de cierre puede
        # aparecer en una columna separada.
        negative = True
        s = s[1:]
    elif s == ")":
        return None

    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()

    try:
        value = float(s)
    except ValueError:
        return None

    return -value if negative else value


def normalize_label(raw: Any) -> str:
    if raw is None:
        return ""
    s = str(raw)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().lower()
    s = s.replace("(", "").replace(")", "")
    return s


def find_year_row(table: pd.DataFrame) -> int | None:
    """Localiza la fila que contiene años fiscales."""
    best_row = None
    best_count = 0

    for idx, row in table.head(8).iterrows():
        years = []
        for cell in row:
            text = str(cell).strip()
            if re.fullmatch(r"(19|20)\d{2}", text):
                years.append(int(text))

        count = len(set(years))
        if count > best_count:
            best_count = count
            best_row = int(idx)

    return best_row if best_count >= 2 else None


def get_year_groups(table: pd.DataFrame, year_row: int) -> dict[int, list[int]]:
    """
    Agrupa columnas por año fiscal.
    """
    groups: dict[int, list[int]] = {}
    current_year: int | None = None

    for col in range(table.shape[1]):
        cell = table.iloc[year_row, col]
        text = str(cell).strip()

        if re.fullmatch(r"(19|20)\d{2}", text):
            current_year = int(text)
            groups.setdefault(current_year, []).append(col)
        else:
            # Si la celda está vacía y la anterior era un año, mantener el grupo
            if current_year is not None and text == "":
                if col > 0:
                    prev = str(table.iloc[year_row, col - 1]).strip()
                    if prev and re.fullmatch(r"(19|20)\d{2}", prev):
                        groups[current_year].append(col)

    return groups


def find_label_row(table: pd.DataFrame, label: str) -> int | None:
    target = normalize_label(label)
    for idx, row in table.iterrows():
        for cell in row:
            if normalize_label(cell) == target:
                return int(idx)
    return None


def extract_value_from_row(table: pd.DataFrame, row_idx: int, cols: list[int]) -> float | None:
    for col in cols:
        value = parse_number(table.iloc[row_idx, col])
        if value is not None:
            return value
    return None


def table_has_required_content(table: pd.DataFrame) -> bool:
    """Comprueba si la tabla contiene los marcadores clave."""
    flat = " ".join(table.astype(str).fillna("").values.ravel()).lower()
    return ("shares sold" in flat) and ("shares repurchased" in flat)


def extract_records(table: pd.DataFrame, filing_date: str, accession: str, doc: str, url: str) -> list[dict[str, Any]]:
    """Extrae registros solo de la tabla correcta."""
    if not table_has_required_content(table):
        return []

    year_row = find_year_row(table)
    if year_row is None:
        return []

    year_groups = get_year_groups(table, year_row)
    if not year_groups:
        return []

    label_rows = {label: find_label_row(table, label) for label in LABELS}

    # Si no se encuentran las etiquetas principales, no es la tabla correcta
    if label_rows["Shares sold"] is None and label_rows["Shares repurchased"] is None:
        return []

    records: list[dict[str, Any]] = []

    for year in sorted(year_groups, reverse=True):
        cols = year_groups[year]

        record = {
            "filing_date": filing_date,
            "accession_number": accession,
            "primary_document": doc,
            "source_url": url,
            "year": year,
            "shares_sold": extract_value_from_row(table, label_rows["Shares sold"], cols) if label_rows["Shares sold"] is not None else None,
            "shares_repurchased": extract_value_from_row(table, label_rows["Shares repurchased"], cols) if label_rows["Shares repurchased"] is not None else None,
            "shares_beginning": extract_value_from_row(table, label_rows["Shares outstanding, beginning of year"], cols) if label_rows["Shares outstanding, beginning of year"] is not None else None,
            "shares_end": extract_value_from_row(table, label_rows["Shares outstanding, end of year"], cols) if label_rows["Shares outstanding, end of year"] is not None else None,
            "proceeds_shares_sold": extract_value_from_row(table, label_rows["Proceeds from shares sold"], cols) if label_rows["Proceeds from shares sold"] is not None else None,
            "value_shares_repurchased": extract_value_from_row(table, label_rows["Value of shares repurchased"], cols) if label_rows["Value of shares repurchased"] is not None else None,
        }
        records.append(record)

    return records


def main() -> None:
    print("Consultando índice EDGAR...")
    r = _get_with_retry(BASE_SUBMISSIONS)
    r.raise_for_status()
    data = r.json()

    filings = data["filings"]["recent"]

    forms = filings["form"]
    dates = filings["filingDate"]
    accessions = filings["accessionNumber"]
    docs = filings["primaryDocument"]

    all_records: list[dict[str, Any]] = []

    for form, filing_date, accession, doc in zip(forms, dates, accessions, docs):
        if form != "N-30B-2":
            continue
        if not doc.lower().endswith((".htm", ".html")):
            continue

        acc_clean = accession.replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{CIK}/"
            f"{acc_clean}/{doc}"
        )

        try:
            print(f"Descargando {filing_date} {accession} ...", end=" ", flush=True)
            resp = _get_with_retry(url)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            html_tables = soup.find_all("table")

            extracted_any = False

            for html_table in html_tables:
                try:
                    table = pd.read_html(StringIO(str(html_table)))[0]
                except Exception:
                    continue

                records = extract_records(
                    table=table,
                    filing_date=filing_date,
                    accession=accession,
                    doc=doc,
                    url=url,
                )
                if records:
                    all_records.extend(records)
                    extracted_any = True
                    print(f"OK ({len(records)} registros)")
                    break

            if not extracted_any:
                print("sin tabla de cambios de acciones")

            time.sleep(0.3)

        except Exception as exc:
            print(f"ERROR: {exc}")

    if not all_records:
        raise RuntimeError("No se extrajeron registros")

    df = pd.DataFrame(all_records)
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df = df.sort_values(["year", "filing_date"], ascending=[True, False])
    df = df.drop_duplicates(subset=["year"], keep="first")
    df = df.sort_values("year", ascending=False)

    df["shares_flow_net"] = df["shares_sold"] + df["shares_repurchased"]
    df["shares_change_check"] = df["shares_end"] - df["shares_beginning"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"\nGuardado: {OUTPUT}")
    print(f"Registros: {len(df)}")
    print(f"Años cubiertos: {df['year'].min()} - {df['year'].max()}")
    print("\nVista previa:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
