from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any

import requests
import yfinance as yf

# Adaptación local
AMUNDI_URL = "https://www.amundietf.es/mapi/ProductAPI/getProductsData"
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_HOLDINGS = BASE_DIR / "outputs" / "amundi_lyxi_holdings.csv"
OUTPUT_VALIDATION = BASE_DIR / "outputs" / "amundi_lyxi_validation.csv"


# ---------------------------------------------------------
# AMUNDI REQUEST
# ---------------------------------------------------------

def build_request_body(isin: str) -> dict[str, Any]:
    return {
        "context": {
            "countryCode": "ESP",
            "countryName": "Spain",
            "googleCountryCode": "ES",
            "domainName": "www.amundietf.es",
            "bcp47Code": "es-ES",
            "languageName": "Spanish",
            "gtmCode": "GTM-W8T6L9X",
            "languageCode": "es",
            "userProfileName": "RETAIL",
            "userProfileSlug": "retail",
            "portalProfileName": None,
            "portalProfileSlug": None
        },
        "productIds": [isin],
        "characteristics": [
            "ISIN",
            "TICKER",
            "SHARE_MARKETING_NAME",
            "BENCHMARK_NAME",
            "BENCHMARK_TICKER",
            "FUND_FUND_NAME",
            "AUM",
            "FUND_AUM",
            "AUM_IN_EURO",
            "FUND_AUM_IN_EURO",
            "TER",
            "TOTAL_EXPENSE_RATIO",
            "CURRENCY",
            "BASE_CURRENCY",
            "ASSET_CLASS",
            "INVESTMENT_ZONE",
            "INVESTMENT_TYPE",
            "NUMBER_OF_COMPONENTS",
            "IS_ACTIVE_ETF",
            "IS_ACTIVELY_MANAGED",
            "REPLICATION_METHODOLOGY",
            "REPLICATION_IS_DIRECT",
            "REPLICATION_IS_SWAP_BASED",
            "INDEX_TRACKED",
            "POSITION_AS_OF_DATE",
            "FUND_BREAKDOWNS_AS_OF_DATE"
        ],
        "historics": [],
        "metrics": [],
        "breakDown": {
            "aggregationFields": [
                "INDEX_TOP10",
                "FUND_TOP10",
                "INDEX_SECTORS",
                "INDEX_COUNTRIES",
                "INDEX_ASSETCLASSES",
                "FUND_ASSETCLASSES",
                "FUND_COUNTRIES",
                "FUND_SECTORS"
            ]
        },
        "productType": "PRODUCT",
        "composition": {
            "compositionFields": [
                "date",
                "type",
                "bbg",
                "isin",
                "name",
                "weight",
                "quantity",
                "currency",
                "sector",
                "country",
                "countryOfRisk"
            ]
        }
    }


def get_amundi_product(isin: str) -> dict[str, Any]:
    body = build_request_body(isin)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Origin": "https://www.amundietf.es",
        "Referer": "https://www.amundietf.es/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        ),
    }
    response = requests.post(AMUNDI_URL, json=body, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    products = data.get("products")
    if not products:
        raise RuntimeError(f"Amundi no devolvió products para {isin}")
    return products[0]


def extract_composition(product: dict[str, Any]) -> list[dict[str, Any]]:
    composition = product.get("composition")
    if not composition:
        raise RuntimeError("El producto no contiene composition.")
    rows = composition.get("compositionData")
    if not rows:
        raise RuntimeError("El producto no contiene compositionData.")
    return rows


# ---------------------------------------------------------
# BBG → YAHOO
# ---------------------------------------------------------

def clean_bbg(bbg: str | None) -> str | None:
    if not bbg:
        return None
    return bbg.strip().upper()


def bbg_to_yahoo(bbg: str | None) -> str | None:
    bbg = clean_bbg(bbg)
    if not bbg:
        return None
    parts = bbg.split()
    if len(parts) >= 2:
        symbol = parts[0]
        market = parts[-1]
        if market == "SM":
            return f"{symbol}.MC"
        if market == "FP":
            return f"{symbol}.PA"
        if market == "GY":
            return f"{symbol}.DE"
        if market == "LN":
            return f"{symbol}.L"
        if market == "NA":
            return f"{symbol}.AS"
    return parts[0]


# ---------------------------------------------------------
# YAHOO SEARCH
# ---------------------------------------------------------

def yahoo_search(query: str, session: requests.Session | None = None) -> list[dict[str, Any]]:
    session = session or requests.Session()
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 10, "newsCount": 0}
    response = session.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    data = response.json()
    return data.get("quotes", [])


def validate_yahoo_ticker(ticker: str | None) -> bool:
    if not ticker:
        return False
    try:
        obj = yf.Ticker(ticker)
        history = obj.history(period="5d", auto_adjust=False)
        return not history.empty
    except Exception:
        return False


# ---------------------------------------------------------
# RESOLVER TICKER
# ---------------------------------------------------------

def resolve_yahoo_ticker(row: dict[str, Any], session: requests.Session) -> tuple[str | None, str]:
    bbg = row.get("bbg")
    name = row.get("name")
    isin = row.get("isin")

    candidate = bbg_to_yahoo(bbg)
    if candidate and validate_yahoo_ticker(candidate):
        return candidate, "BBG_RULE"

    if isin:
        try:
            results = yahoo_search(isin, session)
            for result in results:
                symbol = result.get("symbol")
                if symbol and validate_yahoo_ticker(symbol):
                    return symbol, "YAHOO_ISIN_SEARCH"
        except Exception:
            pass

    if name:
        try:
            results = yahoo_search(name, session)
            for result in results:
                symbol = result.get("symbol")
                if symbol and validate_yahoo_ticker(symbol):
                    return symbol, "YAHOO_NAME_SEARCH"
        except Exception:
            pass

    return None, "NOT_FOUND"


# ---------------------------------------------------------
# NORMALIZACIÓN
# ---------------------------------------------------------

def normalize_holdings(isin: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        characteristics = row.get("compositionCharacteristics", row)
        security_type = characteristics.get("type")
        if security_type != "EQUITY_ORDINARY":
            continue
        security_isin = characteristics.get("isin")
        if not security_isin:
            continue
        normalized.append({
            "etf": isin,
            "ticker": None,
            "name": characteristics.get("name"),
            "weight": characteristics.get("weight"),
            "isin": security_isin,
            "bbg": characteristics.get("bbg"),
            "sector": characteristics.get("sector"),
            "country": characteristics.get("country"),
            "countryOfRisk": characteristics.get("countryOfRisk"),
            "date": characteristics.get("date"),
            "currency": characteristics.get("currency"),
            "quantity": characteristics.get("quantity"),
        })
    return normalized


# ---------------------------------------------------------
# PROCESO COMPLETO
# ---------------------------------------------------------

def extract_etf(isin: str) -> list[dict[str, Any]]:
    print(f"\n[AMUNDI] {isin}")
    product = get_amundi_product(isin)
    rows = extract_composition(product)
    holdings = normalize_holdings(isin, rows)
    print(f"[OK] Posiciones equity encontradas: {len(holdings)}")

    session = requests.Session()
    for i, row in enumerate(holdings, start=1):
        ticker, source = resolve_yahoo_ticker(row, session)
        row["ticker"] = ticker
        row["ticker_source"] = source
        status = "OK" if ticker else "FAIL"
        print(f"[{i:02d}/{len(holdings)}] {status:4} {row['name']} → {ticker or '???'} ({source})")
        time.sleep(0.15)

    return holdings


# ---------------------------------------------------------
# CSV
# ---------------------------------------------------------

def write_csv(rows: list[dict[str, Any]]) -> None:
    OUTPUT_HOLDINGS.parent.mkdir(parents=True, exist_ok=True)

    # CSV limpio
    with OUTPUT_HOLDINGS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["etf", "ticker", "name", "weight"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "etf": row["etf"],
                "ticker": row["ticker"] or "",
                "name": row["name"] or "",
                "weight": f"{float(row['weight']):.10f}" if row["weight"] is not None else "",
            })

    # CSV de validación
    with OUTPUT_VALIDATION.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "etf", "isin", "name", "bbg", "ticker",
            "yahoo_valid", "yahoo_source"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "etf": row["etf"],
                "isin": row.get("isin", ""),
                "name": row.get("name", ""),
                "bbg": row.get("bbg", ""),
                "ticker": row.get("ticker") or "",
                "yahoo_valid": bool(row.get("ticker")),
                "yahoo_source": row.get("ticker_source", ""),
            })

    print(f"\n[CSV] Holdings: {OUTPUT_HOLDINGS}")
    print(f"[CSV] Validación: {OUTPUT_VALIDATION}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae holdings de un ETF Amundi y valida tickers Yahoo.")
    parser.add_argument("isin", help="ISIN Amundi, por ejemplo FR0010251744")
    args = parser.parse_args()

    rows = extract_etf(args.isin)
    write_csv(rows)
