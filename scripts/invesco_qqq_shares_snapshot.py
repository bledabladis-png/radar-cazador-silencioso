# -*- coding: utf-8 -*-
"""
Snapshot diario de shares outstanding de QQQ.

Fuente:
    Invesco DNG API — /prices

Salida:
    outputs/history/qqq_shares_snapshot_history.csv

No calcula flujo primario diario.
Solo acumula observaciones oficiales.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.providers.invesco_client import InvescoClient

TICKER = "QQQ"
CUSIP = "46090E103"

OUTPUT = PROJECT_ROOT / "outputs" / "history" / "qqq_shares_snapshot_history.csv"

COLUMNS = [
    "ticker",
    "cusip",
    "effectiveDate",
    "sharesOutstanding",
    "nav",
    "marketValue",
    "creationUnits",
    "creationUnitsExact",
    "source",
]


def fetch_prices() -> dict:
    client = InvescoClient(cusip=CUSIP)
    return client.prices()


def load_existing() -> list[dict]:
    if not OUTPUT.exists():
        return []
    with OUTPUT.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    prices = fetch_prices()

    effective_date = str(prices.get("effectiveDate", ""))[:10]
    if not effective_date:
        raise RuntimeError("effectiveDate vacío en /prices")

    shares = float(prices.get("sharesOutstanding", 0))
    nav = float(prices.get("nav", 0))
    market_value = float(prices.get("marketValue", 0))

    if shares <= 0 or nav <= 0:
        raise RuntimeError("Datos inválidos de shares/nav")

    rows = load_existing()
    rows = [row for row in rows if row["effectiveDate"] != effective_date]

    record = {
        "ticker": TICKER,
        "cusip": CUSIP,
        "effectiveDate": effective_date,
        "sharesOutstanding": shares,
        "nav": nav,
        "marketValue": market_value,
        "creationUnits": shares / 50_000,
        "creationUnitsExact": (shares % 50_000) == 0,
        "source": "Invesco DNG API",
    }

    rows.append(record)
    rows.sort(key=lambda r: r["effectiveDate"])

    save_rows(rows)

    print(f"Snapshot QQQ guardado: {effective_date} | shares={shares:,.0f} | nav={nav:,.4f}")
    print(f"Total observaciones: {len(rows)}")


if __name__ == "__main__":
    main()
