# -*- coding: utf-8 -*-
"""
Descarga datos crudos de Invesco para los ETFs configurados.

Uso:
    py scripts\\fetch_invesco.py

Configuración:
    config/invesco_etfs.csv

Salida:
    data/invesco/{TICKER}/{YYYY-MM-DD_HHMMSS}.json
"""

from __future__ import annotations

import sys
import csv
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.providers.invesco_client import InvescoClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_ROOT / "config" / "invesco_etfs.csv"
DATA_DIR = PROJECT_ROOT / "data" / "invesco"

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / "invesco_api.log",
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("fetch_invesco")


def load_etfs() -> list[dict[str, str]]:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"No existe: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_json(ticker: str, dataset: dict) -> Path:
    now = datetime.now()
    output_dir = DATA_DIR / ticker.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{now:%Y-%m-%d_%H%M%S}.json"
    output_file = output_dir / filename

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    return output_file


def main() -> None:
    etfs = load_etfs()
    if not etfs:
        raise RuntimeError("invesco_etfs.csv está vacío")

    for etf in etfs:
        ticker = etf["ticker"].strip()
        cusip = etf["cusip"].strip()

        if not ticker or not cusip:
            logger.warning("ETF ignorado: %s", etf)
            continue

        logger.info("================================================")
        logger.info("ETF: %s | CUSIP: %s", ticker, cusip)
        logger.info("================================================")

        try:
            api = InvescoClient(cusip=cusip)
            dataset = api.fetch_all()
            output = save_json(ticker=ticker, dataset=dataset)
            logger.info("Dataset guardado: %s", output)

        except Exception as exc:
            logger.exception("Fallo completo para %s: %s", ticker, exc)


if __name__ == "__main__":
    main()
