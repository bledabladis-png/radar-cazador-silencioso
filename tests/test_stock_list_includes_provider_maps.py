# -*- coding: utf-8 -*-
"""Test de regresion 2026-10-06: get_stock_list() debe incluir todos
los tickers europeos que los providers oficiales soportan.

Bug: DB1.DE, HEI.DE, MUV2.DE estan en config/xetra_ticker_map.csv
pero no aparecen en etf_holdings.csv, index_holdings.csv (fuera del
top-20 del DAX por peso) ni en radar_target_catalog.csv. Sin
descarga -> european_coverage.csv marcaba SIN_DATOS perpetuo.
"""
import csv
from pathlib import Path

from src.stock_data_loader import get_stock_list


ROOT = Path(__file__).resolve().parent.parent


def _read_yahoo_tickers(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [row["yahoo_ticker"] for row in reader if row.get("yahoo_ticker")]


def test_get_stock_list_incluye_todos_los_xetra_del_mapa():
    mapping = _read_yahoo_tickers(ROOT / "config" / "xetra_ticker_map.csv")
    listed = set(get_stock_list())
    missing = [t for t in mapping if t not in listed]
    assert not missing, f"Xetra del mapa no incluidos en get_stock_list: {missing}"


def test_get_stock_list_incluye_todos_los_euronext_del_mapa():
    mapping = _read_yahoo_tickers(ROOT / "config" / "euronext_ticker_map.csv")
    listed = set(get_stock_list())
    missing = [t for t in mapping if t not in listed]
    assert not missing, f"Euronext del mapa no incluidos en get_stock_list: {missing}"


def test_get_stock_list_incluye_todos_los_bme_del_mapa():
    mapping = _read_yahoo_tickers(ROOT / "config" / "bme_ticker_map.csv")
    listed = set(get_stock_list())
    missing = [t for t in mapping if t not in listed]
    assert not missing, f"BME del mapa no incluidos en get_stock_list: {missing}"


def test_caso_concreto_db1_hei_muv2():
    """Caso reportado: los 3 Xetra que quedaban SIN_DATOS."""
    listed = set(get_stock_list())
    for t in ("DB1.DE", "HEI.DE", "MUV2.DE"):
        assert t in listed, f"{t} no esta en get_stock_list()"
