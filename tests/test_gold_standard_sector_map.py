# -*- coding: utf-8 -*-
"""Tests mapping ticker -> sector.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.sector_map import (
    coverage_report,
    load_sector_map,
    sector_map_dict,
)


def _make_holdings_csv(tmp_path):
    """Mini holdings con BRK.B para verificar normalizacion."""
    csv = tmp_path / "holdings.csv"
    csv.write_text(
        "etf,ticker,identifier,weight\n"
        "XLK,AAPL,037833100,15.0\n"
        "XLK,MSFT,594918104,10.0\n"
        "XLK,NVDA,67066G104,8.0\n"
        "XLF,JPM,46647P104,12.0\n"
        "XLF,BRK.B,084670108,9.0\n"
        "XLF,BAC,060505104,7.0\n"
        "XLV,UNH,91324P102,14.0\n"
        "XLV,JNJ,478160104,11.0\n"
        "XLV,PFE,717081103,6.0\n",
        encoding="utf-8",
    )
    return csv


def test_load_sector_map_normaliza_brk_b(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    tickers = set(df["ticker"])
    assert "BRK-B" in tickers
    assert "BRK.B" not in tickers


def test_load_sector_map_columnas(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    for c in ("ticker", "sector", "weight", "etf"):
        assert c in df.columns


def test_load_sector_map_un_ticker_por_sector(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    assert df["ticker"].is_unique


def test_load_sector_map_top_n(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df_all = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    df_2 = load_sector_map(holdings_csv=csv, top_n_per_etf=2)
    assert len(df_all) > len(df_2)
    # top-2 por ETF: los 2 de mayor weight
    assert set(df_2["ticker"]) == {"AAPL", "MSFT", "JPM", "BRK-B", "UNH", "JNJ"}


def test_load_sector_map_filtro_dataset(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    dataset = {"AAPL", "MSFT", "JPM"}
    df = load_sector_map(
        holdings_csv=csv, top_n_per_etf=20, dataset_tickers=dataset,
    )
    assert set(df["ticker"]) == dataset


def test_load_sector_map_fichero_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sector_map(holdings_csv=tmp_path / "nope.csv")


def test_load_sector_map_columnas_faltantes(tmp_path):
    csv = tmp_path / "mal.csv"
    csv.write_text("etf,ticker\nXLK,AAPL\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Falta columna"):
        load_sector_map(holdings_csv=csv)

def test_sector_map_dict(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    m = sector_map_dict(df)
    assert m["AAPL"] == "XLK"
    assert m["JPM"] == "XLF"
    assert m["UNH"] == "XLV"


def test_coverage_report_calcula_correctamente(tmp_path):
    csv = _make_holdings_csv(tmp_path)
    df = load_sector_map(holdings_csv=csv, top_n_per_etf=20)
    dataset = {"AAPL", "MSFT", "JPM", "ZZZ", "YYY"}
    rep = coverage_report(df, dataset)
    assert rep["n_dataset"] == 5
    assert rep["n_con_sector"] == 3
    assert rep["n_sin_sector"] == 2
    assert rep["por_sector"]["XLK"] == 2
    assert rep["por_sector"]["XLF"] == 1
    assert set(rep["sin_sector_sample"]) == {"ZZZ", "YYY"}


def test_sector_map_real_dataset():
    """Test contra datos reales si estan presentes."""
    holdings = ROOT / "data" / "etf_holdings.csv"
    dataset = ROOT / "data" / "stock_prices.parquet"
    if not holdings.exists() or not dataset.exists():
        pytest.skip("dataset no presente")
    df = pd.read_parquet(dataset)
    tickers = set(df.columns.get_level_values(1).unique())
    sm = load_sector_map(dataset_tickers=tickers)
    assert len(sm) >= 200
    assert sm["ticker"].is_unique
    assert (sm["sector"].str.startswith("XL")).all()
    assert sm["sector"].nunique() <= 11


def test_sector_map_real_tiene_brk_b():
    holdings = ROOT / "data" / "etf_holdings.csv"
    dataset = ROOT / "data" / "stock_prices.parquet"
    if not holdings.exists() or not dataset.exists():
        pytest.skip("dataset no presente")
    df = pd.read_parquet(dataset)
    tickers = set(df.columns.get_level_values(1).unique())
    sm = load_sector_map(dataset_tickers=tickers)
    assert "BRK-B" in set(sm["ticker"])
    assert "BRK.B" not in set(sm["ticker"])