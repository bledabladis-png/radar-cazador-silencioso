"""Centinela: _load_declared_tickers multi-fuente (2026-11-07).

Tras commit 1c2c4004, get_stock_list() descarga holdings completos de
sector. El yaml dejo de ser la unica fuente de tickers declarados.
Este fichero verifica que las fuentes canonicas (etf_holdings,
index_holdings, xetra_map) y el yaml estan todas en declared.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import regenerate_radar_catalog as rrc


def test_declared_incluye_etf_holdings():
    csv = ROOT / "data" / "etf_holdings.csv"
    if not csv.exists():
        pytest.skip("etf_holdings.csv no disponible")
    df = pd.read_csv(csv)
    declared = rrc._load_declared_tickers()
    faltantes = [t for t in df["ticker"].dropna().astype(str) if t not in declared]
    assert not faltantes, (
        f"_load_declared_tickers NO incluye {len(faltantes)} tickers de "
        f"etf_holdings.csv: {faltantes[:10]}")


def test_declared_incluye_index_holdings():
    csv = ROOT / "data" / "index_holdings.csv"
    if not csv.exists():
        pytest.skip("index_holdings.csv no disponible")
    df = pd.read_csv(csv)
    declared = rrc._load_declared_tickers()
    faltantes = [t for t in df["ticker"].dropna().astype(str) if t not in declared]
    assert not faltantes, (
        f"_load_declared_tickers NO incluye {len(faltantes)} tickers de "
        f"index_holdings.csv: {faltantes[:10]}")


def test_declared_incluye_xetra():
    csv = ROOT / "config" / "xetra_ticker_map.csv"
    if not csv.exists():
        pytest.skip("xetra_ticker_map.csv no disponible")
    df = pd.read_csv(csv)
    declared = rrc._load_declared_tickers()
    faltantes = [
        t for t in df["yahoo_ticker"].dropna().astype(str)
        if t not in declared
    ]
    assert not faltantes, (
        f"_load_declared_tickers NO incluye {len(faltantes)} tickers Xetra: "
        f"{faltantes[:10]}")


def test_declared_no_incluye_ticker_inventado():
    declared = rrc._load_declared_tickers()
    assert "ZZZFAKE" not in declared
    assert "TICKER_SPURIOUS" not in declared


def test_guard_no_dispara_con_universo_completo(monkeypatch):
    """Simula parquet con 529 tickers radar y verifica 0 no declarados."""
    csv = ROOT / "data" / "etf_holdings.csv"
    if not csv.exists():
        pytest.skip("etf_holdings.csv no disponible")
    df = pd.read_csv(csv)
    radar = sorted(set(df["ticker"].dropna().astype(str)))
    declared = rrc._load_declared_tickers()

    cat = pd.read_csv(ROOT / "data" / "mappings" / "radar_target_catalog.csv", dtype=str)
    cat_tks = set(cat["radar_ticker"].dropna().astype(str))

    nuevos = [t for t in radar if t not in cat_tks]
    no_declarados = [t for t in nuevos if t not in declared]
    assert not no_declarados, (
        f"con universo expandido, {len(no_declarados)} no declarados: "
        f"{no_declarados[:10]}")