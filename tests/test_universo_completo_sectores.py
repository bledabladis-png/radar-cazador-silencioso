"""Centinela: get_stock_list() alineado con holdings completos de sectores USA.

Contexto 2026-11-07: hasta esta fecha get_stock_list truncaba a top-20
por sector, dejando universos distintos por capa (n_total=504 en
sector_concentration vs n_valid=220 efectivos). Esta prueba verifica
que la brecha esta cerrada sin romper el cap de indices USA.
"""
import ast
from pathlib import Path

import pandas as pd
import pytest

from src.stock_data_loader import get_stock_list, normalize_yahoo_ticker
from src.ticker_health import is_eligible


SECTORES = ['XLK','XLF','XLI','XLV','XLY','XLP','XLB','XLU','XLRE','XLC','XLE']


def _holdings():
    csv = Path("data/etf_holdings.csv")
    if not csv.exists():
        pytest.skip("etf_holdings.csv no disponible")
    return pd.read_csv(csv)


def test_get_stock_list_incluye_todos_los_holdings_elegibles_usa():
    """Todos los holdings ELEGIBLES de sectores USA deben estar en el universo.

    WBD (DELISTED 2026-10-06) queda fuera por diseno: filter_eligible() lo excluye.
    """
    holdings = _holdings()
    universo = set(get_stock_list())
    faltantes = []
    for etf in SECTORES:
        sub = holdings[holdings['etf'] == etf]
        if sub.empty:
            continue
        for t in sub['ticker'].tolist():
            norm_t = normalize_yahoo_ticker(t)
            if not is_eligible(norm_t):
                continue
            if norm_t not in universo:
                faltantes.append((etf, t))
    assert not faltantes, (
        f"get_stock_list NO incluye {len(faltantes)} tickers ELEGIBLES "
        f"de sectores (primeros 20): {faltantes[:20]}"
    )


def test_get_stock_list_xlk_74_holdings():
    holdings = _holdings()
    xlk = holdings[holdings['etf'] == 'XLK']['ticker'].tolist()
    assert len(xlk) == 74, f"XLK tiene {len(xlk)} holdings en CSV, esperado 74"
    universo = set(get_stock_list())
    faltan = [t for t in xlk if normalize_yahoo_ticker(t) not in universo]
    assert not faltan, (
        f"XLK: {len(faltan)} de 74 holdings no estan en get_stock_list: {faltan}"
    )


def test_codigo_mantiene_cap_solo_en_indices_usa():
    """AST: en get_stock_list, solo la rama de indices debe tener head(TOP_N).

    Antes habia 2 head(TOP_N) (sectores + indices). Tras el cambio, solo
    debe quedar 1 (indices). Si alguien lo quita del indice, o lo vuelve
    a meter en sectores, este test falla.
    """
    src = Path("src/stock_data_loader.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_stock_list":
            fn = node
            break
    assert fn is not None, "get_stock_list() no encontrada en stock_data_loader.py"

    count = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "head":
                if (node.args
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "TOP_N_SECTOR_COMPONENTS"):
                    count += 1
    assert count == 1, (
        f"get_stock_list tiene {count} head(TOP_N_SECTOR_COMPONENTS), "
        f"esperado exactamente 1 (solo la rama de indices USA)"
    )