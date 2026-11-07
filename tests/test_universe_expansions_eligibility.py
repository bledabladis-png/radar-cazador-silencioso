# -*- coding: utf-8 -*-
"""FASE-A-UNIVERSO-90 (2026-11-01): invariantes del universo de descarga.

Estos tests fijan 4 invariantes:

  1. get_stock_list() aplica filter_eligible() al resultado final. Ningun
     ticker DELISTED llega al universo, independientemente de la fuente.
  2. La foto actual del universo no contiene DELISTED.
  3. WBD (DELISTED 2026-10-06, adquirida por Paramount Skydance) esta
     documentado en ticker_health.yaml y el filtro final lo descarta.
     Si WBD se re-listea en el futuro, aparece sin tocar el YAML.
  4. Cobertura de peso: cada ETF sectorial USA cubre >=90% de su peso
     con los tickers en get_stock_list(). Si SPDR rebalancea y baja el
     peso cubierto por debajo del 90%, el test falla y avisa. Es una
     foto, no un contrato temporal: se actualiza cuando cambia el CSV
     o cuando se decide subir el umbral.
"""
import ast
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stock_data_loader import get_stock_list
from src.instrument_registry import normalize_yahoo_ticker


MIN_COVERAGE_PCT = 90.0


def test_get_stock_list_aplica_filter_eligible_al_final():
    """AST: get_stock_list() devuelve filter_eligible(result), no result."""
    src = (ROOT / "src" / "stock_data_loader.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "get_stock_list":
            has_filter = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Return):
                    v = node.value
                    if isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
                        if v.func.id == "filter_eligible":
                            has_filter = True
                            break
            assert has_filter, (
                "get_stock_list() no aplica filter_eligible() al final. "
                "Tickers DELISTED pueden llegar al universo."
            )
            return
    raise AssertionError("get_stock_list() no encontrada en stock_data_loader.py")


def test_ningun_delisted_llega_a_universo():
    """La foto actual del universo no contiene tickers DELISTED."""
    from src.ticker_health import is_eligible
    universe = get_stock_list()
    delisted = [t for t in universe if not is_eligible(t)]
    assert not delisted, (
        "Tickers DELISTED en get_stock_list(): " + str(delisted) + ". "
        "El filtro final no se esta aplicando."
    )


def test_wbd_delisted_documentado_y_fuera_de_universo():
    """WBD esta documentado en ticker_health.yaml y no llega a get_stock_list().

    2026-11-07: la documentacion de WBD vivia tambien en
    universe_expansions.yaml::xlc_extra_90, seccion eliminada al
    activar el universo completo de sectores (commit 1c2c4004). La
    fuente canonica es ticker_health.yaml.
    """
    import yaml
    yp = ROOT / "data" / "mappings" / "ticker_health.yaml"
    if not yp.exists():
        pytest.skip("ticker_health.yaml no disponible")
    th = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
    tickers = th.get("tickers", {})
    assert "WBD" in tickers, (
        "WBD no documentado en ticker_health.yaml. "
        "Si es intencional, retirar este test."
    )
    assert tickers["WBD"].get("estado") == "DELISTED", (
        "WBD no tiene estado DELISTED en ticker_health.yaml"
    )
    universe = get_stock_list()
    assert "WBD" not in universe, (
        "WBD (DELISTED 2026-10-06) llego a get_stock_list(). "
        "El filtro final no se aplica."
    )


def test_cobertura_peso_90pct_por_etf():
    """Foto actual: cada ETF sectorial USA cubre >=90% de su peso."""
    holdings_path = ROOT / "data" / "etf_holdings.csv"
    if not holdings_path.exists():
        pytest.skip("etf_holdings.csv no disponible")
    universe = set(get_stock_list())
    holdings = pd.read_csv(holdings_path)
    holdings["weight"] = pd.to_numeric(holdings["weight"], errors="coerce")
    holdings["ticker_norm"] = holdings["ticker"].apply(normalize_yahoo_ticker)
    holdings = holdings.dropna(subset=["weight"])

    expected = {"XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
                "XLP", "XLRE", "XLU", "XLV", "XLY"}
    fallos = {}
    detalles = {}
    for etf in expected:
        sub = holdings[holdings["etf"] == etf]
        total = sub["weight"].sum()
        if total <= 0:
            pytest.skip(f"ETF {etf} sin pesos en CSV")
        in_univ = sub[sub["ticker_norm"].isin(universe)]["weight"].sum()
        cov = in_univ / total * 100
        detalles[etf] = round(cov, 2)
        if cov < MIN_COVERAGE_PCT:
            fallos[etf] = round(cov, 2)
    assert not fallos, (
        f"ETFs con cobertura de peso < {MIN_COVERAGE_PCT}%: {fallos}. "
        f"Detalle completo: {detalles}. Si SPDR rebalanceo y bajo el peso, "
        f"actualizar etf_holdings.csv y universe_expansions.yaml."
    )