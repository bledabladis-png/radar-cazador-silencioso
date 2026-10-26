# -*- coding: utf-8 -*-
"""Tests de data/universe_expansions.yaml y su integracion en get_stock_list.

DICT3-TOP20-COBERTURA: grupos con cola larga anaden tickers por encima
del top-20 sin cambiar TOP_N_SECTOR_COMPONENTS. El yaml es la fuente
de verdad, versionada, con cobertura documentada.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

EXP_PATH = Path("data/universe_expansions.yaml")
ETF_HOLDINGS = Path("data/etf_holdings.csv")


def _cargar_yaml():
    assert EXP_PATH.exists(), f"no existe {EXP_PATH}"
    return yaml.safe_load(EXP_PATH.read_text(encoding="utf-8")) or {}


def test_yaml_existe_y_estructura_valida():
    """Cada bloque tiene etf, tickers, coberturas y fecha."""
    exp = _cargar_yaml()
    assert exp, "yaml vacio"
    for key, block in exp.items():
        assert isinstance(block, dict), f"{key}: no es dict"
        for campo in ("etf", "tickers", "fecha",
                      "cobertura_top20_actual",
                      "cobertura_top20_mas_extra",
                      "cobertura_tras_expansion",
                      "cobertura_objetivo"):
            assert campo in block, f"{key}: falta {campo}"
        assert isinstance(block["tickers"], list)
        assert len(block["tickers"]) == block["n_tickers"], (
            f"{key}: n_tickers={block['n_tickers']} != len(tickers)={len(block['tickers'])}")


def test_tickers_sin_duplicados_entre_bloques():
    """Ningun ticker aparece en 2 bloques distintos."""
    exp = _cargar_yaml()
    vistos = {}
    for key, block in exp.items():
        for t in block["tickers"]:
            assert t not in vistos, f"{t} en {key} y {vistos[t]}"
            vistos[t] = key


def test_xli_extra_no_esta_en_top20_xli():
    """Los tickers de xli_extra NO estan en top-20 de XLI (redundancia)."""
    if not ETF_HOLDINGS.exists():
        pytest.skip("etf_holdings.csv no disponible")
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    h = pd.read_csv(ETF_HOLDINGS)
    xli_h = h[h["etf"] == "XLI"].sort_values("weight", ascending=False)
    top20 = set(xli_h.head(20)["ticker"].tolist())
    redundantes = top20 & set(xli["tickers"])
    assert not redundantes, f"tickers redundantes con top-20: {sorted(redundantes)}"


def test_xli_cobertura_objetivo_en_80_90():
    """cobertura_tras_expansion esta en [80, 90] (objetivo del dictamen)."""
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    cob = xli["cobertura_tras_expansion"]
    assert 80.0 <= cob <= 90.0, f"cobertura {cob} fuera de [80, 90]"


def test_xli_cobertura_top20_coincide_con_etf_holdings():
    """cobertura_top20_actual declarada == calculada desde etf_holdings."""
    if not ETF_HOLDINGS.exists():
        pytest.skip("etf_holdings.csv no disponible")
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    h = pd.read_csv(ETF_HOLDINGS)
    xli_h = h[h["etf"] == "XLI"].sort_values("weight", ascending=False).reset_index(drop=True)
    cob_calc = xli_h.head(20)["weight"].sum()
    assert abs(cob_calc - xli["cobertura_top20_actual"]) < 0.5, (
        f"top-20 calculado {cob_calc:.2f} vs yaml {xli['cobertura_top20_actual']}"
    )


def test_xli_cobertura_top20_mas_extra_coincide_con_etf_holdings():
    """cobertura_top20_mas_extra == peso(top-20 UNION xli_extra)."""
    if not ETF_HOLDINGS.exists():
        pytest.skip("etf_holdings.csv no disponible")
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    h = pd.read_csv(ETF_HOLDINGS)
    xli_h = h[h["etf"] == "XLI"].sort_values("weight", ascending=False).reset_index(drop=True)
    top20_tickers = set(xli_h.head(20)["ticker"].tolist())
    todos = top20_tickers | set(xli["tickers"])
    cob_calc = xli_h[xli_h["ticker"].isin(todos)]["weight"].sum()
    assert abs(cob_calc - xli["cobertura_top20_mas_extra"]) < 0.5, (
        f"top-20+extra calculado {cob_calc:.2f} vs yaml {xli['cobertura_top20_mas_extra']}"
    )


def test_xli_cobertura_tras_expansion_coincide_top50():
    """cobertura_tras_expansion == peso de los 50 XLI mas pesados (top-50)."""
    if not ETF_HOLDINGS.exists():
        pytest.skip("etf_holdings.csv no disponible")
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    h = pd.read_csv(ETF_HOLDINGS)
    xli_h = h[h["etf"] == "XLI"].sort_values("weight", ascending=False).reset_index(drop=True)
    cob_calc = xli_h.head(50)["weight"].sum()
    assert abs(cob_calc - xli["cobertura_tras_expansion"]) < 0.5, (
        f"top-50 calculado {cob_calc:.2f} vs yaml {xli['cobertura_tras_expansion']}"
    )


def test_get_stock_list_incluye_xli_extra():
    """get_stock_list devuelve los tickers de xli_extra."""
    exp = _cargar_yaml()
    xli = exp.get("xli_extra")
    if xli is None:
        pytest.skip("xli_extra no en yaml")
    from src.stock_data_loader import get_stock_list
    tickers = set(get_stock_list())
    faltantes = [t for t in xli["tickers"] if t not in tickers]
    assert not faltantes, f"get_stock_list NO incluye: {faltantes}"