# -*- coding: utf-8 -*-
"""Tests de data/universe_expansions.yaml y su integracion en get_stock_list.

2026-11-07: las secciones de sector fueron eliminadas al activar el
universo completo (commit 1c2c4004). Este fichero prueba los bloques
que quedan (expansiones de INDICES USA: SPY, QQQ) y verifica que no
se reintroducen secciones de sector.
"""
from pathlib import Path

import pandas as pd
import pytest
import yaml

EXP_PATH = Path("data/universe_expansions.yaml")
ETF_HOLDINGS = Path("data/etf_holdings.csv")
INDEX_HOLDINGS = Path("data/index_holdings.csv")

SECTORES = {"XLK","XLF","XLI","XLV","XLY","XLP","XLB","XLU","XLRE","XLC","XLE"}


def _holdings_for_etf(etf):
    """Devuelve DataFrame de holdings del ETF (sector o indice)."""
    if ETF_HOLDINGS.exists():
        h = pd.read_csv(ETF_HOLDINGS)
        if (h["etf"] == etf).any():
            return h
    if INDEX_HOLDINGS.exists():
        h = pd.read_csv(INDEX_HOLDINGS)
        if (h["etf"] == etf).any():
            return h
    return None


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


def test_yaml_no_contiene_secciones_de_sector():
    """Centinela 2026-11-07: el yaml solo contiene expansion de INDICES.

    Las secciones de sector (XLK, XLF, etc.) son redundantes desde que
    get_stock_list descarga holdings completos (commit 1c2c4004).
    Si alguien reintroduce una, este test falla y obliga a justificar.
    """
    exp = _cargar_yaml()
    sect_blocks = [
        (k, v.get("etf"))
        for k, v in exp.items()
        if isinstance(v, dict) and v.get("etf") in SECTORES
    ]
    assert not sect_blocks, (
        f"yaml contiene secciones de SECTOR: {sect_blocks}. "
        f"El universo completo de sectores ya no las necesita. "
        f"Si es intencional, anadir justificacion y actualizar este test."
    )


def _todos_bloques():
    exp = _cargar_yaml()
    return [(k, v) for k, v in exp.items() if isinstance(v, dict) and "tickers" in v]


@pytest.mark.parametrize("key,block", _todos_bloques())
def test_generico_no_esta_en_top20_del_etf(key, block):
    """Los tickers ELEGIBLES del bloque NO estan en top-20 del ETF."""
    h = _holdings_for_etf(block["etf"])
    if h is None:
        pytest.skip("holdings no disponibles para " + block["etf"])
    from src.ticker_health import is_eligible
    etf = block["etf"]
    sub = h[h["etf"] == etf].sort_values("weight", ascending=False)
    top20 = set(sub.head(20)["ticker"].tolist())
    elegibles = {t for t in block["tickers"] if is_eligible(t)}
    redundantes = top20 & elegibles
    assert not redundantes, f"{key}: redundantes con top-20: {sorted(redundantes)}"


@pytest.mark.parametrize("key,block", _todos_bloques())
def test_generico_cobertura_top20_coincide(key, block):
    """cobertura_top20_actual declarada == calculada desde holdings."""
    h = _holdings_for_etf(block["etf"])
    if h is None:
        pytest.skip("holdings no disponibles para " + block["etf"])
    etf = block["etf"]
    sub = h[h["etf"] == etf].sort_values("weight", ascending=False).reset_index(drop=True)
    tot = sub["weight"].sum()
    top20 = sub.head(20)["weight"].sum() / tot * 100
    assert abs(top20 - block["cobertura_top20_actual"]) < 0.5, (
        f"{key}: top-20 calculado {top20:.2f} vs yaml {block['cobertura_top20_actual']}"
    )


@pytest.mark.parametrize("key,block", _todos_bloques())
def test_generico_cobertura_top20_mas_extra_coincide(key, block):
    """cobertura_top20_mas_extra == peso(top-20 UNION tickers del bloque)."""
    h = _holdings_for_etf(block["etf"])
    if h is None:
        pytest.skip("holdings no disponibles para " + block["etf"])
    etf = block["etf"]
    sub = h[h["etf"] == etf].sort_values("weight", ascending=False).reset_index(drop=True)
    tot = sub["weight"].sum()
    top20_tickers = set(sub.head(20)["ticker"].tolist())
    todos = top20_tickers | set(block["tickers"])
    cob = sub[sub["ticker"].isin(todos)]["weight"].sum() / tot * 100
    assert abs(cob - block["cobertura_top20_mas_extra"]) < 0.5, (
        f"{key}: top-20+extra calculado {cob:.2f} vs yaml {block['cobertura_top20_mas_extra']}"
    )


@pytest.mark.parametrize("key,block", _todos_bloques())
def test_generico_get_stock_list_incluye_tickers(key, block):
    """get_stock_list devuelve todos los tickers ELEGIBLES del bloque."""
    from src.stock_data_loader import get_stock_list
    from src.ticker_health import is_eligible
    tickers = set(get_stock_list())
    faltantes = [t for t in block["tickers"]
                 if is_eligible(t) and t not in tickers]
    assert not faltantes, f"{key}: get_stock_list NO incluye: {faltantes}"
