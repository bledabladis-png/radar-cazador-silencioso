# -*- coding: utf-8 -*-
"""Test AST H2: rs_z en compute_wls se calcula sobre rs_mom, no sobre rs.

Cierra el falso positivo H2 (dictamen 2026-10-15 + cierre ampliado
2026-10-16): el WLS usa rs_mom (log-return relativo 20d, invariante a
escala). El rs nominal solo se publica como columna informativa, no
alimenta el score. Este test prohibe sustituir rs_mom por rs en la
columna rs_z.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STOCK_LEADER = ROOT / "indicators" / "stock_metrics.py"


def _source_of_fn(name):
    """Extrae el source de la funcion `name` via AST."""
    tree = ast.parse(STOCK_LEADER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(
                STOCK_LEADER.read_text(encoding="utf-8"), node
            )
    raise AssertionError("funcion {0} no encontrada".format(name))


def test_rs_z_usa_rs_mom_no_rs_nominal():
    """En compute_wls, la columna rs_z se calcula sobre rs_mom."""
    src = _source_of_fn("compute_wls")
    # Debe contener el patron: groupby('sector')['rs_mom'].transform(...)
    assert "['rs_mom'].transform" in src, (
        "compute_wls no calcula rs_z sobre rs_mom. "
        "Cambio silencioso detectado."
    )
    # NO debe contener: groupby('sector')['rs'].transform(...)
    patron_nominal = re.compile(
        r"groupby\(['\"]sector['\"]\)\[['\"]rs['\"]\]\.transform"
    )
    assert not patron_nominal.search(src), (
        "compute_wls calcula rs_z sobre rs nominal. "
        "Reintroduce el sesgo de escala que H2 cerro."
    )


def test_rs_mom_es_log_return():
    """compute_stock_metrics calcula rs_mom = log(rs).diff(20)."""
    src = _source_of_fn("compute_stock_metrics")
    # Debe contener log(rs).diff(20)
    assert "np.log(rs).diff(20)" in src, (
        "rs_mom no se calcula como log(rs).diff(20). "
        "Si cambio la formula, revisar invarianza a escala."
    )


def test_columna_rs_informativa_existe():
    """La columna rs nominal sigue publicandose (informativa)."""
    src = _source_of_fn("compute_stock_metrics")
    # B6-ILOC-SIN-GUARD-BARRIDO (2026-11-01): la columna se publica
    # desde rs_last, calculado con guard (rs vacio -> NaN). El contrato
    # que se mantiene es 'la columna rs existe y viene de rs'.
    assert "'rs': rs_last" in src, (
        "la columna rs nominal debe seguir publicandose como informativa"
    )
    assert "rs_last = float(rs.iloc[-1]) if len(rs) > 0 else np.nan" in src, (
        "falta el guard de rs vacio introducido por B6-ILOC"
    )
