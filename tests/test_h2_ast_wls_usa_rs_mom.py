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

STOCK_LEADER = ROOT / "indicators" / "stock_leader.py"


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
    # 'rs': rs.iloc[-1] debe aparecer en el dict de results
    assert "'rs': rs.iloc[-1]" in src, (
        "la columna rs nominal debe seguir publicandose como informativa"
    )