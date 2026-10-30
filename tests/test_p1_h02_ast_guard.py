# -*- coding: utf-8 -*-
"""Contrato AST de p1_h02 tras F3 (2026-10-30).

Verifica estaticamente que p1_h02 aplica `is_eligible` antes de
`_has_data`, para no reintroducir WBD (DELISTED 2026-10-06) en
`sin_datos` cuando en realidad no es "sin datos hoy", es "no existe".
"""
import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit" / "p1_h02_cobertura_efectiva.py"


def _tree():
    return ast.parse(SCRIPT.read_text(encoding="utf-8-sig"))


def test_importa_is_eligible():
    names = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "src.ticker_health":
            names.extend(a.name for a in node.names)
    assert "is_eligible" in names, (
        "p1_h02 no importa is_eligible de src.ticker_health. "
        "WBD (DELISTED) volveria a aparecer en sin_datos."
    )


def test_sin_datos_excluye_delisted_antes_de_has_data():
    tree = _tree()
    # Buscar el for que contiene _has_data y verificar que is_eligible
    # aparece antes en el mismo bloque (o en el mismo bucle)
    found_pair = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        body_calls = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name) and sub.func.id in ("_has_data", "is_eligible"):
                    body_calls.append((sub.lineno, sub.func.id))
        body_calls.sort()
        names = [n for _, n in body_calls]
        if "_has_data" in names and "is_eligible" in names:
            idx_elig = names.index("is_eligible")
            idx_has = names.index("_has_data")
            assert idx_elig < idx_has, (
                f"En el bucle (L{node.lineno}), _has_data se llama antes "
                f"que is_eligible. Orden requerido: elegibilidad primero, "
                f"luego datos."
            )
            found_pair = True
            break
    assert found_pair, (
        "No se encontro bucle con _has_data e is_eligible juntos. "
        "F3 roto o refactor sin actualizar el test."
    )


def test_n_excluidos_en_salida():
    tree = _tree()
    # El dict de salida debe tener clave "n_excluidos"
    keys_encontradas = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "n_excluidos":
                    keys_encontradas.append(node.lineno)
    assert keys_encontradas, (
        "El dict de fila no contiene la clave 'n_excluidos'. "
        "F3 incompleto."
    )