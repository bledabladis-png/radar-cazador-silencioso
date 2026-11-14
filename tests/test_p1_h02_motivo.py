# -*- coding: utf-8 -*-
"""FIX-P1H02-MOTIVO (2026-11-14): sin_datos clasificado por motivo."""
import ast
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent
          / "scripts" / "audit" / "p1_h02_cobertura_efectiva.py")


def _tree():
    return ast.parse(SCRIPT.read_text(encoding="utf-8-sig"))


def test_get_stock_list_deferred_no_top_level():
    for node in _tree().body:
        if isinstance(node, ast.ImportFrom):
            assert "stock_data_loader" not in (node.module or ""), (
                "stock_data_loader debe importarse deferred dentro de main()")


def test_sin_datos_tiene_columna_motivo():
    found = False
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and k.value == "motivo":
                    found = True
                    break
        if found: break
    assert found, "sin_datos.append no incluye 'motivo'"


def test_motivos_cubren_tres_casos():
    src = SCRIPT.read_text(encoding="utf-8-sig")
    for m in ("fuera_universo_descarga", "historia_insuficiente", "fallo_descarga"):
        assert f'"{m}"' in src, f"motivo {m!r} no encontrado"
