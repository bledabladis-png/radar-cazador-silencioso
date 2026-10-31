# -*- coding: utf-8 -*-
# Centinela C5: ratio_names en confirmation.py debe cubrir exactamente
# los ratios que produce compute_cross_asset_ratios. Si el productor
# anade uno nuevo o retira uno, este test falla antes que el render
# lo ignore silenciosamente.

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _producer_ratio_keys():
    src = (ROOT / "indicators" / "cross_asset.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "compute_cross_asset_ratios":
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "ratios":
                            return {k.value for k in node.value.keys}
    raise AssertionError("compute_cross_asset_ratios sin dict ratios")


def _render_ratio_keys():
    src = (ROOT / "src" / "report" / "confirmation.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "render_confirmation":
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == "ratio_names":
                            return {k.value for k in node.value.keys}
    raise AssertionError("render_confirmation sin dict ratio_names")


def test_ratio_names_paridad_con_productor():
    productor = _producer_ratio_keys()
    render = _render_ratio_keys()
    assert render == productor, (
        "Drift entre productor y render. "
        "Solo en productor: " + str(sorted(productor - render)) + ". "
        "Solo en render: " + str(sorted(render - productor)) + "."
    )
