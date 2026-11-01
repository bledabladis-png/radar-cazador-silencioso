# -*- coding: utf-8 -*-
# Centinela C5 (post B6-RATIO-NAMES, 2026-11-01): RATIO_LABELS en el
# productor cubre exactamente los ratios de compute_cross_asset_ratios.
# El render importa del productor, sin dict local.
import ast, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

def _parse(relpath):
    return ast.parse((ROOT / relpath).read_text(encoding="utf-8"))

def _producer_ratios(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "compute_cross_asset_ratios":
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign):
                    for tg in n.targets:
                        if isinstance(tg, ast.Name) and tg.id == "ratios":
                            return {k.value for k in n.value.keys}
    raise AssertionError("sin dict ratios en productor")

def _producer_labels(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                if isinstance(tg, ast.Name) and tg.id == "RATIO_LABELS":
                    return {k.value for k in n.value.keys}
    raise AssertionError("sin RATIO_LABELS en productor")

def _render_imports(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "indicators.cross_asset":
            if "RATIO_LABELS" in [a.name for a in n.names]:
                return True
    return False
def _render_local_ratio_names(tree):
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "render_confirmation":
            for n in ast.walk(fn):
                if isinstance(n, ast.Assign):
                    for tg in n.targets:
                        if isinstance(tg, ast.Name) and tg.id == "ratio_names":
                            return True
    return False

def test_labels_paridad_con_ratios():
    tree = _parse("indicators/cross_asset.py")
    assert _producer_labels(tree) == _producer_ratios(tree), "Drift labels vs ratios en productor"

def test_render_importa_ratio_labels():
    tree = _parse("src/report/confirmation.py")
    assert _render_imports(tree), "confirmation.py no importa RATIO_LABELS"

def test_render_no_define_ratio_names_local():
    tree = _parse("src/report/confirmation.py")
    assert not _render_local_ratio_names(tree), "render define ratio_names local"
