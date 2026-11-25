# -*- coding: utf-8 -*-
"""FIX-TEMPORAL-META-STALE (2026-11-17): refresco post-descarga en run.py.

Contrato:
- run.py importa resolve_all_contracts + build_temporal_meta.
- persist_temporal_contracts_result se llama >=2 veces (early + refresh).
- El refresh ocurre despues de compute_flows_primary (descarga SSGA etc).
- build_temporal_meta se invoca para reconstruir temporal_meta.
"""
import ast
from pathlib import Path

RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def _tree():
    return ast.parse(RUN_PY.read_text(encoding="utf-8-sig"))


def _find_calls(tree, func_name):
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    )


def test_importa_resolve_y_build():
    names = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "src.temporal_contracts":
            names.update(a.name for a in node.names)
    assert "persist_temporal_contracts_result" in names
    assert "resolve_all_contracts" in names
    assert "build_temporal_meta" in names


def test_persist_llamado_al_menos_dos_veces():
    calls = _find_calls(_tree(), "persist_temporal_contracts_result")
    assert len(calls) >= 2, f"persist calls={calls}, esperado >=2 (early + refresh)"


def test_refresh_tras_compute_flows_primary():
    tree = _tree()
    flows = _find_calls(tree, "compute_flows_primary")
    persist = _find_calls(tree, "persist_temporal_contracts_result")
    assert flows, "no se encontro compute_flows_primary"
    assert persist, "no se encontro persist_temporal_contracts_result"
    assert any(p > flows[0] for p in persist), (
        f"persist={persist} debe tener alguna llamada > compute_flows_primary={flows[0]}"
    )


def test_build_temporal_meta_en_run():
    calls = _find_calls(_tree(), "build_temporal_meta")
    assert len(calls) >= 1, "falta llamada a build_temporal_meta para refrescar"


def test_resolve_all_contracts_en_run():
    calls = _find_calls(_tree(), "resolve_all_contracts")
    assert len(calls) >= 1, "falta llamada a resolve_all_contracts"


def test_generate_european_coverage_antes_de_data_quality():
    src = RUN_PY.read_text(encoding="utf-8-sig")
    i_eu = src.index("generate_european_coverage(reference_date=reference_date)")
    i_dq = src.index("data_quality_df = _compute_data_quality(reference_date=reference_date)")
    assert i_eu < i_dq, "generate_european_coverage debe ejecutarse antes de _compute_data_quality"
