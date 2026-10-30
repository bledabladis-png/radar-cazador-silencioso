# -*- coding: utf-8 -*-
"""Contrato AST: detect_cross_module_conflict <-> render_cross_module.

Productor: src/utils.py::detect_cross_module_conflict. Dict literal
con 4 claves: conflict_level, message, blocks, details. details es
nested ({mod: {state, bias_financial, bias_inflation}}), cubierto
por tests de integracion del render existentes.

Consumidor: src/report/alerts.py::render_cross_module. Lee las 4
claves top-level. Nested via details.items() -> mod_info.get(...).
"""
from pathlib import Path

from tests._dict_contract import (
    producer_return_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "src" / "utils.py"
RENDER = REPO_ROOT / "src" / "report" / "alerts.py"

EXPECTED_TOP_LEVEL = frozenset({
    "conflict_level", "message", "blocks", "details",
})


def test_render_cross_module_no_lee_campos_ausentes_en_producer():
    """render_cross_module lee X => X en detect_cross_module_conflict."""
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="detect_cross_module_conflict",
        consumer_module=RENDER,
        consumer_fn="render_cross_module",
        consumer_var="cross_module_conflict",
    )


def test_top_level_cross_module_coincide_con_esperado():
    """El dict top-level publica exactamente EXPECTED_TOP_LEVEL."""
    keys = producer_return_keys(
        PRODUCER, "detect_cross_module_conflict")
    assert keys == EXPECTED_TOP_LEVEL, (
        f"El dict top-level de detect_cross_module_conflict cambio. "
        f"Esperado: {sorted(EXPECTED_TOP_LEVEL)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es una decision consciente, actualizar este test y "
        f"revisar render_cross_module."
    )


def test_render_cross_module_lee_campos_criticos():
    """Regresion: campos que el reporte no puede perder.

    - conflict_level: cabecera OK/WARN/INFO + texto.
    - message: explicacion del nivel.
    - blocks: bloques contables (2/5, 0/5).
    - details: tabla por modulo.
    """
    reads = consumer_read_keys(
        RENDER, "render_cross_module", "cross_module_conflict")
    for campo in ("conflict_level", "message", "blocks", "details"):
        assert campo in reads, (
            f"render_cross_module dejo de leer '{campo}'. El reporte "
            f"perderia informacion del bloque Cross-Module. Si es "
            f"intencional, borrar este test con justificacion."
        )
