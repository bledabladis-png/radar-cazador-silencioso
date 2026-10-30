# -*- coding: utf-8 -*-
"""Contrato AST: state<->return<->render de MTE.

Caso 1fb791ab (2026-10-30): render_mte leia mte_result.get('X') pero el
return de compute_mte no exponia X. Solo vivia en el state JSON.
Detectado en run local 8h despues del fix. Cerrado en 4 commits
(1fb791ab, 5637d45e, 5b96192f, db1e0a0f, fc6f63c4).

Los 4 tests que antes vivian aqui se apoyan ahora en el helper
generico `tests/_dict_contract.py` (commit 1 del plan de cierre de la
clase 1fb791ab). El helper sirve para otros pares productor-consumidor
del sistema, no solo MTE. Esta migracion no cambia el comportamiento:
los 4 tests originales siguen pasando con el helper.
"""
from pathlib import Path

from tests._dict_contract import (
    producer_return_keys,
    producer_state_keys,
    consumer_read_keys,
    assert_render_subset_return,
    assert_state_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE = REPO_ROOT / "indicators" / "mte" / "engine.py"
RENDER = REPO_ROOT / "src" / "report" / "volatility_mte.py"

# Campos del state JSON que viven SOLO en el state por diseno. NO
# deben estar en el return: schema_version y temporal_contract_version
# son metadatos; effective_date / expected_date / coverage ya se
# publican en Provenance + tabla de Alineacion con sesion esperada.
# La whitelist se justifica en test_whitelist_state_only_justificada.
STATE_ONLY_WHITELIST = frozenset({
    "schema_version",
    "temporal_contract_version",
    "effective_date",
    "expected_date",
    "coverage",
})


# --- Downstream: render lee X => X en return ---

def test_render_mte_no_lee_campos_ausentes_en_return():
    """Cualquier clave leida por render_mte debe estar en el return."""
    assert_render_subset_return(
        producer_module=ENGINE, producer_fn="compute_mte",
        consumer_module=RENDER, consumer_fn="render_mte",
        consumer_var="mte_result",
    )


def test_render_mte_lee_futures_status_y_pending():
    """Regresion explicita: los 2 campos que ya fallaron siguen leidos.

    Si un refactor elimina estas lecturas del render, el reporte
    pierde la advertencia de futuros stale o de transicion de
    escenario. El test obliga a revisar el contrato antes de borrar.
    """
    reads = consumer_read_keys(RENDER, "render_mte", "mte_result")
    assert "futures_status" in reads, (
        "render_mte dejo de leer futures_status. El reporte perderia "
        "la advertencia de inputs de futuros stale. Si es intencional, "
        "borrar este test con justificacion."
    )
    assert "pending" in reads, (
        "render_mte dejo de leer pending. El reporte perderia el aviso "
        "de transicion de escenario. Si es intencional, borrar este "
        "test con justificacion."
    )


# --- Upstream: state publica X => X en return ---

def test_compute_mte_state_no_omite_campos_en_return():
    """Cualquier campo del state JSON (no whitelist) esta en return."""
    assert_state_subset_return(
        producer_module=ENGINE,
        state_fn="compute_mte", return_fn="compute_mte",
        whitelist=STATE_ONLY_WHITELIST,
    )


def test_whitelist_state_only_justificada():
    """La whitelist state-only no crece sin justificacion."""
    assert STATE_ONLY_WHITELIST == frozenset({
        "schema_version",
        "temporal_contract_version",
        "effective_date",
        "expected_date",
        "coverage",
    }), (
        "La whitelist de campos state-only cambio. Si es una decision "
        "consciente, actualizar este test con la nueva lista y "
        "justificar en el mensaje del commit. Si no, revisar: puede "
        "ser un campo funcional que deberia estar en el return."
    )


def test_state_y_return_comparten_campos_funcionales():
    """Regresion explicita: futures_status y pending en ambos lados."""
    state_keys = producer_state_keys(ENGINE, "compute_mte")
    return_keys = producer_return_keys(ENGINE, "compute_mte")
    for campo in ("futures_status", "pending"):
        assert campo in state_keys, (
            f"{campo} desaparecio del state JSON. Regresion: el reporte "
            f"no podria pintarlo aunque el operador lo viera en el JSON."
        )
        assert campo in return_keys, (
            f"{campo} desaparecio del return. Regresion de la clase "
            f"1fb791ab: render_mte no podria pintarlo."
        )
