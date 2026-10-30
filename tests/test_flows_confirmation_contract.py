# -*- coding: utf-8 -*-
"""Contratos dict para flows_secondary y confirmation.

Cubre dos pares productor-consumidor donde el productor muta un dict
via subindice (no literal) o compone dinamicamente:

1. flows_secondary: compute_flows_secondary construye flow_synthesis
   con flow_synthesis['X'] = ... (5 claves). render_flujo_sintesis lee
   5 de ellas.
2. confirmation: _compute_confirmation muta confirmation_data['X'] = ...
   (4 claves) + .update(vol_data) (4 claves dinamicas) +
   compute_mte_confirmation anade confirmation_data['mte_scenario'].
   render_confirmation lee 9. Bug real 2026-10-30: mte_scenario
   nunca se escribia (clase 1fb791ab). Fix en compute_mte_confirmation.

Los schemas explicitos (FLOW_SYNTHESIS_SCHEMA,
CONFIRMATION_DATA_SCHEMA) hacen visible el contrato completo,
incluidas las claves dinamicas que el AST no puede ver.
"""
from pathlib import Path

from tests._dict_contract import (
    producer_subscript_keys,
    consumer_read_keys,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FLOWS = REPO_ROOT / "src" / "pipeline" / "flows_secondary.py"
FLOWS_RENDER = REPO_ROOT / "src" / "report" / "flows_international.py"
MTE_CONF = REPO_ROOT / "src" / "pipeline" / "mte_confirmation.py"
CONFIRMATION = REPO_ROOT / "src" / "report" / "confirmation.py"

# --- flows_secondary ---

# 5 claves que compute_flows_secondary escribe en flow_synthesis via
# subindice. AST detecta las 5.
FLOW_SYNTHESIS_SCHEMA = frozenset({
    "flow_proxy_sign",
    "etf_primary_flow_sign",
    "cftc_flow_sign",
    "european_flow_sign",
    "confidence",
})


def test_flows_secondary_producer_coincide_con_schema():
    """producer_subscript_keys(compute_flows_secondary) == schema.

    Si crece, anadir al schema y revisar render. Si decrece, el render
    pierde la fila o el confidence.
    """
    keys = producer_subscript_keys(
        FLOWS, "compute_flows_secondary", "flow_synthesis")
    assert keys == FLOW_SYNTHESIS_SCHEMA, (
        f"flow_synthesis cambio. "
        f"Esperado: {sorted(FLOW_SYNTHESIS_SCHEMA)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es decision consciente, actualizar schema + render."
    )


def test_flows_secondary_render_lee_subset_schema():
    """render_flujo_sintesis lee X => X en FLOW_SYNTHESIS_SCHEMA."""
    reads = consumer_read_keys(
        FLOWS_RENDER, "render_flujo_sintesis", "flow_synthesis")
    missing = reads - FLOW_SYNTHESIS_SCHEMA
    assert not missing, (
        f"render_flujo_sintesis lee {sorted(missing)} pero no estan "
        f"en FLOW_SYNTHESIS_SCHEMA. Bug clase 1fb791ab: el render "
        f"pediria un campo que el producer no escribe (o al reves). "
        f"Anadir al producer y al schema, o quitar del render."
    )


# --- confirmation ---

# 9 claves que render_confirmation lee:
# - t10y3m, ratios, fls, ad: subindice en _compute_confirmation (AST).
# - rv_21d, rv_60d, vrp_21d, vrp_60d: dinamicamente via .update(
#   vol_data) en _compute_confirmation. Declaradas en schema, no AST.
# - mte_scenario: asignada en compute_mte_confirmation (fix 2026-10-30).
CONFIRMATION_DATA_SCHEMA = frozenset({
    "t10y3m",
    "rv_21d", "rv_60d", "vrp_21d", "vrp_60d",
    "ratios", "fls", "ad",
    "mte_scenario",
})

# Subconjunto que AST ve en _compute_confirmation (4 claves).
CONFIRMATION_SUBSCRIPT_AST = frozenset({
    "t10y3m", "ratios", "fls", "ad",
})


def test_confirmation_producer_subscript_es_subset_de_schema():
    """Las claves que _compute_confirmation escribe por subindice
    estan en el schema. Las dinamicas (rv_*, update) estan declaradas
    explicitamente en el schema, no en AST."""
    keys = producer_subscript_keys(
        MTE_CONF, "_compute_confirmation", "confirmation_data")
    assert keys == CONFIRMATION_SUBSCRIPT_AST, (
        f"_compute_confirmation escribe {sorted(keys)} via subindice. "
        f"Esperado: {sorted(CONFIRMATION_SUBSCRIPT_AST)}. "
        f"Si se anade una clave nueva via subindice, actualizar "
        f"CONFIRMATION_SUBSCRIPT_AST y CONFIRMATION_DATA_SCHEMA."
    )
    assert keys <= CONFIRMATION_DATA_SCHEMA


def test_compute_mte_confirmation_popula_mte_scenario():
    """Fix 2026-10-30: compute_mte_confirmation asigna
    confirmation_data['mte_scenario'].

    Bug: render_confirmation L94 leia
    confirmation_data.get('mte_scenario', '') para disparar RECESSION
    CAPITULATION SIGNAL, pero ningun productor escribia la clave. La
    rama nunca se ejecutaba. Si el fix se revierte, este test falla.
    """
    keys = producer_subscript_keys(
        MTE_CONF, "compute_mte_confirmation", "confirmation_data")
    assert "mte_scenario" in keys, (
        "compute_mte_confirmation no asigna 'mte_scenario'. "
        "Regresion del fix 2026-10-30: render_confirmation L94 lee "
        "confirmation_data.get('mte_scenario', '') pero la clave "
        "nunca llega. La rama RECESSION CAPITULATION se pierde."
    )
    assert "mte_scenario" in CONFIRMATION_DATA_SCHEMA


def test_confirmation_render_lee_subset_schema():
    """render_confirmation lee X => X en CONFIRMATION_DATA_SCHEMA.

    Detecta "render lee X que no esta en schema", lo que indica que
    alguien anadio un get() sin anadir la clave al producer/schema.
    """
    reads = consumer_read_keys(
        CONFIRMATION, "render_confirmation", "confirmation_data")
    missing = reads - CONFIRMATION_DATA_SCHEMA
    assert not missing, (
        f"render_confirmation lee {sorted(missing)} pero no estan en "
        f"CONFIRMATION_DATA_SCHEMA. Bug clase 1fb791ab: el render "
        f"pediria un campo que el producer no escribe. Anadir al "
        f"producer y al schema, o quitar del render."
    )


def test_render_confirmation_lee_mte_scenario():
    """Regresion: el render sigue leyendo mte_scenario.

    Si un refactor elimina la lectura, el feature RECESSION
    CAPITULATION desaparece aunque el producer lo publique.
    """
    reads = consumer_read_keys(
        CONFIRMATION, "render_confirmation", "confirmation_data")
    assert "mte_scenario" in reads, (
        "render_confirmation dejo de leer 'mte_scenario'. Se pierde "
        "el RECESSION CAPITULATION SIGNAL. Si es intencional, borrar "
        "este test con justificacion."
    )
