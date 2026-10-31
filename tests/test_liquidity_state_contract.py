# -*- coding: utf-8 -*-
"""Contrato AST: liquidity_state.json <-> consumidores.

Clase de bug 1fb791ab aplicada a liquidity_state. El productor
(regimes/liquidity_v2.py::compute_liquidity_score_v2) construye un
dict local `state = {...}` y lo persiste con STATE_PATH.write_text(
json.dumps(state)). Los consumidores (header.py::render_regimenes,
freshness.py::render_data_freshness) lo leen desde disco con
try/except silencioso:

    try:
        _liq = json.loads(path.read_text())
        _v2 = _liq.get('v2') or {}
        ...
    except (OSError, ValueError, TypeError, KeyError):
        pass

Si el productor deja de publicar 'v2' o 'date', el consumer no
falla: pinta N/D o omite la linea. Este test cierra el contrato
top-level con AST. Los campos nested (v2.severidad_cruda,
v2.diff_bp) se cubren con test de integracion del render (ver
tests/test_report_mte_futures_liquidity_visibles.py::
test_liquidity_severidad_tension_elevada_pinta).
"""
from pathlib import Path

from tests._dict_contract import (
    producer_assign_dict_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "regimes" / "liquidity_v2.py"
HEADER = REPO_ROOT / "src" / "report" / "header.py"
FRESHNESS = REPO_ROOT / "src" / "report" / "freshness.py"

# Claves top-level que el productor publica (del dict literal
# `state = {...}` en compute_liquidity_score_v2).
EXPECTED_TOP_LEVEL = frozenset({
    "score", "date", "regime", "v2",
})


# --- Downstream: header.py::render_regimenes lee X => X en state ---

def test_header_regimenes_lee_campos_presentes_en_state():
    """render_regimenes lee solo claves top-level del state.

    El consumer hace `_liq.get('v2')` para obtener el subdict; las
    claves nested (severidad_cruda, diff_bp) se acceden via _v2 y
    no las ve el AST. Cubiertas por test de integracion del render.
    """
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="compute_liquidity_score_v2",
        consumer_module=HEADER,
        consumer_fn="render_regimenes",
        consumer_var="liquidity_state",
        producer_var="state",
    )


# --- Downstream: freshness.py::render_data_freshness lee X => X ---

def test_freshness_lee_campos_presentes_en_state():
    """render_data_freshness lee solo claves top-level del state."""
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="compute_liquidity_score_v2",
        consumer_module=FRESHNESS,
        consumer_fn="render_data_freshness",
        consumer_var="liq_state",
        producer_var="state",
    )


# --- Regresion: consumer sigue leyendo los campos que ya fallaron ---

def test_header_regimenes_lee_v2():
    """Si un refactor elimina `_liq.get('v2')`, este test falla.

    Sin `v2` el reporte pierde "Severidad: TENSION_ELEVADA
    (diff_bp=...)" aunque el state lo tenga. Bug real en sesion 19
    (LIQUIDITY-SEVERIDAD-OCULTA, 1fb791ab).
    """
    reads = consumer_read_keys(
        HEADER, "render_regimenes", "liquidity_state")
    assert "v2" in reads, (
        "render_regimenes dejo de leer 'v2' del state. El reporte "
        "perderia la linea de Severidad (TENSION_ELEVADA). Si es "
        "intencional, borrar este test con justificacion."
    )


def test_freshness_lee_date():
    """Si un refactor elimina `liq_state.get('date')`, este test falla.

    Sin `date` el reporte pierde la frescura real de FRED (cae a
    N/D). "
    """
    reads = consumer_read_keys(
        FRESHNESS, "render_data_freshness", "liq_state")
    assert "date" in reads, (
        "render_data_freshness dejo de leer 'date' del state. El "
        "reporte perderia la frescura de FRED. Si es intencional, "
        "borrar este test con justificacion."
    )


# --- Upstream parcial: whitelist explicita del top-level ---

def test_top_level_state_coincide_con_esperado():
    """El state top-level publica exactamente EXPECTED_TOP_LEVEL.

    Si alguien anade un campo top-level nuevo, este test falla y
    obliga a revisar los consumers (header, freshness, data_quality)
    para ver si deben publicarlo. Si alguien lo elimina, los
    consumers pierden el campo silenciosamente.
    """
    keys = producer_assign_dict_keys(
        PRODUCER, "compute_liquidity_score_v2", "state")
    assert keys == EXPECTED_TOP_LEVEL, (
        f"El state top-level cambio. "
        f"Esperado: {sorted(EXPECTED_TOP_LEVEL)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es una decision consciente, actualizar este test y "
        f"revisar los consumers (header, freshness, data_quality)."
    )
