# -*- coding: utf-8 -*-
"""Contrato AST: compute_sector_rank_score <-> render_ranking_sectorial_unificado.

Productor: regimes/sector_regime.py::compute_sector_rank_score. Dict
literal con 4 claves top-level: scores, ranking, regime, last_scores.
Retorna None en 2 ramas (returns vacio, bench_close KeyError).

Consumidor: src/report/rankings.py::render_ranking_sectorial_unificado.
Lee 1 de las 4 claves: ranking (lista de (ticker, nombre, score)).
Las otras 3 (scores: df de componentes, regime: str, last_scores:
dict) no se leen aqui; viven en el state del pipeline para el
reporte y otras secciones. No es bug: el render unificado solo pinta
la tabla rankeada.

Top-level: contrato AST puro.
Nested: ranking es lista de tuplas (ya cubierto por
test_report_generator_smoke y otros tests de render).

Nota (2026-10-31): la whitelist declaraba producer
src/pipeline/sectors_base.py::compute_sectors_base, pero ese es un
wrapper que contiene `sector_results` como clave de su dict. El
productor real del sub-dict es compute_sector_rank_score en
regimes/sector_regime.py. Test corregido.
"""
from pathlib import Path

from tests._dict_contract import (
    producer_return_keys,
    consumer_read_keys,
    assert_render_subset_return,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCER = REPO_ROOT / "regimes" / "sector_regime.py"
RENDER = REPO_ROOT / "src" / "report" / "rankings.py"

EXPECTED_TOP_LEVEL = frozenset({
    "scores", "ranking", "regime", "last_scores",
})


def test_render_ranking_sectorial_no_lee_campos_ausentes_en_producer():
    """render_ranking_sectorial_unificado lee X => X en compute_sector_rank_score."""
    assert_render_subset_return(
        producer_module=PRODUCER,
        producer_fn="compute_sector_rank_score",
        consumer_module=RENDER,
        consumer_fn="render_ranking_sectorial_unificado",
        consumer_var="sector_results",
    )


def test_top_level_sector_results_coincide_con_esperado():
    """El dict top-level publica exactamente EXPECTED_TOP_LEVEL.

    Si crece, revisar si el nuevo campo es funcional (debe estar en el
    render o justificarse). Si decrece, los consumidores pierden el
    campo silenciosamente.
    """
    keys = producer_return_keys(PRODUCER, "compute_sector_rank_score")
    assert keys == EXPECTED_TOP_LEVEL, (
        f"El dict top-level de compute_sector_rank_score cambio. "
        f"Esperado: {sorted(EXPECTED_TOP_LEVEL)}. "
        f"Actual: {sorted(keys)}. "
        f"Si es una decision consciente, actualizar este test y "
        f"revisar render_ranking_sectorial_unificado y otros "
        f"consumidores del state."
    )


def test_render_ranking_sectorial_lee_campo_critico():
    """Regresion: ranking es la unica clave que el render unificado usa.

    Si el productor deja de emitirlo o el consumer deja de leerlo,
    la tabla "Rankings Sectoriales" desaparece del reporte.
    """
    reads = consumer_read_keys(
        RENDER, "render_ranking_sectorial_unificado", "sector_results")
    assert "ranking" in reads, (
        "render_ranking_sectorial_unificado dejo de leer 'ranking'. "
        "El reporte perderia la tabla Rankings Sectoriales. Si es "
        "intencional, borrar este test con justificacion."
    )
