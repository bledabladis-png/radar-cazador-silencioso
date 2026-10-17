# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa4-commitC: el bloque INFO Cross-Module
publica Expansion: N/5 cuando hay modulos en expansion.

Antes solo se publicaba Financial Stress y Inflation Pressure. Un caso
con modulos en ambos lados se leia como "solo 1/5 estres", ocultando
la mezcla real. El bloque gana informacion sin cambiar conflict_level.
"""
from src.utils import detect_cross_module_conflict


def test_caso_1_estres_1_expansion_publica_ambos():
    """RECOVERY (exp) + HIGH_STRESS (estres): MIXED, ambos bloques."""
    r = detect_cross_module_conflict(
        "RECOVERY", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert r["conflict_level"] == "MIXED"
    assert "Financial Stress: 1/5" in r["blocks"]
    assert "Expansion: 1/5" in r["blocks"]


def test_caso_solo_estres_sin_expansion():
    """Solo HIGH_STRESS: no debe aparecer linea Expansion."""
    r = detect_cross_module_conflict(
        "MIXED", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert "Financial Stress: 1/5" in r["blocks"]
    assert "Expansion" not in r["blocks"]


def test_caso_solo_expansion_sin_estres():
    """Solo RECOVERY: no debe aparecer linea Financial Stress."""
    r = detect_cross_module_conflict(
        "RECOVERY", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert "Expansion: 1/5" in r["blocks"]
    assert "Financial Stress" not in r["blocks"]


def test_caso_referencia_17_oct_post_commit_a_y_c():
    """SLOWDOWN neutralizado: solo estres, sin Expansion."""
    r = detect_cross_module_conflict(
        "SLOWDOWN", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert r["conflict_level"] == "MIXED"
    assert r["blocks"] == "Financial Stress: 1/5"
    assert "Expansion" not in r["blocks"]


def test_caso_todos_neutrales_sin_bloques():
    """Todos neutrales -> 'Sin bloque de estres dominante'."""
    r = detect_cross_module_conflict(
        "MIXED", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert "Sin bloque de estres dominante" in r["blocks"]
    assert "Expansion" not in r["blocks"]


def test_caso_consensus_expansion_no_publica_estres():
    """CONSENSUS expansion: 5/5. No debe aparecer Financial Stress."""
    r = detect_cross_module_conflict(
        "EXPANSION", "ABUNDANTE", "LOW", "EXPANSION", "EXPANSION",
    )
    assert r["conflict_level"] == "CONSENSUS"
    assert "Expansion: 5/5" in r["blocks"]
    assert "Financial Stress" not in r["blocks"]


def test_orden_de_bloques_estable():
    """Estres primero, expansion segundo, inflacion tercero."""
    r = detect_cross_module_conflict(
        "STAGFLATION", "HIGH_STRESS", "STRESS", "ESTRECHA", "MIXED",
    )
    assert r["conflict_level"] == "DIVERGENCE"
    blocks = r["blocks"]
    idx_stress = blocks.find("Financial Stress")
    idx_infl = blocks.find("Inflation Pressure")
    assert idx_stress >= 0
    assert idx_infl >= 0
    assert idx_stress < idx_infl
