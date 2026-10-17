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


def test_caso_solo_expansion_con_ceros_h16():
    """Solo RECOVERY: Financial 0/5 (H16), Expansion 1/5, Inflation 0/5."""
    r = detect_cross_module_conflict(
        "RECOVERY", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert "Expansion: 1/5" in r["blocks"]
    assert "Financial Stress: 0/5" in r["blocks"]
    assert "Inflation Pressure: 0/5" in r["blocks"]


def test_caso_referencia_17_oct_post_h16():
    """Caso 17-oct con H16: Financial 1/5 + Inflation 0/5, sin Expansion."""
    r = detect_cross_module_conflict(
        "SLOWDOWN", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert r["conflict_level"] == "MIXED"
    assert "Financial Stress: 1/5" in r["blocks"]
    assert "Inflation Pressure: 0/5" in r["blocks"]
    assert "Expansion" not in r["blocks"]


def test_caso_todos_neutrales_muestra_ceros():
    """Todos neutrales: H16 publica Financial 0/5 e Inflation 0/5."""
    r = detect_cross_module_conflict(
        "MIXED", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert "Financial Stress: 0/5" in r["blocks"]
    assert "Inflation Pressure: 0/5" in r["blocks"]
    assert "Expansion" not in r["blocks"]
    assert "Sin bloque de estres dominante" not in r["blocks"]


def test_caso_consensus_expansion_publica_ceros_h16():
    """CONSENSUS expansion: Financial 0/5, Expansion 5/5, Inflation 0/5."""
    r = detect_cross_module_conflict(
        "EXPANSION", "ABUNDANTE", "LOW", "EXPANSION", "EXPANSION",
    )
    assert r["conflict_level"] == "CONSENSUS"
    assert "Expansion: 5/5" in r["blocks"]
    assert "Financial Stress: 0/5" in r["blocks"]
    assert "Inflation Pressure: 0/5" in r["blocks"]


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



def test_h16_cero_no_es_ausencia():
    """H16: contador 0 se publica explicitamente, no se omite."""
    r = detect_cross_module_conflict(
        "MIXED", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED",
    )
    blocks = r["blocks"]
    assert "Financial Stress: 0/5" in blocks
    assert "Inflation Pressure: 0/5" in blocks
    assert "Sin bloque de estres dominante" not in blocks


def test_h16_inflation_pressure_con_valor():
    """H16: si hay presion inflacionaria, se publica su valor."""
    r = detect_cross_module_conflict(
        "STAGFLATION", "HIGH_STRESS", "STRESS", "ESTRECHA", "MIXED",
    )
    assert r["conflict_level"] == "DIVERGENCE"
    assert "Inflation Pressure: 1/5" in r["blocks"]
