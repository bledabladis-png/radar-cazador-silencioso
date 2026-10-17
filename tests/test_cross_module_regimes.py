# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa4-commitA: contrato de clasificacion de
detect_cross_module_conflict.

Cambio de substring match a igualdad exacta. Estados que antes se
clasificaban por coincidencia accidental de substring (SLOWDOWN por
LOW, LOW_STRESS por STRESS) ahora son neutrales por diseno. Los que
se clasificaban por substring y SI deben clasificarse (EXTREME_STRESS,
LIQUIDITY CRISIS) se anaden explicitamente.

Caso de referencia: reporte 2026-10-17.
"""
from src.utils import detect_cross_module_conflict


def _bias(macro, fin="NEUTRAL", vol="NORMAL", liq="NEUTRAL", mte="MIXED"):
    r = detect_cross_module_conflict(macro, fin, vol, liq, mte)
    return r["details"]["macro"]["bias_financial"]


def test_slowdown_es_neutral():
    assert _bias("SLOWDOWN") == 0


def test_low_stress_es_neutral():
    assert _bias("LOW_STRESS") == 0


def test_low_volatilidad_sigue_siendo_expansion():
    r = detect_cross_module_conflict("MIXED", "NEUTRAL", "LOW", "NEUTRAL", "MIXED")
    assert r["details"]["volatility"]["bias_financial"] == 1


def test_extreme_stress_sigue_siendo_estres():
    r = detect_cross_module_conflict("MIXED", "NEUTRAL", "NORMAL", "EXTREME_STRESS", "MIXED")
    assert r["details"]["liquidity"]["bias_financial"] == -1


def test_liquidity_crisis_sigue_siendo_estres():
    r = detect_cross_module_conflict("LIQUIDITY CRISIS", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED")
    assert r["details"]["macro"]["bias_financial"] == -1


def test_recovery_sigue_siendo_expansion():
    r = detect_cross_module_conflict("RECOVERY", "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED")
    assert r["details"]["macro"]["bias_financial"] == 1


def test_high_stress_sigue_siendo_estres():
    r = detect_cross_module_conflict("MIXED", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED")
    assert r["details"]["financial"]["bias_financial"] == -1


def test_caso_referencia_17_oct_post_commit_a():
    r = detect_cross_module_conflict(
        "SLOWDOWN", "HIGH_STRESS", "NORMAL", "NEUTRAL", "MIXED",
    )
    assert r["conflict_level"] == "MIXED"
    assert r["blocks"] == "Financial Stress: 1/5"
    assert "Expansion" not in r["blocks"]


def test_mensaje_divergence_usa_tilde_correcta():
    r = detect_cross_module_conflict(
        "STAGFLATION", "HIGH_STRESS", "STRESS", "ESTRECHA", "MIXED",
    )
    assert r["conflict_level"] == "DIVERGENCE"
    msg = r["message"]
    assert "\u00be" not in msg
    assert "m\u00f3dulos" in msg
    assert "clasificaci\u00f3n" in msg


def test_barrido_estados_completos():
    casos = {
        "EXTREME_STRESS": -1,
        "LIQUIDITY CRISIS": -1,
        "HIGH_STRESS": -1,
        "ESTRECHA": -1,
        "CRISIS": -1,
        "STRESS": -1,
        "ABUNDANTE": 1,
        "EXPANSION": 1,
        "RECOVERY": 1,
        "LOW": 1,
        "GOLDILOCKS": 1,
        "SLOWDOWN": 0,
        "LOW_STRESS": 0,
        "NORMAL": 0,
        "ELEVATED": 0,
        "NEUTRAL": 0,
        "MIXED": 0,
        "DEFLATION": 0,
    }
    for state, esperado in casos.items():
        r = detect_cross_module_conflict(state, "NEUTRAL", "NORMAL", "NEUTRAL", "MIXED")
        actual = r["details"]["macro"]["bias_financial"]
        assert actual == esperado, (
            "estado={!r}: esperado={}, obtenido={}".format(state, esperado, actual)
        )
