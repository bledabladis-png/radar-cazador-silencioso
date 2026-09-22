"""P38 pairwise - FIXTURE SINTETICO (dictamen #77, ciclo B-04).

*** NON-PRODUCTION / CONTRACT_TEST_FIXTURE ***

Este fixture NO representa datos reales de 13F ni del universo
contractual del radar. NO modifica mappings productivos. Su unico
proposito es ejercitar la rama pairwise de P38 (coverage con
TARGET_PAIRWISE no vacio) autorizada por el dictamen #77.

Escenario:
    Q4: FIGI_A=1000 (V) | FIGI_B=500 (V) | FIGI_D=200 (V)
    Q1: FIGI_A=1200 (V) | FIGI_C=400 (V) | FIGI_D=200 (TU)
    V = VERIFIED, TU = TEMPORAL_UNVERIFIED

Resultados esperados:
    TARGET_Q4        = {A, B, D}
    TARGET_Q1        = {A, C, D}
    TARGET_PAIRWISE  = {A, D}
    PAIRED           = {A}
    paired_security_coverage         = 1/2      = 0.5
    paired_weighted_share_coverage   = 1200/1400 = 6/7
"""
from __future__ import annotations

import pytest

from src.institutional_accumulation.aggregation import coverage as cov


FIXTURE_MARK = "NON-PRODUCTION / CONTRACT_TEST_FIXTURE"


def _rec(period, key, figi, weight, op_status="VERIFIED"):
    return cov.PositionRecord(
        period=period,
        observed_security_key=key,
        share_class_figi=figi,
        canonical_security=None,
        resolution_status="CANONICAL",
        operational_mapping_status=op_status,
        weight=weight,
    )


def _fixture_records():
    rq4 = [
        _rec("Q4", "key_A", "FIGI_A", 1000.0),
        _rec("Q4", "key_B", "FIGI_B",  500.0),
        _rec("Q4", "key_D", "FIGI_D",  200.0),
    ]
    rq1 = [
        _rec("Q1", "key_A", "FIGI_A", 1200.0),
        _rec("Q1", "key_C", "FIGI_C",  400.0),
        _rec("Q1", "key_D", "FIGI_D",  200.0,
             op_status="TEMPORAL_UNVERIFIED"),
    ]
    return rq4, rq1


TARGET_Q4 = {"FIGI_A", "FIGI_B", "FIGI_D"}
TARGET_Q1 = {"FIGI_A", "FIGI_C", "FIGI_D"}


def test_fixture_esta_marcado_non_production():
    assert FIXTURE_MARK in __doc__
    assert "NON-PRODUCTION" in __doc__


def test_paired_security_coverage_con_pairwise_no_vacio():
    """TARGET_PAIRWISE = {A, D}; PAIRED = {A}; cobertura = 1/2."""
    rq4, rq1 = _fixture_records()
    result = cov.compute_contractual_coverage(
        target_q4=TARGET_Q4, target_q1=TARGET_Q1,
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] == 0.5
    assert result["coverage_status"] == "VALID"


def test_paired_weighted_con_pesos_asimetricos():
    """w(A)=max(1000,1200)=1200; w(D)=max(200,200)=200.
    denom = 1400, numer = 1200, resultado = 6/7.
    """
    rq4, rq1 = _fixture_records()
    result = cov.compute_contractual_coverage(
        target_q4=TARGET_Q4, target_q1=TARGET_Q1,
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_weighted_share_coverage"] == pytest.approx(6.0 / 7.0)


def test_max_q4_q1_tras_agregacion_por_figi():
    """Dos CUSIPs comparten FIGI_A: agregado Q4=1000, Q1=1200,
    max = 1200 (NO max(CUSIP1)+max(CUSIP2)).
    """
    rq4 = [
        _rec("Q4", "cusip_A1", "FIGI_A", 600.0),
        _rec("Q4", "cusip_A2", "FIGI_A", 400.0),
    ]
    rq1 = [
        _rec("Q1", "cusip_A1", "FIGI_A", 700.0),
        _rec("Q1", "cusip_A2", "FIGI_A", 500.0),
    ]
    agg_q4 = cov.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    agg_q1 = cov.aggregate_positions_by_shareclass_figi(rq1, "Q1")
    assert agg_q4 == {"FIGI_A": 1000.0}
    assert agg_q1 == {"FIGI_A": 1200.0}
    result = cov.compute_contractual_coverage(
        target_q4={"FIGI_A"}, target_q1={"FIGI_A"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_weighted_share_coverage"] == 1.0


def test_temporal_unverified_deja_de_contribuir():
    """FIGI_D en Q1 esta TEMPORAL_UNVERIFIED -> no entra en PAIRED.
    Si estuviera VERIFIED, PAIRED = {A, D} y cobertura = 1.0.
    """
    rq4, rq1 = _fixture_records()
    # Version degradando FIGI_A en Q1 para que TARGET_PAIRWISE = {D}:
    rq1_alt = [
        _rec("Q1", "key_A", "FIGI_A", 1200.0,
             op_status="TEMPORAL_UNVERIFIED"),
        _rec("Q1", "key_C", "FIGI_C",  400.0),
        _rec("Q1", "key_D", "FIGI_D",  200.0,
             op_status="TEMPORAL_UNVERIFIED"),
    ]
    result = cov.compute_contractual_coverage(
        target_q4=TARGET_Q4, target_q1=TARGET_Q1,
        records_q4=rq4, records_q1=rq1_alt,
    )
    # TARGET_PAIRWISE = {A, D}, pero ninguno VERIFIED en Q1 -> 0/2
    assert result["paired_security_coverage"] == 0.0


def test_determinismo_fixture():
    """Misma entrada -> mismo resultado, dos veces."""
    rq4, rq1 = _fixture_records()
    r1 = cov.compute_contractual_coverage(
        target_q4=TARGET_Q4, target_q1=TARGET_Q1,
        records_q4=rq4, records_q1=rq1,
    )
    r2 = cov.compute_contractual_coverage(
        target_q4=TARGET_Q4, target_q1=TARGET_Q1,
        records_q4=rq4, records_q1=rq1,
    )
    assert r1 == r2
