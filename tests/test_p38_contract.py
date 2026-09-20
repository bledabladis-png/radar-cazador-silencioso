"""Tests contractuales P38 (F2.4 aplicado 2026-09-20).

Contrato NIPC_CONTRATOS_SEMANTICOS_v1.md secciones 3 y 4.

API normativa P38:
    compute_contractual_coverage(...)     FUNCION CONTRACTUAL (coverage.py)
    aggregate_positions_by_shareclass_figi Opcion 2 AGREG.
    PositionRecord                        dataclass tipado (F2.4 regla #4)
    compute_nipc_contractual(...)         ruta CONTRACTUAL (nipc.py)
    compute_nipc(...)                     ruta PROXY (legacy)
    compute_coverage_pairwise(...)        ruta PROXY (legacy)

`coverage.py` NO importa de `nipc.py`. `nipc.py` NO importa
`radar_target_catalog` ni `target_universe`. El TARGET se construye
externamente (identity/target_builder.py) y se pasa como parametro.

NO tocan codigo productivo.
"""
import inspect

import pandas as pd
import pytest

from src.institutional_accumulation.aggregation import coverage
from src.institutional_accumulation.aggregation import nipc


# --- Tests legacy (ruta PROXY, evidencia historica) ---


def test_p38_legacy_denominador_cero_unavailable():
    """Ruta PROXY: denominador cero -> UNAVAILABLE.

    Se mantiene como regresion historica. La evidencia contractual
    vive en los tests de compute_contractual_coverage.
    """
    empty = pd.DataFrame()
    result = nipc.compute_coverage_pairwise(empty, empty)
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


def test_p38_legacy_usa_observed_key():
    """Ruta PROXY: opera sobre observed_security_key, no sobre TARGET."""
    curr = pd.DataFrame([
        {
            "observed_security_key": "cusip:A",
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "sshprnamt_total": 100,
        }
    ])
    prev = pd.DataFrame([
        {
            "observed_security_key": "cusip:A",
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "sshprnamt_total": 80,
        }
    ])
    result = nipc.compute_coverage_pairwise(curr, prev)
    assert result["paired_security_coverage"] == 1.0


# --- Tests contractuales P38 (ruta CONTRACTUAL) ---


def test_p38_existe_compute_contractual_coverage():
    """Contrato P38: la funcion contractual existe en coverage.py."""
    assert hasattr(coverage, "compute_contractual_coverage"), (
        "Falta aggregation/coverage.py::compute_contractual_coverage"
    )


def test_p38_contractual_recibe_target_externo():
    """Contrato P38 seccion 8.3: TARGET se recibe, no se construye internamente."""
    sig = inspect.signature(coverage.compute_contractual_coverage)
    params = set(sig.parameters.keys())
    assert "target_q4" in params and "target_q1" in params, (
        "Firma actual: {0}".format(params)
    )


def test_p38_agrega_pesos_por_shareclass_figi():
    """Contrato P38 seccion 3.3: agregar ANTES de max(Q4,Q1).

    Escenario:
        Q4: CUSIP_A=100, CUSIP_B=50  (ambos FIGI_X, ambos VERIFIED)
        Q1: CUSIP_A=80,  CUSIP_B=60  (ambos FIGI_X, ambos VERIFIED)

    Correcto:   max(Q4_total=150, Q1_total=140) = 150.
    Incorrecto: max(100,80) + max(50,60) = 160.
    """
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=50.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=80.0,
        ),
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=60.0,
        ),
    ]
    agg_q4 = coverage.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    agg_q1 = coverage.aggregate_positions_by_shareclass_figi(rq1, "Q1")
    assert agg_q4 == {"FIGI_X": 150.0}
    assert agg_q1 == {"FIGI_X": 140.0}

    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_X"}, target_q1={"FIGI_X"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_weighted_share_coverage"] == 1.0
    assert result["coverage_status"] == "VALID"


def test_p38_solo_verified_contribuye_al_peso():
    """F2.4 AGREG.: solo VERIFIED aporta al peso contractual."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL",
            operational_mapping_status="TEMPORAL_UNVERIFIED",
            weight=50.0,
        ),
    ]
    agg = coverage.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    # Solo el VERIFIED (100.0) entra al peso contractual.
    assert agg == {"FIGI_X": 100.0}


def test_p38_target_pairwise_interseccion_no_union():
    """Contrato P38: TARGET_PAIRWISE = TARGET_Q4 INTERSECT TARGET_Q1."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_A", canonical_security="equity:A",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:B",
            share_class_figi="FIGI_B", canonical_security="equity:B",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    # TARGET_Q4 = {FIGI_A, FIGI_C}; TARGET_Q1 = {FIGI_B, FIGI_C}.
    # Interseccion = {FIGI_C}. FIGI_C no tiene records -> coverage UNAVAILABLE.
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_A", "FIGI_C"}, target_q1={"FIGI_B", "FIGI_C"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] == 0.0
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


def test_p38_cusip_distinto_figi_igual_paired():
    """Modelo A (shareClassFIGI): CUSIP distinto con FIGI comun -> PAIRED."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="figi:BBG...",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=120.0,
        ),
    ]
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_X"}, target_q1={"FIGI_X"},
        records_q4=rq4, records_q1=rq1,
    )
    # FIGI_X es TARGET_PAIRWISE y esta VERIFIED en ambos -> PAIRED.
    assert result["paired_security_coverage"] == 1.0
    assert result["paired_weighted_share_coverage"] == 1.0


def test_p38_figi_distinto_not_paired():
    """FIGI distinto -> NOT_PAIRED."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_A", canonical_security="equity:A",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:B",
            share_class_figi="FIGI_B", canonical_security="equity:B",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=120.0,
        ),
    ]
    # TARGET_Q4={FIGI_A}; TARGET_Q1={FIGI_B}. Interseccion = {}.
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_A"}, target_q1={"FIGI_B"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] == 0.0


def test_p38_unmapped_count_es_int():
    """F2.4 regla #2: unmapped_count es int, no float."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_A", canonical_security="equity:A",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:B",
            share_class_figi=None, canonical_security=None,
            resolution_status="OBSERVED_ONLY", operational_mapping_status="UNRESOLVED",
            weight=50.0,
        ),
    ]
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_A"}, target_q1={"FIGI_A"},
        records_q4=rq4, records_q1=[],
    )
    assert isinstance(result["unmapped_count_previous"], int)
    assert isinstance(result["unmapped_count_current"], int)
    # unmapped_count_previous: registros con FIGI pero no VERIFIED.
    # Aqui: el segundo tiene FIGI=None (no cuenta en all_q4). Por tanto 0.
    assert result["unmapped_count_previous"] == 0


# --- Tests de la ruta contractual de alto nivel (compute_nipc_contractual) ---


def test_p38_compute_nipc_contractual_evidence_class():
    """Contrato: la ruta contractual marca evidence_class=CONTRACTUAL."""
    delta = pd.DataFrame()
    result = nipc.compute_nipc_contractual(
        delta,
        target_q4={"FIGI_X"}, target_q1={"FIGI_X"},
        records_q4=[], records_q1=[],
    )
    assert result["evidence_class"] == "CONTRACTUAL"


def test_p38_compute_nipc_contractual_sin_target_error():
    """F2.4 regla #3: target None en ruta contractual -> error duro."""
    delta = pd.DataFrame()
    with pytest.raises(ValueError, match="target_q4 y target_q1"):
        nipc.compute_nipc_contractual(
            delta,
            target_q4=None, target_q1={"FIGI_X"},
            records_q4=[], records_q1=[],
        )


def test_p38_compute_nipc_legacy_evidence_class_proxy():
    """Regla #3: la ruta legacy marca evidence_class=PROXY."""
    delta = pd.DataFrame()
    result = nipc.compute_nipc(delta)
    assert result["evidence_class"] == "PROXY"
