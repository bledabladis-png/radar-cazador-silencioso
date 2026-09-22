"""Tests contractuales P38 (F2.4 aplicado 2026-09-20).

Cubre tambien A.6.3 (dictamen #72): test contractual de pairing
segun Q12 Modelo A. Ver `test_p38_cusip_distinto_figi_igual_paired`
(L186): CUSIP distinto con shareClassFIGI comun -> PAIRED aunque
canonical_security difiera entre Q4 y Q1.

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
    """FIGI distinto -> TARGET_PAIRWISE vacio -> None (auditor Q5)."""
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
    # TARGET_Q4={FIGI_A}; TARGET_Q1={FIGI_B}. TARGET_PAIRWISE = {}.
    # A2 c4/5 (Q5): TARGET_PAIRWISE vacio -> paired_security_coverage
    # es None (fail-closed), no 0.0.
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_A"}, target_q1={"FIGI_B"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] is None
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


def test_p38_q5_target_q4_vacio_coverage_previous_none():
    """Auditor Q5: TARGET_Q4 vacio -> coverage_previous UNAVAILABLE.

    coverage_current sigue siendo calculable sobre TARGET_Q1.
    """
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    result = coverage.compute_contractual_coverage(
        target_q4=set(), target_q1={"FIGI_X"},
        records_q4=[], records_q1=rq1,
    )
    assert result["coverage_previous"] is None
    assert result["coverage_current"] == 1.0
    assert result["paired_security_coverage"] is None
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


def test_p38_q5_target_q1_vacio_coverage_current_none():
    """Auditor Q5 simetrico: TARGET_Q1 vacio -> coverage_current UNAVAILABLE."""
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=100.0,
        ),
    ]
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_X"}, target_q1=set(),
        records_q4=rq4, records_q1=[],
    )
    assert result["coverage_previous"] == 1.0
    assert result["coverage_current"] is None
    assert result["paired_security_coverage"] is None
    assert result["paired_weighted_share_coverage"] is None
    assert result["coverage_status"] == "UNAVAILABLE"


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


def test_p38_paired_weighted_max_por_figi_con_cobertura_asimetrica():
    """A2 c5/5 (H-06): paired_weighted calcula max(Q4,Q1) POR FIGI.

    Cobertura asimetrica:
      FIGI_X: Q4=1000, Q1=1200 (VERIFIED ambos) -> max=1200, PAIRED
      FIGI_Y: Q4=500,  Q1=300 (solo Q4 VERIFIED) -> max=500, NO PAIRED
    TARGET_PAIRWISE = {FIGI_X, FIGI_Y}
    numer = 1200 (X); denom = 1200 + 500 = 1700
    """
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:X",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1000.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:Y",
            share_class_figi="FIGI_Y", canonical_security="equity:Y",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=500.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:X",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1200.0,
        ),
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:Y",
            share_class_figi="FIGI_Y", canonical_security="equity:Y",
            resolution_status="CANONICAL",
            operational_mapping_status="TEMPORAL_UNVERIFIED",
            weight=300.0,
        ),
    ]
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_X", "FIGI_Y"},
        target_q1={"FIGI_X", "FIGI_Y"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] == 0.5
    assert result["paired_weighted_share_coverage"] == 1200.0 / 1700.0
    assert result["coverage_status"] == "VALID"


def test_p38_agregacion_figi_multiples_cusip_pesos_reales():
    """A2 c5/5: agregacion FIGI con SSHPRNAMT reales (no weight=1.0).

    FIGI_X acumula 2 CUSIPs en Q4 y 2 en Q1:
      Q4: CUSIP_A=1000 + CUSIP_B=500 = 1500
      Q1: CUSIP_A=1200 + CUSIP_B=400 = 1600
    max = 1600. Se agrega ANTES de max (contrato P38 s.3.3).
    """
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1000.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=500.0,
        ),
    ]
    rq1 = [
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:A",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1200.0,
        ),
        coverage.PositionRecord(
            period="Q1", observed_security_key="cusip:B",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=400.0,
        ),
    ]
    agg_q4 = coverage.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    agg_q1 = coverage.aggregate_positions_by_shareclass_figi(rq1, "Q1")
    assert agg_q4 == {"FIGI_X": 1500.0}
    assert agg_q1 == {"FIGI_X": 1600.0}
    result = coverage.compute_contractual_coverage(
        target_q4={"FIGI_X"}, target_q1={"FIGI_X"},
        records_q4=rq4, records_q1=rq1,
    )
    assert result["paired_security_coverage"] == 1.0
    assert result["paired_weighted_share_coverage"] == 1.0


def test_p38_sshprnamt_no_doble_conteo_multiples_keys_mismo_figi():
    """B-06 (auditor #76): cuando N catalog_keys comparten FIGI, el
    total SSHPRNAMT del FIGI se cuenta UNA vez, no N.

    Distribucion correcta del probe (post-fix B-06): una key
    representativa lleva el total del FIGI; las demas llevan 0.0.
    """
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="key1",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1500.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="key2",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=0.0,
        ),
    ]
    agg = coverage.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    assert agg == {"FIGI_X": 1500.0}, (
        "Doble conteo: esperado 1500.0, obtenido " + str(agg)
    )


def test_p38_sshprnamt_doble_conteo_es_reproducible_sin_fix_b06():
    """B-06: documenta el bug que el fix evita.

    Si el probe difundiera el total del FIGI a TODAS las keys
    (comportamiento pre-fix B-06), aggregate sumaria N veces el
    total. Este test lo demuestra como contraejemplo: si alguien
    revierte el fix, este test da la senal.
    """
    rq4 = [
        coverage.PositionRecord(
            period="Q4", observed_security_key="key1",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1500.0,
        ),
        coverage.PositionRecord(
            period="Q4", observed_security_key="key2",
            share_class_figi="FIGI_X", canonical_security="equity:X",
            resolution_status="CANONICAL", operational_mapping_status="VERIFIED",
            weight=1500.0,
        ),
    ]
    agg = coverage.aggregate_positions_by_shareclass_figi(rq4, "Q4")
    # El bug: 2 x 1500.0 = 3000.0. Documentado como contraejemplo.
    assert agg == {"FIGI_X": 3000.0}


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

# --- C9 (2026-09-23): stats observables en aggregate_positions_by_shareclass_figi ---


def _rec(figi, status, weight=1.0, period="Q4"):
    from src.institutional_accumulation.aggregation.coverage import PositionRecord
    return PositionRecord(
        period=period,
        observed_security_key="cusip:TEST",
        share_class_figi=figi,
        canonical_security=None,
        resolution_status="CANONICAL",
        operational_mapping_status=status,
        weight=weight,
    )


def test_p38_aggregate_stats_backward_compatible():
    """Sin return_stats, la firma antigua sigue devolviendo solo dict."""
    from src.institutional_accumulation.aggregation.coverage import (
        aggregate_positions_by_shareclass_figi)
    records = [_rec("FIGI_A", "VERIFIED", 1.0), _rec("FIGI_B", "VERIFIED", 2.0)]
    result = aggregate_positions_by_shareclass_figi(records, "Q4")
    assert isinstance(result, dict)
    assert result == {"FIGI_A": 1.0, "FIGI_B": 2.0}


def test_p38_aggregate_stats_contadores_basicos():
    """Con return_stats, devuelve (agg, stats) con contadores correctos."""
    from src.institutional_accumulation.aggregation.coverage import (
        aggregate_positions_by_shareclass_figi)
    records = [
        _rec("FIGI_A", "VERIFIED", 1.0),
        _rec("FIGI_B", "TEMPORAL_UNVERIFIED", 2.0),
        _rec("FIGI_C", "TEMPORAL_UNVERIFIED", 3.0),
        _rec("FIGI_D", "UNRESOLVED", 4.0),
    ]
    agg, stats = aggregate_positions_by_shareclass_figi(records, "Q4", return_stats=True)
    assert agg == {"FIGI_A": 1.0}
    assert stats["n_received"] == 4
    assert stats["n_verified"] == 1
    assert stats["n_excluded"] == 3
    assert stats["n_temporal_unverified"] == 2
    assert stats["n_period_mismatch"] == 0
    assert stats["n_missing_figi"] == 0
    assert stats["excluded_by_status"] == {"TEMPORAL_UNVERIFIED": 2, "UNRESOLVED": 1}


def test_p38_aggregate_stats_period_mismatch():
    """Records con period != arg se cuentan en n_period_mismatch."""
    from src.institutional_accumulation.aggregation.coverage import (
        aggregate_positions_by_shareclass_figi)
    records = [
        _rec("FIGI_A", "VERIFIED", 1.0, period="Q4"),
        _rec("FIGI_B", "VERIFIED", 2.0, period="Q1"),
        _rec("FIGI_C", "VERIFIED", 3.0, period="Q1"),
    ]
    agg, stats = aggregate_positions_by_shareclass_figi(records, "Q4", return_stats=True)
    assert agg == {"FIGI_A": 1.0}
    assert stats["n_received"] == 3
    assert stats["n_period_mismatch"] == 2
    assert stats["n_verified"] == 1


def test_p38_aggregate_stats_missing_figi():
    """Records con figi vacio se cuentan en n_missing_figi."""
    from src.institutional_accumulation.aggregation.coverage import (
        aggregate_positions_by_shareclass_figi)
    records = [
        _rec("FIGI_A", "VERIFIED", 1.0),
        _rec(None, "VERIFIED", 2.0),
        _rec("", "VERIFIED", 3.0),
    ]
    agg, stats = aggregate_positions_by_shareclass_figi(records, "Q4", return_stats=True)
    assert agg == {"FIGI_A": 1.0}
    assert stats["n_received"] == 3
    assert stats["n_missing_figi"] == 2
    assert stats["n_verified"] == 1


def test_p38_aggregate_stats_vacio():
    """Sin records: contadores a 0, agg vacio."""
    from src.institutional_accumulation.aggregation.coverage import (
        aggregate_positions_by_shareclass_figi)
    agg, stats = aggregate_positions_by_shareclass_figi([], "Q4", return_stats=True)
    assert agg == {}
    assert stats["n_received"] == 0
    assert stats["n_verified"] == 0
    assert stats["n_excluded"] == 0
    assert stats["excluded_by_status"] == {}


def test_p38_coverage_expone_stats_q4_q1():
    """compute_contractual_coverage propaga los contadores de stats."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    records_q4 = [
        _rec("FIGI_A", "VERIFIED", 1.0, period="Q4"),
        _rec("FIGI_B", "TEMPORAL_UNVERIFIED", 2.0, period="Q4"),
    ]
    records_q1 = [
        _rec("FIGI_A", "VERIFIED", 1.0, period="Q1"),
        _rec("FIGI_C", "VERIFIED", 3.0, period="Q1"),
    ]
    result = compute_contractual_coverage(
        {"FIGI_A", "FIGI_B"}, {"FIGI_A", "FIGI_C"},
        records_q4, records_q1)
    assert "n_received_q4" in result
    assert result["n_received_q4"] == 2
    assert result["n_verified_q4"] == 1
    assert result["n_temporal_unverified_q4"] == 1
    assert result["n_excluded_q4"] == 1
    assert result["n_received_q1"] == 2
    assert result["n_verified_q1"] == 2
    assert result["n_temporal_unverified_q1"] == 0
    assert result["n_excluded_q1"] == 0
    assert result["excluded_by_status_q4"] == {"TEMPORAL_UNVERIFIED": 1}
    assert result["excluded_by_status_q1"] == {}

# --- C10 (2026-09-23): coverage_available + coverage_quality ---


def test_p38_coverage_available_false_si_target_pairwise_vacio():
    """Sin TARGET_PAIRWISE: coverage_available=False, quality=UNAVAILABLE."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    result = compute_contractual_coverage(set(), set(), [], [])
    assert result["coverage_available"] is False
    assert result["coverage_quality"] == "UNAVAILABLE"


def test_p38_coverage_available_true_si_target_pairwise_no_vacio():
    """Con TARGET_PAIRWISE: coverage_available=True (aunque sea 0% cubierto)."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    records_q4 = [_rec("FIGI_A", "VERIFIED", 1.0, period="Q4")]
    records_q1 = [_rec("FIGI_A", "VERIFIED", 1.0, period="Q1")]
    # TARGET_PAIRWISE = {FIGI_A, FIGI_B} no vacio.
    result = compute_contractual_coverage(
        {"FIGI_A", "FIGI_B"}, {"FIGI_A", "FIGI_B"},
        records_q4, records_q1)
    assert result["coverage_available"] is True


def test_p38_coverage_quality_complete_si_ambas_dimensiones_altas():
    """COMPLETE exige paired_security y paired_weighted >= 0.95."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    records_q4 = [_rec("FIGI_A", "VERIFIED", 1.0, period="Q4")]
    records_q1 = [_rec("FIGI_A", "VERIFIED", 1.0, period="Q1")]
    result = compute_contractual_coverage(
        {"FIGI_A"}, {"FIGI_A"}, records_q4, records_q1)
    assert result["coverage_quality"] == "COMPLETE"
    assert result["paired_security_coverage"] == 1.0
    assert result["paired_weighted_share_coverage"] == 1.0


def test_p38_coverage_quality_partial_si_security_coverage_baja():
    """Con security coverage < 0.95 pero weighted alto: PARTIAL, no COMPLETE."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    # TARGET_PAIRWISE = {FIGI_A, FIGI_B}, solo FIGI_A VERIFIED.
    # security = 1/2 = 0.5 < 0.95. weighted: FIGI_A peso 100, FIGI_B peso 1.
    # weighted = 100 / 101 = 0.99 >= 0.95. Debe ser PARTIAL igualmente.
    records_q4 = [
        _rec("FIGI_A", "VERIFIED", 100.0, period="Q4"),
        _rec("FIGI_B", "TEMPORAL_UNVERIFIED", 1.0, period="Q4"),
    ]
    records_q1 = [
        _rec("FIGI_A", "VERIFIED", 100.0, period="Q1"),
        _rec("FIGI_B", "TEMPORAL_UNVERIFIED", 1.0, period="Q1"),
    ]
    result = compute_contractual_coverage(
        {"FIGI_A", "FIGI_B"}, {"FIGI_A", "FIGI_B"},
        records_q4, records_q1)
    assert result["coverage_quality"] == "PARTIAL"
    assert result["paired_security_coverage"] == 0.5
    assert result["paired_weighted_share_coverage"] > 0.95


def test_p38_coverage_quality_partial_si_weighted_bajo():
    """Con weighted < 0.95 pero security >= 0.95: PARTIAL, no COMPLETE.

    Construccion: 20 FIGIs en TARGET_PAIRWISE. 19 VERIFIED en AMBOS
    trimestres (peso 1). FIGI_019 VERIFIED solo en Q4 con peso 1000
    y TEMPORAL_UNVERIFIED en Q1.
    security = 19/20 = 0.95 (threshold, pasa).
    weighted: denom incluye _w(FIGI_019) = max(1000, 0) = 1000.
    numer = 19*1 = 19. weighted = 19/1019 ~= 0.0186 < 0.95.
    """
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    targets = {"FIGI_" + str(i).zfill(3) for i in range(20)}
    records_q4 = [_rec(f, "VERIFIED", 1.0, period="Q4")
                  for f in targets if f != "FIGI_019"]
    records_q4.append(_rec("FIGI_019", "VERIFIED", 1000.0, period="Q4"))
    records_q1 = [_rec(f, "VERIFIED", 1.0, period="Q1")
                  for f in targets if f != "FIGI_019"]
    records_q1.append(_rec("FIGI_019", "TEMPORAL_UNVERIFIED", 1000.0, period="Q1"))
    result = compute_contractual_coverage(targets, targets, records_q4, records_q1)
    assert result["coverage_quality"] == "PARTIAL"
    assert result["paired_security_coverage"] == 0.95
    assert result["paired_weighted_share_coverage"] < 0.95


def test_p38_coverage_quality_no_colapsa_con_status():
    """coverage_status sigue siendo VALID aunque quality sea PARTIAL."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    records_q4 = [
        _rec("FIGI_A", "VERIFIED", 1.0, period="Q4"),
        _rec("FIGI_B", "TEMPORAL_UNVERIFIED", 1.0, period="Q4"),
    ]
    records_q1 = list(records_q4)
    result = compute_contractual_coverage(
        {"FIGI_A", "FIGI_B"}, {"FIGI_A", "FIGI_B"},
        records_q4, records_q1)
    # Legacy intacto.
    assert result["coverage_status"] == "VALID"
    # Nuevo, mas estricto.
    assert result["coverage_quality"] == "PARTIAL"


def test_p38_coverage_available_y_quality_presentes_siempre():
    """Ambos campos deben estar siempre en el dict de retorno."""
    from src.institutional_accumulation.aggregation.coverage import (
        compute_contractual_coverage)
    r1 = compute_contractual_coverage(set(), set(), [], [])
    r2 = compute_contractual_coverage(
        {"FIGI_A"}, {"FIGI_A"},
        [_rec("FIGI_A", "VERIFIED", 1.0, period="Q4")],
        [_rec("FIGI_A", "VERIFIED", 1.0, period="Q1")])
    for r in (r1, r2):
        assert "coverage_available" in r
        assert "coverage_quality" in r

