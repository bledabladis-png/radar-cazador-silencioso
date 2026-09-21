# -*- coding: utf-8 -*-
"""Tests estructurales P65 / Commit 1 (modelos + constantes + L1/L2/L3).

NO tocan produccion. Verifican contrato P65 v3 antes de implementar
los Commits 2 y 3.

Sin red. Deterministas. Sin datetime.now().
"""
from src.institutional_accumulation.aggregation import reporting_dedup as rd
from src.institutional_accumulation.sec_13f.identity import relationships as rel


# ---- constantes de transicion ----


def test_transitions_definidas():
    assert rd.TRANSITION_NULL == "NULL"
    assert rd.TRANSITION_HANDOFF == "HANDOFF"
    assert rd.TRANSITION_DUPLICATE_REMOVED == "DUPLICATE_REMOVED"
    assert rd.TRANSITION_DUPLICATION_UNRESOLVED == "DUPLICATION_UNRESOLVED"
    assert len(rd.ALL_TRANSITIONS) == 4


def test_dedup_reasons_definidos():
    assert rd.DEDUP_REASON_INTRA_PERIOD_DUP == "INTRA_PERIOD_DUP"
    assert rd.DEDUP_REASON_POSITION_HANDOFF == "POSITION_SCOPED_HANDOFF"
    assert rd.DEDUP_REASON_REPORTING_CONFLICT == "REPORTING_CONFLICT"
    assert rd.DEDUP_REASON_OVERLAP_UNRESOLVED == "REPORTING_OVERLAP_UNRESOLVED"
    assert len(rd.ALL_DEDUP_REASONS) == 4


def test_dedup_decisions_definidas():
    assert rd.DEDUP_DECISION_KEEP == "KEEP"
    assert rd.DEDUP_DECISION_DROP == "DROP_DUP"
    assert len(rd.ALL_DEDUP_DECISIONS) == 2


def test_evidence_levels_definidos():
    assert rd.EVIDENCE_LEVEL_L1 == "L1"
    assert rd.EVIDENCE_LEVEL_L2 == "L2"
    assert rd.EVIDENCE_LEVEL_L3 == "L3"
    assert len(rd.ALL_EVIDENCE_LEVELS) == 3


def test_evidence_sources_definidos():
    assert rd.EVIDENCE_SOURCE_REPRESENTANTE == "accession_representante"
    assert rd.EVIDENCE_SOURCE_REPRESENTADO == "accession_representado"
    assert rd.EVIDENCE_SOURCE_BOTH == "both"
    assert len(rd.ALL_EVIDENCE_SOURCES) == 3


# ---- frontera semantica ----


def test_forbidden_terms_no_economic_owner():
    """P65 v3: economic_owner_cik PROHIBIDO.

    Ya impuesto por relationships.py::FORBIDDEN_TERMS.
    """
    assert "economic_owner_cik" in rel.FORBIDDEN_TERMS
    assert "owner_cik" in rel.FORBIDDEN_TERMS
    assert "beneficial_owner_cik" in rel.FORBIDDEN_TERMS


def test_reporting_evidence_dataclass_no_economic_owner():
    """ReportingEvidence NO tiene campo economic_owner."""
    fields = rd.ReportingEvidence.__dataclass_fields__.keys()
    assert "economic_owner_cik" not in fields
    assert "reporting_for_manager_cik" in fields


def test_dedup_audit_columns_completas():
    """dedup_audit debe incluir las columnas obligatorias P65 v3."""
    required = {
        "source_line_id", "period", "filing_manager_cik",
        "reporting_for_manager_cik", "canonical_security",
        "dedup_decision", "dedup_reason", "evidence_level",
        "evidence_source", "evidence_accession_representante",
        "evidence_accession_representado", "evidence_reference_seq",
    }
    assert set(rd.DEDUP_AUDIT_COLUMNS) == required


# ---- classify_evidence_level ----


def test_classify_l1_solo_resolved():
    """RESOLVED sin representante -> L1."""
    level = rd.classify_evidence_level(
        filing_manager_cik="A",
        reporting_for_manager_cik="B",
        reference_status="RESOLVED",
    )
    assert level == rd.EVIDENCE_LEVEL_L1


def test_classify_l2_representante():
    """RESOLVED + accession representante -> L2."""
    level = rd.classify_evidence_level(
        filing_manager_cik="A",
        reporting_for_manager_cik="B",
        reference_status="RESOLVED",
        accession_representante="ACC_A",
    )
    assert level == rd.EVIDENCE_LEVEL_L2


def test_classify_l3_evidencia_cruzada():
    """RESOLVED + representante + representado -> L3."""
    level = rd.classify_evidence_level(
        filing_manager_cik="A",
        reporting_for_manager_cik="B",
        reference_status="RESOLVED",
        accession_representante="ACC_A",
        accession_representado="ACC_B",
    )
    assert level == rd.EVIDENCE_LEVEL_L3


def test_classify_none_si_no_resolved():
    """reference_status != RESOLVED -> None."""
    for status in ("NO_REFERENCE", "UNMAPPED_MISSING_IN_OM2",
                   "INVALID_REFERENCE_ZERO", None):
        level = rd.classify_evidence_level(
            filing_manager_cik="A",
            reporting_for_manager_cik="B",
            reference_status=status,
            accession_representante="ACC_A",
            accession_representado="ACC_B",
        )
        assert level is None, f"esperado None para status={status}"


# ---- stubs de orquestacion ----


# ---- build_effective_reporting_snapshot (Commit 2: R1) ----


def _mk_units(rows):
    """DataFrame minimo compatible con UNITS_COLUMNS para tests."""
    import pandas as pd
    defaults = {
        "report_period": "2026-03-31",
        "security_resolution_status": "CANONICAL",
        "canonical_security_kind": "CANONICAL_EQUIVALENCE",
        "operational_mapping_status": "VERIFIED",
        "sshprnamt_total": 100.0,
        "n_source_lines": 1,
    }
    full = []
    for r in rows:
        full.append({**defaults, **r})
    return pd.DataFrame(full)


def _mk_evidence(rows):
    """DataFrame de cross_filing_evidence."""
    import pandas as pd
    cols = ["representante_cik", "representado_cik",
            "accession_representante", "accession_representado",
            "reference_seq", "security_key"]
    return pd.DataFrame(rows, columns=cols)


def _mk_empty_units():
    import pandas as pd
    return pd.DataFrame(columns=[
        "report_period", "filing_manager_cik", "observed_security_key",
        "security_resolution_status", "canonical_security_kind",
        "canonical_security", "operational_mapping_status",
        "discretion_type", "sshprnamt_total", "n_source_lines",
    ])


def test_p65_sin_l3_keep_silencioso():
    """Sin evidencia cruzada L3, dos units mismo security -> KEEP ambos."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=None,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2
    assert len(eff_q1) == 0
    assert len(audit) == 0


def test_p65_l3_unidireccional_es_overlap_unresolved():
    """P65 v3: L3 unidireccional + coexistencia -> OVERLAP_UNRESOLVED + KEEP.

    v1 NO emite DROP_DUP con 13F puro: no podemos saber si la porcion
    de A incluye o excluye la de B. Fail-closed.
    """
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
    ])
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    # Ambos KEEP (no DROP).
    assert len(eff_q4) == 2
    assert len(audit) == 1
    assert audit.iloc[0]["dedup_decision"] == rd.DEDUP_DECISION_KEEP
    assert audit.iloc[0]["dedup_reason"] == rd.DEDUP_REASON_OVERLAP_UNRESOLVED
    assert audit.iloc[0]["evidence_level"] == rd.EVIDENCE_LEVEL_L3


def test_p65_l3_reciproco_reporting_conflict():
    """A->B y B->A -> ambiguedad -> KEEP ambos + REPORTING_CONFLICT."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
        ("B", "A", "ACC_B", "ACC_A", 3, "equity:AAPL"),
    ])
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2  # ambos KEEP
    assert len(audit) == 1
    assert audit.iloc[0]["dedup_decision"] == rd.DEDUP_DECISION_KEEP
    assert audit.iloc[0]["dedup_reason"] == rd.DEDUP_REASON_REPORTING_CONFLICT


def test_p65_distinta_discretion_no_interactua():
    """SOLE vs DFND -> distintos grupos, sin dedup."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "DFND"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
    ])
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2  # sin interaccion


def test_p65_misma_manager_no_dedup():
    """Mismo filing_manager_cik -> sin dedup."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL2",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
    ])
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2


def test_p65_dedup_audit_columnas():
    """El audit trail tiene exactamente las columnas obligatorias."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
    ])
    _, _, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert set(audit.columns) == set(rd.DEDUP_AUDIT_COLUMNS)


def test_p65_units_vacio_no_revienta():
    """Con units vacios, devuelve vacios sin error."""
    eff_q4, eff_q1, audit = rd.build_effective_reporting_snapshot(
        _mk_empty_units(), _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=None,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert eff_q4.empty
    assert eff_q1.empty
    assert audit.empty

# ---- classify_reporting_transition (Commit 3: HANDOFF) ----


def _mk_delta(rows):
    """DataFrame minimo compatible con DELTA_COLUMNS."""
    import pandas as pd
    defaults = {
        "filing_manager_cik": None,
        "canonical_security": None,
        "observed_security_key": None,
        "discretion_type": "SOLE",
        "sshprnamt_current": 0.0,
        "sshprnamt_previous": 0.0,
        "delta_shares": 0.0,
        "match_status": "BOTH",
    }
    full = []
    for r in rows:
        full.append({**defaults, **r})
    return pd.DataFrame(full)


def _mk_units_rf(rows):
    """units con reporting_for_manager_cik explicito."""
    import pandas as pd
    cols = ["filing_manager_cik", "canonical_security", "discretion_type",
            "reporting_for_manager_cik"]
    return pd.DataFrame(rows, columns=cols)


def test_p65_handoff_positivo():
    """Q4 (A filing, B reporting) + Q1 (B filing, B reporting) -> HANDOFF."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT", "delta_shares": -100.0},
        {"filing_manager_cik": "B", "canonical_security": "equity:AAPL",
         "match_status": "NEW", "delta_shares": 100.0},
    ])
    q4 = _mk_units_rf([
        ("A", "equity:AAPL", "SOLE", "B"),
    ])
    q1 = _mk_units_rf([
        ("B", "equity:AAPL", "SOLE", "B"),
    ])
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert out.iloc[0]["reporting_transition"] == rd.TRANSITION_HANDOFF
    assert out.iloc[1]["reporting_transition"] == rd.TRANSITION_HANDOFF
    assert out.iloc[0]["dedup_reason"] == rd.DEDUP_REASON_POSITION_HANDOFF


def test_p65_sin_reporting_for_es_null():
    """Sin reporting_for_manager_cik -> NULL."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT", "delta_shares": -100.0},
        {"filing_manager_cik": "B", "canonical_security": "equity:AAPL",
         "match_status": "NEW", "delta_shares": 100.0},
    ])
    import pandas as pd
    q4 = pd.DataFrame(columns=["filing_manager_cik", "canonical_security",
                                "discretion_type", "reporting_for_manager_cik"])
    q1 = q4.copy()
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert (out["reporting_transition"] == rd.TRANSITION_NULL).all()


def test_p65_reporting_for_cambia_null():
    """Q4 reporting=B, Q1 reporting=C -> NULL."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT"},
        {"filing_manager_cik": "C", "canonical_security": "equity:AAPL",
         "match_status": "NEW"},
    ])
    q4 = _mk_units_rf([("A", "equity:AAPL", "SOLE", "B")])
    q1 = _mk_units_rf([("C", "equity:AAPL", "SOLE", "C")])
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert (out["reporting_transition"] == rd.TRANSITION_NULL).all()


def test_p65_both_no_se_toca():
    """BOTH no debe marcarse HANDOFF."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "BOTH", "delta_shares": 10.0},
    ])
    q4 = _mk_units_rf([("A", "equity:AAPL", "SOLE", "B")])
    q1 = _mk_units_rf([("B", "equity:AAPL", "SOLE", "B")])
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert out.iloc[0]["reporting_transition"] == rd.TRANSITION_NULL


def test_p65_handoff_no_cambia_match_status_ni_delta():
    """El HANDOFF no toca match_status ni delta_shares."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT", "delta_shares": -100.0},
        {"filing_manager_cik": "B", "canonical_security": "equity:AAPL",
         "match_status": "NEW", "delta_shares": 100.0},
    ])
    q4 = _mk_units_rf([("A", "equity:AAPL", "SOLE", "B")])
    q1 = _mk_units_rf([("B", "equity:AAPL", "SOLE", "B")])
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert out.iloc[0]["match_status"] == "EXIT"
    assert out.iloc[1]["match_status"] == "NEW"
    assert out.iloc[0]["delta_shares"] == -100.0
    assert out.iloc[1]["delta_shares"] == 100.0


def test_p65_delta_vacio_no_revienta():
    """delta vacio -> devuelve con columnas nuevas."""
    import pandas as pd
    empty = pd.DataFrame()
    out = rd.classify_reporting_transition(empty, pd.DataFrame(), pd.DataFrame())
    assert "reporting_transition" in out.columns
    assert "dedup_reason" in out.columns


def test_p65_handoff_ambiguedad_null():
    """Q4 con dos managers (A->B y D->B) + Q1 con B->B -> ambiguedad -> NULL.

    Dos representantes distintos (A y D) reclaman haber reportado para el
    mismo manager B en Q4. Sin evidencia para desambiguar, fail-closed.
    """
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT"},
    ])
    q4 = _mk_units_rf([
        ("A", "equity:AAPL", "SOLE", "B"),
        ("D", "equity:AAPL", "SOLE", "B"),
    ])
    q1 = _mk_units_rf([
        ("B", "equity:AAPL", "SOLE", "B"),
    ])
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert out.iloc[0]["reporting_transition"] == rd.TRANSITION_NULL

# ---- Tests del matrix del contrato (2, 4, 5, 12, 13, 15) ----


def test_p65_matrix_4_same_network_separate_hr_keep():
    """Test 4: HR separados bajo mismo control comun -> KEEP.

    Managers bajo control comun pueden presentar 13F-HR separados.
    NO hay dedup automatico sin evidencia cruzada L3.
    """
    units = _mk_units([
        {"filing_manager_cik": "H1", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "H2", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    # Sin cross_filing_evidence: aunque "control comun" existe,
    # no hay evidencia documental L3.
    eff_q4, _, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=None,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2
    assert len(audit) == 0


def test_p65_matrix_5_resolved_sin_scope_keep():
    """Test 5: RESOLVED sin scope de security (sin evidencia L3) -> KEEP."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    # Solamente L1/L2 (sin accession_representado).
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", None, 7, "equity:AAPL"),
    ])
    eff_q4, _, _ = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    # Sin L3 no hay accion.
    assert len(eff_q4) == 2


def test_p65_matrix_12_exit_new_network_only_null():
    """Test 12: EXIT+NEW con network pero sin reporting_for -> NULL."""
    delta = _mk_delta([
        {"filing_manager_cik": "A", "canonical_security": "equity:AAPL",
         "match_status": "EXIT", "delta_shares": -100.0},
        {"filing_manager_cik": "B", "canonical_security": "equity:AAPL",
         "match_status": "NEW", "delta_shares": 100.0},
    ])
    import pandas as pd
    q4 = pd.DataFrame(columns=["filing_manager_cik", "canonical_security",
                                "discretion_type", "reporting_for_manager_cik"])
    q1 = q4.copy()
    out = rd.classify_reporting_transition(delta, q4, q1)
    assert (out["reporting_transition"] == rd.TRANSITION_NULL).all()


def test_p65_matrix_13_l3_sin_overlap_resuelto_no_drop():
    """Test 13: Column 7 -> B + B HR mismo security + overlap -> KEEP."""
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:AAPL",
         "canonical_security": "equity:AAPL", "discretion_type": "SOLE"},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:AAPL"),
    ])
    eff_q4, _, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    # Coexistencia = overlap no resuelto -> KEEP ambos.
    assert len(eff_q4) == 2
    assert audit.iloc[0]["dedup_reason"] == rd.DEDUP_REASON_OVERLAP_UNRESOLVED


def test_p65_matrix_15_partition_not_deduped():
    """Test 15: partition no resuelta -> KEEP (no DROP)."""
    # B Combination con X = 100 propio, A reports for B con X = 50.
    # Sin evidencia de si los 50 son parte de los 100 o porcion adicional.
    units = _mk_units([
        {"filing_manager_cik": "A", "observed_security_key": "cusip:X",
         "canonical_security": "equity:X", "discretion_type": "SOLE",
         "sshprnamt_total": 50.0},
        {"filing_manager_cik": "B", "observed_security_key": "cusip:X",
         "canonical_security": "equity:X", "discretion_type": "SOLE",
         "sshprnamt_total": 100.0},
    ])
    evidence = _mk_evidence([
        ("A", "B", "ACC_A", "ACC_B", 7, "equity:X"),
    ])
    eff_q4, _, audit = rd.build_effective_reporting_snapshot(
        units, _mk_empty_units(),
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    assert len(eff_q4) == 2  # KEEP ambos, no colapsar
    assert audit.iloc[0]["dedup_decision"] == rd.DEDUP_DECISION_KEEP


# ============================================================
# Tests de caracterizacion - funciones sin tests directos (2026-09-22)
# ============================================================


import pytest


@pytest.mark.parametrize(
    "submissiontype,reporttype,esperado",
    [
        ("13F-NT",   "13F NOTICE",              rd.FILING_FAMILY_NOTICE),
        ("13F-NT/A", "13F NOTICE",              rd.FILING_FAMILY_NOTICE),
        ("13F-HR",   "13F COMBINATION REPORT",  rd.FILING_FAMILY_COMBINATION),
        ("13F-HR/A", "13F COMBINATION REPORT",  rd.FILING_FAMILY_COMBINATION),
        ("13f-nt",   "13f notice",              rd.FILING_FAMILY_NOTICE),
        ("  13F-HR  ", "  13F COMBINATION REPORT  ",
         rd.FILING_FAMILY_COMBINATION),
        ("13F-HR",   "13F NOTICE",              None),
        ("13F-NT",   "13F COMBINATION REPORT",  None),
        ("13F-NT",   None,                      None),
        (None,       "13F NOTICE",              None),
        ("",         "",                        None),
        ("10-K",     "13F NOTICE",              None),
    ],
)
def test_classify_filing_family_caracterizacion(submissiontype, reporttype, esperado):
    """14.3.1 PASO 1 - familia documental (NOTICE / COMBINATION / None)."""
    assert rd.classify_filing_family(submissiontype, reporttype) == esperado


@pytest.mark.parametrize(
    "submissiontype,esperado",
    [
        ("13F-NT",   rd.FILING_ROLE_BASE),
        ("13F-HR",   rd.FILING_ROLE_BASE),
        ("13F-NT/A", rd.FILING_ROLE_AMENDMENT),
        ("13F-HR/A", rd.FILING_ROLE_AMENDMENT),
        ("13f-hr",   rd.FILING_ROLE_BASE),
        ("  13F-HR/A  ", rd.FILING_ROLE_AMENDMENT),
        ("13F-HR-A", None),
        ("10-K",     None),
        (None,       None),
        ("",         None),
    ],
)
def test_classify_filing_role_caracterizacion(submissiontype, esperado):
    """14.3.1 PASO 2 - rol BASE / AMENDMENT / None."""
    assert rd.classify_filing_role(submissiontype) == esperado


def test_evaluate_l3_true_cuando_todas_las_condiciones_se_cumplen():
    """14.3 - L3 True cuando R1 AND R2 AND (R3==TRUE) AND (R4==MATCH) AND R5."""
    assert rd.evaluate_l3(
        True, True, rd.R3_TRUE, rd.R4_MATCH, True,
    ) is True


@pytest.mark.parametrize(
    "r1,r2,r3,r4,r5",
    [
        (False, True,  rd.R3_TRUE, rd.R4_MATCH,  True),   # R1 falla
        (True,  False, rd.R3_TRUE, rd.R4_MATCH,  True),   # R2 falla
        (True,  True,  'N/D',      rd.R4_MATCH,  True),   # R3 N/D
        (True,  True,  'CONFLICT', rd.R4_MATCH,  True),   # R3 CONFLICT
        (True,  True,  'FALSE',    rd.R4_MATCH,  True),   # R3 FALSE
        (True,  True,  rd.R3_TRUE, 'N/D',        True),   # R4 N/D
        (True,  True,  rd.R3_TRUE, 'CONFLICT',   True),   # R4 CONFLICT
        (True,  True,  rd.R3_TRUE, 'NO_MATCH',   True),   # R4 NO_MATCH
        (True,  True,  rd.R3_TRUE, rd.R4_MATCH,  False),  # R5 falla
        (False, False, rd.R3_TRUE, rd.R4_MATCH,  False),  # todo falla
    ],
)
def test_evaluate_l3_false_y_no_colapso_estados(r1, r2, r3, r4, r5):
    """14.3 - L3 False cuando cualquier R falla. N/D y CONFLICT no se
    colapsan a True (contrato: fall-closed respecto de DROP_DUP).
    """
    assert rd.evaluate_l3(r1, r2, r3, r4, r5) is False
