# -*- coding: utf-8 -*-
"""Tests estructurales P65 / Commit 1 (modelos + constantes + L1/L2/L3).

NO tocan produccion. Verifican contrato P65 v3 antes de implementar
los Commits 2 y 3.

Sin red. Deterministas. Sin datetime.now().
"""
import pytest

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


def test_build_effective_reporting_snapshot_stub():
    """Commit 1: la funcion existe pero NO implementada."""
    with pytest.raises(NotImplementedError):
        rd.build_effective_reporting_snapshot(
            units_q4=None, units_q1=None,
            relationships_q4=None, relationships_q1=None,
            cross_filing_evidence=None,
            period_q4="2025-12-31", period_q1="2026-03-31",
        )


def test_classify_reporting_transition_stub():
    """Commit 1: la funcion existe pero NO implementada."""
    with pytest.raises(NotImplementedError):
        rd.classify_reporting_transition(
            delta_df=None, effective_q4=None, effective_q1=None,
        )
