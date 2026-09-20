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


def test_classify_reporting_transition_stub():
    """Commit 3 pendiente: la funcion existe pero NO implementada."""
    with pytest.raises(NotImplementedError):
        rd.classify_reporting_transition(
            delta_df=None, effective_q4=None, effective_q1=None,
        )


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


def test_p65_l3_una_direccion_drop_representado():
    """A representa a B sobre AAPL -> DROP B, KEEP A."""
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
    assert len(eff_q4) == 1
    assert eff_q4.iloc[0]["filing_manager_cik"] == "A"
    assert len(audit) == 1
    assert audit.iloc[0]["dedup_decision"] == rd.DEDUP_DECISION_DROP
    assert audit.iloc[0]["dedup_reason"] == rd.DEDUP_REASON_INTRA_PERIOD_DUP
    assert audit.iloc[0]["evidence_level"] == rd.EVIDENCE_LEVEL_L3
    assert audit.iloc[0]["filing_manager_cik"] == "B"


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
