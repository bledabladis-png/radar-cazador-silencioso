"""P65 - Reporting dedup (Manager Duplication Contract).

F2.4 (2026-09-20) + dictamen P65 v3 GO CONDICIONADO.

Frontera semantica:
    REPORTING RELATIONSHIP  !=  REPORTING NETWORK
                            !=  DEDUP AUTHORIZATION
                            !=  ECONOMIC OWNERSHIP

Los 3 niveles de evidencia:
    L1 REPORTING_EDGE           relacion documental
    L2 POSITION_SCOPED_EDGE     relacion vinculada a security concreta
    L3 DEDUP_AUTHORIZATION      evidencia cruzada permite afirmar duplicidad

Solo L3 autoriza DROP_DUP. Sin L3 -> KEEP.

Esta capa:
  - NO toca MATCH_KEY.
  - NO toca C2.
  - NO toca delta_shares.py.
  - NO toca nipc.py.
  - NO toca coverage.py.
  - NO toca relationships.py.

Dos fases:
  - build_effective_reporting_snapshot  (PRE-delta, Commit 2)
  - classify_reporting_transition       (POST-delta, Commit 3)

Commit 1 (actual): modelos + constantes + classify_evidence_level.
Commits 2-3: implementacion completa.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# --- Transiciones de reporting (post-delta) ---

TRANSITION_NULL = "NULL"
TRANSITION_HANDOFF = "HANDOFF"
TRANSITION_DUPLICATE_REMOVED = "DUPLICATE_REMOVED"
TRANSITION_DUPLICATION_UNRESOLVED = "DUPLICATION_UNRESOLVED"

ALL_TRANSITIONS = (
    TRANSITION_NULL,
    TRANSITION_HANDOFF,
    TRANSITION_DUPLICATE_REMOVED,
    TRANSITION_DUPLICATION_UNRESOLVED,
)


# --- Razones de dedup (pre-delta) ---

DEDUP_REASON_INTRA_PERIOD_DUP = "INTRA_PERIOD_DUP"
DEDUP_REASON_POSITION_HANDOFF = "POSITION_SCOPED_HANDOFF"
DEDUP_REASON_REPORTING_CONFLICT = "REPORTING_CONFLICT"
DEDUP_REASON_OVERLAP_UNRESOLVED = "REPORTING_OVERLAP_UNRESOLVED"

ALL_DEDUP_REASONS = (
    DEDUP_REASON_INTRA_PERIOD_DUP,
    DEDUP_REASON_POSITION_HANDOFF,
    DEDUP_REASON_REPORTING_CONFLICT,
    DEDUP_REASON_OVERLAP_UNRESOLVED,
)


# --- Decisiones de dedup ---

DEDUP_DECISION_KEEP = "KEEP"
DEDUP_DECISION_DROP = "DROP_DUP"

ALL_DEDUP_DECISIONS = (
    DEDUP_DECISION_KEEP,
    DEDUP_DECISION_DROP,
)


# --- Niveles de evidencia ---

EVIDENCE_LEVEL_L1 = "L1"
EVIDENCE_LEVEL_L2 = "L2"
EVIDENCE_LEVEL_L3 = "L3"

ALL_EVIDENCE_LEVELS = (
    EVIDENCE_LEVEL_L1,
    EVIDENCE_LEVEL_L2,
    EVIDENCE_LEVEL_L3,
)


# --- Fuentes de evidencia (para dedup_audit) ---

EVIDENCE_SOURCE_REPRESENTANTE = "accession_representante"
EVIDENCE_SOURCE_REPRESENTADO = "accession_representado"
EVIDENCE_SOURCE_BOTH = "both"

ALL_EVIDENCE_SOURCES = (
    EVIDENCE_SOURCE_REPRESENTANTE,
    EVIDENCE_SOURCE_REPRESENTADO,
    EVIDENCE_SOURCE_BOTH,
)


# --- Columnas del audit trail ---

DEDUP_AUDIT_COLUMNS = (
    "source_line_id",
    "period",
    "filing_manager_cik",
    "reporting_for_manager_cik",
    "canonical_security",
    "dedup_decision",
    "dedup_reason",
    "evidence_level",
    "evidence_source",
    "evidence_accession_representante",
    "evidence_accession_representado",
    "evidence_reference_seq",
)


# --- Dataclass tipada ---

@dataclass(frozen=True)
class ReportingEvidence:
    """Evidencia de reporting entre managers para una linea concreta.

    F2.4 regla #4: estructura tipada en lugar de columnas paralelas.

    NO es economic_owner. NO es Reporting_network_id. Es una relacion
    documental de reporting con su nivel de evidencia.
    """
    filing_manager_cik: str
    reporting_for_manager_cik: Optional[str] = None
    security_key: Optional[str] = None
    accession_representante: Optional[str] = None
    accession_representado: Optional[str] = None
    reference_seq: Optional[int] = None
    level: Optional[str] = None   # L1 | L2 | L3 | None


# --- Clasificacion de evidencia (logica pura de Commit 1) ---

def classify_evidence_level(
    filing_manager_cik: Optional[str],
    reporting_for_manager_cik: Optional[str],
    reference_status: Optional[str],
    accession_representante: Optional[str] = None,
    accession_representado: Optional[str] = None,
    reference_seq: Optional[int] = None,
) -> Optional[str]:
    """Clasifica una relacion en L1 / L2 / L3 / None.

    Reglas (P65 v3 contrato seccion 14.3):
      - reference_status != RESOLVED -> None.
      - RESOLVED + accession_representado -> L3 (evidencia cruzada).
      - RESOLVED + accession_representante (sin representado) -> L2.
      - RESOLVED solo -> L1.

    Nota: un HR normal puede ser representante de otros managers. NO se
    exige que el filing sea Combination/NT.
    """
    if reference_status != "RESOLVED":
        return None
    if accession_representante and accession_representado:
        return EVIDENCE_LEVEL_L3
    if accession_representante:
        return EVIDENCE_LEVEL_L2
    return EVIDENCE_LEVEL_L1


# --- Helpers internos ---

def _empty_audit():
    return pd.DataFrame(columns=list(DEDUP_AUDIT_COLUMNS))


def _build_l3_index(cross_filing_evidence):
    """Index {(representante_cik, representado_cik): evidencia_dict}."""
    if cross_filing_evidence is None or cross_filing_evidence.empty:
        return {}
    idx = {}
    for _, row in cross_filing_evidence.iterrows():
        rep = str(row.get("representante_cik", "")).strip()
        rdo = str(row.get("representado_cik", "")).strip()
        if not rep or not rdo:
            continue
        idx[(rep, rdo)] = {
            "accession_representante": row.get("accession_representante"),
            "accession_representado": row.get("accession_representado"),
            "reference_seq": row.get("reference_seq"),
            "security_key": row.get("security_key"),
        }
    return idx


def _make_audit_row(
    unit, period, dedup_decision, dedup_reason, evidence_level,
    evidence_source, acc_rep, acc_repres, ref_seq,
):
    """Construye una fila de dedup_audit."""
    return {
        "source_line_id": (
            str(unit.get("observed_security_key", "")) + "@"
            + str(unit.get("filing_manager_cik", ""))
        ),
        "period": period,
        "filing_manager_cik": unit.get("filing_manager_cik"),
        "reporting_for_manager_cik": unit.get("filing_manager_cik"),
        "canonical_security": unit.get("canonical_security"),
        "dedup_decision": dedup_decision,
        "dedup_reason": dedup_reason,
        "evidence_level": evidence_level,
        "evidence_source": evidence_source,
        "evidence_accession_representante": acc_rep,
        "evidence_accession_representado": acc_repres,
        "evidence_reference_seq": ref_seq,
    }


def _apply_intra_period_dedup(units, l3_index, *, period):
    """Aplica R1 (7 requisitos) a un periodo.

    Devuelve (effective_units, audit_df).
    Sin L3 -> KEEP silencioso (sin audit).
    Con L3 (una sola direccion) -> DROP_DUP del representado.
    Ambiguedad reciproca -> KEEP + REPORTING_CONFLICT.
    """
    if units is None or units.empty:
        return units, _empty_audit()

    u = units.copy().reset_index(drop=True)
    u["_rid"] = range(len(u))
    u["_decision"] = DEDUP_DECISION_KEEP
    u["_reason"] = None

    audit_rows = []

    group_cols = ["canonical_security", "discretion_type"]
    for _, grp in u.groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        for i, r1 in enumerate(rows):
            for r2 in rows[i + 1:]:
                fm1 = str(r1.get("filing_manager_cik", "")).strip()
                fm2 = str(r2.get("filing_manager_cik", "")).strip()
                if not fm1 or not fm2 or fm1 == fm2:
                    continue

                e12 = l3_index.get((fm1, fm2))
                e21 = l3_index.get((fm2, fm1))

                if e12 and e21:
                    # Ambiguedad: ambos lados reclaman representacion.
                    audit_rows.append(_make_audit_row(
                        r1, period, DEDUP_DECISION_KEEP,
                        DEDUP_REASON_REPORTING_CONFLICT,
                        EVIDENCE_LEVEL_L3, EVIDENCE_SOURCE_BOTH,
                        e12["accession_representante"],
                        e12["accession_representado"],
                        e12["reference_seq"],
                    ))
                    continue

                if e12:
                    # fm1 representa a fm2 -> DROP r2.
                    u.loc[r2["_rid"], "_decision"] = DEDUP_DECISION_DROP
                    u.loc[r2["_rid"], "_reason"] = DEDUP_REASON_INTRA_PERIOD_DUP
                    audit_rows.append(_make_audit_row(
                        r2, period, DEDUP_DECISION_DROP,
                        DEDUP_REASON_INTRA_PERIOD_DUP,
                        EVIDENCE_LEVEL_L3, EVIDENCE_SOURCE_BOTH,
                        e12["accession_representante"],
                        e12["accession_representado"],
                        e12["reference_seq"],
                    ))
                    continue

                if e21:
                    u.loc[r1["_rid"], "_decision"] = DEDUP_DECISION_DROP
                    u.loc[r1["_rid"], "_reason"] = DEDUP_REASON_INTRA_PERIOD_DUP
                    audit_rows.append(_make_audit_row(
                        r1, period, DEDUP_DECISION_DROP,
                        DEDUP_REASON_INTRA_PERIOD_DUP,
                        EVIDENCE_LEVEL_L3, EVIDENCE_SOURCE_BOTH,
                        e21["accession_representante"],
                        e21["accession_representado"],
                        e21["reference_seq"],
                    ))

    effective = u[u["_decision"] == DEDUP_DECISION_KEEP].drop(
        columns=["_rid", "_decision", "_reason"]
    ).reset_index(drop=True)

    audit = pd.DataFrame(audit_rows, columns=list(DEDUP_AUDIT_COLUMNS)) if audit_rows else _empty_audit()
    return effective, audit


# --- Interfaces de orquestacion ---

def build_effective_reporting_snapshot(
    units_q4,
    units_q1,
    relationships_q4,
    relationships_q1,
    cross_filing_evidence,
    *,
    period_q4: str,
    period_q1: str,
):
    """PRE-delta (Commit 2).

    Devuelve (effective_q4, effective_q1, dedup_audit).

    Aplica dedup intra-periodo segun R1 (7 requisitos). Sin L3 -> KEEP.
    NO modifica delta_shares ni MATCH_KEY.

    - units_q4 / units_q1: salida de compute_reported_position_units.
    - relationships_q4 / relationships_q1: salida de build_canonical_relationship.
      NO usados directamente en Commit 2 (reservados para auditoria de
      contradicciones en Commit 4).
    - cross_filing_evidence: DataFrame con evidencia cruzada L3.
      Columnas: representante_cik, representado_cik, accession_representante,
      accession_representado, reference_seq, security_key.
    """
    l3_index = _build_l3_index(cross_filing_evidence)

    eff_q4, audit_q4 = _apply_intra_period_dedup(units_q4, l3_index, period=period_q4)
    eff_q1, audit_q1 = _apply_intra_period_dedup(units_q1, l3_index, period=period_q1)

    if audit_q4.empty and audit_q1.empty:
        audit = _empty_audit()
    else:
        audit = pd.concat([audit_q4, audit_q1], ignore_index=True)

    return eff_q4, eff_q1, audit


def classify_reporting_transition(
    delta_df,
    effective_q4,
    effective_q1,
):
    """POST-delta (Commit 3).

    Anade columnas reporting_transition + dedup_reason.
    NO modifica match_status ni delta_shares.
    """
    raise NotImplementedError(
        "Commit 3 (post-delta). Ver iae/P64_P65_EXPEDIENTE.md seccion 2.17."
    )
