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


# --- Interfaces de orquestacion (stubs de Commit 1) ---

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
    """
    raise NotImplementedError(
        "Commit 2 (pre-delta). Ver iae/P64_P65_EXPEDIENTE.md seccion 2.17."
    )


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
