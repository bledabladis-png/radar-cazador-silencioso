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

    v1: NO emite DROP_DUP con evidencia 13F pura. La coexistencia de dos
    filings distintos (A reporting B, B reporting B) sobre la misma
    security es OVERLAP por definicion: no podemos saber si la porcion
    de A incluye o excluye la de B. El requisito 7 del contrato v3
    ("sin overlap no resuelto") NO es demostrable con 13F aislado.

    Reglas v1:
      - L3 reciproco (A->B y B->A) -> REPORTING_CONFLICT + KEEP.
      - L3 unidireccional + coexistencia -> REPORTING_OVERLAP_UNRESOLVED + KEEP.
      - Sin L3 -> KEEP silencioso (sin audit).

    DROP_DUP queda como capacidad diferida v2 (requiere evidencia
    cuantitativa externa que desambigue porcion propia vs delegada).

    Optimizacion: si l3_index esta vacio, no hay trabajo que hacer
    (ni conflictos ni overlap). Se devuelve sin iterar (fail-fast).
    Ademas, se agrupan solo las (canonical, discretion) que aparecen
    en alguna arista L3 y se agrupan solo por managers, no por pares
    de filas.
    """
    if units is None or units.empty:
        return units, _empty_audit()

    # Fast path: sin evidencia cruzada no hay nada que decidir.
    if not l3_index:
        return units, _empty_audit()

    # Precomputar el conjunto de managers implicados en alguna arista L3.
    managers_in_l3 = set()
    for (a, b) in l3_index.keys():
        managers_in_l3.add(a)
        managers_in_l3.add(b)

    # Reducir a filas candidatas (manager implicado en L3).
    cand_mask = units["filing_manager_cik"].astype(str).str.strip().isin(managers_in_l3)
    cand = units[cand_mask]
    if cand.empty:
        return units, _empty_audit()

    u = units.copy().reset_index(drop=True)
    u["_rid"] = range(len(u))

    audit_rows = []
    group_cols = ["canonical_security", "discretion_type"]

    for _, grp in u[cand_mask].groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        # Reduce a managers unicos presentes en el grupo.
        m_set = grp["filing_manager_cik"].astype(str).str.strip().unique().tolist()
        m_set = [m for m in m_set if m and m in managers_in_l3]
        if len(m_set) < 2:
            continue
        # Recorrer pares de managers, no pares de filas.
        m_set_sorted = sorted(m_set)
        for i, fm1 in enumerate(m_set_sorted):
            for fm2 in m_set_sorted[i + 1:]:
                e12 = l3_index.get((fm1, fm2))
                e21 = l3_index.get((fm2, fm1))
                if not e12 and not e21:
                    continue
                r1 = grp[grp["filing_manager_cik"].astype(str).str.strip() == fm1].iloc[0]
                if e12 and e21:
                    audit_rows.append(_make_audit_row(
                        r1, period, DEDUP_DECISION_KEEP,
                        DEDUP_REASON_REPORTING_CONFLICT,
                        EVIDENCE_LEVEL_L3, EVIDENCE_SOURCE_BOTH,
                        e12["accession_representante"],
                        e12["accession_representado"],
                        e12["reference_seq"],
                    ))
                    continue
                ev = e12 if e12 else e21
                audit_rows.append(_make_audit_row(
                    r1, period, DEDUP_DECISION_KEEP,
                    DEDUP_REASON_OVERLAP_UNRESOLVED,
                    EVIDENCE_LEVEL_L3, EVIDENCE_SOURCE_BOTH,
                    ev["accession_representante"],
                    ev["accession_representado"],
                    ev["reference_seq"],
                ))

    effective = u.drop(columns=["_rid"]).reset_index(drop=True)
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


def _build_units_index(units, canonical, discretion):
    """Index {(canonical, discretion): [(filing_manager, reporting_for), ...]}.

    Solo considera filas con reporting_for_manager_cik no None ni vacio.
    """
    if units is None or units.empty:
        return {}
    if "reporting_for_manager_cik" not in units.columns:
        return {}
    idx = {}
    for _, row in units.iterrows():
        can = row.get(canonical)
        dis = row.get(discretion)
        fm = row.get("filing_manager_cik")
        rf = row.get("reporting_for_manager_cik")
        if can is None or dis is None or fm is None or rf is None:
            continue
        fm_s = str(fm).strip()
        rf_s = str(rf).strip()
        if not fm_s or not rf_s:
            continue
        key = (str(can), str(dis))
        idx.setdefault(key, []).append((fm_s, rf_s))
    return idx


def _find_handoff_pairs(idx_q4, idx_q1, key):
    """Devuelve lista de managers con handoff demostrado.

    HANDOFF:
      Q4: (filing=A, reporting=B)
      Q1: (filing=B, reporting=B)
      con A != B.

    Sin ambiguedad: exactamente 1 par (A, B) valido. Si Q4 tiene dos
    representantes distintos (A y D) reclamando representar a B, hay
    ambiguedad y se devuelve [] (fail-closed).
    """
    q4 = idx_q4.get(key, [])
    q1 = idx_q1.get(key, [])
    if not q4 or not q1:
        return []

    valid_pairs = set()
    for fm_a, rf_b in q4:
        for fm_b, rf_b2 in q1:
            if rf_b != rf_b2:
                continue
            if fm_b != rf_b2:
                continue
            if fm_a == fm_b:
                continue
            # Q4: A reporta para B. Q1: B reporta para si mismo.
            valid_pairs.add((fm_a, fm_b, rf_b))

    if len(valid_pairs) == 1:
        return [list(valid_pairs)[0][2]]
    return []


def classify_reporting_transition(
    delta_df,
    effective_q4,
    effective_q1,
):
    """POST-delta (Commit 3).

    Anade columnas reporting_transition + dedup_reason.
    NO modifica match_status ni delta_shares.

    Regla HANDOFF (P65 v3 contrato seccion 14.5):
      Q4: filing_manager=A, reporting_for_manager=B
      Q1: filing_manager=B, reporting_for_manager=B
      mismo canonical_security, mismo discretion_type.
      A != B.

    Sin ambiguedad: solo si hay exactamente 1 B que cumple.
    Si falta reporting_for_manager_cik -> NULL (fail-closed).
    """
    if delta_df is None:
        return pd.DataFrame()
    out = delta_df.copy()
    out["reporting_transition"] = TRANSITION_NULL
    out["dedup_reason"] = None
    if out.empty:
        return out

    idx_q4 = _build_units_index(effective_q4, "canonical_security", "discretion_type")
    idx_q1 = _build_units_index(effective_q1, "canonical_security", "discretion_type")

    for i, row in out.iterrows():
        ms = row.get("match_status")
        if ms not in ("EXIT", "NEW"):
            continue
        can = row.get("canonical_security")
        dis = row.get("discretion_type")
        if can is None or dis is None:
            continue
        key = (str(can), str(dis))
        handoffs = _find_handoff_pairs(idx_q4, idx_q1, key)
        if not handoffs:
            continue
        out.at[i, "reporting_transition"] = TRANSITION_HANDOFF
        out.at[i, "dedup_reason"] = DEDUP_REASON_POSITION_HANDOFF

    return out


# ============================================================================
# P66 - determinacion del filing efectivo y evidencia R4 (§14.3 reformulada)
# ----------------------------------------------------------------------------
# Ciclo P66 (GO CONTRACTUAL #40). Implementa la seccion 14.3 del contrato
# NIPC_CONTRATOS_SEMANTICOS_v1.md.
#
# ALCANCE:
#   14.3.6 canonicalizacion Form13FFileNumber
#   14.3.5 mapping FormNum -> CIK
#   14.3.1 R3(B, period) tri-state (PASO 0 + PASO 1-4)
#   14.3.2 validacion de cadena de amendments
#   14.3.3 semantica RESTATEMENT / NEW HOLDINGS
#   14.3.4 R4 candidate_A + CONFLICT > MATCH + N/D != NO_MATCH
#   L3 booleano explicito (R1 AND R2 AND R3==TRUE AND R4==MATCH AND R5)
#
# NO TOCA:
#   - Logica P65 existente (classify_evidence_level, _apply_intra_period_dedup,
#     build_effective_reporting_snapshot, classify_reporting_transition).
#   - DROP_DUP. Sigue diferido v2. R4 por si solo no autoriza DROP_DUP.
# ============================================================================


# --- Estados contractuales 14.3 ---

R3_TRUE = "TRUE"
R3_FALSE = "FALSE"
R3_ND = "N/D"

R4_MATCH = "MATCH"
R4_NO_MATCH = "NO_MATCH"
R4_ND = "N/D"
R4_CONFLICT = "CONFLICT"

FORMNUM_STATUS_IDENTITY_RESOLVED = "IDENTITY_RESOLVED"
FORMNUM_STATUS_UNRESOLVED = "UNRESOLVED"
FORMNUM_STATUS_CONFLICT = "CONFLICT"

AMENDMENT_TYPE_RESTATEMENT = "RESTATEMENT"
AMENDMENT_TYPE_NEW_HOLDINGS = "NEW_HOLDINGS"
ALL_AMENDMENT_TYPES = (AMENDMENT_TYPE_RESTATEMENT, AMENDMENT_TYPE_NEW_HOLDINGS)


# --- 14.3.6 Canonicalizacion Form13FFileNumber ---

FORMNUM_PREFIX_CANONICAL = "028"
FORMNUM_VALID_PREFIXES = frozenset(("28", "028"))
FORMNUM_SUFFIX_WIDTH = 5


def canonicalize_form13f_filenumber(value):
    """14.3.6. Canonicaliza un Form 13F File Number.

    Regla IAE (correspondencia observada con COVERPAGE, no formato SEC
    universal):
      <prefijo>-<sufijo>
      prefijo in {"28", "028"}  -> "028"
      cualquier otro prefijo    -> None (UNRESOLVED)
      sufijo: strip leading zeros;
              si len(sufijo_sin_zeros) > 5 -> None (no truncar)
              sino -> padding a 5 digitos

    Devuelve str canonico o None si UNRESOLVED.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or "-" not in s:
        return None
    parts = s.split("-")
    if len(parts) != 2:
        return None
    prefijo = parts[0].strip()
    sufijo = parts[1].strip()
    if prefijo not in FORMNUM_VALID_PREFIXES:
        return None
    if not prefijo.isdigit() or not sufijo.isdigit():
        return None
    sufijo_sin_zeros = sufijo.lstrip("0") or "0"
    if len(sufijo_sin_zeros) > FORMNUM_SUFFIX_WIDTH:
        return None
    return "{0}-{1}".format(
        FORMNUM_PREFIX_CANONICAL,
        sufijo_sin_zeros.zfill(FORMNUM_SUFFIX_WIDTH),
    )


# --- 14.3.1 PASO 1: familia documental ---

FILING_FAMILY_NOTICE = "NOTICE"
FILING_FAMILY_COMBINATION = "COMBINATION"
ALL_FILING_FAMILIES = (FILING_FAMILY_NOTICE, FILING_FAMILY_COMBINATION)

FILING_ROLE_BASE = "BASE"
FILING_ROLE_AMENDMENT = "AMENDMENT"
ALL_FILING_ROLES = (FILING_ROLE_BASE, FILING_ROLE_AMENDMENT)


def classify_filing_family(submissiontype, reporttype):
    """14.3.1 PASO 1. Familia documental (NOTICE / COMBINATION).

    NOTICE:       SUBMISSIONTYPE in {13F-NT, 13F-NT/A} AND REPORTTYPE = 13F NOTICE
    COMBINATION:  SUBMISSIONTYPE in {13F-HR, 13F-HR/A} AND REPORTTYPE = 13F COMBINATION REPORT

    Devuelve "NOTICE" / "COMBINATION" / None.
    """
    st = (str(submissiontype or "")).strip().upper()
    rt = (str(reporttype or "")).strip().upper()
    if st in ("13F-NT", "13F-NT/A") and rt == "13F NOTICE":
        return FILING_FAMILY_NOTICE
    if st in ("13F-HR", "13F-HR/A") and rt == "13F COMBINATION REPORT":
        return FILING_FAMILY_COMBINATION
    return None


def classify_filing_role(submissiontype):
    """14.3.1 PASO 2. Rol BASE / AMENDMENT por SUBMISSIONTYPE.

    ISAMENDMENT es SOLO control de coherencia; no se usa aqui.
    """
    st = (str(submissiontype or "")).strip().upper()
    if st in ("13F-NT", "13F-HR"):
        return FILING_ROLE_BASE
    if st in ("13F-NT/A", "13F-HR/A"):
        return FILING_ROLE_AMENDMENT
    return None


# --- 14.3.5 Mapping FormNum -> CIK ---

def build_formnum_cik_mapping(coverpage_df, period):
    """14.3.5. Construye mapping FormNum normalizado -> CIK.

    Devuelve dict:
        {formnum_canonico: {
            "status": "IDENTITY_RESOLVED" | "CONFLICT",
            "cik": str | None,
            "ciks": list[str],
            "source_accessions": list[str],
        }}
    """
    out = {}
    if coverpage_df is None or coverpage_df.empty:
        return out
    df = coverpage_df.copy()
    for col in ("FORM13FFILENUMBER", "CIK", "PERIODOFREPORT"):
        if col not in df.columns:
            return out
    df = df[df["PERIODOFREPORT"].astype(str).str.strip() == str(period).strip()]
    df = df[df["FORM13FFILENUMBER"].notna() & df["CIK"].notna()]
    groups = {}
    for _, row in df.iterrows():
        canon = canonicalize_form13f_filenumber(row.get("FORM13FFILENUMBER"))
        if canon is None:
            continue
        cik = str(row.get("CIK")).strip()
        if not cik:
            continue
        acc = row.get("ACCESSION_NUMBER")
        groups.setdefault(canon, {"ciks": set(), "accessions": set()})
        groups[canon]["ciks"].add(cik)
        if acc is not None:
            groups[canon]["accessions"].add(str(acc))
    for canon, g in groups.items():
        ciks = sorted(g["ciks"])
        if len(ciks) == 1:
            out[canon] = {
                "status": FORMNUM_STATUS_IDENTITY_RESOLVED,
                "cik": ciks[0],
                "ciks": ciks,
                "source_accessions": sorted(g["accessions"]),
            }
        else:
            out[canon] = {
                "status": FORMNUM_STATUS_CONFLICT,
                "cik": None,
                "ciks": ciks,
                "source_accessions": sorted(g["accessions"]),
            }
    return out


def resolve_formnum_to_cik(formnum, mapping):
    """14.3.5. Resuelve un FormNum a su CIK contractual.

    Devuelve (status, cik):
        ("IDENTITY_RESOLVED", cik)
        ("UNRESOLVED", None)   FormNum no canonicalizable o ausente
        ("CONFLICT", None)     resuelve a >1 CIK
    """
    canon = canonicalize_form13f_filenumber(formnum)
    if canon is None:
        return (FORMNUM_STATUS_UNRESOLVED, None)
    entry = (mapping or {}).get(canon)
    if entry is None:
        return (FORMNUM_STATUS_UNRESOLVED, None)
    if entry["status"] == FORMNUM_STATUS_CONFLICT:
        return (FORMNUM_STATUS_CONFLICT, None)
    return (entry["status"], entry.get("cik"))


# --- 14.3.2 Validacion de cadena de amendments ---

def validate_amendment_chain(amendments):
    """14.3.2. Valida la cadena de amendments de una base.

    Reglas (fail-closed):
        - AMENDMENTNO entero 1..99
        - AMENDMENTTYPE in {RESTATEMENT, NEW_HOLDINGS}
        - numeros unicos
        - secuencia sin huecos desde 1 hasta N

    Devuelve "OK" si la cadena es valida (incluida vacia) o "N/D".
    """
    if amendments is None:
        return "OK"
    items = list(amendments)
    if not items:
        return "OK"
    nums = []
    for a in items:
        raw = a.get("AMENDMENTNO")
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return "N/D"
        if n < 1 or n > 99:
            return "N/D"
        atype = str(a.get("AMENDMENTTYPE") or "").strip().upper()
        if atype not in ALL_AMENDMENT_TYPES:
            return "N/D"
        nums.append(n)
    if len(nums) != len(set(nums)):
        return "N/D"
    if sorted(nums) != list(range(1, len(nums) + 1)):
        return "N/D"
    return "OK"


# --- 14.3.1 R3(B, period) tri-state ---

def resolve_r3(filings, *, period, scope_completeness_verified):
    """14.3.1. Determina R3(B, period) tri-state.

    filings: iterable de dicts con claves:
        PERIODOFREPORT, SUBMISSIONTYPE, REPORTTYPE,
        AMENDMENTNO, AMENDMENTTYPE

    scope_completeness_verified: bool. Corresponde al PASO 0.
        False -> R3 = N/D inmediatamente.
        True  -> procede a PASO 1-4.

    Devuelve "TRUE" / "FALSE" / "N/D".
    """
    # PASO 0: completitud del scope.
    if not scope_completeness_verified:
        return R3_ND

    # PASO 1: filtrar filings R4 del periodo.
    r4_filings = []
    for f in (filings or []):
        p = str(f.get("PERIODOFREPORT") or "").strip()
        if p != str(period).strip():
            continue
        fam = classify_filing_family(
            f.get("SUBMISSIONTYPE"), f.get("REPORTTYPE"),
        )
        if fam is None:
            continue
        r4_filings.append((fam, f))

    if not r4_filings:
        return R3_FALSE

    # PASO 2: clasificar por rol.
    bases = []
    amendments = []
    families = set()
    for fam, f in r4_filings:
        families.add(fam)
        rol = classify_filing_role(f.get("SUBMISSIONTYPE"))
        if rol == FILING_ROLE_BASE:
            bases.append((fam, f))
        elif rol == FILING_ROLE_AMENDMENT:
            amendments.append((fam, f))
        else:
            return R3_ND

    # PASO 3: pluralidad de familias documentales.
    if len(families) > 1:
        return R3_ND

    # PASO 4: conteo y validacion.
    if len(bases) == 0 and len(amendments) == 0:
        return R3_FALSE
    if len(bases) == 0 and len(amendments) >= 1:
        return R3_ND
    if len(bases) > 1:
        return R3_ND

    # BASE == 1: validar cadena de amendments.
    chain = [
        {"AMENDMENTNO": a.get("AMENDMENTNO"),
         "AMENDMENTTYPE": a.get("AMENDMENTTYPE")}
        for _, a in amendments
    ]
    if validate_amendment_chain(chain) != "OK":
        return R3_ND
    return R3_TRUE


# --- 14.3.4 R4(A, B) estados y candidate_A ---

def _row_identity_state(cik_r, fn_status, fn_cik, fn_ciks, a_cik):
    """Estado interno de una fila OTHERMANAGER respecto de A.

    Devuelve tupla (is_candidate, identity_ok, has_conflict):
        is_candidate: candidate_A(r)
        identity_ok:  IDENTITY_RESOLVED segun 0.3
        has_conflict: CONFLICT local (INCONSISTENT o FormNum->>1CIK sobre A)
    """
    is_candidate = (cik_r == a_cik) or (a_cik in fn_ciks)
    inconsistent = (
        bool(cik_r)
        and fn_status == FORMNUM_STATUS_IDENTITY_RESOLVED
        and fn_cik != cik_r
    )
    conflict_fn = (
        fn_status == FORMNUM_STATUS_CONFLICT and a_cik in fn_ciks
    )
    identity_ok = (
        (bool(cik_r) and (fn_status is None or fn_cik == cik_r))
        or (not cik_r and fn_status == FORMNUM_STATUS_IDENTITY_RESOLVED)
    )
    return is_candidate, identity_ok, (inconsistent or conflict_fn)


def resolve_r4(a_cik, othermanager_rows, formnum_mapping):
    """14.3.4. Determina R4(A, B).

    a_cik: CIK del manager A.
    othermanager_rows: iterable de dicts con claves CIK, FORM13FFILENUMBER
        (ACCESSION_NUMBER opcional). Es el contenido de OTHERMANAGER del
        filing efectivo de B.
    formnum_mapping: dict producido por build_formnum_cik_mapping.

    Devuelve "MATCH" / "NO_MATCH" / "N/D" / "CONFLICT".

    Precondicion: el filing efectivo B ya ha sido determinado por R3.
    Si R3 != TRUE, el caller no debe invocar esta funcion.
    """
    a_cik_s = str(a_cik or "").strip()
    if not othermanager_rows:
        return R4_ND  # OTHERMANAGER ausente/vacio -> N/D

    has_match = False
    has_nd = False
    has_conflict = False

    for r in othermanager_rows:
        cik_r = str(r.get("CIK") or "").strip()
        fn_raw = r.get("FORM13FFILENUMBER")
        fn_status = None
        fn_cik = None
        fn_ciks = []

        if fn_raw is not None and str(fn_raw).strip():
            canon = canonicalize_form13f_filenumber(fn_raw)
            if canon is not None:
                entry = (formnum_mapping or {}).get(canon)
                if entry is not None:
                    fn_status = entry["status"]
                    fn_cik = entry.get("cik")
                    fn_ciks = entry.get("ciks") or []

        is_cand, identity_ok, local_conflict = _row_identity_state(
            cik_r, fn_status, fn_cik, fn_ciks, a_cik_s,
        )

        if is_cand:
            if local_conflict:
                has_conflict = True
            elif identity_ok:
                has_match = True
            else:
                has_nd = True
        else:
            # Fila no-candidate: solo puede aportar N/D si su identidad no
            # esta resuelta. Un conflicto en fila no-candidate NO contamina.
            if not identity_ok and not local_conflict:
                has_nd = True

    if has_conflict:
        return R4_CONFLICT
    if has_match:
        return R4_MATCH
    if has_nd:
        return R4_ND
    return R4_NO_MATCH


# --- 14.3 L3 booleano explicito ---

def evaluate_l3(r1, r2, r3, r4, r5):
    """14.3. L3(A, B, S, period) = R1 AND R2 AND (R3==TRUE)
    AND (R4==MATCH) AND R5.

    R3 y R4 llegan como estados string. N/D y CONFLICT NUNCA se
    convierten internamente en False; se preservan como estado.
    Si R3 != TRUE o R4 != MATCH, el resultado es False.
    """
    return (
        bool(r1)
        and bool(r2)
        and r3 == R3_TRUE
        and r4 == R4_MATCH
        and bool(r5)
    )
