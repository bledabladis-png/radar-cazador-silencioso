"""Amendments + canonical snapshot (FA-2.4).

Dictamen FA-2.4 Gate 0 (2026-09-19):
  - RESTATEMENT -> REPLACE snapshot previo.
  - NEW HOLDINGS -> ADD entradas.
  - Orden: period -> original/amendment -> AMENDMENTNO -> FILING_DATE -> ACCESSION.
  - NT/NT-A no contribuyen a holdings (solo grafo).
  - Snapshot composicional. Lineage obligatorio.
  - Sin heuristicas de resolucion.

Especificacion: INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md.
Fuera de FA-2.4: NIPC, breadth, clasificacion, cross-validation N-PORT.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd


# --- Estrategias (7) ---
STRATEGY_SINGLE_HR = "SINGLE_HR"
STRATEGY_SINGLE_NOTICE = "SINGLE_NOTICE"
STRATEGY_HR_PLUS_RESTATEMENT = "HR_PLUS_RESTATEMENT"
STRATEGY_HR_PLUS_NEW_HOLDINGS = "HR_PLUS_NEW_HOLDINGS"
STRATEGY_HR_CHAIN_RESTATEMENT = "HR_CHAIN_RESTATEMENT"
STRATEGY_HR_COMPOSITE = "HR_COMPOSITE"
STRATEGY_NOTICE_AMENDED = "NOTICE_AMENDED"
STRATEGY_REVIEW_REQUIRED = "REVIEW_REQUIRED"

ALL_STRATEGIES = (
    STRATEGY_SINGLE_HR,
    STRATEGY_SINGLE_NOTICE,
    STRATEGY_HR_PLUS_RESTATEMENT,
    STRATEGY_HR_PLUS_NEW_HOLDINGS,
    STRATEGY_HR_CHAIN_RESTATEMENT,
    STRATEGY_HR_COMPOSITE,
    STRATEGY_NOTICE_AMENDED,
    STRATEGY_REVIEW_REQUIRED,
)

# --- Status (5) ---
STATUS_CANONICAL = "CANONICAL"
STATUS_CANONICAL_COMPOSITE = "CANONICAL_COMPOSITE"
STATUS_NO_HOLDINGS = "NO_HOLDINGS"
STATUS_SOURCE_ANOMALY = "SOURCE_ANOMALY"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"

ALL_STATUSES = (
    STATUS_CANONICAL,
    STATUS_CANONICAL_COMPOSITE,
    STATUS_NO_HOLDINGS,
    STATUS_SOURCE_ANOMALY,
    STATUS_REVIEW_REQUIRED,
)

# --- Anomalias (A1..A5) ---
ANOMALY_NT_AUGMENTED = "SOURCE_ANOMALY_NT_AUGMENTED"
ANOMALY_HR_WITH_AMENDMENT_FLAGS = "SOURCE_ANOMALY_HR_WITH_AMENDMENT_FLAGS"
ANOMALY_AMBIGUOUS_ORDER = "AMBIGUOUS_AMENDMENT_ORDER"
ANOMALY_AMBIGUOUS_BASE = "AMBIGUOUS_BASE"
ANOMALY_REVIEW_REQUIRED = "REVIEW_REQUIRED"

ALL_ANOMALIES = (
    ANOMALY_NT_AUGMENTED,
    ANOMALY_HR_WITH_AMENDMENT_FLAGS,
    ANOMALY_AMBIGUOUS_ORDER,
    ANOMALY_AMBIGUOUS_BASE,
    ANOMALY_REVIEW_REQUIRED,
)

# --- Operaciones ---
OP_REPLACE = "REPLACE"
OP_ADD = "ADD"
OP_NO_HOLDINGS = "NO_HOLDINGS"

HOLDINGS_TYPES = ("13F-HR", "13F-HR/A")
NOTICE_TYPES = ("13F-NT", "13F-NT/A")
AMENDMENT_TYPES = ("13F-HR/A", "13F-NT/A")


def _norm_str(v):
    if pd.isna(v):
        return None
    return str(v).strip()


def _parse_amendment_no(v):
    if pd.isna(v):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def order_filings(sub_df, cov_df=None):
    """Devuelve SUBMISSION + AMENDMENTNO/AMENDMENTTYPE/ISAMENDMENT + _order.

    Orden canonico (dictamen P-AM.2):
      1. period (externo, agrupacion)
      2. original (0) vs amendment (1)
      3. AMENDMENTNO (NaN -> 0)
      4. FILING_DATE
      5. ACCESSION_NUMBER
    """
    for col in ("ACCESSION_NUMBER", "CIK", "PERIODOFREPORT", "FILING_DATE", "SUBMISSIONTYPE"):
        if col not in sub_df.columns:
            raise KeyError("SUBMISSION sin columna " + col)

    out = sub_df.copy()

    if cov_df is not None:
        cov_cols = [c for c in ("ACCESSION_NUMBER", "AMENDMENTNO", "AMENDMENTTYPE", "ISAMENDMENT") if c in cov_df.columns]
        out = out.merge(cov_df[cov_cols], on="ACCESSION_NUMBER", how="left")

    for col in ("AMENDMENTNO", "AMENDMENTTYPE", "ISAMENDMENT"):
        if col not in out.columns:
            out[col] = None

    out["_is_amendment"] = out["SUBMISSIONTYPE"].isin(AMENDMENT_TYPES).astype(int)
    out["_amendment_no"] = out["AMENDMENTNO"].apply(_parse_amendment_no)
    out["_amendment_no_filled"] = (
        pd.to_numeric(out["_amendment_no"], errors="coerce").fillna(0).astype(int)
    )
    out["_filing_date"] = pd.to_datetime(out["FILING_DATE"], errors="coerce")

    out = out.sort_values(
        by=["CIK", "PERIODOFREPORT", "_is_amendment", "_amendment_no_filled", "_filing_date", "ACCESSION_NUMBER"],
        kind="mergesort",
    ).reset_index(drop=True)
    out["_order"] = out.groupby(["CIK", "PERIODOFREPORT"]).cumcount()
    return out


def _classify_strategy_from_types(types):
    """types: lista de SUBMISSIONTYPE ordenada (original primero)."""
    n = len(types)
    if n == 1:
        if types[0] == "13F-HR":
            return STRATEGY_SINGLE_HR
        if types[0] == "13F-NT":
            return STRATEGY_SINGLE_NOTICE
        return STRATEGY_REVIEW_REQUIRED
    # >1 filing: primer elemento debe ser original (HR o NT)
    if types[0] == "13F-HR":
        rest = types[1:]
        if all(t == "13F-HR/A" for t in rest):
            if len(rest) == 1:
                return STRATEGY_HR_PLUS_RESTATEMENT  # o NEW_HOLDINGS, refine fuera
            return STRATEGY_HR_CHAIN_RESTATEMENT
        return STRATEGY_REVIEW_REQUIRED
    if types[0] == "13F-NT":
        if all(t == "13F-NT/A" for t in types[1:]):
            return STRATEGY_NOTICE_AMENDED
        return STRATEGY_REVIEW_REQUIRED
    return STRATEGY_REVIEW_REQUIRED


def classify_strategy(filings_group):
    """Clasifica un grupo (CIK, PERIOD) segun SUBMISSIONTYPE + AMENDMENTTYPE.

    filings_group: DataFrame ya ordenado por order_filings (o compatible).
    Devuelve una de las 7 estrategias + REVIEW_REQUIRED.
    """
    if len(filings_group) == 0:
        return STRATEGY_REVIEW_REQUIRED

    types = filings_group["SUBMISSIONTYPE"].tolist()
    base = _classify_strategy_from_types(types)

    # Refinar HR_PLUS_RESTATEMENT vs HR_PLUS_NEW_HOLDINGS segun AMENDMENTTYPE
    if base == STRATEGY_HR_PLUS_RESTATEMENT:
        if "AMENDMENTTYPE" in filings_group.columns:
            at = _norm_str(filings_group.iloc[1].get("AMENDMENTTYPE"))
            if at == "NEW HOLDINGS":
                return STRATEGY_HR_PLUS_NEW_HOLDINGS
        return STRATEGY_HR_PLUS_RESTATEMENT

    if base == STRATEGY_HR_CHAIN_RESTATEMENT:
        # HR + HR/A + HR/A: si el ultimo es NEW HOLDINGS -> HR_COMPOSITE
        if "AMENDMENTTYPE" in filings_group.columns and len(filings_group) >= 2:
            last = _norm_str(filings_group.iloc[-1].get("AMENDMENTTYPE"))
            if last == "NEW HOLDINGS":
                return STRATEGY_HR_COMPOSITE
        return STRATEGY_HR_CHAIN_RESTATEMENT

    return base


def detect_base_filing(filings_group):
    """Detecta el filing base de holdings.

    Reglas (I2):
      13F-HR -> candidato a base.
      13F-NT -> no contiene holdings.
      >1 candidato base -> AMBIGUOUS_BASE.

    Devuelve (accession|None, status).
    """
    if len(filings_group) == 0:
        return (None, STATUS_REVIEW_REQUIRED)
    hra = filings_group[filings_group["SUBMISSIONTYPE"] == "13F-HR"]
    if len(hra) == 0:
        # NT puro
        if filings_group.iloc[0]["SUBMISSIONTYPE"] in NOTICE_TYPES:
            return (None, STATUS_NO_HOLDINGS)
        return (None, STATUS_REVIEW_REQUIRED)
    if len(hra) > 1:
        return (None, "AMBIGUOUS_BASE")
    acc = str(hra.iloc[0]["ACCESSION_NUMBER"]).strip()
    return (acc, STATUS_CANONICAL)


def _build_lineage(filings_group):
    """Devuelve lista de dicts con operation y applied por filing."""
    lineage = []
    current_applied = []  # accessions que forman el snapshot tras procesar
    for _, row in filings_group.iterrows():
        acc = str(row["ACCESSION_NUMBER"]).strip()
        stype = row["SUBMISSIONTYPE"]
        atype = _norm_str(row.get("AMENDMENTTYPE"))
        ano = _parse_amendment_no(row.get("AMENDMENTNO"))

        op = None
        applied = False
        reason = ""

        if stype == "13F-HR":
            op = OP_REPLACE
            applied = True
            current_applied = [acc]
        elif stype == "13F-HR/A":
            if atype == "RESTATEMENT":
                op = OP_REPLACE
                applied = True
                current_applied = [acc]
            elif atype == "NEW HOLDINGS":
                op = OP_ADD
                applied = True
                current_applied = current_applied + [acc]
            else:
                op = OP_NO_HOLDINGS
                applied = False
                reason = "unknown_amendment_type"
        elif stype in NOTICE_TYPES:
            op = OP_NO_HOLDINGS
            applied = False
            reason = "notice_does_not_contribute_holdings"
        else:
            op = OP_NO_HOLDINGS
            applied = False
            reason = "unknown_submission_type"

        lineage.append({
            "accession": acc,
            "submission_type": stype,
            "amendment_no": ano,
            "amendment_type": atype,
            "operation": op,
            "applied": bool(applied),
            "reason": reason,
        })

    # Marcar applied=False retroactivo para accessions superados por REPLACE posterior
    final_accessions = set(current_applied)
    for entry in lineage:
        if entry["applied"] and entry["accession"] not in final_accessions:
            entry["applied"] = False
            entry["reason"] = "superseded_by_later_replace"
    return lineage, current_applied


def _detect_group_anomalies(filings_group, strategy):
    """Devuelve lista de dicts con anomalias del grupo."""
    anomalies = []

    # A3: unique (CIK, period, AMENDMENTNO)
    amendments = filings_group[filings_group["SUBMISSIONTYPE"].isin(AMENDMENT_TYPES)]
    if "AMENDMENTNO" in amendments.columns:
        nos = amendments["AMENDMENTNO"].apply(_parse_amendment_no).dropna().tolist()
        dup_nos = [n for n, c in Counter(nos).items() if c > 1]
        if dup_nos:
            anomalies.append({
                "anomaly_code": ANOMALY_AMBIGUOUS_ORDER,
                "detail": "duplicate AMENDMENTNO=" + str(sorted(dup_nos)),
            })

    # A1: NT/A con NEW HOLDINGS
    for _, row in filings_group.iterrows():
        if row["SUBMISSIONTYPE"] == "13F-NT/A":
            at = _norm_str(row.get("AMENDMENTTYPE"))
            if at == "NEW HOLDINGS":
                anomalies.append({
                    "anomaly_code": ANOMALY_NT_AUGMENTED,
                    "accession": str(row["ACCESSION_NUMBER"]).strip(),
                    "detail": "13F-NT/A with AMENDMENTTYPE=NEW HOLDINGS",
                })

    # A2: HR con flags de amendment
    for _, row in filings_group.iterrows():
        if row["SUBMISSIONTYPE"] == "13F-HR":
            ano = _parse_amendment_no(row.get("AMENDMENTNO"))
            at = _norm_str(row.get("AMENDMENTTYPE"))
            is_amd = _norm_str(row.get("ISAMENDMENT"))
            if ano is not None or at is not None or is_amd == "Y":
                anomalies.append({
                    "anomaly_code": ANOMALY_HR_WITH_AMENDMENT_FLAGS,
                    "accession": str(row["ACCESSION_NUMBER"]).strip(),
                    "detail": "13F-HR with AMENDMENTNO/AMENDMENTTYPE/ISAMENDMENT",
                })

    # A4: ambiguedad de base
    hra = filings_group[filings_group["SUBMISSIONTYPE"] == "13F-HR"]
    if len(hra) > 1:
        anomalies.append({
            "anomaly_code": ANOMALY_AMBIGUOUS_BASE,
            "detail": "multiple 13F-HR candidates for base",
        })

    # A5: review
    if strategy == STRATEGY_REVIEW_REQUIRED:
        anomalies.append({
            "anomaly_code": ANOMALY_REVIEW_REQUIRED,
            "detail": "sequence not recognized",
        })

    return anomalies


def _status_from_strategy(strategy, anomalies):
    if strategy == STRATEGY_REVIEW_REQUIRED:
        return STATUS_REVIEW_REQUIRED
    if strategy == STRATEGY_SINGLE_NOTICE:
        return STATUS_NO_HOLDINGS
    if strategy == STRATEGY_NOTICE_AMENDED:
        return STATUS_SOURCE_ANOMALY
    if strategy in (STRATEGY_HR_PLUS_NEW_HOLDINGS, STRATEGY_HR_COMPOSITE):
        return STATUS_CANONICAL_COMPOSITE
    if strategy in (STRATEGY_SINGLE_HR, STRATEGY_HR_PLUS_RESTATEMENT, STRATEGY_HR_CHAIN_RESTATEMENT):
        # si hay anomalias no bloqueantes, sigue siendo canonico
        return STATUS_CANONICAL
    return STATUS_REVIEW_REQUIRED


def apply_amendments(dfs, *, period):
    """Orquesta la canonicalizacion. Devuelve dict.

    dfs: dict con los 7 TSVs filtrados a period.
    period: str YYYY-MM-DD o Timestamp.

    Devuelve:
      canonical_snapshot  dict[str, DataFrame] con los TSVs aplicados
      applied_accessions  set[str] de accessions que forman el snapshot
      lineage             DataFrame
      anomalies           DataFrame
      per_cik_period      DataFrame
    """
    sub = dfs["SUBMISSION"]
    cov = dfs.get("COVERPAGE")

    ordered = order_filings(sub, cov)

    lineage_rows = []
    anomaly_rows = []
    per_group_rows = []
    applied_accessions_global = set()

    for (cik, per), grp in ordered.groupby(["CIK", "PERIODOFREPORT"], sort=False):
        grp = grp.sort_values("_order").reset_index(drop=True)
        strategy = classify_strategy(grp)
        line, applied_accs = _build_lineage(grp)
        anomalies = _detect_group_anomalies(grp, strategy)
        status = _status_from_strategy(strategy, anomalies)
        base_acc, base_status = detect_base_filing(grp)

        for entry in line:
            entry["CIK"] = str(cik)
            entry["report_period"] = str(pd.Timestamp(per).date())
            lineage_rows.append(entry)

        for anom in anomalies:
            anom["CIK"] = str(cik)
            anom["report_period"] = str(pd.Timestamp(per).date())
            anomaly_rows.append(anom)

        if status in (STATUS_CANONICAL, STATUS_CANONICAL_COMPOSITE):
            applied_accessions_global.update(applied_accs)

        per_group_rows.append({
            "CIK": str(cik),
            "report_period": str(pd.Timestamp(per).date()),
            "strategy": strategy,
            "status": status,
            "base_accession": base_acc,
            "base_status": base_status,
            "applied_accessions": "|".join(applied_accs),
            "applied_count": len(applied_accs),
            "filing_count": int(len(grp)),
            "amendment_count": int(grp["SUBMISSIONTYPE"].isin(AMENDMENT_TYPES).sum()),
        })

    # Construir canonical snapshot: filtrar los 7 TSVs a applied_accessions_global
    canonical_snapshot = {}
    for name, df in dfs.items():
        if "ACCESSION_NUMBER" not in df.columns:
            canonical_snapshot[name] = df.copy()
            continue
        acc = df["ACCESSION_NUMBER"].astype(str).str.strip()
        canonical_snapshot[name] = df.loc[acc.isin(applied_accessions_global)].reset_index(drop=True)

    lineage_df = pd.DataFrame(lineage_rows)
    anomalies_df = pd.DataFrame(anomaly_rows)
    per_group_df = pd.DataFrame(per_group_rows)

    return {
        "canonical_snapshot": canonical_snapshot,
        "applied_accessions": applied_accessions_global,
        "lineage": lineage_df,
        "anomalies": anomalies_df,
        "per_cik_period": per_group_df,
    }


def compute_strategy_counts(per_cik_period_df):
    """Devuelve dict estrategia -> count."""
    if per_cik_period_df.empty:
        return {}
    return per_cik_period_df["strategy"].value_counts().to_dict()


def compute_status_counts(per_cik_period_df):
    """Devuelve dict status -> count."""
    if per_cik_period_df.empty:
        return {}
    return per_cik_period_df["status"].value_counts().to_dict()
