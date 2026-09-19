"""Reporting relationships - framework 3 niveles (FA-2.3 fix).

Dictamen FA-2.3 caracterizacion C (2026-09-19):
  - FK real: INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQUENCENUMBER.
  - Split comma-separated (1:N).
  - 6 estados de token (clasificacion congelada).
  - source_line_id preservado.
  - Multi-edge NO divide holding economicamente.
  - Sin fallback por CIK ni por nombre.
  - Atribucion PROVISIONAL hasta FA-2.4 (canonicalizacion).

Estados de token:
  RESOLVED                      SEQ numerico presente en OM2.
  NO_REFERENCE                  raw == "0" o "NONE" (campo completo).
  INVALID_REFERENCE_ZERO        token "0" embebido en lista.
  INVALID_REFERENCE_NONNUMERIC  token no parseable como int.
  INVALID_OUT_OF_DOMAIN         SEQ fuera de NUMBER(3) (> 999).
  UNMAPPED_MISSING_IN_OM2       SEQ en dominio pero ausente de OM2.

Fuera de FA-2.3: canonicalizacion de amendments (FA-2.4).
Atribucion definitiva: FA-2.3.bis. NIPC: post Gate FA-2.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd


PROVISIONAL_FLAG = True
SEQ_MAX_DOMAIN = 999  # NUMBER(3)

STATUS_RESOLVED = "RESOLVED"
STATUS_NO_REFERENCE = "NO_REFERENCE"
STATUS_INVALID_ZERO = "INVALID_REFERENCE_ZERO"
STATUS_INVALID_NONNUMERIC = "INVALID_REFERENCE_NONNUMERIC"
STATUS_INVALID_OUT_OF_DOMAIN = "INVALID_OUT_OF_DOMAIN"
STATUS_UNMAPPED_MISSING_OM2 = "UNMAPPED_MISSING_IN_OM2"

ALL_STATUSES = (
    STATUS_RESOLVED,
    STATUS_NO_REFERENCE,
    STATUS_INVALID_ZERO,
    STATUS_INVALID_NONNUMERIC,
    STATUS_INVALID_OUT_OF_DOMAIN,
    STATUS_UNMAPPED_MISSING_OM2,
)

NO_REFERENCE_LITERALS = frozenset({"0", "NONE", "NAN", "NAT", "N/A", "NA"})

CANONICAL_KEY_COLUMNS = (
    "filing_manager_cik",
    "included_manager_cik",
    "discretion_type",
    "report_period",
)

EDGE_COLUMNS = (
    "ACCESSION_NUMBER",
    "INFOTABLE_SK",
    "manager_sequence",
    "included_manager_cik",
    "reference_status",
    "raw_othermanager",
)

FORBIDDEN_TERMS = (
    "economic_owner_cik",
    "owner_cik",
    "beneficial_owner_cik",
)


def _norm_str(v):
    if pd.isna(v):
        return None
    return str(v).strip()


def build_filing_manager_index(submission_df, coverpage_df=None):
    """Indice ACCESSION -> filing_manager_cik (SUBMISSION.CIK)."""
    for col in ("ACCESSION_NUMBER", "CIK"):
        if col not in submission_df.columns:
            raise KeyError("SUBMISSION sin columna " + col)
    idx = submission_df[["ACCESSION_NUMBER", "CIK"]].copy()
    idx = idx.rename(columns={"CIK": "filing_manager_cik"})
    idx["filing_manager_cik"] = idx["filing_manager_cik"].astype("string").str.strip()
    idx = idx.drop_duplicates(subset=["ACCESSION_NUMBER"], keep="first")
    return idx.reset_index(drop=True)


def build_om2_seq_index(othermanager2_df):
    """Indice (ACCESSION, SEQ) -> CIK desde OTHERMANAGER2.

    Devuelve dict[accession_str] -> dict[int_seq] -> CIK|None.
    """
    for col in ("ACCESSION_NUMBER", "SEQUENCENUMBER"):
        if col not in othermanager2_df.columns:
            raise KeyError("OTHERMANAGER2 sin columna " + col)
    idx = defaultdict(dict)
    for acc, seq, cik in zip(
        othermanager2_df["ACCESSION_NUMBER"],
        othermanager2_df["SEQUENCENUMBER"],
        othermanager2_df["CIK"] if "CIK" in othermanager2_df.columns else [None]*len(othermanager2_df),
    ):
        if pd.isna(seq):
            continue
        try:
            seq_i = int(seq)
        except (ValueError, TypeError):
            continue
        acc_s = str(acc).strip() if pd.notna(acc) else None
        if acc_s is None:
            continue
        cik_s = str(cik).strip() if pd.notna(cik) else None
        # Si ya existe, no sobreescribir (keep first)
        if seq_i not in idx[acc_s]:
            idx[acc_s][seq_i] = cik_s
    return dict(idx)


def build_provenance_index(othermanager_df):
    """Indice de provenance desde OTHERMANAGER (SK interno).

    Uso: reconstruccion del grafo reporting-for.
    NO es la FK de Column 7 (esa va por OTHERMANAGER2).
    """
    for col in ("ACCESSION_NUMBER", "OTHERMANAGER_SK"):
        if col not in othermanager_df.columns:
            raise KeyError("OTHERMANAGER sin columna " + col)
    out = othermanager_df[["ACCESSION_NUMBER", "OTHERMANAGER_SK"]].copy()
    if "CIK" in othermanager_df.columns:
        out["reporting_for_cik"] = othermanager_df["CIK"].astype("string").str.strip()
    if "NAME" in othermanager_df.columns:
        out["reporting_for_name"] = othermanager_df["NAME"].astype("string").str.strip()
    out["OTHERMANAGER_SK"] = pd.to_numeric(out["OTHERMANAGER_SK"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["OTHERMANAGER_SK"])
    out = out.drop_duplicates(subset=["ACCESSION_NUMBER", "OTHERMANAGER_SK"], keep="first")
    return out.reset_index(drop=True)


def classify_token(raw_token, accession, om2_index):
    """Clasifica un token individual de Column 7.

    raw_token: str con el token ya separado (sin comas).
    accession: str del filing.
    om2_index: dict[accession] -> dict[seq] -> cik.

    Devuelve (status, seq_or_none, cik_or_none).
    """
    s = str(raw_token).strip()
    if s == "":
        return (STATUS_NO_REFERENCE, None, None)
    if s in NO_REFERENCE_LITERALS:
        # Token "0" embebido o literal NONE dentro de lista
        if s == "0":
            return (STATUS_INVALID_ZERO, None, None)
        return (STATUS_NO_REFERENCE, None, None)
    try:
        seq = int(s)
    except (ValueError, TypeError):
        return (STATUS_INVALID_NONNUMERIC, None, None)
    if seq < 0 or seq > SEQ_MAX_DOMAIN:
        return (STATUS_INVALID_OUT_OF_DOMAIN, None, None)
    cik = om2_index.get(str(accession), {}).get(seq)
    if cik is None and seq not in om2_index.get(str(accession), {}):
        return (STATUS_UNMAPPED_MISSING_OM2, seq, None)
    # Resuelto aunque cik sea None (manager presente sin CIK en OM2)
    return (STATUS_RESOLVED, seq, cik)


def _classify_raw_field(raw, accession, om2_index):
    """Clasifica un campo raw completo (puede tener comas).

    Devuelve (status_campo, lista_tokens).
    status_campo se usa para NO_REFERENCE global.
    lista_tokens = [(status, seq, cik), ...].
    """
    if pd.isna(raw):
        return (STATUS_NO_REFERENCE, [])
    s = str(raw).strip()
    if s == "":
        return (STATUS_NO_REFERENCE, [])
    if s.upper() in NO_REFERENCE_LITERALS:
        if s == "0":
            return (STATUS_NO_REFERENCE, [])
        return (STATUS_NO_REFERENCE, [])
    tokens = [t.strip() for t in s.split(",")]
    tokens = [t for t in tokens if t != ""]
    classified = []
    for t in tokens:
        classified.append(classify_token(t, accession, om2_index))
    return (None, classified)


def explode_othermanager_edges(infotable_df, om2_df, *, return_metrics=False):
    """Expande INFOTABLE a nivel de edge (source_line, manager).

    infotable_df: DataFrame con ACCESSION_NUMBER, INFOTABLE_SK, OTHERMANAGER.
    om2_df: DataFrame OTHERMANAGER2 con ACCESSION_NUMBER, SEQUENCENUMBER, CIK.

    Devuelve DataFrame con columnas EDGE_COLUMNS.
    Si return_metrics=True: (edges_df, metrics_dict).

    Invariantes:
      - source_line_id (ACCESSION, INFOTABLE_SK) NO se pierde.
      - NO_REFERENCE global produce 1 edge con seq=None y status NO_REFERENCE.
      - Multi-manager produce N edges (uno por token resoluble/no-resoluble).
    """
    for col in ("ACCESSION_NUMBER", "INFOTABLE_SK", "OTHERMANAGER"):
        if col not in infotable_df.columns:
            raise KeyError("INFOTABLE sin columna " + col)

    om2_index = build_om2_seq_index(om2_df)

    rows = []
    for acc, itsk, raw in zip(
        infotable_df["ACCESSION_NUMBER"],
        infotable_df["INFOTABLE_SK"],
        infotable_df["OTHERMANAGER"],
    ):
        acc_s = str(acc).strip()
        itsk_s = str(itsk).strip()
        status_field, tokens = _classify_raw_field(raw, acc_s, om2_index)
        if status_field == STATUS_NO_REFERENCE and not tokens:
            # Fila con 0 / NONE / null: 1 edge NO_REFERENCE
            rows.append((acc_s, itsk_s, None, None, STATUS_NO_REFERENCE, _norm_str(raw)))
            continue
        if not tokens:
            # No deberia ocurrir, pero por robustez
            rows.append((acc_s, itsk_s, None, None, STATUS_NO_REFERENCE, _norm_str(raw)))
            continue
        for status, seq, cik in tokens:
            rows.append((acc_s, itsk_s, seq, cik, status, _norm_str(raw)))

    edges = pd.DataFrame(rows, columns=list(EDGE_COLUMNS))

    if return_metrics:
        return edges, compute_edge_metrics(edges, infotable_df)
    return edges


def compute_edge_metrics(edges_df, infotable_df=None):
    """Metricas de cobertura y clasificacion (dictamen FA-2.3)."""
    status_counts = edges_df["reference_status"].value_counts(dropna=False).to_dict()
    metrics = {
        "edge_count_total": int(len(edges_df)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }
    if infotable_df is not None:
        source_lines = set(
            zip(
                infotable_df["ACCESSION_NUMBER"].astype(str),
                infotable_df["INFOTABLE_SK"].astype(str),
            )
        )
        edges_source_lines = set(
            zip(edges_df["ACCESSION_NUMBER"].astype(str), edges_df["INFOTABLE_SK"].astype(str))
        )
        metrics["source_line_count"] = int(len(source_lines))
        metrics["unique_source_line_id"] = int(len(edges_source_lines))
        metrics["invalid_source_line_edges"] = int(
            len(edges_source_lines - source_lines)
        )

    resolved = int((edges_df["reference_status"] == STATUS_RESOLVED).sum())
    unmapped = int((edges_df["reference_status"] == STATUS_UNMAPPED_MISSING_OM2).sum())
    invalid_nn = int((edges_df["reference_status"] == STATUS_INVALID_NONNUMERIC).sum())
    invalid_z = int((edges_df["reference_status"] == STATUS_INVALID_ZERO).sum())
    invalid_ood = int((edges_df["reference_status"] == STATUS_INVALID_OUT_OF_DOMAIN).sum())
    no_ref = int((edges_df["reference_status"] == STATUS_NO_REFERENCE).sum())

    denominator = resolved + unmapped
    metrics["resolved_edges"] = resolved
    metrics["unmapped_missing_om2"] = unmapped
    metrics["invalid_reference_non_numeric"] = invalid_nn
    metrics["invalid_reference_zero"] = invalid_z
    metrics["invalid_out_of_domain"] = invalid_ood
    metrics["no_reference"] = no_ref
    metrics["resolution_rate"] = (
        float(resolved) / float(denominator) if denominator > 0 else 0.0
    )
    return metrics


def check_edge_uniqueness(edges_df):
    """Verifica unicidad de (ACCESSION, INFOTABLE_SK, manager_sequence).

    Los NO_REFERENCE con seq=None se cuentan por separado.
    Devuelve dict con duplicados (deberia estar vacio).
    """
    sub = edges_df[edges_df["manager_sequence"].notna()].copy()
    sub["manager_sequence"] = sub["manager_sequence"].astype(int)
    key = sub[["ACCESSION_NUMBER", "INFOTABLE_SK", "manager_sequence"]]
    dups = key[key.duplicated(keep=False)]
    return {
        "duplicate_canonical_edges": int(len(dups)),
        "unique_canonical_edges": int(len(key.drop_duplicates())),
    }


def build_canonical_relationship(infotable_df, om2_df, submission_df, *,
                                  report_period, coverpage_df=None):
    """Pipeline completo: edges + filing_manager_cik + canonical key.

    Devuelve DataFrame source-line + edge + columnas canonicas:
      ACCESSION_NUMBER, INFOTABLE_SK, manager_sequence,
      included_manager_cik, reference_status, raw_othermanager,
      filing_manager_cik, discretion_type, report_period,
      canonical_reporting_relationship_key, attribution_status.
    """
    filing_idx = build_filing_manager_index(submission_df, coverpage_df)
    edges = explode_othermanager_edges(infotable_df, om2_df)

    # Anexar discretion_type (por INFOTABLE_SK)
    if "INVESTMENTDISCRETION" in infotable_df.columns:
        disc = infotable_df[["ACCESSION_NUMBER", "INFOTABLE_SK", "INVESTMENTDISCRETION"]].copy()
        disc = disc.rename(columns={"INVESTMENTDISCRETION": "discretion_type"})
        # Normalizar tipos para merge (parquet trae INFOTABLE_SK como int64;
        # edges lo tiene como string tras explode).
        disc["ACCESSION_NUMBER"] = disc["ACCESSION_NUMBER"].astype(str).str.strip()
        disc["INFOTABLE_SK"] = disc["INFOTABLE_SK"].astype(str).str.strip()
        edges["ACCESSION_NUMBER"] = edges["ACCESSION_NUMBER"].astype(str).str.strip()
        edges["INFOTABLE_SK"] = edges["INFOTABLE_SK"].astype(str).str.strip()
        edges = edges.merge(
            disc, on=["ACCESSION_NUMBER", "INFOTABLE_SK"], how="left"
        )
    else:
        edges["discretion_type"] = None

    edges = edges.merge(filing_idx, on="ACCESSION_NUMBER", how="left")

    period_str = str(pd.Timestamp(report_period).date())
    edges["report_period"] = period_str

    def _mk_key(row):
        f = row["filing_manager_cik"]
        i = row["included_manager_cik"]
        d = row["discretion_type"]
        p = row["report_period"]
        f_s = "" if pd.isna(f) else str(f)
        i_s = "" if pd.isna(i) else str(i)
        d_s = "" if pd.isna(d) else str(d)
        return f_s + "|" + i_s + "|" + d_s + "|" + p

    edges["canonical_reporting_relationship_key"] = edges.apply(_mk_key, axis=1)
    edges["attribution_status"] = "PROVISIONAL"
    return edges.reset_index(drop=True)


def compute_source_line_count(infotable_df):
    """Cuenta source_line_id unicos (ACCESSION, INFOTABLE_SK)."""
    return int(
        infotable_df[["ACCESSION_NUMBER", "INFOTABLE_SK"]]
        .drop_duplicates().shape[0]
    )
