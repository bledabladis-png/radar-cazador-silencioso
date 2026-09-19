"""Reporting relationships - framework 3 niveles (FA-2.3).

Dictamen FA-2.0 (2026-09-19):
  - B.1 aprobada como canonical_reporting_relationship_key.
  - Clave: (filing_manager_cik, included_manager_cik|null,
           discretion_type, report_period).
  - OTROS nombres PROHIBIDOS en codigo y docs (hasta FA-2.4):
      economic_owner_cik, owner_cik, beneficial_owner_cik.
  - OTHERMANAGER2 = provenance, NO entra en la clave.
  - Atribucion PROVISIONAL hasta que FA-2.4 determine el snapshot
    canonico del trimestre (canonicalizacion de amendments).

Grafo:
    SUBMISSION.CIK + COVERPAGE.FILINGMANAGER -> filing_manager_cik
    INFOTABLE.OTHERMANAGER -> OTHERMANAGER.OTHERMANAGER_SK
                            -> included_manager_cik
    INFOTABLE.INVESTMENTDISCRETION -> discretion_type
    OTHERMANAGER2 = provenance (no en clave)

Fuera de FA-2.3: canonicalizacion de amendments (FA-2.4),
atribucion definitiva (FA-2.3.bis), NIPC, breadth.
"""
from __future__ import annotations

import pandas as pd


PROVISIONAL_FLAG = True  # FA-2.3 siempre provisional (dictamen seccion 9)

CANONICAL_KEY_COLUMNS = (
    "filing_manager_cik",
    "included_manager_cik",
    "discretion_type",
    "report_period",
)

FORBIDDEN_TERMS = (
    "economic_owner_cik",
    "owner_cik",
    "beneficial_owner_cik",
)


def build_filing_manager_index(submission_df, coverpage_df):
    """Indice ACCESSION -> filing_manager_cik.

    Fuente: SUBMISSION.CIK (filer CIK). Un accession, un filing manager.
    Devuelve DataFrame con columnas: ACCESSION_NUMBER, filing_manager_cik.
    """
    if "ACCESSION_NUMBER" not in submission_df.columns:
        raise KeyError("SUBMISSION sin ACCESSION_NUMBER")
    if "CIK" not in submission_df.columns:
        raise KeyError("SUBMISSION sin CIK")
    idx = submission_df[["ACCESSION_NUMBER", "CIK"]].copy()
    idx = idx.rename(columns={"CIK": "filing_manager_cik"})
    idx["filing_manager_cik"] = idx["filing_manager_cik"].astype("string").str.strip()
    idx = idx.drop_duplicates(subset=["ACCESSION_NUMBER"], keep="first")
    return idx.reset_index(drop=True)


def build_included_manager_index(othermanager_df):
    """Indice (ACCESSION, OTHERMANAGER_SK) -> included_manager_cik.

    Fuente: OTHERMANAGER. FK desde INFOTABLE.OTHERMANAGER.
    Devuelve DataFrame con columnas:
      ACCESSION_NUMBER, OTHERMANAGER_SK, included_manager_cik.
    """
    required = ("ACCESSION_NUMBER", "OTHERMANAGER_SK", "CIK")
    for col in required:
        if col not in othermanager_df.columns:
            raise KeyError("OTHERMANAGER sin columna " + col)
    idx = othermanager_df[["ACCESSION_NUMBER", "OTHERMANAGER_SK", "CIK"]].copy()
    idx = idx.rename(columns={"CIK": "included_manager_cik"})
    idx["OTHERMANAGER_SK"] = pd.to_numeric(idx["OTHERMANAGER_SK"], errors="coerce").astype("Int64")
    idx["included_manager_cik"] = idx["included_manager_cik"].astype("string").str.strip()
    idx = idx.dropna(subset=["OTHERMANAGER_SK"])
    idx = idx.drop_duplicates(subset=["ACCESSION_NUMBER", "OTHERMANAGER_SK"], keep="first")
    return idx.reset_index(drop=True)


def build_provenance_index(othermanager2_df):
    """Indice de provenance (OTHERMANAGER2). NO entra en la clave canonica.

    Devuelve DataFrame: ACCESSION_NUMBER, SEQUENCENUMBER, supermanager_cik.
    Uso exclusivo: trazabilidad / reconstruccion del grafo.
    """
    required = ("ACCESSION_NUMBER", "SEQUENCENUMBER", "CIK")
    for col in required:
        if col not in othermanager2_df.columns:
            raise KeyError("OTHERMANAGER2 sin columna " + col)
    idx = othermanager2_df[["ACCESSION_NUMBER", "SEQUENCENUMBER", "CIK"]].copy()
    idx = idx.rename(columns={"CIK": "supermanager_cik"})
    idx["SEQUENCENUMBER"] = pd.to_numeric(idx["SEQUENCENUMBER"], errors="coerce").astype("Int64")
    idx["supermanager_cik"] = idx["supermanager_cik"].astype("string").str.strip()
    idx = idx.dropna(subset=["SEQUENCENUMBER"])
    idx = idx.drop_duplicates(subset=["ACCESSION_NUMBER", "SEQUENCENUMBER"], keep="first")
    return idx.reset_index(drop=True)


def assign_canonical_key(infotable_df, filing_idx, included_idx, *, report_period):
    """Anade columnas de atribucion canonica (PROVISIONAL).

    infotable_df: DataFrame con INFOTABLE.
    filing_idx: salida de build_filing_manager_index.
    included_idx: salida de build_included_manager_index.
    report_period: str|Timestamp, periodo canonico.

    Devuelve copia con columnas:
      filing_manager_cik
      included_manager_cik
      discretion_type
      report_period
      canonical_reporting_relationship_key
      attribution_status = "PROVISIONAL"
    """
    if "ACCESSION_NUMBER" not in infotable_df.columns:
        raise KeyError("INFOTABLE sin ACCESSION_NUMBER")
    if "OTHERMANAGER" not in infotable_df.columns:
        raise KeyError("INFOTABLE sin OTHERMANAGER")
    if "INVESTMENTDISCRETION" not in infotable_df.columns:
        raise KeyError("INFOTABLE sin INVESTMENTDISCRETION")

    period_str = str(pd.Timestamp(report_period).date())

    out = infotable_df.copy()
    out["_OTHERMANAGER_SK"] = pd.to_numeric(out["OTHERMANAGER"], errors="coerce").astype("Int64")

    out = out.merge(
        filing_idx, on="ACCESSION_NUMBER", how="left"
    )
    out = out.merge(
        included_idx,
        left_on=["ACCESSION_NUMBER", "_OTHERMANAGER_SK"],
        right_on=["ACCESSION_NUMBER", "OTHERMANAGER_SK"],
        how="left",
    )

    out["discretion_type"] = out["INVESTMENTDISCRETION"].astype("string").str.strip()
    out["report_period"] = period_str

    # Canonical key: tupla serializada a string (para agrupar/dedup).
    def _mk_key(row):
        f = row["filing_manager_cik"]
        i = row["included_manager_cik"]
        d = row["discretion_type"]
        p = row["report_period"]
        f_s = "" if pd.isna(f) else str(f)
        i_s = "" if pd.isna(i) else str(i)
        d_s = "" if pd.isna(d) else str(d)
        return f_s + "|" + i_s + "|" + d_s + "|" + p

    out["canonical_reporting_relationship_key"] = out.apply(_mk_key, axis=1)
    out["attribution_status"] = "PROVISIONAL"

    out = out.drop(columns=["_OTHERMANAGER_SK", "OTHERMANAGER_SK"], errors="ignore")
    return out.reset_index(drop=True)


def assert_no_forbidden_terms():
    """Verifica que el modulo no contiene terminos prohibidos por dictamen."""
    import inspect
    src = inspect.getsource(__import__(__name__, fromlist=["x"]))
    lower = src.lower()
    for term in FORBIDDEN_TERMS:
        if term in lower:
            # Auto-exclusion: la lista blanca de la constante misma.
            occurrences = lower.count(term)
            if occurrences > 1:
                raise AssertionError("Termino prohibido en codigo: " + term)
    return True
