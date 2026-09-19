"""Filtro temporal canonico por SUBMISSION.PERIODOFREPORT.

FA-2.1 - Dictamen auditor FA-2.0 (2026-09-19):
  - Campo canonico: SUBMISSION.PERIODOFREPORT.
  - REPORTCALENDARORQUARTER es control de coherencia, no sustituto.
  - NO introducir: CUSIP resolver, manager graph, NIPC, economic owner.

Contrato:
  filter_by_period(dfs, period="2026-03-31") -> dict[str, DataFrame]
    - Filtra los 7 TSVs al periodo indicado.
    - Propagacion por ACCESSION_NUMBER.
    - Un ACCESSION_NUMBER solo se conserva si su filing esta en el
      periodo.
"""
from __future__ import annotations

import pandas as pd


CANONICAL_PERIOD_FIELD = "PERIODOFREPORT"
COHERENCE_PERIOD_FIELD = "REPORTCALENDARORQUARTER"
FULL_PERIOD = "2026-03-31"
TABLES_WITH_ACCESSION = (
    "SUBMISSION", "COVERPAGE", "SUMMARYPAGE",
    "OTHERMANAGER", "OTHERMANAGER2", "SIGNATURE", "INFOTABLE",
)


def _normalize_period(period):
    """Convierte el periodo a Timestamp tz-naive normalizado."""
    try:
        ts = pd.Timestamp(period)
    except Exception as exc:
        raise ValueError("period invalido: " + repr(period)) from exc
    return ts.normalize()


def _compute_stats(sub, mask, period):
    """Estadisticas del filtro. Requeridas por el dictamen."""
    sub_in = sub.loc[mask]
    types = sub_in["SUBMISSIONTYPE"].value_counts(dropna=False).to_dict()
    ciks = sub_in["CIK"].nunique()
    return {
        "period": period,
        "filings_total": int(len(sub)),
        "filings_in_period": int(mask.sum()),
        "filings_outside_period": int((~mask).sum()),
        "submissiontype_counts": {str(k): int(v) for k, v in types.items()},
        "ciks_in_period": int(ciks),
    }


def filter_by_period(dfs, period=FULL_PERIOD, *, return_stats=False):
    """Filtra los 7 TSVs al periodo canonico.

    dfs: dict[str, DataFrame] con los 7 TSVs.
    period: str "YYYY-MM-DD". Default: FULL_PERIOD = "2026-03-31".
    return_stats: si True, devuelve (dfs_filtrados, stats).

    Devuelve dict[str, DataFrame] filtrado. Preserva las columnas.
    Sin imputacion. Los ACCESSION_NUMBER sin filing en periodo se
    descartan de todos los TSVs.

    Raises:
      KeyError si dfs no contiene SUBMISSION.
      ValueError si period no es parseable.
    """
    if "SUBMISSION" not in dfs:
        raise KeyError("dfs debe contener SUBMISSION")
    sub = dfs["SUBMISSION"]
    if CANONICAL_PERIOD_FIELD not in sub.columns:
        raise KeyError("SUBMISSION sin columna " + CANONICAL_PERIOD_FIELD)

    period_norm = _normalize_period(period)
    period_field = sub[CANONICAL_PERIOD_FIELD]
    if not pd.api.types.is_datetime64_any_dtype(period_field):
        period_field = pd.to_datetime(period_field, errors="coerce")
    mask = period_field.dt.normalize() == period_norm

    accessions_keep = set(
        sub.loc[mask, "ACCESSION_NUMBER"].astype(str).tolist()
    )

    filtered = {}
    for name, df in dfs.items():
        if name not in TABLES_WITH_ACCESSION:
            filtered[name] = df.copy()
            continue
        if "ACCESSION_NUMBER" not in df.columns:
            filtered[name] = df.copy()
            continue
        if name == "SUBMISSION":
            filtered[name] = df.loc[mask].reset_index(drop=True)
            continue
        acc = df["ACCESSION_NUMBER"].astype(str)
        filtered[name] = df.loc[acc.isin(accessions_keep)].reset_index(drop=True)

    if return_stats:
        return filtered, _compute_stats(sub, mask, period)
    return filtered
