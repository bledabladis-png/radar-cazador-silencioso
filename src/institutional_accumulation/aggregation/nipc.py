"""NIPC - Net Institutional Position Change.

Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
(v1.2) secciones 3.14, 10, 11.

Contrato de pureza (spec 11.3):
  - NO leen ficheros (reciben DataFrames).
  - NO escriben ficheros.
  - NO usan datetime.now() como fecha de observacion.
  - Deterministas y testeables.

NIPC (spec 0):
  NIPC = sum_i DeltaShares_i
  DeltaShares_i = Shares_i,t - Shares_i,t-1

Reglas duras:
  - NIPC total suma solo filas con delta observable
    (match_status in {BOTH, NEW, EXIT}).
  - UNRESOLVED_IDENTITY no contribuye al total.
  - Breakdown por discretion (SOLE + DFND + OTR) preservado.
  - Coverage pairwise obligatorio (spec 3.14): 6 metricas.
  - Status por defecto INSUFFICIENT hasta que el auditor fije
    thresholds (NIPC_COVERAGE_POLICY.md). READY solo si ambos
    thresholds se pasan y estan definidos.
"""
from __future__ import annotations

import pandas as pd

from .delta_shares import (
    STATUS_BOTH,
    STATUS_EXIT,
    STATUS_NEW,
    STATUS_UNRESOLVED_IDENTITY,
)

# --- Estados de status NIPC (spec 3.12) ---

STATUS_READY = "READY"
STATUS_INSUFFICIENT = "INSUFFICIENT"
STATUS_CONFLICT = "CONFLICT"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_TEMPORAL_UNVERIFIED = "TEMPORAL_UNVERIFIED"

ALL_NIPC_STATUSES = (
    STATUS_READY,
    STATUS_INSUFFICIENT,
    STATUS_CONFLICT,
    STATUS_AMBIGUOUS,
    STATUS_UNRESOLVED,
    STATUS_TEMPORAL_UNVERIFIED,
)

DISCRETION_TYPES = ("SOLE", "DFND", "OTR")


def _sum_delta(delta_df, mask=None):
    """Suma delta_shares ignorando NaN.

    P31: devuelve siempre float (nunca None). Si no hay filas validas,
    devuelve 0.0. La informacion de disponibilidad se expone en
    compute_nipc a traves del campo nipc_total_available.
    """
    if delta_df is None or delta_df.empty:
        return 0.0
    df = delta_df if mask is None else delta_df[mask]
    if df.empty:
        return 0.0
    s = pd.to_numeric(df["delta_shares"], errors="coerce").dropna()
    return float(s.sum()) if len(s) > 0 else 0.0


def _count_status(delta_df, status):
    if delta_df is None or delta_df.empty:
        return 0
    return int((delta_df["match_status"] == status).sum())


def compute_nipc(delta_df, *, discretion_breakdown=True):
    """Calcula NIPC total y breakdown por discretion.

    delta_df: salida de compute_delta_shares.
    Devuelve dict con:
      nipc_total
      nipc_total_available   bool: True si n_delta_observable > 0
      nipc_sole, nipc_dfnd, nipc_otr  (si discretion_breakdown)
      n_both, n_new, n_exit, n_unresolved_identity
      n_delta_observable   (BOTH + NEW + EXIT)

    P31: nipc_total sigue siendo 0.0 cuando no hay datos (compatibilidad),
    pero nipc_total_available=False distingue "no hay datos" de "neto = 0".
    """
    if delta_df is None or delta_df.empty:
        result = {
            "nipc_total": 0.0,
            "nipc_total_available": False,
            "n_both": 0,
            "n_new": 0,
            "n_exit": 0,
            "n_unresolved_identity": 0,
            "n_delta_observable": 0,
        }
        if discretion_breakdown:
            for d in DISCRETION_TYPES:
                result["nipc_" + d.lower()] = 0.0
        return result

    observable_mask = delta_df["match_status"].isin(
        [STATUS_BOTH, STATUS_NEW, STATUS_EXIT]
    )
    n_delta_observable = int(observable_mask.sum())
    result = {
        "nipc_total": _sum_delta(delta_df, observable_mask),
        "nipc_total_available": bool(n_delta_observable > 0),
        "n_both": _count_status(delta_df, STATUS_BOTH),
        "n_new": _count_status(delta_df, STATUS_NEW),
        "n_exit": _count_status(delta_df, STATUS_EXIT),
        "n_unresolved_identity": _count_status(
            delta_df, STATUS_UNRESOLVED_IDENTITY
        ),
        "n_delta_observable": n_delta_observable,
    }

    if discretion_breakdown:
        for d in DISCRETION_TYPES:
            m = observable_mask & (delta_df["discretion_type"] == d)
            result["nipc_" + d.lower()] = _sum_delta(delta_df, m)

    return result


def _mapped_mask(units_df):
    """True donde security_resolution_status == CANONICAL."""
    if units_df is None or units_df.empty:
        return pd.Series(dtype=bool)
    return units_df["security_resolution_status"] == "CANONICAL"


def compute_coverage_pairwise(units_current, units_previous):
    """6 metricas de cobertura pairwise (spec 3.14).

    Devuelve dict con:
      coverage_previous, coverage_current
      paired_security_coverage, paired_weighted_share_coverage
      unmapped_weight_previous, unmapped_weight_current

    Definicion formal (spec 3.14):
      paired_security_coverage
        = |{S : S tiene mapping en Q4 Y en Q1}|
          / |{S : S existe en Q4 O en Q1}|
      paired_weighted_share_coverage
        = sum(SSHPRNAMT de S con mapping en ambos)
          / sum(SSHPRNAMT de S con posicion en ambos)

    units_*: DataFrame con UNITS_COLUMNS (spec 11.2).
    """
    empty = {
        "coverage_previous": 0.0,
        "coverage_current": 0.0,
        "paired_security_coverage": 0.0,
        "paired_weighted_share_coverage": 0.0,
        "unmapped_weight_previous": 0.0,
        "unmapped_weight_current": 0.0,
    }
    if units_current is None or units_current.empty:
        if units_previous is None or units_previous.empty:
            return empty

    # Normalizar
    curr = units_current if units_current is not None else pd.DataFrame()
    prev = units_previous if units_previous is not None else pd.DataFrame()

    def _securities(df):
        if df.empty:
            return set()
        return set(df["observed_security_key"].dropna().astype(str))

    def _mapped_securities(df):
        if df.empty:
            return set()
        return set(
            df.loc[_mapped_mask(df), "observed_security_key"]
            .dropna().astype(str)
        )

    sec_c = _securities(curr)
    sec_p = _securities(prev)
    mapped_c = _mapped_securities(curr)
    mapped_p = _mapped_securities(prev)

    # --- coverage per periodo ---
    def _cov(mapped, all_):
        return (len(mapped) / len(all_)) if all_ else 0.0

    coverage_current = _cov(mapped_c, sec_c)
    coverage_previous = _cov(mapped_p, sec_p)

    # --- paired ---
    union_sec = sec_c | sec_p
    both_mapped_sec = mapped_c & mapped_p
    paired_sec_cov = (
        len(both_mapped_sec) / len(union_sec) if union_sec else 0.0
    )

    # --- ponderado por SSHPRNAMT ---
    def _weighted(df, subset_keys):
        if df.empty or not subset_keys:
            return 0.0
        sub = df[df["observed_security_key"].astype(str).isin(subset_keys)]
        if sub.empty:
            return 0.0
        return float(pd.to_numeric(sub["sshprnamt_total"], errors="coerce").sum())

    # para cada security, tomar SSHPRNAMT max entre periodos (evita doble conteo)
    all_units = pd.concat(
        [curr, prev], ignore_index=True,
    ) if not curr.empty or not prev.empty else pd.DataFrame()
    if not all_units.empty:
        by_sec = (
            all_units.groupby("observed_security_key", dropna=False)
            ["sshprnamt_total"]
            .apply(lambda s: float(pd.to_numeric(s, errors="coerce").max()))
            .to_dict()
        )
        denom = sum(by_sec.get(k, 0.0) for k in union_sec)
        numer = sum(by_sec.get(k, 0.0) for k in both_mapped_sec)
        paired_weighted = (numer / denom) if denom > 0 else 0.0
        unmapped_prev = 1.0 - coverage_previous
        unmapped_curr = 1.0 - coverage_current
    else:
        paired_weighted = 0.0
        unmapped_prev = 1.0
        unmapped_curr = 1.0

    return {
        "coverage_previous": coverage_previous,
        "coverage_current": coverage_current,
        "paired_security_coverage": paired_sec_cov,
        "paired_weighted_share_coverage": paired_weighted,
        "unmapped_weight_previous": unmapped_prev,
        "unmapped_weight_current": unmapped_curr,
    }


def _derive_status(coverage, *, threshold_1=None, threshold_2=None,
                   n_conflict=0, n_ambiguous=0, n_total=0):
    """Deriva status NIPC segun spec 3.12.

    Orden:
      1. n_total == 0 -> UNRESOLVED.
      2. n_conflict > 0 -> CONFLICT.
      3. n_ambiguous > 0 -> AMBIGUOUS.
      4. thresholds UNDEFINED -> INSUFFICIENT.
      5. ambos thresholds pasados -> READY.
      6. en otro caso -> INSUFFICIENT.
    """
    if n_total == 0:
        return STATUS_UNRESOLVED
    if n_conflict > 0:
        return STATUS_CONFLICT
    if n_ambiguous > 0:
        return STATUS_AMBIGUOUS
    if threshold_1 is None or threshold_2 is None:
        return STATUS_INSUFFICIENT
    if (coverage["paired_security_coverage"] >= threshold_1
            and coverage["paired_weighted_share_coverage"] >= threshold_2):
        return STATUS_READY
    return STATUS_INSUFFICIENT


def compute_nipc_and_coverage(
    delta_df,
    units_current=None,
    units_previous=None,
    *,
    threshold_1=None,
    threshold_2=None,
):
    """Wrapper: NIPC + breakdown + coverage pairwise + status derivado.

    Devuelve dict con:
      nipc_total, nipc_sole, nipc_dfnd, nipc_otr
      n_both, n_new, n_exit, n_unresolved_identity, n_delta_observable
      coverage_previous, coverage_current,
      paired_security_coverage, paired_weighted_share_coverage,
      unmapped_weight_previous, unmapped_weight_current
      status
    """
    nipc = compute_nipc(delta_df, discretion_breakdown=True)
    coverage = compute_coverage_pairwise(units_current, units_previous)

    n_total = (
        nipc["n_both"] + nipc["n_new"] + nipc["n_exit"]
        + nipc["n_unresolved_identity"]
    )
    n_conflict = 0
    n_ambiguous = 0
    if units_current is not None and not units_current.empty:
        n_conflict = int(
            (units_current["security_resolution_status"] == "CONFLICT").sum()
        )
        n_ambiguous = int(
            (units_current["security_resolution_status"] == "AMBIGUOUS").sum()
        )

    status = _derive_status(
        coverage,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        n_conflict=n_conflict,
        n_ambiguous=n_ambiguous,
        n_total=n_total,
    )

    out = dict(nipc)
    out.update(coverage)
    out["status"] = status
    return out