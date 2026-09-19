"""DeltaShares - agregacion de shares por reported_position_unit.

Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md
(v1.2) secciones 4, 5, 6, 11.

Contrato de pureza (spec 11.3):
  - NO leen ficheros (reciben DataFrames).
  - NO escriben ficheros.
  - NO usan datetime.now() como fecha de observacion.
  - Deterministas y testeables.

Reported position unit (spec 4.1):
  (report_period, filing_manager_cik, canonical_security, discretion_type)

Reglas duras:
  - SSHPRNAMT entra UNA SOLA VEZ por source line.
  - PROHIBIDO multiplicar SSHPRNAMT por numero de edges OTHERMANAGER.
  - PROHIBIDO incluir report_period en la match key interperiodo.
  - Discretion: incluir SOLE + DFND + OTR (spec 5.2).
  - Filtro base: SSHPRNAMTTYPE == "SH" AND PUTCALL IS NULL (spec 5.1).
  - match_status in {BOTH, NEW, EXIT, UNRESOLVED_IDENTITY}.
  - NEW asume previous=0; EXIT asume current=0.
  - UNRESOLVED_IDENTITY no fabrica delta.
"""
from __future__ import annotations

import pandas as pd

MATCH_KEY = ("filing_manager_cik", "canonical_security", "discretion_type")
VALID_DISCRETIONS = ("SOLE", "DFND", "OTR")

STATUS_BOTH = "BOTH"
STATUS_NEW = "NEW"
STATUS_EXIT = "EXIT"
STATUS_UNRESOLVED_IDENTITY = "UNRESOLVED_IDENTITY"

ALL_MATCH_STATUSES = (
    STATUS_BOTH,
    STATUS_NEW,
    STATUS_EXIT,
    STATUS_UNRESOLVED_IDENTITY,
)

CANONICAL = "CANONICAL"

UNITS_COLUMNS = (
    "report_period",
    "filing_manager_cik",
    "observed_security_key",
    "security_resolution_status",
    "canonical_security_kind",
    "canonical_security",
    "discretion_type",
    "sshprnamt_total",
    "n_source_lines",
)

DELTA_COLUMNS = (
    "filing_manager_cik",
    "canonical_security",
    "observed_security_key",
    "discretion_type",
    "sshprnamt_current",
    "sshprnamt_previous",
    "delta_shares",
    "match_status",
)


def _filter_canonical(infotable_df):
    """Aplica filtro SH + PUTCALL NULL y devuelve copia con columnas minimas.

    NO filtra por discretion (eso se hace despues en la agregacion).
    """
    required = ("ACCESSION_NUMBER", "INFOTABLE_SK", "CUSIP",
                "SSHPRNAMTTYPE", "PUTCALL", "SSHPRNAMT",
                "INVESTMENTDISCRETION")
    for c in required:
        if c not in infotable_df.columns:
            raise KeyError("INFOTABLE sin columna " + c)
    df = infotable_df.copy()
    mask_sh = df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = df["PUTCALL"].isna()
    df = df[mask_sh & mask_null].copy()
    # SSHPRNAMT a float64 nullable
    df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce")
    df = df.dropna(subset=["SSHPRNAMT_f"])
    return df


def _attach_filing_manager(df, submission_df):
    """Anade filing_manager_cik via ACCESSION_NUMBER -> SUBMISSION.CIK."""
    if "ACCESSION_NUMBER" not in submission_df.columns:
        raise KeyError("SUBMISSION sin ACCESSION_NUMBER")
    if "CIK" not in submission_df.columns:
        raise KeyError("SUBMISSION sin CIK")
    idx = submission_df[["ACCESSION_NUMBER", "CIK"]].drop_duplicates(
        subset=["ACCESSION_NUMBER"], keep="first"
    ).copy()
    idx["ACCESSION_NUMBER"] = idx["ACCESSION_NUMBER"].astype(str).str.strip()
    idx["filing_manager_cik"] = idx["CIK"].astype(str).str.strip()
    out = df.copy()
    out["ACCESSION_NUMBER"] = out["ACCESSION_NUMBER"].astype(str).str.strip()
    out = out.merge(idx[["ACCESSION_NUMBER", "filing_manager_cik"]],
                    on="ACCESSION_NUMBER", how="left")
    return out


def _attach_identity(df, identity_results):
    """Anade columnas de identidad via observed_security_key (dict).

    identity_results: dict {cusip: {observed_security_key, security_resolution_status,
                                    canonical_security_kind, canonical_security}}.
    Si None, rellena con OBSERVED_ONLY / NULL canonical.
    """
    out = df.copy()
    cusips = out["CUSIP"].astype(str).str.strip()

    if not identity_results:
        out["observed_security_key"] = cusips.apply(lambda c: "cusip:" + c)
        out["security_resolution_status"] = "OBSERVED_ONLY"
        out["canonical_security_kind"] = "OBSERVED_CUSIP_ONLY"
        out["canonical_security"] = None
        return out

    def _get(c, field):
        entry = identity_results.get(c)
        if not entry:
            return None
        return entry.get(field)

    out["observed_security_key"] = cusips.apply(
        lambda c: identity_results.get(c, {}).get("observed_security_key", "cusip:" + c)
    )
    out["security_resolution_status"] = cusips.apply(
        lambda c: identity_results.get(c, {}).get(
            "security_resolution_status", "OBSERVED_ONLY")
    )
    out["canonical_security_kind"] = cusips.apply(
        lambda c: identity_results.get(c, {}).get(
            "canonical_security_kind", "OBSERVED_CUSIP_ONLY")
    )
    out["canonical_security"] = cusips.apply(
        lambda c: _get(c, "canonical_security")
    )
    return out


def compute_reported_position_units(
    infotable_df,
    submission_df,
    coverpage_df=None,
    *,
    report_period,
    identity_results=None,
):
    """Agrega SSHPRNAMT por reported_position_unit.

    Devuelve DataFrame con columnas UNITS_COLUMNS.

    Una fila por (report_period, filing_manager_cik, observed_security_key,
    discretion_type). El campo canonical_security se hereda del resolver;
    puede ser NULL.

    Prohibicion: NO multiplicar SSHPRNAMT por numero de edges
    OTHERMANAGER. La agregacion es sobre source lines del filing manager,
    no sobre los included managers del grafo.

    coverpage_df: NO usado en este ciclo (firma estable para extension).
    report_period: string "YYYY-MM-DD".
    identity_results: dict {cusip: ...} de security_identity.resolve_batch_identities.
    """
    df = _filter_canonical(infotable_df)
    df = _attach_filing_manager(df, submission_df)
    df = _attach_identity(df, identity_results)

    # normalizar discretion
    df["discretion_type"] = df["INVESTMENTDISCRETION"].astype(str).str.strip()
    df = df[df["discretion_type"].isin(VALID_DISCRETIONS)].copy()

    group_cols = [
        "filing_manager_cik",
        "observed_security_key",
        "security_resolution_status",
        "canonical_security_kind",
        "canonical_security",
        "discretion_type",
    ]

    agg = (
        df.groupby(group_cols, dropna=False)
        .agg(
            sshprnamt_total=("SSHPRNAMT_f", "sum"),
            n_source_lines=("INFOTABLE_SK", "size"),
        )
        .reset_index()
    )
    agg["report_period"] = str(report_period)
    agg = agg[list(UNITS_COLUMNS)]
    return agg.reset_index(drop=True)


def _split_canonical(units_df):
    """Separa filas con canonical_security resuelto vs no resuelto.

    Solo filas con security_resolution_status == CANONICAL participan
    en el match interperiodo. El resto va a UNRESOLVED_IDENTITY.
    """
    if units_df is None or units_df.empty:
        return pd.DataFrame(columns=UNITS_COLUMNS), pd.DataFrame(columns=UNITS_COLUMNS)
    mask = units_df["security_resolution_status"] == CANONICAL
    return units_df[mask].copy(), units_df[~mask].copy()


def _agg_by_match_key(units_df, value_name):
    """Agrupa por MATCH_KEY. Colapsa variantes observed del mismo canonical."""
    if units_df.empty:
        return pd.DataFrame(columns=list(MATCH_KEY) + [value_name])
    out = (
        units_df.groupby(list(MATCH_KEY), dropna=False)
        .agg(**{value_name: ("sshprnamt_total", "sum")})
        .reset_index()
    )
    return out


def compute_delta_shares(units_current, units_previous):
    """FULL OUTER JOIN por MATCH_KEY. Devuelve DataFrame con DELTA_COLUMNS.

    Match key: (filing_manager_cik, canonical_security, discretion_type).
    SIN report_period (spec 4.7).

    Filas con status != CANONICAL NO participan del match: se emiten
    como UNRESOLVED_IDENTITY con delta_shares = None.

    Reglas:
      - BOTH: sshprnamt_current y sshprnamt_previous observados.
      - NEW: previous = 0.
      - EXIT: current = 0.
      - UNRESOLVED_IDENTITY: delta_shares = None (no fabrica delta).
    """
    cur_canon, cur_unres = _split_canonical(units_current)
    prev_canon, prev_unres = _split_canonical(units_previous)

    cur_agg = _agg_by_match_key(cur_canon, "sshprnamt_current")
    prev_agg = _agg_by_match_key(prev_canon, "sshprnamt_previous")

    merged = pd.merge(
        cur_agg, prev_agg,
        on=list(MATCH_KEY), how="outer", indicator=True,
    )
    merged["sshprnamt_current"] = (
        pd.to_numeric(merged["sshprnamt_current"], errors="coerce")
        .fillna(0.0)
        .astype("Float64")
    )
    merged["sshprnamt_previous"] = (
        pd.to_numeric(merged["sshprnamt_previous"], errors="coerce")
        .fillna(0.0)
        .astype("Float64")
    )
    merged["delta_shares"] = (
        merged["sshprnamt_current"] - merged["sshprnamt_previous"]
    ).astype("Float64")

    def _map_status(m):
        if m == "both":
            return STATUS_BOTH
        if m == "left_only":
            return STATUS_NEW
        return STATUS_EXIT

    merged["match_status"] = merged["_merge"].apply(_map_status)
    merged = merged.drop(columns=["_merge"])
    merged["observed_security_key"] = None

    # Unresolved: por cada fila de units, emitir UNRESOLVED_IDENTITY
    unres_rows = []
    for df, side in ((cur_unres, "current"), (prev_unres, "previous")):
        if df.empty:
            continue
        for _, r in df.iterrows():
            unres_rows.append({
                "filing_manager_cik": r["filing_manager_cik"],
                "canonical_security": None,
                "observed_security_key": r["observed_security_key"],
                "discretion_type": r["discretion_type"],
                "sshprnamt_current": (
                    float(r["sshprnamt_total"]) if side == "current" else 0.0
                ),
                "sshprnamt_previous": (
                    float(r["sshprnamt_total"]) if side == "previous" else 0.0
                ),
                "delta_shares": pd.NA,
                "match_status": STATUS_UNRESOLVED_IDENTITY,
            })

    if unres_rows:
        unres_df = pd.DataFrame(unres_rows, columns=list(DELTA_COLUMNS))
        unres_df["sshprnamt_current"] = unres_df["sshprnamt_current"].astype("Float64")
        unres_df["sshprnamt_previous"] = unres_df["sshprnamt_previous"].astype("Float64")
        unres_df["delta_shares"] = unres_df["delta_shares"].astype("Float64")
        merged = pd.concat([merged, unres_df], ignore_index=True)

    merged = merged[list(DELTA_COLUMNS)]
    return merged.reset_index(drop=True)