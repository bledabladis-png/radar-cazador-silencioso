"""§5.5 - operational_universe (NIPC spec v1.4).

Dictamen habilitante: #65 (2026-09-21).
Encadena los filtros §5.1 -> §5.3 -> §5.4 -> §5.5 y materializa el
universo operativo contractual.

Invariantes contractuales (#65):
  - estrictamente restrictivo: cada etapa es incremental.
  - no duplica §5.3/§5.4: reutiliza sus funciones contractuales.
  - §5.5 consume identity_results P61 ya computados (arquitectura C).
  - PIT: identity_results debe corresponder al mismo period_iso.
  - fail-closed: ausencia, ambiguedad, conflicto o sin vigencia -> fuera.

No activa compute_nipc_contractual. No modifica security_identity.py
ni temporal_validity.py. No resuelve identidad internamente.

Contrato de pureza (spec 11.3):
  NO leen ficheros.
  NO escriben ficheros.
  NO usan datetime.now().
  Deterministas.
"""
from __future__ import annotations

from src.institutional_accumulation import security_type as st
from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl


# --- Estados contractuales de identity_results (P61) ---
RESOLUTION_CANONICAL = "CANONICAL"
MAPPING_VERIFIED = "VERIFIED"


# --- Columnas requeridas en infotable_df ---
REQUIRED_INFOTABLE_COLUMNS = (
    "CUSIP",
    "TITLEOFCLASS",
    "SSHPRNAMTTYPE",
    "PUTCALL",
)


# --- Columnas de trazabilidad anadidas a la salida ---
DIAGNOSTIC_COLUMNS = (
    "_security_type",
    "_security_type_status",
    "_elig_status",
    "_security_resolution_status",
    "_operational_mapping_status",
    "_canonical_security",
    "_canonical_security_kind",
    "_observed_security_key",
    "_ticker_mapped",
)


def _check_required_columns(df):
    missing = [c for c in REQUIRED_INFOTABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "infotable_df no contiene columnas requeridas: "
            + ", ".join(missing)
        )


def _apply_5_1(df):
    """Filtro §5.1: SSHPRNAMTTYPE == 'SH' AND PUTCALL IS NULL."""
    mask_sh = df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = df["PUTCALL"].isna()
    return df[mask_sh & mask_null].copy()


def _apply_5_3(df, official_df):
    """Filtro §5.3: section13f_eligible in {ADDED, ACTIVE}.

    Reutiliza sl.resolve_eligibility. No reimplementa.
    """
    cusips = df["CUSIP"].astype(str).unique()
    elig = sl.resolve_eligibility(cusips, official_df)
    df = df.copy()
    df["_elig_status"] = df["CUSIP"].astype(str).map(
        lambda c: elig.get(c, {}).get("status", sl.STATUS_NOT_IN_LIST)
    )
    eligible_mask = df["_elig_status"].isin(sl.ELIGIBLE_STATES)
    return df[eligible_mask].copy()


def _apply_5_4(df):
    """Filtro §5.4: security_type_status == RESOLVED_EQUITY.

    Reutiliza st.classify_title_of_class. No reimplementa.
    """
    df = df.copy()
    tm = {}
    for toc in df["TITLEOFCLASS"].unique():
        tm[toc] = st.classify_title_of_class(toc)
    df["_security_type"] = df["TITLEOFCLASS"].map(lambda x: tm[x][0])
    df["_security_type_status"] = df["TITLEOFCLASS"].map(lambda x: tm[x][1])
    mask = df["_security_type_status"] == st.STATUS_RESOLVED_EQUITY
    return df[mask].copy()


def _apply_5_5(df, identity_results):
    """Filtro §5.5: CANONICAL AND VERIFIED (contrato #65).

    Unico ticker operativo, deterministico y PIT-valido.
    Cualquier otro estado -> excluido (fail-closed).
    """
    df = df.copy()

    def _get(cusip, key):
        rec = identity_results.get(str(cusip))
        if rec is None:
            return None
        return rec.get(key)

    df["_security_resolution_status"] = df["CUSIP"].astype(str).map(
        lambda c: _get(c, "security_resolution_status")
    )
    df["_operational_mapping_status"] = df["CUSIP"].astype(str).map(
        lambda c: _get(c, "operational_mapping_status")
    )
    df["_canonical_security"] = df["CUSIP"].astype(str).map(
        lambda c: _get(c, "canonical_security")
    )
    df["_canonical_security_kind"] = df["CUSIP"].astype(str).map(
        lambda c: _get(c, "canonical_security_kind")
    )
    df["_observed_security_key"] = df["CUSIP"].astype(str).map(
        lambda c: _get(c, "observed_security_key")
    )

    mask = (
        (df["_security_resolution_status"] == RESOLUTION_CANONICAL)
        & (df["_operational_mapping_status"] == MAPPING_VERIFIED)
    )
    df = df[mask].copy()
    df["_ticker_mapped"] = True
    return df


def build_operational_universe(
    infotable_df,
    official_df,
    period_iso,
    *,
    identity_results,
    identity_period_iso=None,
):
    """Construye el operational_universe §5.5 (dictamen #65).

    Cadena: §5.1 -> §5.3 -> §5.4 -> §5.5. Estrictamente restrictiva.

    Parametros:
      infotable_df        DataFrame con columnas REQUIRED_INFOTABLE_COLUMNS.
      official_df         DataFrame de load_official_list para el periodo.
      period_iso          str YYYY-MM-DD del periodo evaluado.
      identity_results    dict {cusip: dict P61}. OBLIGATORIO.
                          Debe corresponder al mismo period_iso.
      identity_period_iso str | None. Si se pasa, debe coincidir con
                          period_iso. Si no coincide -> ValueError.

    Devuelve DataFrame con columnas originales + DIAGNOSTIC_COLUMNS.
    Universo vacio -> DataFrame vacio con las columnas esperadas.

    Excepciones:
      ValueError si infotable_df no contiene columnas requeridas.
      ValueError si identity_results is None.
      ValueError si identity_period_iso != period_iso.
    """
    if identity_results is None:
        raise ValueError(
            "identity_results es obligatorio (§5.5 arquitectura C, dictamen #65)"
        )
    if identity_period_iso is not None:
        if str(identity_period_iso) != str(period_iso):
            raise ValueError(
                "identity_results corresponde a period "
                + repr(identity_period_iso)
                + ", no a "
                + repr(period_iso)
                + " (§5.5 PIT, dictamen #65)"
            )
    _check_required_columns(infotable_df)

    df = _apply_5_1(infotable_df)
    df = _apply_5_3(df, official_df)
    df = _apply_5_4(df)
    df = _apply_5_5(df, identity_results)
    return df