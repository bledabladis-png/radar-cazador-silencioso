"""Resolucion de identidad canonica de security (C1-revisada, dictamen v1.1).

Dictamen habilitante: docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_
ESPECIFICACION_V11_DICTAMEN.md (Q-ESP-V11-1) + spec v1.2 seccion 3.13.

Responsabilidad:
  observed_CUSIP (+ report_period) -> canonical_security (o NULL).

Prohibiciones explicitas (dictamen C1-revisada):
  - NO usar `ticker:<ticker>` como canonical_security.
  - NO inferir equivalence sin entry explicita en cusip_equivalence.csv.
  - NO resolver heuristicamente por NAMEOFISSUER ni proximidad de CUSIP.
  - NO presentar `cusip:<CUSIP>` como identidad canonica estable sin
    evidencia.

Dos capas de estado independientes:
  security_resolution_status (5): CANONICAL | OBSERVED_ONLY | UNRESOLVED
                                   | AMBIGUOUS | CONFLICT
  canonical_security_kind    (4): CANONICAL_FIGI | CANONICAL_EQUIVALENCE
                                   | OBSERVED_CUSIP_ONLY | UNRESOLVED

Regla dura:
  canonical_security = NULL salvo status == CANONICAL.

Cadena de autoridad (dictamen v1.1 seccion 3.13):
  1. cusip_equivalence.csv (equivalence explicita con vigencia temporal).
  2. crosswalk_internal (cusip_ticker_exceptions + etf_holdings).
  3. OpenFIGI shareClassFIGI (opcional, no enchufado en este ciclo).
  4. Sin evidencia -> OBSERVED_ONLY o UNRESOLVED.

OpenFIGI NO se usa en este ciclo (sin API key). El kwarg `figi_lookup`
acepta un callable opcional para enchufarlo en el futuro sin cambios
de firma.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Estados (C1-revisada) ---

STATUS_CANONICAL = "CANONICAL"
STATUS_OBSERVED_ONLY = "OBSERVED_ONLY"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_CONFLICT = "CONFLICT"

ALL_STATUSES = (
    STATUS_CANONICAL,
    STATUS_OBSERVED_ONLY,
    STATUS_UNRESOLVED,
    STATUS_AMBIGUOUS,
    STATUS_CONFLICT,
)

# --- Kinds (C1-revisada) ---

KIND_CANONICAL_FIGI = "CANONICAL_FIGI"
KIND_CANONICAL_EQUIVALENCE = "CANONICAL_EQUIVALENCE"
KIND_OBSERVED_CUSIP_ONLY = "OBSERVED_CUSIP_ONLY"
KIND_UNRESOLVED = "UNRESOLVED"

ALL_KINDS = (
    KIND_CANONICAL_FIGI,
    KIND_CANONICAL_EQUIVALENCE,
    KIND_OBSERVED_CUSIP_ONLY,
    KIND_UNRESOLVED,
)

# --- Esquemas ---

EQUIVALENCE_COLUMNS = (
    "CUSIP_A",
    "canonical_security",
    "valid_from",
    "valid_to",
    "source",
    "reason",
    "verified_by",
    "source_document",
)

EQUIVALENCE_REQUIRED = EQUIVALENCE_COLUMNS[:7]
CROSSWALK_COLUMNS = ("CUSIP", "ticker", "valid_from", "valid_to", "source")

DEFAULT_EQUIVALENCE_PATH = Path("data/mappings/cusip_equivalence.csv")
DEFAULT_EXCEPTIONS_PATH = Path("data/mappings/cusip_ticker_exceptions.csv")
DEFAULT_ETF_HOLDINGS_PATH = Path("data/etf_holdings.csv")

VERIFIED_BY_VALUES = frozenset({"SEC", "OpenFIGI", "manual"})
FORBIDDEN_SOURCES = frozenset({"manual", "internet", "known", "unknown"})


# --- Helpers publicos ---

def observed_security_key(cusip):
    """Devuelve la clave tecnica por periodo: cusip:<CUSIP>."""
    s = str(cusip).strip() if cusip is not None else ""
    return f"cusip:{s}"


def _normalize_canonical(value, identity_type):
    """Normaliza canonical_security segun P60.

    El tipo de identidad es OBLIGATORIO y lo declara la fuente.
    Prohibido inferir el tipo desde el valor desnudo.

    identity_type == "TICKER" -> "equity:<ticker>"
    identity_type == "FIGI"   -> "figi:<FIGI>"
    identity_type == "CUSIP"  -> None (no produce canonical)
    identity_type == "ISIN"   -> None (fuera de alcance v1)

    None / NaN / cadena vacia -> None.
    identity_type None o invalido -> ValueError.
    """
    VALID = ("TICKER", "FIGI", "CUSIP", "ISIN")
    if identity_type is None:
        raise ValueError("identity_type es obligatorio (P60)")
    if identity_type not in VALID:
        raise ValueError(
            "identity_type invalido: " + repr(identity_type) +
            ". Validos: " + str(VALID)
        )
    if value is None:
        return None
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    # Un prefijo explicito en el valor (equity:/figi:) es declaracion
    # de la propia fuente; se respeta por encima de identity_type.
    if s.startswith("equity:") or s.startswith("figi:"):
        return s
    if identity_type in ("CUSIP", "ISIN"):
        return None
    if identity_type == "FIGI":
        return "figi:" + s
    # TICKER
    return "equity:" + s


# --- Carga de tablas ---

def load_cusip_equivalence(path=DEFAULT_EQUIVALENCE_PATH):
    """Carga y valida la tabla de equivalencias CUSIP.

    Si el fichero no existe, devuelve DataFrame vacio con columnas
    EQUIVALENCE_COLUMNS. No es error.

    Validaciones obligatorias:
      - Columnas requeridas presentes.
      - valid_from <= valid_to cuando valid_to no es NULL.
      - verified_by en VALID_VERIFIED_BY.
      - source no en FORBIDDEN_SOURCES.
      - Sin solapes por CUSIP_A.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(EQUIVALENCE_COLUMNS))

    df = pd.read_csv(path, dtype=str)
    missing = [c for c in EQUIVALENCE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Columnas faltantes en cusip_equivalence: " + str(missing))
    if "source_document" not in df.columns:
        df["source_document"] = None

    df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce")
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    if df["valid_from"].isna().any():
        raise ValueError("valid_from no parseable en cusip_equivalence")

    both = df["valid_to"].notna()
    bad = both & (df["valid_from"] > df["valid_to"])
    if bad.any():
        raise ValueError(
            "valid_from > valid_to en filas: " + str(df.index[bad].tolist())
        )

    src = df["source"].astype(str).str.strip().str.lower()
    forb = src.isin(FORBIDDEN_SOURCES)
    if forb.any():
        raise ValueError(
            "source no verificable en filas: " + str(df.index[forb].tolist())
        )

    vb = df["verified_by"].astype(str).str.strip()
    bad_vb = ~vb.isin(VERIFIED_BY_VALUES)
    if bad_vb.any():
        raise ValueError(
            "verified_by invalido en filas: " + str(df.index[bad_vb].tolist())
        )

    _check_no_overlap_equivalence(df)
    return df


def _check_no_overlap_equivalence(df):
    """Comprueba que no hay solapes temporales por CUSIP_A."""
    if df.empty:
        return
    for cusip, group in df.groupby("CUSIP_A"):
        g = group.sort_values("valid_from").reset_index(drop=True)
        for i in range(len(g) - 1):
            a_to = g.loc[i, "valid_to"]
            b_from = g.loc[i + 1, "valid_from"]
            if pd.isna(a_to):
                raise ValueError(
                    "Solape: " + str(cusip) +
                    " tiene valid_to NULL seguido de otra fila"
                )
            if b_from <= a_to:
                raise ValueError(
                    "Solape: " + str(cusip) +
                    " [" + str(g.loc[i, "valid_from"]) + ".." + str(a_to) +
                    "] vs [" + str(b_from) + "..]"
                )


def load_crosswalk_internal(
    exceptions_path=DEFAULT_EXCEPTIONS_PATH,
    etf_holdings_path=DEFAULT_ETF_HOLDINGS_PATH,
):
    """Carga el crosswalk interno combinado.

    Fuentes:
      - cusip_ticker_exceptions.csv (con vigencia temporal).
      - etf_holdings.csv (sin vigencia; activo siempre).

    Devuelve DataFrame con columnas CROSSWALK_COLUMNS.
    """
    rows = []

    p1 = Path(exceptions_path)
    if p1.exists():
        e = pd.read_csv(p1, dtype=str)
        for _, r in e.iterrows():
            rows.append({
                "CUSIP": str(r["CUSIP"]).strip(),
                "ticker": str(r["ticker"]).strip(),
                "valid_from": r.get("valid_from"),
                "valid_to": r.get("valid_to"),
                "source": "exceptions",
            })

    p2 = Path(etf_holdings_path)
    if p2.exists():
        h = pd.read_csv(p2, dtype=str)
        for _, r in h.iterrows():
            rows.append({
                "CUSIP": str(r["identifier"]).strip(),
                "ticker": str(r["ticker"]).strip(),
                "valid_from": None,
                "valid_to": None,
                "source": "etf_holdings",
            })

    df = pd.DataFrame(rows, columns=list(CROSSWALK_COLUMNS))
    if not df.empty:
        df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce")
        df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    return df


# --- Resolucion ---

def _find_active_equivalence(cusip, report_period, eq_df):
    """Devuelve lista de canonical_security activos para (cusip, period)."""
    if eq_df is None or eq_df.empty:
        return []
    sub = eq_df[eq_df["CUSIP_A"].astype(str).str.strip() == str(cusip).strip()]
    if sub.empty:
        return []
    try:
        period = pd.Timestamp(report_period).normalize()
    except Exception:
        return []
    active = sub[
        (sub["valid_from"] <= period)
        & (sub["valid_to"].isna() | (sub["valid_to"] >= period))
    ]
    return (
        active["canonical_security"].dropna().astype(str).str.strip()
        .unique().tolist()
    )


def _find_active_crosswalk(cusip, report_period, cw_df):
    """Devuelve lista de tickers activos para (cusip, period)."""
    if cw_df is None or cw_df.empty:
        return []
    sub = cw_df[cw_df["CUSIP"].astype(str).str.strip() == str(cusip).strip()]
    if sub.empty:
        return []
    try:
        period = pd.Timestamp(report_period).normalize()
    except Exception:
        return []
    active = sub[
        (sub["valid_from"].isna() | (sub["valid_from"] <= period))
        & (sub["valid_to"].isna() | (sub["valid_to"] >= period))
    ]
    return (
        active["ticker"].dropna().astype(str).str.strip()
        .unique().tolist()
    )


def resolve_security_identity(
    cusip,
    report_period,
    *,
    equivalence_df=None,
    crosswalk_internal_df=None,
    figi_lookup=None,
):
    """Resuelve la identidad canonica de un CUSIP para un report_period.

    Devuelve dict con:
      observed_security_key
      security_resolution_status  (5 estados)
      canonical_security_kind     (4 estados)
      canonical_security          (str o None)
      evidence                    (dict)

    Prioridad (C1-revisada 3.13):
      1. equivalence_df  -> CANONICAL + CANONICAL_EQUIVALENCE
      2. crosswalk_internal_df -> CANONICAL + CANONICAL_EQUIVALENCE
      3. figi_lookup (opcional) -> CANONICAL + CANONICAL_FIGI
      4. OBSERVED_ONLY / UNRESOLVED
    """
    cusip_s = str(cusip).strip() if cusip is not None else ""
    osk = observed_security_key(cusip_s) if cusip_s else "cusip:"
    result = {
        "observed_security_key": osk,
        "security_resolution_status": STATUS_UNRESOLVED,
        "canonical_security_kind": KIND_UNRESOLVED,
        "canonical_security": None,
        "evidence": {},
    }
    if not cusip_s:
        return result

    # 1. equivalence
    eq_canon = _find_active_equivalence(cusip_s, report_period, equivalence_df)
    if len(eq_canon) > 1:
        result["security_resolution_status"] = STATUS_AMBIGUOUS
        result["canonical_security_kind"] = KIND_OBSERVED_CUSIP_ONLY
        result["evidence"] = {"source": "equivalence", "candidates": eq_canon}
        return result
    if len(eq_canon) == 1:
        result["security_resolution_status"] = STATUS_CANONICAL
        result["canonical_security_kind"] = KIND_CANONICAL_EQUIVALENCE
        result["canonical_security"] = _normalize_canonical(eq_canon[0], "TICKER")
        result["evidence"] = {"source": "equivalence"}
        return result

    # 2. crosswalk_internal
    cw_tickers = _find_active_crosswalk(cusip_s, report_period, crosswalk_internal_df)
    if len(cw_tickers) > 1:
        result["security_resolution_status"] = STATUS_CONFLICT
        result["canonical_security_kind"] = KIND_OBSERVED_CUSIP_ONLY
        result["evidence"] = {
            "source": "crosswalk_internal",
            "candidates": cw_tickers,
        }
        return result
    if len(cw_tickers) == 1:
        result["security_resolution_status"] = STATUS_CANONICAL
        result["canonical_security_kind"] = KIND_CANONICAL_EQUIVALENCE
        result["canonical_security"] = _normalize_canonical(cw_tickers[0], "TICKER")
        result["evidence"] = {
            "source": "crosswalk_internal",
            "ticker": cw_tickers[0],
        }
        return result

    # 3. figi_lookup (opcional, no enchufado en este ciclo)
    if figi_lookup is not None:
        try:
            figi = figi_lookup(cusip_s, report_period)
        except Exception:
            figi = None
        if figi:
            result["security_resolution_status"] = STATUS_CANONICAL
            result["canonical_security_kind"] = KIND_CANONICAL_FIGI
            result["canonical_security"] = "figi:" + str(figi)
            result["evidence"] = {"source": "openfigi"}
            return result

    # 4. observed only
    result["security_resolution_status"] = STATUS_OBSERVED_ONLY
    result["canonical_security_kind"] = KIND_OBSERVED_CUSIP_ONLY
    return result


def resolve_batch_identities(
    cusips,
    report_period,
    *,
    equivalence_df=None,
    crosswalk_internal_df=None,
    figi_lookup=None,
):
    """Version batch. Devuelve dict {cusip: resultado}."""
    return {
        str(c).strip(): resolve_security_identity(
            c, report_period,
            equivalence_df=equivalence_df,
            crosswalk_internal_df=crosswalk_internal_df,
            figi_lookup=figi_lookup,
        )
        for c in cusips
    }


def compute_identity_coverage(
    cusips,
    report_period,
    *,
    equivalence_df=None,
    crosswalk_internal_df=None,
    figi_lookup=None,
):
    """Metricas agregadas de resolucion de identidad.

    Devuelve dict con conteos por status y kind, y pct_canonical.
    """
    from collections import Counter

    results = resolve_batch_identities(
        cusips, report_period,
        equivalence_df=equivalence_df,
        crosswalk_internal_df=crosswalk_internal_df,
        figi_lookup=figi_lookup,
    )
    status_counts = Counter(r["security_resolution_status"] for r in results.values())
    kind_counts = Counter(r["canonical_security_kind"] for r in results.values())
    n_total = len(results)
    n_canonical = status_counts.get(STATUS_CANONICAL, 0)
    return {
        "n_total": n_total,
        "n_canonical": n_canonical,
        "n_observed_only": status_counts.get(STATUS_OBSERVED_ONLY, 0),
        "n_unresolved": status_counts.get(STATUS_UNRESOLVED, 0),
        "n_ambiguous": status_counts.get(STATUS_AMBIGUOUS, 0),
        "n_conflict": status_counts.get(STATUS_CONFLICT, 0),
        "by_status": dict(status_counts),
        "by_kind": dict(kind_counts),
        "pct_canonical": (n_canonical / n_total) if n_total > 0 else 0.0,
    }