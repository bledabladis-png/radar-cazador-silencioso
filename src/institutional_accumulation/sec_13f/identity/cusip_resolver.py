"""Resolver CUSIP -> ticker con vigencia temporal.

FA-2.2 - Dictamen auditor FA-2.0 (2026-09-19):
  - Q-C: columnas obligatorias + 4 controles.
  - Q-D: modelo multi-fila con vigencia temporal.
  - Firma: resolver(cusip, report_period), NO resolver(cusip).

Controles obligatorios del dictamen:
  A. Sin solapes incompatibles para (CUSIP, ticker).
  B. source verificable (no "manual", "internet", "known").
  C. No inferencia silenciosa.
  D. title_of_class no decorativo.

Fuera de FA-2.2: reporting relationships (FA-2.3), amendments (FA-2.4).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = (
    "CUSIP", "ticker", "valid_from", "valid_to",
    "source", "reason", "title_of_class", "verified_by",
)

FORBIDDEN_SOURCES = frozenset({"manual", "internet", "known", "unknown"})
VALID_VERIFIED_BY = frozenset({"SEC", "OpenFIGI", "manual"})
DEFAULT_EXCEPTIONS_PATH = Path("data/mappings/cusip_ticker_exceptions.csv")


def load_exceptions(path=DEFAULT_EXCEPTIONS_PATH):
    """Carga y valida tabla de excepciones CUSIP.

    Valida:
      - Columnas obligatorias presentes.
      - valid_from <= valid_to cuando valid_to != NULL.
      - source no en FORBIDDEN_SOURCES.
      - verified_by en VALID_VERIFIED_BY.
      - Sin solapes para la misma clave (CUSIP, ticker).

    Devuelve DataFrame. Si el fichero no existe, devuelve DataFrame
    vacio con las columnas correctas (no es error: la tabla se puede
    poblar despues).
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    df = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Columnas faltantes en excepciones: " + str(missing))

    df["valid_from"] = pd.to_datetime(df["valid_from"], errors="coerce")
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    if df["valid_from"].isna().any():
        raise ValueError("valid_from no parseable en excepciones")

    both = df["valid_to"].notna()
    bad = both & (df["valid_from"] > df["valid_to"])
    if bad.any():
        raise ValueError("valid_from > valid_to en filas: " + str(df.index[bad].tolist()))

    src_norm = df["source"].astype(str).str.strip().str.lower()
    forbidden = src_norm.isin(FORBIDDEN_SOURCES)
    if forbidden.any():
        raise ValueError("source no verificable en filas: " + str(df.index[forbidden].tolist()))

    vb = df["verified_by"].astype(str).str.strip()
    bad_vb = ~vb.isin(VALID_VERIFIED_BY)
    if bad_vb.any():
        raise ValueError("verified_by invalido en filas: " + str(df.index[bad_vb].tolist()))

    _check_no_overlap(df)
    return df


def _check_no_overlap(df):
    """Comprueba que no hay solapes para (CUSIP, ticker)."""
    if df.empty:
        return
    for (cusip, ticker), group in df.groupby(["CUSIP", "ticker"]):
        g = group.sort_values("valid_from").reset_index(drop=True)
        for i in range(len(g) - 1):
            a_to = g.loc[i, "valid_to"]
            b_from = g.loc[i + 1, "valid_from"]
            if pd.isna(a_to):
                raise ValueError(
                    "Solape: " + str(cusip) + "/" + str(ticker) +
                    " tiene valid_to NULL seguido de otra fila"
                )
            if b_from <= a_to:
                raise ValueError(
                    "Solape: " + str(cusip) + "/" + str(ticker) +
                    " [" + str(g.loc[i, "valid_from"]) + ".." + str(a_to) + "] vs [" +
                    str(b_from) + "..]"
                )


def resolve_cusip(cusip, report_period, exceptions_df):
    """Resuelve un CUSIP a ticker segun report_period.

    cusip: str. report_period: str|Timestamp. exceptions_df: DataFrame.
    Devuelve ticker (str) o None si no hay match activo.
    """
    if exceptions_df is None or exceptions_df.empty:
        return None
    cusip_str = str(cusip).strip()
    try:
        period = pd.Timestamp(report_period).normalize()
    except Exception:
        return None

    sub = exceptions_df[exceptions_df["CUSIP"].astype(str).str.strip() == cusip_str]
    if sub.empty:
        return None

    active = sub[
        (sub["valid_from"] <= period)
        & (sub["valid_to"].isna() | (sub["valid_to"] >= period))
    ]
    if active.empty:
        return None

    # Si hay varias filas activas (distinto ticker), es config inconsistente.
    tickers = active["ticker"].astype(str).unique()
    if len(tickers) > 1:
        raise ValueError(
            "Multiples tickers activos para CUSIP " + cusip_str +
            " en periodo " + str(period) + ": " + str(list(tickers))
        )
    return str(active.iloc[0]["ticker"])


def resolve_batch(df, report_period, exceptions_df, *, cusip_col="CUSIP"):
    """Anade columna `ticker_resolved` a un DataFrame con CUSIPs.

    Devuelve copia del df con la columna nueva. Sin modificar el original.
    Los CUSIPs sin match quedan con ticker_resolved=None (no se imputa).
    """
    if cusip_col not in df.columns:
        raise KeyError("Columna " + cusip_col + " no presente")
    out = df.copy()
    out["ticker_resolved"] = out[cusip_col].apply(
        lambda c: resolve_cusip(c, report_period, exceptions_df)
    )
    return out
