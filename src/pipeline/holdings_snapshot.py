# -*- coding: utf-8 -*-
"""Snapshot append-only de holdings (PIT-fix auditor externo 2026-10-25).

Problema
--------
`data/etf_holdings.csv` y `data/index_holdings.csv` son estaticos (sin
columna temporal). El historico Wyckoff (`sector_wyckoff_distribution.csv`)
se regenera con la composicion vigente en el momento del run, aplicada
retroactivamente -> survivorship bias (H-PIT del dictamen auditor externo
2026-10-25).

Solucion
--------
En cada run, anadir la composicion actual a
`outputs/history/holdings_snapshots.csv` con la sesion del dia. Solo se
anade si la composicion de algun (source_file, etf) cambia respecto al
ultimo snapshot registrado. Append-only, nunca sobreescribe historia.

Schema: date, source_file, etf, ticker, name, identifier, weight, ticker_hash.

En 6-12 meses habra historia PIT real. Regeneraciones futuras usaran
holdings_snapshots filtrando date <= as_of.

Contrato: docs/auditoria/wyckoff/68_pit_holdings_2026-10-25.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline.output_writer import persist, skip
from src.market_calendar import last_expected_market_date


HOLDINGS_ETF_PATH = "data/etf_holdings.csv"
HOLDINGS_INDEX_PATH = "data/index_holdings.csv"
OUTPUT_NAME = "holdings_snapshots"

SNAPSHOT_COLUMNS = [
    "date", "source_file", "etf", "ticker", "name",
    "identifier", "weight", "ticker_hash",
]


def _ticker_hash(tickers) -> str:
    """SHA256 de la lista ordenada de tickers. Estable entre runs."""
    joined = "|".join(sorted(str(t) for t in tickers))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _normalize(df: pd.DataFrame, source_file: str, session_date: str) -> pd.DataFrame:
    """Normaliza un CSV de holdings a SNAPSHOT_COLUMNS (sin date)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    out = pd.DataFrame()
    out["source_file"] = [source_file] * len(df)
    out["etf"] = df.get("etf", "").astype(str)
    out["ticker"] = df.get("ticker", "").astype(str)
    out["name"] = df["name"].astype(str) if "name" in df.columns else ""
    out["identifier"] = df["identifier"].astype(str) if "identifier" in df.columns else ""
    out["weight"] = pd.to_numeric(df.get("weight", 0.0), errors="coerce").fillna(0.0)
    return out


def _build_snapshot_df(session_date, etf_df, index_df) -> pd.DataFrame:
    """Construye el DataFrame del snapshot para la sesion."""
    parts = []
    etf_norm = _normalize(etf_df, "etf_holdings", session_date)
    idx_norm = _normalize(index_df, "index_holdings", session_date)
    for part in (etf_norm, idx_norm):
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    df = pd.concat(parts, ignore_index=True)
    df["date"] = session_date

    # Hash por (source_file, etf): mismo valor en todas las filas del grupo.
    hash_map = {}
    for (src, etf), grp in df.groupby(["source_file", "etf"]):
        hash_map[(src, etf)] = _ticker_hash(grp["ticker"].tolist())
    df["ticker_hash"] = df.apply(
        lambda r: hash_map[(r["source_file"], r["etf"])], axis=1,
    )
    return df[SNAPSHOT_COLUMNS]


def _load_latest_hashes(snapshot_path: Path) -> dict:
    """Dict[(source_file, etf)] -> ultimo ticker_hash registrado."""
    if not snapshot_path.exists():
        return {}
    try:
        df = pd.read_csv(snapshot_path)
    except (OSError, pd.errors.ParserError, ValueError):
        return {}
    if df.empty or "ticker_hash" not in df.columns:
        return {}
    # Ultimo hash por (source_file, etf): tomamos el de la fila con date_max del grupo.
    out = {}
    for (src, etf), grp in df.groupby(["source_file", "etf"]):
        try:
            last_row = grp.sort_values("date").iloc[-1]
            out[(src, etf)] = str(last_row["ticker_hash"])
        except (KeyError, IndexError, ValueError):
            continue
    return out


def _read_csv_safe(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def compute_and_persist_holdings_snapshot(
    reference_date=None,
    etf_holdings_df: Optional[pd.DataFrame] = None,
    index_holdings_df: Optional[pd.DataFrame] = None,
    path_override=None,
) -> Optional[pd.DataFrame]:
    """Lee composiciones, compara con ultimo snapshot, escribe si cambia.

    Args:
        reference_date: fecha de run (se resuelve a sesion NYSE).
        etf_holdings_df, index_holdings_df: opcionales (tests). Si no,
            se leen de HOLDINGS_ETF_PATH y HOLDINGS_INDEX_PATH.
        path_override: ruta alternativa para tests.

    Returns:
        DataFrame escrito (solo filas nuevas) o None si skip.
    """
    session = last_expected_market_date(pd.Timestamp(reference_date)) if reference_date else last_expected_market_date()
    session_str = pd.Timestamp(session).strftime("%Y-%m-%d")

    etf_df = etf_holdings_df if etf_holdings_df is not None else _read_csv_safe(HOLDINGS_ETF_PATH)
    idx_df = index_holdings_df if index_holdings_df is not None else _read_csv_safe(HOLDINGS_INDEX_PATH)

    if (etf_df is None or etf_df.empty) and (idx_df is None or idx_df.empty):
        skip(OUTPUT_NAME, "EMPTY_DF", path_override=path_override)
        return None

    today = _build_snapshot_df(session_str, etf_df, idx_df)
    if today.empty:
        skip(OUTPUT_NAME, "EMPTY_DF", path_override=path_override)
        return None

    # Ruta del CSV (para comparar con historia antes de escribir).
    if path_override is not None:
        snapshot_path = Path(path_override)
    else:
        from src.pipeline.output_registry import get_spec
        snapshot_path = Path(get_spec(OUTPUT_NAME).path)

    latest = _load_latest_hashes(snapshot_path)

    # Filtrar grupos cuyo hash coincide con el ultimo registrado.
    def _is_changed(row):
        key = (row["source_file"], row["etf"])
        return latest.get(key) != row["ticker_hash"]

    mask = today.apply(_is_changed, axis=1)
    new_rows = today[mask].reset_index(drop=True)

    if new_rows.empty:
        skip(OUTPUT_NAME, "SKIP_CONDITION", path_override=path_override)
        return None

    persist(
        OUTPUT_NAME,
        new_rows,
        date_value=pd.Timestamp(session_str),
        sort_by=["source_file", "etf", "ticker"],
        path_override=path_override,
    )
    return new_rows