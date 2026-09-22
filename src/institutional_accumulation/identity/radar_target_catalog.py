"""RADAR_TARGET_CATALOG - identidad estable de los 242 tickers del radar.

Contrato (dictamen F2.3 del auditor):
  - Fuente: OpenFIGI /v3/mapping (TICKER, exchCode=US).
  - Identidad: shareClassFIGI (estable frente a corporate actions).
  - Provenance: source + source_date por fila.
  - Independiente del crosswalk interno (etf_holdings.csv,
    cusip_radar_crosswalk.csv).
  - El catalog es INPUT del resolver, no output. Construirlo NO consume
    el resolver evaluado.

Columnas:
  radar_ticker, figi, share_class_figi, composite_figi,
  ticker_from_openfigi, name, security_type, market_sector,
  exch_code, source, source_date, status

status:
  OK      -> figi + share_class_figi presentes
  PARTIAL -> figi pero sin share_class_figi
  MISS    -> sin hit en OpenFIGI

No usa datetime.now(). source_date lo aporta el caller.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .openfigi_client import extract_stable_identity

SOURCE_NAME = "openfigi:/v3/mapping"
COLUMNS = (
    "radar_ticker",
    "figi",
    "share_class_figi",
    "composite_figi",
    "ticker_from_openfigi",
    "name",
    "security_type",
    "market_sector",
    "exch_code",
    "source",
    "source_date",
    "status",
)


def load_radar_tickers(stock_prices_path: Path) -> list[str]:
    """Devuelve los tickers del radar USA (242 por construccion actual).

    Regla: sin sufijo europeo, sin ^, sin =X.
    """
    sp = pd.read_parquet(stock_prices_path)
    cols = sp.columns.get_level_values(-1).unique()
    return sorted(
        c for c in cols
        if isinstance(c, str)
        and "." not in c
        and not c.startswith("^")
        and not c.endswith("=X")
    )


def build_from_probe_result(
    result_json_path: Path,
    *,
    source_date: str,
) -> pd.DataFrame:
    """Construye el catalog leyendo un result_radar.json ya generado.

    No consulta OpenFIGI. Determinista.
    """
    raw = json.loads(Path(result_json_path).read_text(encoding="utf-8"))
    rows = []
    for tk, hit in raw.items():
        ident = extract_stable_identity(hit)
        if ident is None:
            rows.append({
                "radar_ticker": tk,
                "figi": None,
                "share_class_figi": None,
                "composite_figi": None,
                "ticker_from_openfigi": None,
                "name": None,
                "security_type": None,
                "market_sector": None,
                "exch_code": None,
                "source": SOURCE_NAME,
                "source_date": source_date,
                "status": "MISS",
            })
            continue
        has_figi = bool(ident.get("figi"))
        has_scf = bool(ident.get("share_class_figi"))
        status = "OK" if (has_figi and has_scf) else ("PARTIAL" if has_figi else "MISS")
        rows.append({
            "radar_ticker": tk,
            "figi": ident.get("figi"),
            "share_class_figi": ident.get("share_class_figi"),
            "composite_figi": ident.get("composite_figi"),
            "ticker_from_openfigi": ident.get("ticker"),
            "name": ident.get("name"),
            "security_type": ident.get("security_type"),
            "market_sector": ident.get("market_sector"),
            "exch_code": ident.get("exch_code"),
            "source": SOURCE_NAME,
            "source_date": source_date,
            "status": status,
        })
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return df.sort_values("radar_ticker").reset_index(drop=True)


def write_catalog(df: pd.DataFrame, out_path: Path) -> None:
    """Escribe el catalog a CSV con UTF-8 sin BOM y LF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        df.to_csv(f, index=False)


def coverage_summary(df: pd.DataFrame) -> dict:
    """Cuenta filas por status y % con shareClassFIGI."""
    n = len(df)
    if n == 0:
        return {
            "n_total": 0,
            "n_ok": 0,
            "n_partial": 0,
            "n_miss": 0,
            "pct_with_scf": 0.0,
        }
    n_ok = int((df["status"] == "OK").sum())
    n_partial = int((df["status"] == "PARTIAL").sum())
    n_miss = int((df["status"] == "MISS").sum())
    pct = float(100.0 * df["share_class_figi"].notna().sum() / n)
    return {
        "n_total": int(n),
        "n_ok": n_ok,
        "n_partial": n_partial,
        "n_miss": n_miss,
        "pct_with_scf": round(pct, 4),
    }
