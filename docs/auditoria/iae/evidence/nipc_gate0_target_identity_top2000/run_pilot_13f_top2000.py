"""Piloto: TARGET_UNIVERSE_Q1 sobre top 2000 CUSIPs del 13F 2026Q1.

Autorizado por dictamen F2.3-bis + dictamen de autorizacion TOP 2000.

Convencion temporal aprobada:
  RADAR_SNAPSHOT_DATE = 2026-09-19
  13F_OBSERVATION_PERIOD = 2026-03-31
  RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

Metricas: count + weight (SSHPRNAMT) por status.
Diagnosticos: duplicate_shareClassFIGI, multiple hits, conflicts.

Determinista. Sin datetime.now().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.institutional_accumulation.identity.target_universe import (
    load_catalog,
    resolve_cusips,
)

SRC_13F = Path(r"D:\13f_probe\processed\2026Q1")
CATALOG = ROOT / "data" / "mappings" / "radar_target_catalog.csv"
HERE = Path(__file__).parent
OUT_SAMPLE = HERE / "pilot_cusips_sample_top2000.csv"
OUT_CSV = HERE / "pilot_13f_top2000.csv"
OUT_JSON = HERE / "pilot_13f_summary_2000.json"
OUT_DIAG = HERE / "catalog_diagnostics.json"

PERIOD = "2026-03-31"
TOP_N = 2000
SOURCE_DATE = "2026-09-19"
RADAR_SNAPSHOT_DATE = "2026-09-19"
RADAR_MEMBERSHIP_MODE = "CURRENT_RETROSPECTIVE"

def build_cusip_sample() -> pd.DataFrame:
    info = pd.read_parquet(SRC_13F / "INFOTABLE.parquet")
    sub = pd.read_parquet(SRC_13F / "SUBMISSION.parquet")

    print(f"INFOTABLE rows: {len(info)}")
    print(f"SUBMISSION rows: {len(sub)}")

    if "PERIODOFREPORT" in sub.columns:
        sub["PERIODOFREPORT"] = pd.to_datetime(sub["PERIODOFREPORT"], errors="coerce")
        period_mask = sub["PERIODOFREPORT"] == pd.Timestamp(PERIOD)
        sub_p = sub[period_mask][["ACCESSION_NUMBER"]]
        print(f"SUBMISSION filings en periodo: {len(sub_p)}")
        info = info.merge(sub_p, on="ACCESSION_NUMBER", how="inner")
        print(f"INFOTABLE tras filtro periodo: {len(info)}")

    if "SSHPRNAMTTYPE" in info.columns:
        info = info[info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"]
    if "PUTCALL" in info.columns:
        info = info[info["PUTCALL"].isna()]
    print(f"Tras SH + PUTCALL NULL: {len(info)}")

    if "SSHPRNAMT" in info.columns:
        info["SSHPRNAMT"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
        info = info.dropna(subset=["SSHPRNAMT"])

    agg = (
        info.groupby("CUSIP", dropna=False)
        .agg(sshprnamt_sum=("SSHPRNAMT", "sum"), n_lines=("SSHPRNAMT", "size"))
        .reset_index()
        .sort_values("sshprnamt_sum", ascending=False)
    )
    print(f"CUSIPs unicos: {len(agg)}")

    top = agg.head(TOP_N).copy()
    print(f"Muestra (top {TOP_N}): {len(top)} CUSIPs")
    return top

def catalog_diagnostics(catalog_df: pd.DataFrame) -> dict:
    """Diagnostico de unicidad y conflictos del catalogo radar."""
    print("=== Diagnosticos del catalogo ===")

    scf_counts = defaultdict(list)
    for _, row in catalog_df.iterrows():
        scf = row.get("share_class_figi")
        ticker = row.get("radar_ticker")
        if pd.notna(scf) and scf:
            scf_counts[scf].append(ticker)

    duplicate_scf = {scf: tickers for scf, tickers in scf_counts.items() if len(tickers) > 1}

    multiple_hits = 0
    conflicting = 0
    for _, row in catalog_df.iterrows():
        ticker = row.get("radar_ticker")
        oi_ticker = row.get("ticker_from_openfigi")
        if pd.notna(oi_ticker) and oi_ticker and pd.notna(ticker):
            if str(oi_ticker).upper() != str(ticker).upper():
                conflicting += 1

    diag = {
        "n_rows": int(len(catalog_df)),
        "n_unique_share_class_figi": int(len(scf_counts)),
        "duplicate_shareClassFIGI_count": int(len(duplicate_scf)),
        "duplicate_shareClassFIGI_detail": {
            scf: tickers for scf, tickers in list(duplicate_scf.items())[:20]
        },
        "multiple_openfigi_hits": int(multiple_hits),
        "conflicting_candidates": int(conflicting),
        "radar_snapshot_date": RADAR_SNAPSHOT_DATE,
        "radar_membership_mode": RADAR_MEMBERSHIP_MODE,
    }
    print(f"  duplicate_shareClassFIGI: {len(duplicate_scf)}")
    print(f"  conflicting_candidates: {conflicting}")
    return diag

def weighted_metrics(sample: pd.DataFrame, resolved: pd.DataFrame) -> dict:
    """Metricas ponderadas por SSHPRNAMT."""
    merged = sample.merge(resolved, left_on="CUSIP", right_on="cusip", how="left")
    total_count = len(merged)
    total_weight = float(merged["sshprnamt_sum"].sum())

    def stats(mask: pd.Series) -> dict:
        sub = merged[mask]
        c = int(len(sub))
        w = float(sub["sshprnamt_sum"].sum())
        return {
            "count": c,
            "shares": w,
            "pct_count": round(c / total_count * 100, 4) if total_count else 0.0,
            "pct_weight": round(w / total_weight * 100, 4) if total_weight else 0.0,
        }

    status = merged["status"].fillna("UNKNOWN")
    result = {
        "n_total": total_count,
        "total_weight": total_weight,
        "target_true": stats(status == "OK"),
        "target_false": stats(status == "NOT_IN_RADAR"),
        "no_id": stats(status == "NO_ID"),
        "error": stats(status == "ERROR"),
        "period": PERIOD,
        "sample_size": total_count,
        "source_date": SOURCE_DATE,
        "radar_snapshot_date": RADAR_SNAPSHOT_DATE,
        "radar_membership_mode": RADAR_MEMBERSHIP_MODE,
    }
    return result, merged

def main() -> None:
    print("=== Construyendo muestra CUSIP 13F (top 2000) ===")
    sample = build_cusip_sample()
    sample.to_csv(OUT_SAMPLE, index=False)

    print()
    print("=== Cargando RADAR_TARGET_CATALOG ===")
    catalog = load_catalog(CATALOG)
    print(f"Radar en catalogo: {len(catalog)}")

    print()
    print("=== Diagnosticos del catalogo ===")
    diag = catalog_diagnostics(catalog)
    OUT_DIAG.write_text(
        json.dumps(diag, indent=2, default=str),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=== Resolviendo CUSIPs via OpenFIGI ===")
    cusips = sample["CUSIP"].astype(str).tolist()
    resolved = resolve_cusips(cusips, catalog, source_date=SOURCE_DATE)
    print(f"Resueltos: {len(resolved)}")

    summary, merged = weighted_metrics(sample, resolved)
    merged.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=== SUMMARY (count + weight) ===")
    for key in ["target_true", "target_false", "no_id", "error"]:
        s = summary[key]
        print(f"  {key}: count={s['count']} ({s['pct_count']}%) "
              f"shares={s['shares']:.2f} ({s['pct_weight']}%)")

    print()
    print("=== PRIMEROS 15 TARGET ===")
    tgt = merged[merged["target_membership"] == True][["CUSIP", "radar_ticker", "sshprnamt_sum"]].head(15)
    print(tgt.to_string(index=False))


if __name__ == "__main__":
    main()