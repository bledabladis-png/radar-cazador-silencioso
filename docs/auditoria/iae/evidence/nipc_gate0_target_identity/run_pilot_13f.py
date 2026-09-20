"""Piloto: TARGET_UNIVERSE_Q1 sobre top 500 CUSIPs del 13F 2026Q1.

Direccion autorizada por dictamen F2.3:
  CUSIP_13F -> OpenFIGI -> shareClassFIGI -> RADAR_TARGET_CATALOG
  -> target_membership.

Muestra: top 500 CUSIPs por suma SSHPRNAMT en 2026Q1
(SH + PUTCALL NULL). No procesa el universo completo (24k).

Determinista. Sin datetime.now().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.institutional_accumulation.identity.target_universe import (
    load_catalog,
    membership_summary,
    resolve_cusips,
)

SRC_13F = Path(r"D:\13f_probe\processed\2026Q1")
CATALOG = ROOT / "data" / "mappings" / "radar_target_catalog.csv"
HERE = Path(__file__).parent
OUT_CSV = HERE / "pilot_13f_top500.csv"
OUT_JSON = HERE / "pilot_13f_summary.json"

PERIOD = "2026-03-31"
TOP_N = 500
SOURCE_DATE = "2026-09-19"


def build_cusip_sample() -> pd.DataFrame:
    info = pd.read_parquet(SRC_13F / "INFOTABLE.parquet")
    sub = pd.read_parquet(SRC_13F / "SUBMISSION.parquet")

    print("INFOTABLE cols:", list(info.columns))
    print("SUBMISSION cols:", list(sub.columns))
    print(f"INFOTABLE rows: {len(info)}")

    # Filtrar SUBMISSION por periodo
    if "PERIODOFREPORT" in sub.columns:
        sub["PERIODOFREPORT"] = pd.to_datetime(sub["PERIODOFREPORT"], errors="coerce")
        period_mask = sub["PERIODOFREPORT"] == pd.Timestamp(PERIOD)
        sub_p = sub[period_mask][["ACCESSION_NUMBER"]]
        print(f"SUBMISSION filings en periodo: {len(sub_p)}")
        info = info.merge(sub_p, on="ACCESSION_NUMBER", how="inner")
        print(f"INFOTABLE tras filtro periodo: {len(info)}")

    # SH + PUTCALL NULL
    if "SSHPRNAMTTYPE" in info.columns:
        info = info[info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"]
    if "PUTCALL" in info.columns:
        info = info[info["PUTCALL"].isna()]
    print(f"Tras SH + PUTCALL NULL: {len(info)}")

    # SSHPRNAMT numerico
    if "SSHPRNAMT" in info.columns:
        info["SSHPRNAMT"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce")
        info = info.dropna(subset=["SSHPRNAMT"])

    # Agregar por CUSIP
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


def main() -> None:
    print("=== Construyendo muestra CUSIP 13F ===")
    sample = build_cusip_sample()
    sample.to_csv(HERE / "pilot_cusips_sample.csv", index=False)

    print()
    print("=== Cargando RADAR_TARGET_CATALOG ===")
    catalog = load_catalog(CATALOG)
    print(f"Radar en catalogo: {len(catalog)}")

    print()
    print("=== Resolviendo CUSIPs via OpenFIGI ===")
    cusips = sample["CUSIP"].astype(str).tolist()
    resolved = resolve_cusips(cusips, catalog, source_date=SOURCE_DATE)
    print(f"Resueltos: {len(resolved)}")

    merged = sample.merge(resolved, left_on="CUSIP", right_on="cusip", how="left")
    merged = merged.drop(columns=["cusip"])
    merged.to_csv(OUT_CSV, index=False)

    summary = membership_summary(resolved)
    summary["period"] = PERIOD
    summary["sample_size"] = len(sample)
    summary["source_date"] = SOURCE_DATE
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8", newline="\n")

    print()
    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    print("=== PRIMEROS 15 TARGET ===")
    tgt = merged[merged["target_membership"] == True][["CUSIP", "radar_ticker", "sshprnamt_sum"]].head(15)
    print(tgt.to_string(index=False))


if __name__ == "__main__":
    main()
