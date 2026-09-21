"""A.6.4-v2 - compute_contractual_coverage sobre top2000 (TARGET real).

Reejecuta la evidencia del piloto TOP 2000 bajo la ruta CONTRACTUAL
P38, en lugar del proxy legacy.

Fuente de TARGET: pilot_13f_top2000.csv (2000 CUSIPs, 1776 FIGIs
unicos resueltos por OpenFIGI). El piloto es evidencia ya cerrada
(F2.3-bis). Este probe solo valida que la ruta contractual consume
correctamente el TARGET.

Los records se construyen desde el canonical Q4/Q1, filtrando a
los CUSIPs del piloto y heredando el FIGI resuelto por el piloto.
Esto aísla el probe de la capa P61 (que solo resuelve un subconjunto
reducido) y mide exclusivamente la ruta contractual P38.

Determinista. Sin datetime.now().
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.aggregation import coverage as cov

PILOT_CSV = ROOT / "docs" / "auditoria" / "iae" / "evidence" / \
            "nipc_gate0_target_identity_top2000" / "pilot_13f_top2000.csv"
DATA = ROOT / "data" / "sec_13f" / "processed"
HERE = Path(__file__).parent

# (folder, period_iso, period_label) - period_label es la convencion
# que espera aggregate_positions_by_shareclass_figi (Q4 / Q1).
PERIODS = [
    ("2025Q4", "2025-12-31", "Q4"),
    ("2026Q1", "2026-03-31", "Q1"),
]
TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]


def load_canonical(folder, period_iso):
    d = DATA / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    filtered = filter_by_period(dfs, period_iso)
    return apply_amendments(filtered, period=period_iso)["canonical_snapshot"]


def load_pilot():
    """Devuelve (pilot_df, figi_by_cusip, target_figi_set)."""
    df = pd.read_csv(PILOT_CSV, dtype=str, keep_default_na=False)
    ok = df[df["target_membership"].astype(str).str.lower() == "true"].copy()
    figi_by_cusip = {
        str(r["CUSIP"]): str(r["share_class_figi"])
        for _, r in ok.iterrows()
        if r["share_class_figi"]
    }
    target_figi = set(figi_by_cusip.values())
    return df, figi_by_cusip, target_figi


def build_records(snapshot, period_label, figi_by_cusip):
    """Construye PositionRecords agregados por CUSIP desde el canonical.

    Hereda el FIGI resuelto por el piloto (OpenFIGI). No usa P61.
    """
    info = snapshot["INFOTABLE"].copy()
    mask_sh = info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = info["PUTCALL"].isna()
    info = info[mask_sh & mask_null].copy()
    info["CUSIP_s"] = info["CUSIP"].astype(str)
    info = info[info["CUSIP_s"].isin(figi_by_cusip.keys())]
    info["SSHPRNAMT_f"] = pd.to_numeric(info["SSHPRNAMT"], errors="coerce").fillna(0.0)

    records = []
    for cusip_s, grp in info.groupby("CUSIP_s"):
        figi = figi_by_cusip[cusip_s]
        records.append(cov.PositionRecord(
            period=period_label,
            observed_security_key="cusip:" + cusip_s,
            share_class_figi=figi,
            canonical_security="figi:" + figi,
            resolution_status="CANONICAL",
            operational_mapping_status="VERIFIED",
            weight=float(grp["SSHPRNAMT_f"].sum()),
        ))
    return records


def main():
    print("=== A.6.4-v2 - compute_contractual_coverage sobre top2000 ===")

    pilot_df, figi_by_cusip, target_figi = load_pilot()
    print("CUSIPs target_membership=True: " + str(len(figi_by_cusip)))
    print("FIGIs unicos en TARGET: " + str(len(target_figi)))

    records_by_period = {}
    for folder, period_iso, period_label in PERIODS:
        print()
        print("--- " + folder + " (" + period_iso + ") ---")
        snap = load_canonical(folder, period_iso)
        records = build_records(snap, period_label, figi_by_cusip)
        with_figi = sum(1 for r in records if r.share_class_figi)
        total_w = sum(r.weight for r in records)
        print("  records: " + str(len(records)))
        print("  records con FIGI: " + str(with_figi))
        print("  SSHPRNAMT total: " + f"{total_w:,.0f}")
        records_by_period[folder] = records

    target_q4 = target_figi
    target_q1 = target_figi
    rq4 = records_by_period["2025Q4"]
    rq1 = records_by_period["2026Q1"]

    print()
    print("=== compute_contractual_coverage ===")
    result = cov.compute_contractual_coverage(target_q4, target_q1, rq4, rq1)
    for k, v in result.items():
        print("  " + str(k).ljust(35) + " " + str(v))

    out_json = HERE / "result.json"
    out_json.write_text(
        json.dumps({
            "target_figi_count": len(target_figi),
            "records_q4": len(rq4),
            "records_q1": len(rq1),
            "result": dict(result),
        }, indent=2, default=str),
        encoding="utf-8", newline="\n",
    )
    print()
    print("OK -> " + str(out_json))


if __name__ == "__main__":
    main()