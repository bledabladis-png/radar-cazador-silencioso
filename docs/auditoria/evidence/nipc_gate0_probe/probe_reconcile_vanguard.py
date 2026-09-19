# Probe reconciliacion cifra Vanguard (Q-MINI hallazgo cuantitativo).
# Deterministico. Sin datetime.now().

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period

PROBE_DIR = Path(r"D:\13f_probe\processed")
TSVS = ["SUBMISSION","COVERPAGE","SUMMARYPAGE","OTHERMANAGER",
        "OTHERMANAGER2","SIGNATURE","INFOTABLE"]
TARGET_CIMS = ["0002100119","0002100121","0000102909"]


def _sep(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def load_raw(q, period):
    d = PROBE_DIR / q
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    filtered = filter_by_period(dfs, period)
    return filtered


def canon(q, period):
    d = PROBE_DIR / q
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    filtered = filter_by_period(dfs, period)
    am = apply_amendments(filtered, period=period)
    return filtered, am


def sum_shares(info_df, sub_df, target_ciks):
    m = (info_df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info_df["PUTCALL"].isna())
    df = info_df[m].copy()
    df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce").fillna(0.0)
    s = sub_df[["ACCESSION_NUMBER","CIK"]].drop_duplicates(subset=["ACCESSION_NUMBER"]).copy()
    s["ACCESSION_NUMBER"] = s["ACCESSION_NUMBER"].astype(str).str.strip()
    s["CIK"] = s["CIK"].astype(str).str.zfill(10)
    df["ACCESSION_NUMBER"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    df = df.merge(s, on="ACCESSION_NUMBER", how="left")
    df = df[df["CIK"].isin(target_ciks)]
    out = {}
    for cik, grp in df.groupby("CIK"):
        out[cik] = float(grp["SSHPRNAMT_f"].sum())
    return out


def main():
    _sep("RECONCILIACION VANGUARD - Q1 2026")

    filtered, am = canon("2026Q1", "2026-03-31")
    canonical = am["canonical_snapshot"]

    print("Applied accessions (del canonical_snapshot):", len(am["applied_accessions"]))
    print()
    print("=== per_cik_period de Vanguard ===")
    pcp = am["per_cik_period"]
    pcp_vg = pcp[pcp["CIK"].isin(TARGET_CIMS)]
    print(pcp_vg.to_string())

    print()
    print("=== Lineage de Vanguard (todas las transiciones) ===")
    lin = am["lineage"]
    lin_vg = lin[lin["CIK"].isin(TARGET_CIMS)]
    cols = [c for c in ["CIK","accession","submission_type","amendment_no","amendment_type",
                        "operation","applied","reason"] if c in lin_vg.columns]
    print(lin_vg[cols].to_string())

    print()
    print("=== Suma SSHPRNAMT raw (todos los filings del periodo) ===")
    raw_shares = sum_shares(filtered["INFOTABLE"], filtered["SUBMISSION"], TARGET_CIMS)
    for cik in TARGET_CIMS:
        print(f"  CIK={cik}  raw_shares={raw_shares.get(cik, 0.0):>20,.0f}")

    print()
    print("=== Suma SSHPRNAMT canonical (solo applied accessions) ===")
    canon_shares = sum_shares(canonical["INFOTABLE"], canonical["SUBMISSION"], TARGET_CIMS)
    for cik in TARGET_CIMS:
        print(f"  CIK={cik}  canonical_shares={canon_shares.get(cik, 0.0):>20,.0f}")

    print()
    print("=== Diferencia (doble conteo) ===")
    for cik in TARGET_CIMS:
        r = raw_shares.get(cik, 0.0)
        c = canon_shares.get(cik, 0.0)
        print(f"  CIK={cik}  raw-canon={r - c:>20,.0f}  ratio={r/c if c else 0:.4f}")

    print()
    print("=== Detalle por ACCESSION para CIK 0002100119 ===")
    filtered_q1 = filtered
    sub_0119 = filtered_q1["SUBMISSION"].copy()
    sub_0119["CIK"] = sub_0119["CIK"].astype(str).str.zfill(10)
    sub_0119 = sub_0119[sub_0119["CIK"] == "0002100119"]
    print(sub_0119[["ACCESSION_NUMBER","SUBMISSIONTYPE","FILING_DATE"]].to_string())
    info_0119 = filtered_q1["INFOTABLE"].copy()
    info_0119["ACCESSION_NUMBER"] = info_0119["ACCESSION_NUMBER"].astype(str).str.strip()
    info_0119 = info_0119[info_0119["ACCESSION_NUMBER"].isin(sub_0119["ACCESSION_NUMBER"].astype(str).str.strip())]
    m = (info_0119["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info_0119["PUTCALL"].isna())
    info_0119 = info_0119[m].copy()
    info_0119["SSHPRNAMT_f"] = pd.to_numeric(info_0119["SSHPRNAMT"], errors="coerce").fillna(0.0)
    by_acc = info_0119.groupby("ACCESSION_NUMBER")["SSHPRNAMT_f"].sum()
    print()
    for acc, s in by_acc.items():
        applied = "APPLIED" if acc in am["applied_accessions"] else "SUPERSEDED"
        print(f"  ACC={acc}  shares={s:>20,.0f}  {applied}")
    print()
    print(f"  Suma raw:       {by_acc.sum():>20,.0f}")
    print(f"  Suma applied:   {by_acc[[a for a in by_acc.index if a in am['applied_accessions']]].sum():>20,.0f}")

    _sep("FIN RECONCILIACION")


if __name__ == "__main__":
    main()