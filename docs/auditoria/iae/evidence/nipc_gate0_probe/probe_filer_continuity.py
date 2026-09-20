# Probe filer continuity (Q-PROBE-5) Q4 2025 -> Q1 2026.
# Deterministico. Sin datetime.now().
# Uso: py probe_filer_continuity.py

import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period

PROBE_DIR = Path(r"D:\13f_probe\processed")
PERIODS = {
    "Q4_2025": {"dir": PROBE_DIR / "2025Q4", "period": "2025-12-31"},
    "Q1_2026": {"dir": PROBE_DIR / "2026Q1", "period": "2026-03-31"},
}
TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]
TOP_N = 20


def _sep(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_and_canonicalize(q):
    meta = PERIODS[q]
    d = meta["dir"]
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    filtered = filter_by_period(dfs, meta["period"])
    am = apply_amendments(filtered, period=meta["period"])
    return am["canonical_snapshot"], meta["period"]


def shares_by_cik(info_df, sub_df):
    m = (info_df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info_df["PUTCALL"].isna())
    df = info_df[m].copy()
    df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce").fillna(0.0)
    sub = sub_df[["ACCESSION_NUMBER", "CIK"]].drop_duplicates(subset=["ACCESSION_NUMBER"]).copy()
    sub["ACCESSION_NUMBER"] = sub["ACCESSION_NUMBER"].astype(str).str.strip()
    sub["CIK"] = sub["CIK"].astype(str).str.zfill(10)
    df["ACCESSION_NUMBER"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    df = df.merge(sub, on="ACCESSION_NUMBER", how="left")
    df["CIK"] = df["CIK"].fillna("")
    return df.groupby("CIK", dropna=False)["SSHPRNAMT_f"].sum()


def filing_types_by_cik(sub_df):
    sub = sub_df.copy()
    sub["CIK"] = sub["CIK"].astype(str).str.zfill(10)
    return sub.groupby("CIK")["SUBMISSIONTYPE"].apply(lambda s: sorted(set(s))).to_dict()


def cik_names(sub_df, cover_df):
    sub = sub_df.copy()
    sub["CIK"] = sub["CIK"].astype(str).str.zfill(10)
    acc_to_cik = dict(zip(sub["ACCESSION_NUMBER"].astype(str).str.strip(), sub["CIK"]))
    if cover_df is None or cover_df.empty:
        return {}
    cov = cover_df.copy()
    cov["ACCESSION_NUMBER"] = cov["ACCESSION_NUMBER"].astype(str).str.strip()
    cov["CIK"] = cov["ACCESSION_NUMBER"].map(acc_to_cik)
    cov = cov[cov["CIK"].notna()]
    names = {}
    for cik, grp in cov.groupby("CIK"):
        n = grp["FILINGMANAGER_NAME"].dropna()
        if len(n) > 0:
            names[cik] = str(n.iloc[0]).strip()
    return names


def load_submission_raw(q):
    p = PERIODS[q]["dir"] / "SUBMISSION.parquet"
    df = pd.read_parquet(p).copy()
    df["PERIODOFREPORT"] = pd.to_datetime(df["PERIODOFREPORT"], errors="coerce").dt.normalize()
    df = df[df["PERIODOFREPORT"] == pd.Timestamp(PERIODS[q]["period"])].copy()
    df["CIK"] = df["CIK"].astype(str).str.zfill(10)
    df["ACCESSION_NUMBER"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    return df


def load_othermanager_raw(q):
    p = PERIODS[q]["dir"] / "OTHERMANAGER.parquet"
    df = pd.read_parquet(p)
    df["ACCESSION_NUMBER"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    if "CIK" in df.columns:
        df["CIK_ref"] = df["CIK"].astype(str).str.zfill(10)
    else:
        df["CIK_ref"] = None
    if "NAME" not in df.columns:
        df["NAME"] = None
    return df


def classify_filer(present_q4, present_q1, types_q4, types_q1, nt_refs):
    if present_q4 and present_q1:
        has_nt_q4 = any(t == "13F-NT" for t in types_q4)
        has_nt_q1 = any(t == "13F-NT" for t in types_q1)
        has_hr_q4 = any(t.startswith("13F-HR") for t in types_q4)
        has_hr_q1 = any(t.startswith("13F-HR") for t in types_q1)
        if (has_nt_q4 and has_hr_q1) or (has_hr_q4 and has_nt_q1):
            return "NT_TO_HR_RELATION_OBSERVED" if nt_refs else "FILER_DISCONTINUITY"
        return "CONTINUOUS_FILER"
    if present_q4 != present_q1:
        return "FILER_DISCONTINUITY"
    return "UNRESOLVED"


def main():
    _sep("CARGANDO Q4 2025 y Q1 2026")
    snaps = {}
    for q in ("Q4_2025", "Q1_2026"):
        snap, period = load_and_canonicalize(q)
        snaps[q] = {"snap": snap, "period": period}
        print(f"{q} period={period} SUBMISSION={len(snap['SUBMISSION'])} INFOTABLE={len(snap['INFOTABLE'])}")

    shares = {}
    types = {}
    names = {}
    for q in ("Q4_2025", "Q1_2026"):
        snap = snaps[q]["snap"]
        shares[q] = shares_by_cik(snap["INFOTABLE"], snap["SUBMISSION"])
        types[q] = filing_types_by_cik(snap["SUBMISSION"])
        names[q] = cik_names(snap["SUBMISSION"], snap.get("COVERPAGE"))

    _sep(f"TOP {TOP_N} CIKs por sum(SSHPRNAMT) - Q4 2025")
    top_q4 = shares["Q4_2025"].sort_values(ascending=False).head(TOP_N)
    for cik, s in top_q4.items():
        print(f"  CIK={cik}  shares={s:>18,.0f}  name={names['Q4_2025'].get(cik, '?')[:50]}")

    _sep(f"TOP {TOP_N} CIKs por sum(SSHPRNAMT) - Q1 2026")
    top_q1 = shares["Q1_2026"].sort_values(ascending=False).head(TOP_N)
    for cik, s in top_q1.items():
        print(f"  CIK={cik}  shares={s:>18,.0f}  name={names['Q1_2026'].get(cik, '?')[:50]}")

    universe = sorted(set(top_q4.index) | set(top_q1.index))
    _sep(f"UNION TOP {TOP_N} (n={len(universe)}) - clasificacion observable")

    # NT refs
    sub_raw_q4 = load_submission_raw("Q4_2025")
    sub_raw_q1 = load_submission_raw("Q1_2026")
    om_raw_q4 = load_othermanager_raw("Q4_2025")
    om_raw_q1 = load_othermanager_raw("Q1_2026")

    nt_refs_by_acc = {}
    for q, sub_raw, om_raw in (("Q4_2025", sub_raw_q4, om_raw_q4),
                                ("Q1_2026", sub_raw_q1, om_raw_q1)):
        nt = sub_raw[sub_raw["SUBMISSIONTYPE"].isin(["13F-NT", "13F-NT/A"])]
        for _, row in nt.iterrows():
            acc = row["ACCESSION_NUMBER"]
            cik = row["CIK"]
            refs = om_raw[om_raw["ACCESSION_NUMBER"] == acc]["CIK_ref"].dropna().tolist()
            nt_refs_by_acc.setdefault(cik, []).extend(refs)

    statuses = Counter()
    for cik in universe:
        present_q4 = cik in types["Q4_2025"]
        present_q1 = cik in types["Q1_2026"]
        t_q4 = types["Q4_2025"].get(cik, [])
        t_q1 = types["Q1_2026"].get(cik, [])
        refs = nt_refs_by_acc.get(cik, [])
        cls = classify_filer(present_q4, present_q1, t_q4, t_q1, refs)
        statuses[cls] += 1
        s_q4 = shares["Q4_2025"].get(cik, 0.0)
        s_q1 = shares["Q1_2026"].get(cik, 0.0)
        print(f"  CIK={cik}  Q4={'si' if present_q4 else 'no'}  Q1={'si' if present_q1 else 'no'}"
              f"  Q4_shares={s_q4:>15,.0f}  Q1_shares={s_q1:>15,.0f}  {cls}")

    _sep("RESUMEN ESTADOS")
    for cls, n in sorted(statuses.items()):
        print(f"  {cls}: {n}")

    _sep("CASO VANGUARD - NT padre declara managers")
    vg_parent = "0000102909"
    nt_vg_q1 = sub_raw_q1[(sub_raw_q1["CIK"] == vg_parent) & (sub_raw_q1["SUBMISSIONTYPE"].isin(["13F-NT", "13F-NT/A"]))]
    print(f"NT filings Q1 del padre: {len(nt_vg_q1)}")
    for _, row in nt_vg_q1.iterrows():
        acc = row["ACCESSION_NUMBER"]
        refs = om_raw_q1[om_raw_q1["ACCESSION_NUMBER"] == acc]
        print(f"  ACC={acc}  refs_OM={len(refs)}")
        for _, r in refs.iterrows():
            ref_cik = r["CIK_ref"]
            ref_name = r["NAME"]
            s_q4 = shares["Q4_2025"].get(ref_cik, 0.0)
            s_q1 = shares["Q1_2026"].get(ref_cik, 0.0)
            t_q4 = types["Q4_2025"].get(ref_cik, [])
            t_q1 = types["Q1_2026"].get(ref_cik, [])
            print(f"    CIK={ref_cik}  name={str(ref_name)[:45]:<45}  "
                  f"Q4_types={t_q4}  Q4_shares={s_q4:>15,.0f}  "
                  f"Q1_types={t_q1}  Q1_shares={s_q1:>15,.0f}")

    _sep("FIN DEL MINI-PROBE")


if __name__ == "__main__":
    main()