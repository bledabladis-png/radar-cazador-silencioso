# Probe filer continuity TOP 50 (Q-MINI-5).
# Deterministico. Sin datetime.now().
# Taxonomia Q-MINI-2: filer_status + nt_to_hr_relation_observed + targets.

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
TOP_N = 50


def _sep(t):
    print()
    print("=" * 72)
    print(t)
    print("=" * 72)


def load_canonical(q):
    meta = PERIODS[q]
    d = meta["dir"]
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    filtered = filter_by_period(dfs, meta["period"])
    am = apply_amendments(filtered, period=meta["period"])
    return am["canonical_snapshot"]


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
    df["CIK_ref"] = df["CIK"].astype(str).str.zfill(10) if "CIK" in df.columns else None
    if "NAME" not in df.columns:
        df["NAME"] = None
    return df


def shares_by_cik(info_df, sub_df):
    m = (info_df["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info_df["PUTCALL"].isna())
    df = info_df[m].copy()
    df["SSHPRNAMT_f"] = pd.to_numeric(df["SSHPRNAMT"], errors="coerce").fillna(0.0)
    s = sub_df[["ACCESSION_NUMBER", "CIK"]].drop_duplicates(subset=["ACCESSION_NUMBER"]).copy()
    s["ACCESSION_NUMBER"] = s["ACCESSION_NUMBER"].astype(str).str.strip()
    s["CIK"] = s["CIK"].astype(str).str.zfill(10)
    df["ACCESSION_NUMBER"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    df = df.merge(s, on="ACCESSION_NUMBER", how="left")
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


def classify_filer(present_q4, present_q1):
    if present_q4 and present_q1:
        return "CONTINUOUS_FILER"
    if present_q4 != present_q1:
        return "FILER_DISCONTINUITY"
    return "UNRESOLVED"


def main():
    _sep("CARGANDO Q4 2025 y Q1 2026")
    snaps = {}
    for q in ("Q4_2025", "Q1_2026"):
        snaps[q] = load_canonical(q)
        print(f"{q} SUBMISSION={len(snaps[q]['SUBMISSION'])} INFOTABLE={len(snaps[q]['INFOTABLE'])}")

    shares = {}
    types = {}
    names = {}
    for q in ("Q4_2025", "Q1_2026"):
        snap = snaps[q]
        shares[q] = shares_by_cik(snap["INFOTABLE"], snap["SUBMISSION"])
        types[q] = filing_types_by_cik(snap["SUBMISSION"])
        names[q] = cik_names(snap["SUBMISSION"], snap.get("COVERPAGE"))

    _sep(f"TOP {TOP_N} CIKs por sum(SSHPRNAMT) - Q4 2025")
    top_q4 = shares["Q4_2025"].sort_values(ascending=False).head(TOP_N)
    for cik, s in top_q4.items():
        print(f"  CIK={cik}  shares={s:>18,.0f}  name={names['Q4_2025'].get(cik, '?')[:55]}")

    _sep(f"TOP {TOP_N} CIKs por sum(SSHPRNAMT) - Q1 2026")
    top_q1 = shares["Q1_2026"].sort_values(ascending=False).head(TOP_N)
    for cik, s in top_q1.items():
        print(f"  CIK={cik}  shares={s:>18,.0f}  name={names['Q1_2026'].get(cik, '?')[:55]}")

    universe = sorted(set(top_q4.index) | set(top_q1.index))
    _sep(f"UNION TOP {TOP_N} (n={len(universe)})")

    # NT refs
    nt_refs_by_cik_q4 = {}
    nt_refs_by_cik_q1 = {}
    sub_raw_q4 = load_submission_raw("Q4_2025")
    sub_raw_q1 = load_submission_raw("Q1_2026")
    om_raw_q4 = load_othermanager_raw("Q4_2025")
    om_raw_q1 = load_othermanager_raw("Q1_2026")

    for q, sub_raw, om_raw, target in (("Q4_2025", sub_raw_q4, om_raw_q4, nt_refs_by_cik_q4),
                                        ("Q1_2026", sub_raw_q1, om_raw_q1, nt_refs_by_cik_q1)):
        nt = sub_raw[sub_raw["SUBMISSIONTYPE"].isin(["13F-NT", "13F-NT/A"])]
        for _, row in nt.iterrows():
            acc = row["ACCESSION_NUMBER"]
            cik = row["CIK"]
            if cik not in universe:
                continue
            refs = om_raw[om_raw["ACCESSION_NUMBER"] == acc][["CIK_ref", "NAME"]].dropna(subset=["CIK_ref"])
            if cik not in target:
                target[cik] = []
            for _, r in refs.iterrows():
                target[cik].append({"cik": r["CIK_ref"], "name": str(r["NAME"])})

    _sep("CLASIFICACION FILER STATUS + NT_TO_HR")
    statuses = Counter()
    rows = []
    for cik in universe:
        present_q4 = cik in types["Q4_2025"]
        present_q1 = cik in types["Q1_2026"]
        filer_status = classify_filer(present_q4, present_q1)
        refs_q4 = nt_refs_by_cik_q4.get(cik, [])
        refs_q1 = nt_refs_by_cik_q1.get(cik, [])
        all_refs = refs_q4 + refs_q1
        nt_observed = len(all_refs) > 0
        statuses[filer_status] += 1
        rows.append({
            "cik": cik,
            "present_q4": present_q4,
            "present_q1": present_q1,
            "shares_q4": shares["Q4_2025"].get(cik, 0.0),
            "shares_q1": shares["Q1_2026"].get(cik, 0.0),
            "filer_status": filer_status,
            "nt_to_hr_observed": nt_observed,
            "nt_targets": [r["cik"] for r in all_refs],
            "name": names["Q1_2026"].get(cik) or names["Q4_2025"].get(cik) or "?",
        })

    for r in rows:
        print(f"  CIK={r['cik']}  Q4={'si' if r['present_q4'] else 'no'}"
              f"  Q1={'si' if r['present_q1'] else 'no'}"
              f"  Q4_sh={r['shares_q4']:>15,.0f}  Q1_sh={r['shares_q1']:>15,.0f}"
              f"  {r['filer_status']}"
              f"  nt_hr={r['nt_to_hr_observed']}"
              f"  targets={r['nt_targets'][:3]}")

    _sep("RESUMEN ESTADOS FILER (Q-MINI-2)")
    for s, n in sorted(statuses.items()):
        print(f"  {s}: {n}")

    n_discont = statuses.get("FILER_DISCONTINUITY", 0)
    n_total = sum(statuses.values())
    print(f"  Total: {n_total}")
    print(f"  pct_discontinuity: {n_discont/n_total if n_total else 0:.4f}")

    _sep("FILER_DISCONTINUITY - DETALLE")
    for r in rows:
        if r["filer_status"] == "FILER_DISCONTINUITY":
            print(f"  CIK={r['cik']}  name={r['name'][:60]}")
            print(f"    Q4_present={r['present_q4']}  Q4_sh={r['shares_q4']:>18,.0f}")
            print(f"    Q1_present={r['present_q1']}  Q1_sh={r['shares_q1']:>18,.0f}")
            print(f"    nt_to_hr_observed={r['nt_to_hr_observed']}")
            print(f"    nt_targets={r['nt_targets']}")

    _sep("NT_TO_HR OBSERVED - DETALLE")
    for r in rows:
        if r["nt_to_hr_observed"]:
            print(f"  CIK={r['cik']}  name={r['name'][:60]}  filer_status={r['filer_status']}")
            print(f"    targets={r['nt_targets']}")

    _sep("FIN TOP 50")


if __name__ == "__main__":
    main()