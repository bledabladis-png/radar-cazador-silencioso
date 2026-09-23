"""Reproduce §12.3 con semantica contractual correcta.

El TARGET de cada periodo = keys del catalogo radar con sshprnamt
observado en ese periodo (VERIFIED + CUSIP en crosswalk). Las keys sin
observacion no entran al TargetUniverse del periodo.
"""
from __future__ import annotations
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, resolve_batch_identities)
from src.institutional_accumulation import operational_universe as ou
from src.institutional_accumulation.identity import period_state as ps
from src.institutional_accumulation.identity.target_builder import (
    build_target, TargetUniverse)
from src.institutional_accumulation.aggregation import catalog_p38_adapter as ca
from src.institutional_accumulation.aggregation import coverage as cov

DATA = ROOT / "data" / "sec_13f" / "processed"
MAPPINGS = ROOT / "data" / "mappings"
OUT = ROOT / "outputs" / "audit" / "contractual_coverage"
OFFICIAL = Path(r"D:\13f_probe\official_list_13f")
TSVS = ("SUBMISSION","COVERPAGE","SUMMARYPAGE","OTHERMANAGER",
        "OTHERMANAGER2","SIGNATURE","INFOTABLE")
PERIODS = {"2025Q4": "2025-12-31", "2026Q1": "2026-03-31"}


def subset_universe(u: TargetUniverse, keys: set) -> TargetUniverse:
    """TargetUniverse reducido a las keys indicadas."""
    ks = set(keys)
    return TargetUniverse(
        version_id=u.version_id,
        period_end=u.period_end,
        catalog_version_id=u.catalog_version_id,
        catalog_sha256=u.catalog_sha256,
        declared_keys=ks,
        ticker_by_key={k: u.ticker_by_key[k] for k in ks},
        figi_by_key={k: u.figi_by_key[k] for k in ks},
        row_uid_by_key={k: u.row_uid_by_key[k] for k in ks},
        key_by_row_uid={u.row_uid_by_key[k]: k for k in ks},
    )


def load_canonical(folder, iso):
    d = DATA / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    f = filter_by_period(dfs, iso)
    return apply_amendments(f, period=iso)["canonical_snapshot"]


def git_head():
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("REPRODUCCION §12.3 (semantica contractual estricta)")
    print("=" * 72)

    # Universo catalogo
    snap = pd.read_csv(MAPPINGS / "radar_target_catalog.csv", dtype=str)
    mem = pd.read_csv(MAPPINGS / "catalog_membership.csv", dtype=str)
    asg = pd.read_csv(MAPPINGS / "catalog_assignments.csv", dtype=str)
    version_id = str(mem["version_id"].iloc[0])
    u_full = build_target(snap, mem, asg, version_id=version_id,
                          period_end="2026-03-31",
                          catalog_version_id="cat", catalog_sha256="a"*64)
    print("\nUniverso catalogo: " + str(len(u_full.declared_keys)) + " keys")

    # CUSIP -> FIGI radar via crosswalk
    figi_by_ticker = dict(zip(snap["radar_ticker"], snap["share_class_figi"]))
    cw_csv = pd.read_csv(MAPPINGS / "cusip_radar_crosswalk.csv", dtype=str)
    cusip_to_figi = {}
    for _, r in cw_csv.iterrows():
        tk = str(r["ticker"]).strip()
        f = figi_by_ticker.get(tk)
        if pd.notna(f) and f:
            cusip_to_figi[str(r["CUSIP"]).strip()] = f
    figi_to_key = {u_full.figi_by_key[k]: k for k in u_full.declared_keys}
    print("CUSIPs mapeados: " + str(len(cusip_to_figi)))
    print("FIGIs radar -> keys: " + str(len(figi_to_key)))

    # Por periodo
    per_period = {}
    for folder, iso in PERIODS.items():
        print("\n[" + folder + "]")
        sc = load_canonical(folder, iso)
        cw = load_crosswalk_internal()
        cusips = sc["INFOTABLE"]["CUSIP"].astype(str).unique()
        eqp = MAPPINGS / "cusip_equivalence.csv"
        eq = pd.read_csv(eqp, dtype=str) if eqp.exists() else None
        idr = resolve_batch_identities(cusips, iso, equivalence_df=eq,
            crosswalk_internal_df=cw, figi_lookup=None)
        off = sl.load_official_list(OFFICIAL / ("13flist_" + folder + ".txt"))
        oper = ou.build_operational_universe(
            sc["INFOTABLE"], off, iso,
            identity_results=idr, identity_period_iso=iso)
        print("  filas oper: " + str(len(oper)))

        # Filtrar por VERIFIED + CUSIP en crosswalk
        op = oper.copy()
        op["_figi_radar"] = op["CUSIP"].astype(str).str.strip().map(cusip_to_figi)
        if "_operational_mapping_status" in op.columns:
            op = op[op["_operational_mapping_status"] == "VERIFIED"]
        op = op[op["_figi_radar"].notna()]

        # Agregar sshprnamt por FIGI
        sshp = pd.to_numeric(op["SSHPRNAMT"], errors="coerce").fillna(0.0)
        sh_by_figi = {}
        for f, v in zip(op["_figi_radar"], sshp):
            sh_by_figi[f] = sh_by_figi.get(f, 0.0) + float(v)

        # FIGI -> key -> sshprnamt
        sh_by_key = {}
        for f, v in sh_by_figi.items():
            k = figi_to_key.get(f)
            if k:
                sh_by_key[k] = v
        print("  figis con sshprnamt: " + str(len(sh_by_figi)))
        print("  keys con sshprnamt: " + str(len(sh_by_key)))

        per_period[folder] = {
            "keys": set(sh_by_key.keys()),
            "sh_by_key": sh_by_key,
            "n_oper": len(oper),
        }

    # Sub-universos por periodo
    print("\nSub-universos:")
    u_q4 = subset_universe(u_full, per_period["2025Q4"]["keys"])
    u_q1 = subset_universe(u_full, per_period["2026Q1"]["keys"])
    print("  Q4 universo: " + str(len(u_q4.declared_keys)))
    print("  Q1 universo: " + str(len(u_q1.declared_keys)))

    pairwise = per_period["2025Q4"]["keys"] & per_period["2026Q1"]["keys"]
    print("  pairwise: " + str(len(pairwise)))

    # State por periodo (todas las keys del sub-universo son RESOLVED por def)
    st_q4 = ps.build_period_state(
        u_q4,
        operational_evidence={k: "VERIFIED" for k in u_q4.declared_keys},
        sshprnamt_evidence=per_period["2025Q4"]["sh_by_key"])
    st_q1 = ps.build_period_state(
        u_q1,
        operational_evidence={k: "VERIFIED" for k in u_q1.declared_keys},
        sshprnamt_evidence=per_period["2026Q1"]["sh_by_key"])

    # Adapter
    print("\nAdapter...")
    t4, t1, r4, r1, feas = ca.catalog_to_p38_targets(
        u_q4, u_q1, state_q4=st_q4, state_q1=st_q1,
        pairwise_keys=pairwise)
    print("  feasibility: " + str(feas))
    print("  TARGET_Q4: " + str(len(t4)))
    print("  TARGET_Q1: " + str(len(t1)))
    print("  records_q4: " + str(len(r4)))
    print("  records_q1: " + str(len(r1)))

    # Coverage
    print("\nCoverage:")
    result = cov.compute_contractual_coverage(
        target_q4=t4, target_q1=t1, records_q4=r4, records_q1=r1)
    for k, v in result.items():
        print("  " + k + ": " + str(v))

    # JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT / (ts + "_coverage.json")
    payload = {
        "run_id": ts, "git_head": git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
        "universe_size": len(u_full.declared_keys),
        "q4_keys_observed": len(per_period["2025Q4"]["keys"]),
        "q1_keys_observed": len(per_period["2026Q1"]["keys"]),
        "pairwise_size": len(pairwise),
        "adapter": {
            "n_target_q4": len(t4), "n_target_q1": len(t1),
            "n_records_q4": len(r4), "n_records_q1": len(r1),
            "feasibility": str(feas),
        },
        "coverage": result,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nJSON: " + str(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())