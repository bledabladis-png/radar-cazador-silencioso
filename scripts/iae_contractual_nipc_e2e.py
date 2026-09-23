"""E.1 - Validacion E2E de compute_nipc_contractual contra §12.5.

Encadena la cadena contractual completa:
  canonical -> identities -> units -> delta -> filtro radar
  -> build_target + catalog_to_p38_targets
  -> compute_nipc_contractual

No reimplementa la logica de NIPC ni de coverage: invoca
compute_nipc_contractual con los inputs producidos por el pipeline.

Salida esperada (identica a §12.5):
  delta_radar_rows = 553321
  nipc_total       = -4316734936.0
  nipc_sole        = -32484713330.0
  nipc_dfnd        = +28317652408.0
  nipc_otr         = -149674014.0
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
from src.institutional_accumulation.aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares)
from src.institutional_accumulation.aggregation.nipc import compute_nipc_contractual

DATA = ROOT / "data" / "sec_13f" / "processed"
MAPPINGS = ROOT / "data" / "mappings"
OUT = ROOT / "outputs" / "audit" / "iae_e1_contractual_nipc"
OFFICIAL = Path(r"D:\13f_probe\official_list_13f")
TSVS = ("SUBMISSION","COVERPAGE","SUMMARYPAGE","OTHERMANAGER",
        "OTHERMANAGER2","SIGNATURE","INFOTABLE")
PERIODS = {"2025Q4": "2025-12-31", "2026Q1": "2026-03-31"}

EXPECTED = {
    "delta_radar_rows": 553321,
    "nipc_total": -4316734936.0,
    "nipc_sole":  -32484713330.0,
    "nipc_dfnd":   28317652408.0,
    "nipc_otr":    -149674014.0,
}


def git_head():
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"

def load_canonical(folder, iso):
    d = DATA / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    f = filter_by_period(dfs, iso)
    return apply_amendments(f, period=iso)["canonical_snapshot"]


def build_identities(snap, iso):
    cusips = snap["INFOTABLE"]["CUSIP"].astype(str).unique()
    cw = load_crosswalk_internal()
    eqp = MAPPINGS / "cusip_equivalence.csv"
    eq = pd.read_csv(eqp, dtype=str) if eqp.exists() else None
    return resolve_batch_identities(cusips, iso, equivalence_df=eq,
        crosswalk_internal_df=cw, figi_lookup=None)


def subset_universe(u, keys):
    ks = set(keys)
    return TargetUniverse(
        version_id=u.version_id, period_end=u.period_end,
        catalog_version_id=u.catalog_version_id,
        catalog_sha256=u.catalog_sha256,
        declared_keys=ks,
        ticker_by_key={k: u.ticker_by_key[k] for k in ks},
        figi_by_key={k: u.figi_by_key[k] for k in ks},
        row_uid_by_key={k: u.row_uid_by_key[k] for k in ks},
        key_by_row_uid={u.row_uid_by_key[k]: k for k in ks},
    )


def ticker_of(s):
    x = s.fillna("").astype(str).str.strip()
    ok = x.str.startswith("equity:")
    return x.str.replace("equity:", "", regex=False).where(ok, None)

def run_pipeline():
    """Ejecuta la cadena contractual completa y devuelve el dict resultado."""
    # 1. Canonical
    print("[1/7] Cargando canonical Q4/Q1...")
    s4 = load_canonical("2025Q4", PERIODS["2025Q4"])
    s1 = load_canonical("2026Q1", PERIODS["2026Q1"])

    # 2. Identidades P61
    print("[2/7] Resolviendo identidades P61...")
    i4 = build_identities(s4, PERIODS["2025Q4"])
    i1 = build_identities(s1, PERIODS["2026Q1"])

    # 3. Units
    print("[3/7] compute_reported_position_units...")
    u4 = compute_reported_position_units(
        s4["INFOTABLE"], s4["SUBMISSION"],
        report_period=PERIODS["2025Q4"], identity_results=i4)
    u1 = compute_reported_position_units(
        s1["INFOTABLE"], s1["SUBMISSION"],
        report_period=PERIODS["2026Q1"], identity_results=i1)

    # 4. Delta full
    print("[4/7] compute_delta_shares (full)...")
    delta = compute_delta_shares(u1, u4)
    print("  delta_full rows: " + str(len(delta)))

    # 5. Delta radar
    print("[5/7] Filtrando delta al radar...")
    cat = pd.read_csv(MAPPINGS / "radar_target_catalog.csv", dtype=str)
    radar_tickers = set(cat["radar_ticker"].dropna().astype(str).str.strip())
    d = delta.copy()
    d["_ticker"] = ticker_of(d["canonical_security"])
    radar = d[d["_ticker"].isin(radar_tickers)].copy()
    print("  delta_radar rows: " + str(len(radar)))

    # 6. Universo + target + adapter
    print("[6/7] Construyendo universo + adapter...")
    snap = pd.read_csv(MAPPINGS / "radar_target_catalog.csv", dtype=str)
    mem = pd.read_csv(MAPPINGS / "catalog_membership.csv", dtype=str)
    asg = pd.read_csv(MAPPINGS / "catalog_assignments.csv", dtype=str)
    version_id = str(mem["version_id"].iloc[0])
    u_full = build_target(snap, mem, asg, version_id=version_id,
                          period_end="2026-03-31",
                          catalog_version_id="cat", catalog_sha256="a"*64)
    figi_by_ticker = dict(zip(snap["radar_ticker"], snap["share_class_figi"]))
    cw_csv = pd.read_csv(MAPPINGS / "cusip_radar_crosswalk.csv", dtype=str)
    cusip_to_figi = {}
    for _, r in cw_csv.iterrows():
        tk = str(r["ticker"]).strip()
        f = figi_by_ticker.get(tk)
        if pd.notna(f) and f:
            cusip_to_figi[str(r["CUSIP"]).strip()] = f
    figi_to_key = {u_full.figi_by_key[k]: k for k in u_full.declared_keys}

    # Sub-universos por periodo
    per_period = {}
    for folder, iso, units in (("2025Q4", PERIODS["2025Q4"], u4),
                                ("2026Q1", PERIODS["2026Q1"], u1)):
        # Reutilizamos las identidades ya resueltas (i4/i1)
        sc = load_canonical(folder, iso)
        idmap = i4 if folder == "2025Q4" else i1
        off = sl.load_official_list(OFFICIAL / ("13flist_" + folder + ".txt"))
        oper = ou.build_operational_universe(
            sc["INFOTABLE"], off, iso,
            identity_results=idmap, identity_period_iso=iso)
        op = oper.copy()
        op["_figi_radar"] = op["CUSIP"].astype(str).str.strip().map(cusip_to_figi)
        if "_operational_mapping_status" in op.columns:
            op = op[op["_operational_mapping_status"] == "VERIFIED"]
        op = op[op["_figi_radar"].notna()]
        sshp = pd.to_numeric(op["SSHPRNAMT"], errors="coerce").fillna(0.0)
        sh_by_figi = {}
        for f, v in zip(op["_figi_radar"], sshp):
            sh_by_figi[f] = sh_by_figi.get(f, 0.0) + float(v)
        sh_by_key = {}
        for f, v in sh_by_figi.items():
            k = figi_to_key.get(f)
            if k:
                sh_by_key[k] = v
        per_period[folder] = {"keys": set(sh_by_key.keys()),
                              "sh_by_key": sh_by_key}

    u_q4 = subset_universe(u_full, per_period["2025Q4"]["keys"])
    u_q1 = subset_universe(u_full, per_period["2026Q1"]["keys"])
    pairwise = per_period["2025Q4"]["keys"] & per_period["2026Q1"]["keys"]

    st_q4 = ps.build_period_state(
        u_q4,
        operational_evidence={k: "VERIFIED" for k in u_q4.declared_keys},
        sshprnamt_evidence=per_period["2025Q4"]["sh_by_key"])
    st_q1 = ps.build_period_state(
        u_q1,
        operational_evidence={k: "VERIFIED" for k in u_q1.declared_keys},
        sshprnamt_evidence=per_period["2026Q1"]["sh_by_key"])

    t4, t1, r4, r1, feas = ca.catalog_to_p38_targets(
        u_q4, u_q1, state_q4=st_q4, state_q1=st_q1,
        pairwise_keys=pairwise)
    print("  feasibility: " + str(feas))

    # 7. compute_nipc_contractual
    print("[7/7] compute_nipc_contractual...")
    result = compute_nipc_contractual(
        radar, target_q4=t4, target_q1=t1, records_q4=r4, records_q1=r1)

    return {
        "delta_radar_rows": int(len(radar)),
        "delta_full_rows": int(len(delta)),
        "target_q4_size": int(len(t4)),
        "target_q1_size": int(len(t1)),
        "feasibility": str(feas),
        "result": result,
    }

def reconcile(result, expected):
    """Compara resultado con §12.5. Devuelve dict de checks."""
    r = result["result"]
    checks = []
    def check(name, actual, exp, tol=0.0):
        ok = abs(actual - exp) <= tol if isinstance(exp, (int, float)) \
             else actual == exp
        checks.append({"name": name, "actual": actual, "expected": exp, "pass": ok})

    check("delta_radar_rows", result["delta_radar_rows"], expected["delta_radar_rows"])
    check("nipc_total", r.get("nipc_total"), expected["nipc_total"])
    check("nipc_sole",  r.get("nipc_sole"),  expected["nipc_sole"])
    check("nipc_dfnd",  r.get("nipc_dfnd"),  expected["nipc_dfnd"])
    check("nipc_otr",   r.get("nipc_otr"),   expected["nipc_otr"])
    check("evidence_class", r.get("evidence_class"), "CONTRACTUAL")
    return checks


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("E.1 - VALIDACION E2E compute_nipc_contractual vs §12.5")
    print("=" * 72)

    # Primera ejecucion
    print("\n--- Ejecucion 1 ---")
    r1 = run_pipeline()
    checks1 = reconcile(r1, EXPECTED)

    # Segunda ejecucion (determinismo)
    print("\n--- Ejecucion 2 (determinismo) ---")
    r2 = run_pipeline()

    det_total = (r1["result"]["nipc_total"] == r2["result"]["nipc_total"])
    det_rows  = (r1["delta_radar_rows"] == r2["delta_radar_rows"])

    print("\n" + "=" * 72)
    print("RECONCILIACION §12.5")
    print("=" * 72)
    for c in checks1:
        m = "PASS" if c["pass"] else "FAIL"
        print(f"  [{m}] {c['name']:20s} actual={c['actual']}  esperado={c['expected']}")
    print(f"  [{'PASS' if det_total else 'FAIL'}] determinismo_nipc_total")
    print(f"  [{'PASS' if det_rows else 'FAIL'}] determinismo_delta_radar_rows")

    all_pass = all(c["pass"] for c in checks1) and det_total and det_rows

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT / (ts + "_iae_e1.json")
    payload = {
        "run_id": ts,
        "git_head": git_head(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pandas_version": pd.__version__,
        "expected_12_5": EXPECTED,
        "execution_1": {k: v for k, v in r1.items() if k != "result"},
        "execution_1_result": {k: v for k, v in r1["result"].items()
                                if not isinstance(v, (list, dict)) or k in ("status","evidence_class")},
        "determinism": {"nipc_total": det_total, "delta_radar_rows": det_rows},
        "checks": checks1,
        "passed": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nJSON: " + str(out))
    print("RESULTADO: " + ("PASS" if all_pass else "FAIL"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())