"""A.1 v3 - Reconciliacion B1 definitiva.

Conclusion de v2: delta_full esta partido en dos subpoblaciones disjuntas:
  - UNRESOLVED: osk=cusip:XXX, cs=None
  - CANONICAL:  osk=None,        cs=equity:XXX
Split A (por CUSIP) es inviable. Split B (por ticker) es el correcto.
Este script:
  1. Verifica la particion disjunta.
  2. Lista el radar ticker ausente en delta_B.
  3. Congela JSON definitivo con los numeros para §12.6.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, resolve_batch_identities)
from src.institutional_accumulation.aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares)
from src.institutional_accumulation.aggregation.nipc import compute_nipc

DATA_DIR = ROOT / "data" / "sec_13f" / "processed"
MAPPINGS = ROOT / "data" / "mappings"
OUT_DIR = ROOT / "outputs" / "audit" / "b1_reconciliation"
TSVS = ("SUBMISSION","COVERPAGE","SUMMARYPAGE","OTHERMANAGER",
        "OTHERMANAGER2","SIGNATURE","INFOTABLE")
PERIODS = {"2025Q4": "2025-12-31", "2026Q1": "2026-03-31"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest().upper()


def git_head() -> str:
    try:
        r = subprocess.run(["git","rev-parse","HEAD"], cwd=str(ROOT),
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def load_canonical(folder: str, period_iso: str) -> dict:
    d = DATA_DIR / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    f = filter_by_period(dfs, period_iso)
    return apply_amendments(f, period=period_iso)["canonical_snapshot"]


def build_identities(snap: dict, period_iso: str) -> dict:
    cusips = snap["INFOTABLE"]["CUSIP"].astype(str).unique()
    cw = load_crosswalk_internal()
    eqp = MAPPINGS / "cusip_equivalence.csv"
    eq = pd.read_csv(eqp, dtype=str) if eqp.exists() else None
    return resolve_batch_identities(cusips, period_iso,
        equivalence_df=eq, crosswalk_internal_df=cw, figi_lookup=None)


def ticker_of(s: pd.Series) -> pd.Series:
    x = s.fillna("").astype(str).str.strip()
    ok = x.str.startswith("equity:")
    return x.str.replace("equity:", "", regex=False).where(ok, None)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("A.1 v3 - RECONCILIACION B1 DEFINITIVA")
    print("=" * 72)

    print("\n[1/6] Canonical + identidades + units + delta...")
    s4 = load_canonical("2025Q4", PERIODS["2025Q4"])
    s1 = load_canonical("2026Q1", PERIODS["2026Q1"])
    i4 = build_identities(s4, PERIODS["2025Q4"])
    i1 = build_identities(s1, PERIODS["2026Q1"])
    u4 = compute_reported_position_units(s4["INFOTABLE"], s4["SUBMISSION"],
            report_period=PERIODS["2025Q4"], identity_results=i4)
    u1 = compute_reported_position_units(s1["INFOTABLE"], s1["SUBMISSION"],
            report_period=PERIODS["2026Q1"], identity_results=i1)
    delta = compute_delta_shares(u1, u4)
    print("  delta_full filas: " + str(len(delta)))

    print("\n[2/6] Verificando particion disjunta...")
    osk_ok = delta["observed_security_key"].fillna("").astype(str).str.startswith("cusip:")
    cs_ok  = delta["canonical_security"].fillna("").astype(str).str.startswith("equity:")
    both    = int((osk_ok & cs_ok).sum())
    neither = int((~osk_ok & ~cs_ok).sum())
    print("  filas con ambos:   " + str(both))
    print("  filas con ninguno: " + str(neither))
    partition_ok = (both == 0) and (neither == 0)
    print("  particion limpia:  " + ("PASS" if partition_ok else "FAIL"))

    print("\n[3/6] Cargando radar...")
    cat = pd.read_csv(MAPPINGS / "radar_target_catalog.csv", dtype=str)
    radar_tickers = set(cat["radar_ticker"].dropna().astype(str).str.strip())
    print("  radar_tickers: " + str(len(radar_tickers)))

    print("\n[4/6] Split B y ticker ausente...")
    d = delta.copy()
    d["_ticker"] = ticker_of(d["canonical_security"])
    mask_B = d["_ticker"].isin(radar_tickers)
    radar_B = d[mask_B].copy()
    comp_B  = d[~mask_B].copy()
    tickers_presentes = set(radar_B["_ticker"].dropna().unique())
    ausentes = sorted(radar_tickers - tickers_presentes)
    print("  radar_B filas: " + str(len(radar_B))
          + "  comp_B: " + str(len(comp_B)))
    print("  tickers en radar_B: " + str(len(tickers_presentes))
          + "  ausentes: " + str(len(ausentes)))
    if ausentes:
        print("  lista ausentes: " + ", ".join(ausentes))

    print("\n[5/6] NIPC...")
    r_full = compute_nipc(delta)
    r_rB   = compute_nipc(radar_B)
    r_cB   = compute_nipc(comp_B)
    print("  full:        " + str(r_full["nipc_total"]))
    print("  radar_B:     " + str(r_rB["nipc_total"]))
    print("  comp_B:      " + str(r_cB["nipc_total"]))

    sum_B  = r_rB["nipc_total"] + r_cB["nipc_total"]
    ok_sum = sum_B == r_full["nipc_total"]
    ok_ne  = len(radar_B) > 0

    print("\n" + "=" * 72)
    print("ASSERTS DEFINITIVOS")
    print("=" * 72)
    print("  particion disjunta:       " + ("PASS" if partition_ok else "FAIL"))
    print("  radar_B + comp_B == full: " + ("PASS" if ok_sum else "FAIL")
          + "  (" + str(sum_B) + " vs " + str(r_full["nipc_total"]) + ")")
    print("  radar_B no vacio:         " + ("PASS" if ok_ne else "FAIL"))

    passed = bool(partition_ok and ok_sum and ok_ne)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / (ts + "_b1_final.json")
    payload = {
        "run_id": ts,
        "git_head": git_head(),
        "pandas_version": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_q4_infotable": sha256(DATA_DIR / "2025Q4" / "INFOTABLE.parquet"),
        "sha256_q1_infotable": sha256(DATA_DIR / "2026Q1" / "INFOTABLE.parquet"),
        "split_oficial": "B_por_canonical_security_ticker",
        "particion_delta": {
            "unresolved_osk": int(osk_ok.sum()),
            "canonical_cs":   int(cs_ok.sum()),
            "ambos": both, "ninguno": neither,
        },
        "counts": {
            "delta_full_rows": len(delta),
            "radar_B_rows": len(radar_B), "comp_B_rows": len(comp_B),
            "radar_tickers_catalogo": len(radar_tickers),
            "radar_tickers_presentes": len(tickers_presentes),
            "radar_tickers_ausentes": ausentes,
        },
        "nipc": {
            "full": r_full, "radar": r_rB, "complemento": r_cB,
        },
        "reconciliacion_12_6": {
            "radar_anterior":      -4316734936.0,
            "complemento_anterior":  164048507.0,
            "full_anterior":      -1404191665.0,
            "radar_nuevo":          r_rB["nipc_total"],
            "complemento_nuevo":    r_cB["nipc_total"],
            "full_nuevo":           r_full["nipc_total"],
        },
        "asserts": {
            "particion_disjunta": partition_ok,
            "radar_plus_comp_eq_full": ok_sum,
            "radar_non_empty": ok_ne,
        },
        "passed": passed,
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nJSON: " + str(out))
    print("RESULTADO: " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())