"""A.1 - Auditoria de unicidad de identidad canonica por shareClassFIGI.

Verifica empiricamente si una misma shareClassFIGI puede llegar al delta
bajo mas de un canonical_security (equity:TICKER vs figi:FIGI).

No modifica el motor. Post-procesa units/delta producidos por el pipeline
de iae_reconciliation_b1.py.
"""
from __future__ import annotations
import hashlib, json, subprocess, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
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
OUT_DIR = ROOT / "outputs" / "audit" / "iae_identity_audit"
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

def load_catalog_index():
    """Devuelve (ticker_to_scf, scf_to_ticker, composite_to_ticker)."""
    cat = pd.read_csv(MAPPINGS / "radar_target_catalog.csv", dtype=str)
    ticker_to_scf = defaultdict(set)
    scf_to_ticker = {}
    composite_to_ticker = {}
    for _, r in cat.iterrows():
        t = str(r.get("radar_ticker", "")).strip() or None
        scf = str(r.get("share_class_figi", "")).strip() or None
        comp = str(r.get("composite_figi", "")).strip() or None
        if t and scf and scf != "nan":
            ticker_to_scf[t].add(scf)
            scf_to_ticker.setdefault(scf, t)
        if t and comp and comp != "nan":
            composite_to_ticker.setdefault(comp, t)
    return dict(ticker_to_scf), scf_to_ticker, composite_to_ticker


def resolve_row(cs, ticker_to_scf, scf_to_ticker, composite_to_ticker):
    """Resuelve canonical_security a (scf, ticker, status)."""
    if not isinstance(cs, str) or not cs.strip():
        return None, None, "no_canonical"
    cs = cs.strip()
    if cs.startswith("equity:"):
        t = cs[len("equity:"):]
        scfs = ticker_to_scf.get(t, set())
        if len(scfs) == 0:
            return None, t, "unresolved_ref"
        if len(scfs) > 1:
            return tuple(sorted(scfs)), t, "equity_multi"
        return next(iter(scfs)), t, "equity_ok"
    if cs.startswith("figi:"):
        f = cs[len("figi:"):]
        if f in scf_to_ticker:
            return f, scf_to_ticker[f], "figi_scf"
        if f in composite_to_ticker:
            t = composite_to_ticker[f]
            scfs = ticker_to_scf.get(t, set())
            if len(scfs) == 1:
                return next(iter(scfs)), t, "figi_composite"
            if len(scfs) > 1:
                return tuple(sorted(scfs)), t, "equity_multi"
        return None, None, "unresolved_ref"
    return None, None, "unresolved_ref"

def audit_units(units, ticker_to_scf, scf_to_ticker, composite_to_ticker):
    rows = []
    for _, r in units.iterrows():
        cs = r.get("canonical_security")
        scf, tick, status = resolve_row(cs, ticker_to_scf, scf_to_ticker, composite_to_ticker)
        rows.append({
            "report_period": r.get("report_period"),
            "filing_manager_cik": r.get("filing_manager_cik"),
            "observed_security_key": r.get("observed_security_key"),
            "canonical_security": cs,
            "discretion_type": r.get("discretion_type"),
            "sshprnamt_total": r.get("sshprnamt_total"),
            "scf_resuelta": scf,
            "ticker_ref": tick,
            "status": status,
        })
    return pd.DataFrame(rows)


def find_fragmentation(audit_df):
    """Group by scf_resuelta y cuenta set(canonical_security)."""
    frag = {}
    df = audit_df.copy()
    df["_scf_key"] = df["scf_resuelta"].apply(
        lambda s: s if isinstance(s, str) else None)
    for scf, grp in df[df["_scf_key"].notna()].groupby("_scf_key"):
        cs_set = set(grp["canonical_security"].dropna().astype(str).unique())
        if len(cs_set) > 1:
            frag[scf] = {
                "canonical_set": sorted(cs_set),
                "n_rows": int(len(grp)),
                "cusips": sorted(set(grp["observed_security_key"].dropna().astype(str))),
                "tickers": sorted(set(grp["ticker_ref"].dropna().astype(str))),
            }
    return frag


def normalize_canonical(cs, ticker_to_scf, scf_to_ticker, composite_to_ticker):
    scf, tick, status = resolve_row(cs, ticker_to_scf, scf_to_ticker, composite_to_ticker)
    if status in ("figi_scf", "figi_composite") and tick:
        return "equity:" + tick, True
    return cs, False


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

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("A.1 - AUDITORIA DE UNICIDAD POR shareClassFIGI")
    print("=" * 72)

    ticker_to_scf, scf_to_ticker, composite_to_ticker = load_catalog_index()
    print("Catalogo: tickers=", len(ticker_to_scf),
          "scf=", len(scf_to_ticker),
          "composite=", len(composite_to_ticker))

    print("\n[1/5] Pipeline units + delta...")
    s4 = load_canonical("2025Q4", PERIODS["2025Q4"])
    s1 = load_canonical("2026Q1", PERIODS["2026Q1"])
    i4 = build_identities(s4, PERIODS["2025Q4"])
    i1 = build_identities(s1, PERIODS["2026Q1"])
    u4 = compute_reported_position_units(s4["INFOTABLE"], s4["SUBMISSION"],
            report_period=PERIODS["2025Q4"], identity_results=i4)
    u1 = compute_reported_position_units(s1["INFOTABLE"], s1["SUBMISSION"],
            report_period=PERIODS["2026Q1"], identity_results=i1)
    units_all = pd.concat([u4, u1], ignore_index=True)
    print("  units rows:", len(units_all))

    print("\n[2/5] Resolucion de canonical_security por fila...")
    audit = audit_units(units_all, ticker_to_scf, scf_to_ticker, composite_to_ticker)
    status_counts = dict(Counter(audit["status"]))
    print("  status counts:", status_counts)

    print("\n[3/5] Buscando fragmentacion por shareClassFIGI...")
    frag = find_fragmentation(audit)
    print("  scf fragmentadas:", len(frag))
    for scf, info in sorted(frag.items())[:30]:
        print(f"    {scf}: {info['canonical_set']} rows={info['n_rows']} cusips={info['cusips']}")

    cusips_frag = set()
    for info in frag.values():
        cusips_frag.update(info["cusips"])
    units_rows_frag = sum(info["n_rows"] for info in frag.values())

    print("\n[4/5] NIPC actual vs normalizado...")
    delta_actual = compute_delta_shares(u1, u4)
    u4n = u4.copy()
    u1n = u1.copy()
    for u in (u4n, u1n):
        u["canonical_security"] = u["canonical_security"].apply(
            lambda cs: normalize_canonical(cs, ticker_to_scf, scf_to_ticker, composite_to_ticker)[0]
            if isinstance(cs, str) else cs)
    delta_norm = compute_delta_shares(u1n, u4n)
    r_actual = compute_nipc(delta_actual)
    r_norm = compute_nipc(delta_norm)
    delta_nipc = {
        "total": r_norm["nipc_total"] - r_actual["nipc_total"],
        "sole":  r_norm.get("nipc_sole", 0) - r_actual.get("nipc_sole", 0),
        "dfnd":  r_norm.get("nipc_dfnd", 0) - r_actual.get("nipc_dfnd", 0),
        "otr":   r_norm.get("nipc_otr", 0) - r_actual.get("nipc_otr", 0),
    }
    print("  NIPC actual:     ", r_actual["nipc_total"])
    print("  NIPC normalizado:", r_norm["nipc_total"])
    print("  Delta NIPC total:", delta_nipc["total"])

    print("\n[5/5] Persistir JSON + evaluacion de gates...")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / (ts + "_identity_audit.json")
    figi_used = (status_counts.get("figi_scf", 0)
                 + status_counts.get("figi_composite", 0))
    passed = (len(frag) == 0) and (figi_used == 0)

    payload = {
        "run_id": ts,
        "git_head": git_head(),
        "pandas_version": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_q4_infotable": sha256(DATA_DIR / "2025Q4" / "INFOTABLE.parquet"),
        "sha256_q1_infotable": sha256(DATA_DIR / "2026Q1" / "INFOTABLE.parquet"),
        "catalogo": {
            "n_tickers": len(ticker_to_scf),
            "n_share_class_figi": len(scf_to_ticker),
            "n_composite_figi": len(composite_to_ticker),
            "tickers_con_multi_scf": sorted(
                t for t, scfs in ticker_to_scf.items() if len(scfs) > 1),
        },
        "units_status_counts": status_counts,
        "fragmentacion": {
            "n_shareclass_figis_fragmented": len(frag),
            "n_tickers_fragmented": len({t for info in frag.values() for t in info["tickers"]}),
            "n_cusips_affected": len(cusips_frag),
            "n_units_rows_affected": units_rows_frag,
            "detalle": frag,
        },
        "nipc": {
            "actual": {k: r_actual.get(k) for k in
                       ("nipc_total","nipc_sole","nipc_dfnd","nipc_otr")},
            "normalizado": {k: r_norm.get(k) for k in
                            ("nipc_total","nipc_sole","nipc_dfnd","nipc_otr")},
            "delta": delta_nipc,
        },
        "gates": {
            "n_shareclass_fragmented_cero": len(frag) == 0,
            "figi_namespace_no_usado_en_e2e": figi_used == 0,
            "gate_2_cerrado_para_config_e2e_auditada": passed,
            "info": {
                "unresolved_ref": status_counts.get("unresolved_ref", 0),
                "nota": ("unresolved_ref = equity:X con X fuera del radar; "
                         "es esperado, no es fragmentacion."),
            },
        },
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\nJSON: " + str(out))
    print("RESULTADO: " + ("PASS" if passed else "REVISAR"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())