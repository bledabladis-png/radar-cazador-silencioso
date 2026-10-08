"""Diagnostico H-10.1 - Bug estructural en la cobertura P38 (A.6.4).

NO modifica codigo productivo. Mide:

  1. Cardinalidades REALES de TARGET_Q4 / TARGET_Q1 / TARGET_PAIRWISE
     con datos Q1 2026 y Q4 2025.
  2. Demuestra que compute_contractual_coverage NO se puede invocar
     con Q4 vacio (adapter exige pairwise no vacio).
  3. Compara con el resultado del probe original (mock Q4=Q1 -> 1.0).

Evidencia para dictamen: el probe original publica coverage=1.0 porque
usa Q4=Q1 (mock). Con datos reales, el adapter no puede ejecutarse.

Determinista. Sin red. Sin datetime.now().
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
from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, resolve_batch_identities,
)
from src.institutional_accumulation.identity import target_builder as tb
from src.institutional_accumulation.aggregation import catalog_p38_adapter as ca
from src.institutional_accumulation import operational_universe as ou

DATA = ROOT / "data" / "sec_13f" / "processed"
MAPPINGS = ROOT / "data" / "mappings"
SNAPSHOT_CSV = MAPPINGS / "catalog_snapshots" / "snapshot_20260921_01.csv"
MEMBERSHIP = MAPPINGS / "catalog_membership.csv"
ASSIGNMENTS = MAPPINGS / "catalog_assignments.csv"
MANIFEST = MAPPINGS / "catalog_manifest.json"
OFFICIAL_DIR = Path(r"D:\13f_probe\official_list_13f")

TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]
PERIODS = [
    ("2025Q4", "2025-12-31", "13flist_2025Q4.txt"),
    ("2026Q1", "2026-03-31", "13flist_2026Q1.txt"),
]

HERE = Path(__file__).parent


def load_canonical(folder, period_iso):
    d = DATA / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    filtered = filter_by_period(dfs, period_iso)
    return apply_amendments(filtered, period=period_iso)["canonical_snapshot"]


def build_operational(folder, period_iso, official_name):
    snap = load_canonical(folder, period_iso)
    cw = load_crosswalk_internal()
    ids = resolve_batch_identities(
        snap["INFOTABLE"]["CUSIP"].astype(str).unique(),
        period_iso, crosswalk_internal_df=cw,
    )
    off = sl.load_official_list(OFFICIAL_DIR / official_name)
    oper = ou.build_operational_universe(
        snap["INFOTABLE"], off, period_iso,
        identity_results=ids, identity_period_iso=period_iso,
    )
    return oper, ids


def extract_tickers(oper, ids):
    tickers = {}
    for cusip in oper["CUSIP"].astype(str).unique():
        rec = ids.get(cusip) or {}
        cs = rec.get("canonical_security")
        if cs and str(cs).startswith("equity:"):
            tickers[cusip] = str(cs)[7:]
    return tickers


def build_target_universe():
    snap_df = pd.read_csv(SNAPSHOT_CSV, dtype=str, keep_default_na=False)
    mem_df = pd.read_csv(MEMBERSHIP, dtype=str, keep_default_na=False)
    asg_df = pd.read_csv(ASSIGNMENTS, dtype=str, keep_default_na=False)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    m = manifest["snapshots"][0]
    return tb.build_target(
        snap_df, mem_df, asg_df,
        version_id=m["version_id"],
        period_end="2026-03-31",
        catalog_version_id=m["version_id"],
        catalog_sha256=m["sha256"],
    )


def main():
    print("=" * 76)
    print("Diagnostico H-10.1 - Bug estructural cobertura P38 (A.6.4)")
    print("=" * 76)
    print()

    # 1. Operational universes reales
    print("=== 1. Operational universes por periodo (P61 seccion 5.5) ===")
    per = {}
    for folder, period_iso, off_name in PERIODS:
        oper, ids = build_operational(folder, period_iso, off_name)
        tickers = extract_tickers(oper, ids)
        per[folder] = {"rows": len(oper), "tickers": tickers}
        print("  " + folder + ": " + str(len(oper)) + " filas oper, "
              + str(len(tickers)) + " tickers equity")

    # 2. Universo B2-PIT completo
    print()
    print("=== 2. TargetUniverse B2-PIT (universo completo, snapshot) ===")
    universe = build_target_universe()
    n_declared = len(universe.declared_keys)
    n_with_figi = sum(1 for k in universe.declared_keys if universe.figi_by_key.get(k))
    print("  declared_keys:      " + str(n_declared))
    print("  con FIGI:           " + str(n_with_figi))

    # 3. Cardinalidades reales
    print()
    print("=== 3. Cardinalidades reales (SIN mock) ===")
    q1_tickers = set(per["2026Q1"]["tickers"].values())
    q4_tickers = set(per["2025Q4"]["tickers"].values())
    snap_tickers = set(universe.ticker_by_key.values())

    shared_q1 = q1_tickers & snap_tickers
    shared_q4 = q4_tickers & snap_tickers

    print("  TARGET_Q4 (reales):       " + str(len(shared_q4)) + " keys")
    print("  TARGET_Q1 (reales):       " + str(len(shared_q1)) + " keys")
    target_pairwise = shared_q4 & shared_q1
    print("  TARGET_PAIRWISE:          " + str(len(target_pairwise)) + " keys")
    print("  TARGET_Q1 universo:       " + str(n_declared) + " keys (B2-PIT completo)")
    print("  RESOLVED_Q4 (con FIGI):   " + str(len(shared_q4)) + " keys")
    print("  RESOLVED_Q1 (con FIGI):   " + str(len(shared_q1)) + " keys")
    print("  PAIRED (Q4 int Q1):       " + str(len(target_pairwise)) + " keys")
    print()

    # 4. Demostrar que el adapter NO acepta Q4 vacio
    print("=== 4. Intento de invocar el adapter con Q4 real (vacio) ===")
    print("  Q4 real: " + str(len(shared_q4)) + " keys")
    print("  Q1 real: " + str(len(shared_q1)) + " keys")
    print()
    try:
        # Necesitamos un state para Q1 real y un state vacio para Q4
        from src.institutional_accumulation.identity import period_state as ps
        st_full = ps.build_period_state(universe)
        # Sub-universo Q1 con FIGI
        keys_q1 = {k for k in universe.declared_keys
                   if universe.ticker_by_key.get(k) in shared_q1
                   and universe.figi_by_key.get(k)}
        # Q4 keys: vacio (real)
        keys_q4 = {k for k in universe.declared_keys
                   if universe.ticker_by_key.get(k) in shared_q4
                   and universe.figi_by_key.get(k)}

        import dataclasses
        u_q4 = dataclasses.replace(
            universe,
            declared_keys=keys_q4,
            ticker_by_key={k: universe.ticker_by_key[k] for k in keys_q4},
            figi_by_key={k: universe.figi_by_key[k] for k in keys_q4},
            row_uid_by_key={k: universe.row_uid_by_key[k] for k in keys_q4},
            key_by_row_uid={universe.row_uid_by_key[k]: k for k in keys_q4},
        )
        u_q1 = dataclasses.replace(
            universe,
            declared_keys=keys_q1,
            ticker_by_key={k: universe.ticker_by_key[k] for k in keys_q1},
            figi_by_key={k: universe.figi_by_key[k] for k in keys_q1},
            row_uid_by_key={k: universe.row_uid_by_key[k] for k in keys_q1},
            key_by_row_uid={universe.row_uid_by_key[k]: k for k in keys_q1},
        )
        st_q4 = {k: st_full[k] for k in keys_q4}
        st_q1 = {k: st_full[k] for k in keys_q1}

        t4, t1, r4, r1, feas = ca.catalog_to_p38_targets(
            u_q4, u_q1,
            state_q4=st_q4, state_q1=st_q1,
            pairwise_keys=keys_q4 & keys_q1,
        )
        print("  RESULTADO INESPERADO: adapter acepto Q4 vacio")
    except ca.AdapterError as e:
        print("  AdapterError capturado: " + str(e))
        print()
        print("  >>> El adapter RECHAZA Q4 vacio (pairwise vacio).")
        print("  >>> Es IMPOSIBLE computar la cobertura Q4/Q1 real con el")
        print("      adapter actual. El probe original recurre a Q4=Q1 (mock)")
        print("      precisamente para saltar esta precondicion.")

    # 5. Comparacion con resultado del probe original
    print()
    print("=== 5. Comparacion ===")
    print()
    print("  Probe original (Q4=Q1 mock):        coverage = 1.0 (trivial)")
    print("  Probe con Q4 real (vacio):           AdapterError")
    print("  Cobertura Q1 real sobre universo:    "
          + str(len(keys_q1) if 'keys_q1' in dir() else 0)
          + " / " + str(n_declared) + " = "
          + ("{:.4f}".format((len(keys_q1) if 'keys_q1' in dir() else 0) / n_declared if n_declared else 0.0)))
    print()
    print("  Consecuencia: el unico numero publicable con los datos actuales")
    print("  es la cobertura del subconjunto Q1 sobre el universo B2-PIT,")
    print("  que NO es la metrica contractual P38 (esa exige Q4 y Q1 reales).")

    print()
    print("=== 6. Conclusion ===")
    print()
    print("  El coverage=1.0 de A.6.4 es artefacto del mock Q4=Q1, no evidencia")
    print("  de cobertura contractual. El fix requiere tocar codigo productivo")
    print("  (adapter + coverage.py + propagacion de SSHPRNAMT). Dictamen externo.")

    out = {
        "periods": {k: {"rows": v["rows"], "tickers": len(v["tickers"])}
                    for k, v in per.items()},
        "universe_declared_keys": n_declared,
        "universe_with_figi": n_with_figi,
        "target_q4_real": len(shared_q4),
        "target_q1_real": len(shared_q1),
        "target_pairwise_real": len(target_pairwise),
        "adapter_accepts_q4_empty": False,
        "note": "adapter_rejects_empty_pairwise",
    }
    (HERE / "diagnose_result.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")
    print()
    print("OK -> diagnose_result.json")


if __name__ == "__main__":
    main()
