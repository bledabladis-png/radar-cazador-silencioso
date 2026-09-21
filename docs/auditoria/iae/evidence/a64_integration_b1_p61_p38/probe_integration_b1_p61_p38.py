"""A.6.4 - Integracion B1 + P61 + P38 (dictamen #73).

Demuestra la cadena end-to-end sobre el subconjunto que atraviesa
realmente las tres capas:
  P61 (CANONICAL + VERIFIED) ∩ snapshot B2-PIT -> B1 (build_target)
  -> period_state -> catalog_p38_adapter -> compute_contractual_coverage.

Q1 2026: subconjunto real -> resultado.
Q4 2025: fail-closed, sin fabricar TARGET historico.

El snapshot B2-PIT (valid_from=2026-09-21) NO cubre Q1/Q4 por PIT.
La integracion se ejecuta invocando build_target directamente (no
target_catalog_as_of). La parte PIT del catalog no se evalua aqui:
requiere snapshots historicos inexistentes. Documentado en README.

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
from src.institutional_accumulation.identity import period_state as ps
from src.institutional_accumulation.aggregation import coverage as cov
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
    ("2025Q4", "2025-12-31", "2025Q4", "13flist_2025Q4.txt"),
    ("2026Q1", "2026-03-31", "2026Q1", "13flist_2026Q1.txt"),
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
    return snap, ids, oper


def extract_tickers(oper, ids):
    """Tickers equity:<T> del operational_universe."""
    tickers = {}
    for cusip in oper["CUSIP"].astype(str).unique():
        rec = ids.get(cusip) or {}
        cs = rec.get("canonical_security")
        if cs and str(cs).startswith("equity:"):
            t = str(cs)[7:]
            tickers[cusip] = t
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


def filter_universe_by_tickers(universe, ticker_set):
    """Devuelve un dict {key: ticker} restringido a un set de tickers."""
    out = {}
    for k in universe.declared_keys:
        t = universe.ticker_by_key.get(k)
        if t and t in ticker_set:
            out[k] = t
    return out


def build_sub_state(universe, keys_subset, *, sshprnamt_evidence=None,
                    operational_evidence=None):
    """Construye state para las keys del subconjunto.

    Usa figi_by_key del universo. Default: RESOLVED + RESOLVED_OBSERVED
    + operational_mapping_status=UNRESOLVED + sshprnamt=None.
    A2 c2/5: sshprnamt_evidence es dict {catalog_key: float}, NO
    {shareClassFIGI: float}. El state vive por catalog_key.
    A2 c5/5: operational_evidence es dict {catalog_key: status}.
    El probe lo pasa con VERIFIED para las keys del operational_
    universe (5.5), que garantiza CANONICAL + VERIFIED por
    construccion. El adapter PROPAGA lo que el state declara.
    """
    st_full = ps.build_period_state(
        universe,
        sshprnamt_evidence=sshprnamt_evidence,
        operational_evidence=operational_evidence,
    )
    return {k: st_full[k] for k in keys_subset}


def build_records_for_subset(keys_subset, universe, state, tickers_by_key,
                              period_label):
    """PositionRecords para el subconjunto, con FIGI del snapshot.

    A2 c5/5: el adapter productivo PROPAGA operational_mapping_status
    y weight desde el state (commit 3, H-07 + H-10.1). Este helper
    del probe debe ser coherente: PROPAGA del state, no fabrica
    VERIFIED ni weight=1.0.

    El probe declara VERIFIED en el state via operational_evidence
    porque las keys vienen del operational_universe (5.5), que
    garantiza CANONICAL + VERIFIED. Aqui solo se propaga.
    """
    records = []
    for k in sorted(keys_subset):
        s = state[k]
        records.append(cov.PositionRecord(
            period=period_label,
            observed_security_key=k,
            share_class_figi=s.figi,
            canonical_security="equity:" + tickers_by_key[k],
            resolution_status=s.identity_status,
            operational_mapping_status=s.operational_mapping_status,
            weight=(float(s.sshprnamt) if s.sshprnamt is not None else 0.0),
        ))
    return records


def main():
    print("=" * 72)
    print("A.6.4 - Integracion B1 + P61 + P38 (dictamen #73)")
    print("=" * 72)

    # --- 1. §5.5 por periodo ---
    print()
    print("=== 1. Operational universes por periodo (P61) ===")
    per_data = {}
    for folder, period_iso, label, off_name in PERIODS:
        snap, ids, oper = build_operational(folder, period_iso, off_name)
        tickers = extract_tickers(oper, ids)
        per_data[label] = {
            "folder": folder,
            "period_iso": period_iso,
            "oper_rows": len(oper),
            "tickers_by_cusip": tickers,
            "ticker_set": set(tickers.values()),
            "snap": snap,
        }
        print("  " + label + ": " + str(len(oper)) + " filas oper, "
              + str(len(tickers)) + " tickers equity")

    # --- 2. TargetUniverse desde snapshot B2-PIT ---
    print()
    print("=== 2. TargetUniverse (B1) ===")
    universe = build_target_universe()
    print("  declared_keys: " + str(len(universe.declared_keys)))

    # --- 2b. SSHPRNAMT efectivo por catalog_key (A2 c2/5) ---
    # El resolver NO devuelve share_class_figi: el FIGI contractual
    # vive en universe.figi_by_key (B2-PIT). Construimos:
    #   ticker_to_figi: {radar_ticker: share_class_figi}
    #   figi_to_keys:   {share_class_figi: [catalog_key, ...]}
    # Luego agregamos SSHPRNAMT por FIGI sobre el INFOTABLE del
    # canonical_snapshot post-amendments, y replicamos a las keys.
    ticker_to_figi = {}
    figi_to_keys = {}
    for k in universe.declared_keys:
        f = universe.figi_by_key.get(k)
        t = universe.ticker_by_key.get(k)
        if f:
            if t:
                ticker_to_figi[t] = f
            figi_to_keys.setdefault(f, []).append(k)

    for label, d in per_data.items():
        figi_by_cusip = {}
        for cusip, t in d["tickers_by_cusip"].items():
            f = ticker_to_figi.get(t)
            if f:
                figi_by_cusip[str(cusip)] = str(f)
        sh_by_figi = tb.extract_sshprnamt_by_figi(
            d["snap"]["INFOTABLE"], figi_by_cusip,
        )
        # B-06 fix (auditor #76): SSHPRNAMT total del FIGI se asigna a UNA
        # key representativa (la primera en orden lexicografico). Las
        # demas keys del mismo FIGI quedan en None (weight=0.0 en record).
        # Asi aggregate_positions_by_shareclass_figi suma el total del
        # FIGI exactamente UNA vez, no N veces (donde N = keys por FIGI).
        sh_by_key = {}
        for f, v in sh_by_figi.items():
            ks = sorted(figi_to_keys.get(f, []))
            if ks:
                sh_by_key[ks[0]] = v
        d["sshprnamt_by_figi"] = sh_by_figi
        d["sshprnamt_by_key"] = sh_by_key
        print("  SSHPRNAMT " + label + ": "
              + str(len(sh_by_figi)) + " FIGIs, "
              + str(len(sh_by_key)) + " keys representativas")

    # --- 3. Cruce ---
    print()
    print("=== 3. Cruce §5.5 <-> snapshot ===")
    q1_tickers = per_data["2026Q1"]["ticker_set"]
    q4_tickers = per_data["2025Q4"]["ticker_set"]
    snap_tickers = set(universe.ticker_by_key.values())
    shared_q1 = q1_tickers & snap_tickers
    shared_q4 = q4_tickers & snap_tickers
    print("  tickers §5.5 Q1:     " + str(len(q1_tickers)))
    print("  tickers §5.5 Q4:     " + str(len(q4_tickers)))
    print("  tickers snapshot:    " + str(len(snap_tickers)))
    print("  shared Q1:           " + str(len(shared_q1)))
    print("  shared Q4:           " + str(len(shared_q4)))

    # --- 4. Sub-universo por periodo ---
    print()
    print("=== 4. Ejecucion cadena B1 + P38 ===")
    result = {}
    for label, shared in [("2026Q1", shared_q1), ("2025Q4", shared_q4)]:
        print()
        print("--- " + label + " ---")
        keys_all = filter_universe_by_tickers(universe, shared)
        keys_sub = {k for k in keys_all if universe.figi_by_key.get(k)}
        dropped = len(keys_all) - len(keys_sub)
        print("  keys en subconjunto: " + str(len(keys_all))
              + " (con FIGI: " + str(len(keys_sub))
              + ", sin FIGI: " + str(dropped) + ")")
        if not keys_sub:
            print("  sin subconjunto: fail-closed")
            result[label] = {"keys": 0, "p38": None, "fail_closed": True}
            continue
        tickers_by_key = {k: universe.ticker_by_key[k] for k in keys_sub}
        st_q = build_sub_state(
            universe, keys_sub,
            sshprnamt_evidence=per_data[label]["sshprnamt_by_key"],
        )

        records_q = build_records_for_subset(
            keys_sub, universe, st_q, tickers_by_key, "Q1")
        target_figi = {st_q[k].figi for k in keys_sub if st_q[k].figi}
        print("  FIGIs unicos: " + str(len(target_figi)))
        result[label] = {
            "keys": len(keys_sub),
            "target_figi": len(target_figi),
            "records": len(records_q),
            "fail_closed": False,
        }

    # --- 5. P38 real: Q4 y Q1 sin mock (fix A2 c5/5, H-05/H-06) ---
    # Q4 y Q1 son universos DISTINTOS. Se invoca el adapter real
    # con (u_q4, u_q1). Cuando un lado esta vacio, NO se fabrica
    # coverage=1.0: se aplica Q5 literal del auditor:
    #   - coverage_previous = UNAVAILABLE si TARGET_Q4 vacio
    #   - coverage_current  = calculable sobre TARGET_Q1
    #   - paired_*          = UNAVAILABLE si TARGET_PAIRWISE vacio
    print()
    print("=== 5. compute_contractual_coverage (Q4 y Q1 reales) ===")

    import dataclasses

    def _sub_universe(u, shared):
        keys = {k for k in filter_universe_by_tickers(u, shared)
                if u.figi_by_key.get(k)}
        if not keys:
            return None, set()
        usub = dataclasses.replace(
            u,
            declared_keys=set(keys),
            ticker_by_key={k: u.ticker_by_key[k] for k in keys},
            figi_by_key={k: u.figi_by_key[k] for k in keys},
            row_uid_by_key={k: u.row_uid_by_key[k] for k in keys},
            key_by_row_uid={u.row_uid_by_key[k]: k for k in keys},
        )
        return usub, keys

    u_q4, keys_q4 = _sub_universe(universe, shared_q4)
    u_q1, keys_q1 = _sub_universe(universe, shared_q1)
    pairwise_keys = keys_q4 & keys_q1

    print("  catalog_keys Q4: " + str(len(keys_q4)))
    print("  catalog_keys Q1: " + str(len(keys_q1)))
    print("  TARGET_PAIRWISE (catalog_keys): " + str(len(pairwise_keys)))

    def _build_side(u, keys, period_label):
        if u is None or not keys:
            return None, set(), []
        st = build_sub_state(
            u, keys,
            sshprnamt_evidence=per_data[period_label]["sshprnamt_by_key"],
            operational_evidence={k: "VERIFIED" for k in keys},
        )
        tb_key = {k: u.ticker_by_key[k] for k in keys}
        tgt = {st[k].figi for k in keys if st[k].figi}
        recs = build_records_for_subset(
            keys, u, st, tb_key, period_label)
        return st, tgt, recs

    st_q4_side, t4, r4 = _build_side(u_q4, keys_q4, "2025Q4")
    st_q1_side, t1, r1 = _build_side(u_q1, keys_q1, "2026Q1")

    if u_q4 is not None and u_q1 is not None and pairwise_keys:
        t4, t1, r4, r1, feas = ca.catalog_to_p38_targets(
            u_q4, u_q1,
            state_q4=st_q4_side, state_q1=st_q1_side,
            pairwise_keys=pairwise_keys,
        )
        print("  feasibility: " + str(feas))
        p38_result = cov.compute_contractual_coverage(
            t4, t1, r4, r1)
    else:
        if u_q4 is None:
            reason = "TARGET_Q4 vacio"
        elif u_q1 is None:
            reason = "TARGET_Q1 vacio"
        else:
            reason = "TARGET_PAIRWISE vacio"
        print("  fail-closed: " + reason)
        p38_result = cov.compute_contractual_coverage(t4, t1, r4, r1)
    for k, v in p38_result.items():
        print("  " + str(k).ljust(35) + " " + str(v))

    # H-06: publicar cardinalidades REALES (auditoria 2026-09-21).
    res_q4_figi = {r.share_class_figi for r in r4
                   if r.share_class_figi
                   and r.operational_mapping_status == "VERIFIED"}
    res_q1_figi = {r.share_class_figi for r in r1
                   if r.share_class_figi
                   and r.operational_mapping_status == "VERIFIED"}
    cardinalities = {
        "target_q4_figi": len(t4),
        "target_q1_figi": len(t1),
        "target_pairwise_figi": len(t4 & t1),
        "records_q4": len(r4),
        "records_q1": len(r1),
        "records_q4_verified": sum(
            1 for r in r4 if r.operational_mapping_status == "VERIFIED"
        ),
        "records_q1_verified": sum(
            1 for r in r1 if r.operational_mapping_status == "VERIFIED"
        ),
        "paired_figi": len((t4 & t1) & res_q4_figi & res_q1_figi),
    }
    print()
    print("=== 5b. Cardinalidades reales (H-06) ===")
    for k, v in cardinalities.items():
        print("  " + k.ljust(30) + " " + str(v))

    # --- 6. Q4 fail-closed (fix H-05) ---
    print()
    print("=== 6. Q4 fail-closed ===")
    if result.get("2025Q4", {}).get("fail_closed", True):
        print("  Q4 sin subconjunto -> fail-closed correcto")
        assert p38_result.get("coverage_previous") is None, \
            "H-05: Q4 vacio debe producir coverage_previous=None"
        assert p38_result.get("paired_security_coverage") is None, \
            "H-05: Q4 vacio debe producir paired_security_coverage=None"
    else:
        print("  Q4 tiene subconjunto -> revisar")

    out = {
        "periods": {k: dict(
            {kk: vv for kk, vv in v.items()
             if kk not in ("tickers_by_cusip", "ticker_set", "snap",
                           "sshprnamt_by_figi", "sshprnamt_by_key")},
            n_sshprnamt_by_figi=len(v.get("sshprnamt_by_figi", {})),
            n_sshprnamt_by_key=len(v.get("sshprnamt_by_key", {})),
        ) for k, v in per_data.items()},
        "universe_keys": len(universe.declared_keys),
        "shared_q1": len(shared_q1),
        "shared_q4": len(shared_q4),
        "result_periods": result,
        "cardinalities": cardinalities,
        "p38_pairwise": dict(p38_result),
        "p38_q1": dict(p38_result),
    }
    (HERE / "result.json").write_text(
        json.dumps(out, indent=2, default=str),
        encoding="utf-8", newline="\n")
    print()
    print("OK -> result.json")


if __name__ == "__main__":
    main()