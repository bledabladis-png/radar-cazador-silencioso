"""Probe P65 end-to-end: Q4 2025 -> Q1 2026.

- Solo imprime por consola. NO escribe a repo.
- Determinista. NO usa datetime.now().
- Fechas literales: "2025-12-31", "2026-03-31".
- Inyecta cross_filing_evidence = vacio (fail-closed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, load_cusip_equivalence,
    resolve_batch_identities,
)
from src.institutional_accumulation.aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares,
)
from src.institutional_accumulation.aggregation.reporting_dedup import (
    build_effective_reporting_snapshot,
    classify_reporting_transition,
)

PROBE_DIR = Path(r"D:\13f_probe\processed")
OFFICIAL_DIR = Path(r"D:\13f_probe\official_list_13f")

PERIODS = {
    "Q4_2025": {
        "dir": PROBE_DIR / "2025Q4",
        "period": "2025-12-31",
        "list": OFFICIAL_DIR / "13flist_2025Q4.txt",
    },
    "Q1_2026": {
        "dir": PROBE_DIR / "2026Q1",
        "period": "2026-03-31",
        "list": OFFICIAL_DIR / "13flist_2026q1.txt",
    },
}

TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]


def _sep(title):
    print(flush=True)
    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def load_period(q):
    meta = PERIODS[q]
    d = meta["dir"]
    if not d.exists():
        raise FileNotFoundError(f"Probe dir no existe: {d}")
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    return dfs, meta["period"]


def main():
    _sep("P65 PROBE - Q4 2025 -> Q1 2026")

    cw = load_crosswalk_internal()
    eq = load_cusip_equivalence()
    print(f"crosswalk_internal rows: {len(cw)}")
    print(f"cusip_equivalence rows:  {len(eq)}")

    results = {}
    for q in ("Q4_2025", "Q1_2026"):
        _sep(f"PERIODO {q}")
        dfs, period = load_period(q)
        print(f"period: {period}")
        print(f"SUBMISSION filas: {len(dfs['SUBMISSION'])}")
        print(f"INFOTABLE filas:  {len(dfs['INFOTABLE'])}")

        print(f"[{q}] filter_by_period...", flush=True)
        filtered = filter_by_period(dfs, period)
        print(f"[{q}] apply_amendments...", flush=True)
        am = apply_amendments(filtered, period=period)
        print(f"[{q}] amendments OK", flush=True)
        snap = am["canonical_snapshot"]
        info = snap["INFOTABLE"]
        sub = snap["SUBMISSION"]
        print(f"canonical_snapshot SUBMISSION: {len(sub)}")
        print(f"canonical_snapshot INFOTABLE:  {len(info)}")

        mask = (info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info["PUTCALL"].isna())
        info_sh = info[mask]
        cusips = info_sh["CUSIP"].astype(str).str.strip().unique().tolist()
        print(f"CUSIPs unicos (SH+null): {len(cusips)}")

        print(f"[{q}] resolve_batch_identities ({len(cusips)} cusips)...", flush=True)
        identity_results = resolve_batch_identities(
            cusips, period,
            equivalence_df=eq, crosswalk_internal_df=cw,
        )
        print(f"[{q}] identity OK", flush=True)

        print(f"[{q}] compute_reported_position_units ({len(info_sh)} rows)...", flush=True)
        units = compute_reported_position_units(
            info_sh, sub, snap.get("COVERPAGE"),
            report_period=period, identity_results=identity_results,
        )
        print(f"[{q}] units: {len(units)}", flush=True)

        # Reporting_for: no materializable en v1 -> None.
        units = units.copy()
        units["reporting_for_manager_cik"] = None

        results[q] = {
            "period": period,
            "units": units,
        }

    _sep("DEDUP PRE-DELTA")
    u_q4 = results["Q4_2025"]["units"]
    u_q1 = results["Q1_2026"]["units"]

    # cross_filing_evidence vacio: fail-closed.
    evidence_empty = pd.DataFrame(columns=[
        "representante_cik", "representado_cik",
        "accession_representante", "accession_representado",
        "reference_seq", "security_key",
    ])

    eff_q4, eff_q1, audit = build_effective_reporting_snapshot(
        u_q4, u_q1,
        relationships_q4=None, relationships_q1=None,
        cross_filing_evidence=evidence_empty,
        period_q4="2025-12-31", period_q1="2026-03-31",
    )
    print(f"effective_q4: {len(eff_q4)}  effective_q1: {len(eff_q1)}")
    print(f"dedup_audit rows: {len(audit)}")
    if not audit.empty:
        print(f"dedup_decision counts: {audit['dedup_decision'].value_counts().to_dict()}")
        print(f"dedup_reason counts:   {audit['dedup_reason'].value_counts().to_dict()}")
    else:
        print("dedup_decision counts: {} (vacio - sin L3 no hay filas audit)")

    _sep("DELTA SHARES")
    delta = compute_delta_shares(eff_q1, eff_q4)
    print(f"delta rows: {len(delta)}")
    print(f"match_status counts: {delta['match_status'].value_counts().to_dict()}")

    _sep("REPORTING TRANSITION (POST-DELTA)")
    delta_t = classify_reporting_transition(delta, eff_q4, eff_q1)
    print(f"reporting_transition counts: {delta_t['reporting_transition'].value_counts().to_dict()}")
    if "dedup_reason" in delta_t.columns:
        non_null = delta_t["dedup_reason"].dropna()
        print(f"dedup_reason non-null: {len(non_null)}")
        if len(non_null) > 0:
            print(f"dedup_reason counts: {non_null.value_counts().to_dict()}")

    _sep("FIN DEL PROBE")


if __name__ == "__main__":
    main()
