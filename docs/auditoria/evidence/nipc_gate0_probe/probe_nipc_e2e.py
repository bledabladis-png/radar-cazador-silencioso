"""Probe end-to-end NIPC: Q4 2025 -> Q1 2026.

- Solo imprime por consola. NO escribe a repo.
- Determinista. NO usa datetime.now().
- Fechas literales: "2025-12-31", "2026-03-31".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Macro_Sectorial")
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.sec13f_list import (
    load_official_list, resolve_eligibility, compute_eligibility_coverage,
)
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, load_cusip_equivalence,
    resolve_batch_identities, compute_identity_coverage,
)
from src.institutional_accumulation.aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares,
)
from src.institutional_accumulation.aggregation.nipc import (
    compute_nipc_and_coverage, compute_nipc,
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
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_period(q):
    meta = PERIODS[q]
    d = meta["dir"]
    if not d.exists():
        raise FileNotFoundError(f"Probe dir no existe: {d}")
    dfs = {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}
    return dfs, meta["period"]


def prep_period(dfs, period):
    # 1) filter_by_period
    filtered = filter_by_period(dfs, period)
    # 2) apply_amendments
    am = apply_amendments(filtered, period=period)
    snap = am["canonical_snapshot"]
    return snap, am


def main():
    _sep("CROSSWALK / EQUIVALENCE")
    cw = load_crosswalk_internal()
    print(f"crosswalk_internal rows: {len(cw)}  (exceptions + etf_holdings)")
    eq = load_cusip_equivalence()
    print(f"cusip_equivalence rows:  {len(eq)}")

    results = {}
    for q in ("Q4_2025", "Q1_2026"):
        _sep(f"PERIODO {q}")
        dfs, period = load_period(q)
        print(f"period: {period}")
        print(f"SUBMISSION filas crudas:  {len(dfs['SUBMISSION'])}")
        print(f"INFOTABLE filas crudas:   {len(dfs['INFOTABLE'])}")

        snap, am = prep_period(dfs, period)
        info = snap["INFOTABLE"]
        sub = snap["SUBMISSION"]
        print(f"canonical_snapshot SUBMISSION: {len(sub)}")
        print(f"canonical_snapshot INFOTABLE:  {len(info)}")

        # filtro SH+null para medir universo técnico
        mask = (info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info["PUTCALL"].isna())
        info_sh = info[mask]
        cusips = info_sh["CUSIP"].astype(str).str.strip().unique().tolist()
        print(f"CUSIPs únicos (SH+null): {len(cusips)}")

        # SEC Official List
        sec = load_official_list(PERIODS[q]["list"])
        print(f"SEC Official List rows:   {len(sec)}  (lineas)")
        print(f"SEC Official List CUSIPs: {sec['cusip'].nunique()}")

        elig_cov = compute_eligibility_coverage(cusips, sec)
        print(f"eligibility: total={elig_cov['n_total']}  eligible={elig_cov['n_eligible']}"
              f"  not_in_list={elig_cov['n_not_in_list']}  deleted={elig_cov['n_deleted']}"
              f"  active={elig_cov['n_active']}  added={elig_cov['n_added']}"
              f"  conflict={elig_cov['n_conflict']}  pct_elig={elig_cov['pct_eligible']:.4f}")

        # Identidad: resolver sobre todos los CUSIPs SH+null
        identity_results = resolve_batch_identities(
            cusips, period,
            equivalence_df=eq, crosswalk_internal_df=cw,
        )
        id_cov = compute_identity_coverage(
            cusips, period,
            equivalence_df=eq, crosswalk_internal_df=cw,
        )
        print(f"identity: n_total={id_cov['n_total']}  canonical={id_cov['n_canonical']}"
              f"  observed_only={id_cov['n_observed_only']}  conflict={id_cov['n_conflict']}"
              f"  ambiguous={id_cov['n_ambiguous']}  unresolved={id_cov['n_unresolved']}"
              f"  pct_canonical={id_cov['pct_canonical']:.4f}")
        print(f"by_kind: {id_cov['by_kind']}")

        # Units (todos los CUSIPs SH+null con identidad asignada)
        units = compute_reported_position_units(
            info_sh, sub, snap.get("COVERPAGE"),
            report_period=period, identity_results=identity_results,
        )
        print(f"reported_position_units: {len(units)}")
        n_can = int((units["security_resolution_status"] == "CANONICAL").sum())
        print(f"  units canonical: {n_can}")

        # filtra units a eligibles
        elig_set = {c for c, v in resolve_eligibility(cusips, sec).items() if v["eligible"]}
        units_elig = units[units["observed_security_key"].str.replace("cusip:", "", regex=False).isin(elig_set)]
        print(f"  units eligible:  {len(units_elig)}")

        results[q] = {
            "period": period,
            "units": units,
            "units_elig": units_elig,
            "identity_results": identity_results,
            "elig_set": elig_set,
        }

    # ---- Delta + NIPC ----
    _sep("DELTA SHARES + NIPC")
    u_q4_all = results["Q4_2025"]["units"]
    u_q1_all = results["Q1_2026"]["units"]

    for scope in ("ALL", "ELIGIBLE"):
        u_q4 = u_q4_all if scope == "ALL" else results["Q4_2025"]["units_elig"]
        u_q1 = u_q1_all if scope == "ALL" else results["Q1_2026"]["units_elig"]
        print()
        print(f"--- scope={scope} ---")
        print(f"  Q4 units: {len(u_q4)}   Q1 units: {len(u_q1)}")

        delta = compute_delta_shares(u_q1, u_q4)
        print(f"  delta filas: {len(delta)}")
        print(f"  match_status: {delta['match_status'].value_counts().to_dict()}")

        nipc_only = compute_nipc(delta)
        print(f"  NIPC total (observable): {nipc_only['nipc_total']:.0f}")
        print(f"  nipc_sole={nipc_only['nipc_sole']:.0f}"
              f"  nipc_dfnd={nipc_only['nipc_dfnd']:.0f}"
              f"  nipc_otr={nipc_only['nipc_otr']:.0f}")

        cov = compute_nipc_and_coverage(delta, u_q1, u_q4)
        print(f"  coverage_previous={cov['coverage_previous']:.4f}"
              f"  coverage_current={cov['coverage_current']:.4f}")
        print(f"  paired_security_coverage={cov['paired_security_coverage']:.4f}"
              f"  paired_weighted_share_coverage={cov['paired_weighted_share_coverage']:.4f}")
        print(f"  unmapped_weight_previous={cov['unmapped_weight_previous']:.4f}"
              f"  unmapped_weight_current={cov['unmapped_weight_current']:.4f}")
        print(f"  STATUS: {cov['status']}")

        # Top 10 por |delta|
        obs = delta[delta["match_status"].isin(["BOTH", "NEW", "EXIT"])].copy()
        obs["abs_delta"] = obs["delta_shares"].abs()
        top = obs.nlargest(10, "abs_delta")[
            ["filing_manager_cik", "canonical_security", "discretion_type",
             "sshprnamt_previous", "sshprnamt_current", "delta_shares", "match_status"]
        ]
        print("  TOP 10 |delta|:")
        for _, r in top.iterrows():
            print(f"    FM={r['filing_manager_cik']}  "
                  f"sec={r['canonical_security']}  "
                  f"disc={r['discretion_type']}  "
                  f"prev={r['sshprnamt_previous']:.0f}  "
                  f"curr={r['sshprnamt_current']:.0f}  "
                  f"delta={r['delta_shares']:.0f}  "
                  f"({r['match_status']})")

    _sep("FIN DEL PROBE")


if __name__ == "__main__":
    main()