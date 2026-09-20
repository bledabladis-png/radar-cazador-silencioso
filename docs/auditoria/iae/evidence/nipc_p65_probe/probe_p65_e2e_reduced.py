"""Probe P65 end-to-end REDUCIDO: Q4 2025 -> Q1 2026.

- Mismo flujo que probe_p65_e2e.py pero sobre un subconjunto de ~200 CIKs.
- Datos reales SEC 13F. NO sintetico.
- Determinista. Sin datetime.now(). Solo imprime.
- Sirve para validar el pipeline P65 en tiempos razonables.
- El probe completo (todas las filas) es viable pero tarda ~10 min.
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

N_CIKS = 200

PERIODS = {
    "Q4_2025": {"dir": PROBE_DIR / "2025Q4", "period": "2025-12-31"},
    "Q1_2026": {"dir": PROBE_DIR / "2026Q1", "period": "2026-03-31"},
}

TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]


def _sep(title):
    print(flush=True)
    print("=" * 72, flush=True)
    print(title, flush=True)
    print("=" * 72, flush=True)


def load_reduced(q):
    d = PERIODS[q]["dir"]
    sub = pd.read_parquet(d / "SUBMISSION.parquet")
    # Determinista: primeros N_CIKS por ACCESSION_NUMBER ascendente.
    sub = sub.sort_values("ACCESSION_NUMBER").reset_index(drop=True)
    accs_keep = set(sub["ACCESSION_NUMBER"].astype(str).head(N_CIKS).tolist())
    sub = sub[sub["ACCESSION_NUMBER"].astype(str).isin(accs_keep)].reset_index(drop=True)
    dfs = {"SUBMISSION": sub}
    for name in TSVS:
        if name == "SUBMISSION":
            continue
        df = pd.read_parquet(d / (name + ".parquet"))
        if "ACCESSION_NUMBER" in df.columns:
            df = df[df["ACCESSION_NUMBER"].astype(str).isin(accs_keep)].reset_index(drop=True)
        dfs[name] = df
    return dfs, PERIODS[q]["period"]


def main():
    _sep("P65 PROBE REDUCIDO - Q4 2025 -> Q1 2026")
    print(f"N_CIKS (max accessions): {N_CIKS}", flush=True)

    cw = load_crosswalk_internal()
    eq = load_cusip_equivalence()
    print(f"crosswalk_internal rows: {len(cw)}", flush=True)
    print(f"cusip_equivalence rows:  {len(eq)}", flush=True)

    results = {}
    for q in ("Q4_2025", "Q1_2026"):
        _sep(f"PERIODO {q}")
        dfs, period = load_reduced(q)
        print(f"period: {period}", flush=True)
        print(f"SUBMISSION filas: {len(dfs['SUBMISSION'])}", flush=True)
        print(f"INFOTABLE filas:  {len(dfs['INFOTABLE'])}", flush=True)

        filtered = filter_by_period(dfs, period)
        am = apply_amendments(filtered, period=period)
        snap = am["canonical_snapshot"]
        info = snap["INFOTABLE"]
        sub = snap["SUBMISSION"]
        print(f"canonical_snapshot SUBMISSION: {len(sub)}", flush=True)
        print(f"canonical_snapshot INFOTABLE:  {len(info)}", flush=True)

        mask = (info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH") & (info["PUTCALL"].isna())
        info_sh = info[mask]
        cusips = info_sh["CUSIP"].astype(str).str.strip().unique().tolist()
        print(f"CUSIPs unicos: {len(cusips)}", flush=True)

        identity_results = resolve_batch_identities(
            cusips, period,
            equivalence_df=eq, crosswalk_internal_df=cw,
        )

        units = compute_reported_position_units(
            info_sh, sub, snap.get("COVERPAGE"),
            report_period=period, identity_results=identity_results,
        )
        print(f"units: {len(units)}", flush=True)

        units = units.copy()
        units["reporting_for_manager_cik"] = None

        results[q] = {"period": period, "units": units}

    _sep("DEDUP PRE-DELTA")
    u_q4 = results["Q4_2025"]["units"]
    u_q1 = results["Q1_2026"]["units"]

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
    print(f"effective_q4: {len(eff_q4)}  effective_q1: {len(eff_q1)}", flush=True)
    print(f"dedup_audit rows: {len(audit)}", flush=True)

    _sep("DELTA SHARES")
    delta = compute_delta_shares(eff_q1, eff_q4)
    print(f"delta rows: {len(delta)}", flush=True)
    print(f"match_status counts: {delta['match_status'].value_counts().to_dict()}", flush=True)

    _sep("REPORTING TRANSITION (POST-DELTA)")
    delta_t = classify_reporting_transition(delta, eff_q4, eff_q1)
    print(f"reporting_transition counts: {delta_t['reporting_transition'].value_counts().to_dict()}", flush=True)

    _sep("FIN DEL PROBE")


if __name__ == "__main__":
    main()
