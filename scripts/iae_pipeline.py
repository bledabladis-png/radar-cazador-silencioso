"""Orquestador productivo IAE - cadena §5.1 -> §5.5.

Materializa la cadena completa de ingestion + identity + operational
universe sobre los parquets canonicos de 13F.

NO invoca compute_nipc_contractual (sin callers productivos por
dictamen #64/#65). NO abre OpenFIGI. NO fija thresholds.

Uso:
    py scripts/iae_pipeline.py --period 2026-03-31 --folder 2026Q1

Referencia: PROMPT_MAESTRO v6.53. Modulos consumidos (no modificados):
  filter_by_period, apply_amendments, load_crosswalk_internal,
  load_cusip_equivalence, resolve_batch_identities, load_official_list,
  build_operational_universe.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.institutional_accumulation.sec_13f.identity.temporal_filter import filter_by_period
from src.institutional_accumulation.sec_13f.identity.amendments import apply_amendments
from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal,
    resolve_batch_identities,
)
from src.institutional_accumulation import operational_universe as ou

DATA_DIR = ROOT / "data" / "sec_13f" / "processed"
OFFICIAL_DIR = Path(os.environ.get(
    "IAE_OFFICIAL_DIR",
    str(ROOT / "data" / "sec_13f" / "official_list_13f")))
TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]

PERIODS = {
    "2025Q4": {"period": "2025-12-31",
               "official": OFFICIAL_DIR / "13flist_2025Q4.txt"},
    "2026Q1": {"period": "2026-03-31",
               "official": OFFICIAL_DIR / "13flist_2026Q1.txt"},
}


def load_canonical(folder, period_iso):
    """Carga los 7 TSVs + filter_by_period + apply_amendments."""
    d = DATA_DIR / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    filtered = filter_by_period(dfs, period_iso)
    snap = apply_amendments(filtered, period=period_iso)["canonical_snapshot"]
    return snap


def build_identities(snapshot, period_iso):
    """Resuelve identidad P61 para todos los CUSIPs del canonical."""
    cusips = snapshot["INFOTABLE"]["CUSIP"].astype(str).unique()
    cw = load_crosswalk_internal()
    eq = None
    eq_path = ROOT / "data" / "mappings" / "cusip_equivalence.csv"
    if eq_path.exists():
        eq = pd.read_csv(eq_path, dtype=str)
    return resolve_batch_identities(
        cusips, period_iso,
        equivalence_df=eq,
        crosswalk_internal_df=cw,
        figi_lookup=None,
    )


def report(period_iso, folder, snapshot, identities, oper):
    """Reporte por etapa del pipeline."""
    print()
    print("=" * 72)
    print("IAE PIPELINE - " + folder + " (" + period_iso + ")")
    print("=" * 72)

    info = snapshot["INFOTABLE"]
    print()
    print("Etapa 0 - canonical (post amendments):")
    print("  filas INFOTABLE:      " + str(len(info)))
    print("  CUSIPs unicos:        " + str(info["CUSIP"].nunique()))

    mask_sh = info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH"
    mask_null = info["PUTCALL"].isna()
    u51 = info[mask_sh & mask_null]
    print()
    print("Etapa 1 - seccion 5.1 (SH + PUTCALL NULL):")
    print("  filas:                " + str(len(u51)))

    print()
    print("Etapa 2 - identidad P61:")
    status_counts = {}
    for v in identities.values():
        s = v.get("security_resolution_status", "?")
        status_counts[s] = status_counts.get(s, 0) + 1
    for s, n in sorted(status_counts.items(), key=lambda x: -x[1]):
        print("  " + str(s).ljust(20) + " " + str(n))

    print()
    print("Etapa 3 - operational universe (secciones 5.1+5.3+5.4+5.5):")
    print("  filas operacionales:  " + str(len(oper)))

    if len(u51) > 0:
        pct = 100.0 * len(oper) / len(u51)
        print("  pct operational:      " + str(round(pct, 4)) + "%")

    # Top TITLEOFCLASS en oper
    if len(oper) > 0 and "TITLEOFCLASS" in oper.columns:
        print()
        print("  Top 10 TITLEOFCLASS en operational:")
        vc = oper["TITLEOFCLASS"].astype(str).value_counts().head(10)
        for t, n in vc.items():
            print("    " + f"{n:>7}" + "  " + str(t)[:55])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, choices=list(PERIODS.keys()))
    args = ap.parse_args()

    folder = args.folder
    period_iso = PERIODS[folder]["period"]
    official_path = PERIODS[folder]["official"]

    print("Cargando canonical " + folder + "...")
    snap = load_canonical(folder, period_iso)

    print("Resolviendo identidades P61...")
    identities = build_identities(snap, period_iso)
    print("  " + str(len(identities)) + " CUSIPs resueltos")

    if not official_path.exists():
        print("WARN: Official List no existe: " + str(official_path))
        print("      Saltando seccion 5.3. El operational sera vacio.")
        return

    official_df = sl.load_official_list(official_path)
    print("  Official List: " + str(len(official_df)) + " filas")

    print("Construyendo operational_universe...")
    oper = ou.build_operational_universe(
        snap["INFOTABLE"], official_df, period_iso,
        identity_results=identities,
        identity_period_iso=period_iso,
    )

    report(period_iso, folder, snap, identities, oper)


if __name__ == "__main__":
    main()