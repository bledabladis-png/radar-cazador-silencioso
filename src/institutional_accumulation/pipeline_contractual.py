"""Pipeline contractual IAE - cadena §10.1 -> §10.7.

Ejecuta la cadena contractual completa:

  canonical -> identities -> units -> delta -> filtro radar
  -> build_target + catalog_to_p38_targets
  -> compute_nipc_contractual

Devuelve el resultado en un dict plano. No imprime. No escribe a
disco. El consumidor (script CLI, pipeline de reporte) decide como
presentar y persistir.

Uso:
    from src.institutional_accumulation.pipeline_contractual import (
        run_contractual_nipc)
    result = run_contractual_nipc()
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .sec_13f.identity.temporal_filter import filter_by_period
from .sec_13f.identity.amendments import apply_amendments
from .sec_13f.identity import sec13f_list as sl
from .sec_13f.identity.security_identity import (
    load_crosswalk_internal, resolve_batch_identities)
from . import operational_universe as ou
from .identity import period_state as ps
from .identity.target_builder import build_target, TargetUniverse
from .aggregation import catalog_p38_adapter as ca
from .aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares)
from .aggregation.nipc import compute_nipc_contractual

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = ROOT / "data" / "sec_13f" / "processed"
DEFAULT_MAPPINGS = ROOT / "data" / "mappings"
DEFAULT_OFFICIAL = Path(os.environ.get(
    "IAE_OFFICIAL_DIR", r"D:\13f_probe\official_list_13f"))
TSVS = ("SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE")

def load_canonical(folder, iso, *, data_dir=None):
    """Carga los 7 TSVs + filter_by_period + apply_amendments."""
    d = Path(data_dir or DEFAULT_DATA) / folder
    dfs = {n: pd.read_parquet(d / (n + ".parquet")) for n in TSVS}
    f = filter_by_period(dfs, iso)
    return apply_amendments(f, period=iso)["canonical_snapshot"]


def build_identities(snap, iso, *, mappings_dir=None):
    """Resuelve identidad P61 para todos los CUSIPs del canonical."""
    cusips = snap["INFOTABLE"]["CUSIP"].astype(str).unique()
    cw = load_crosswalk_internal()
    mp = Path(mappings_dir or DEFAULT_MAPPINGS)
    eqp = mp / "cusip_equivalence.csv"
    eq = pd.read_csv(eqp, dtype=str) if eqp.exists() else None
    return resolve_batch_identities(cusips, iso, equivalence_df=eq,
        crosswalk_internal_df=cw, figi_lookup=None)


def ticker_of(series):
    """Extrae el ticker de canonical_security de forma equity:X.

    Devuelve None para filas no canonical o con prefijo distinto.
    """
    x = series.fillna("").astype(str).str.strip()
    ok = x.str.startswith("equity:")
    return x.str.replace("equity:", "", regex=False).where(ok, None)


def _subset_universe(u, keys):
    """TargetUniverse reducido a las keys indicadas."""
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

def run_contractual_nipc(
    folders=("2025Q4", "2026Q1"),
    periods=("2025-12-31", "2026-03-31"),
    *,
    data_dir=None,
    mappings_dir=None,
    official_dir=None,
):
    """Ejecuta la cadena contractual y devuelve el resultado como dict.

    folders: (folder_prev, folder_curr) p.ej. ("2025Q4", "2026Q1").
    periods: (iso_prev, iso_curr) p.ej. ("2025-12-31", "2026-03-31").
    data_dir / mappings_dir / official_dir: overrides opcionales.

    Devuelve dict con NIPC + coverage + contadores + status +
    evidence_class. No imprime, no escribe.
    """
    mp = Path(mappings_dir or DEFAULT_MAPPINGS)
    off_dir = Path(official_dir or DEFAULT_OFFICIAL)
    f_prev, f_curr = folders
    iso_prev, iso_curr = periods

    # 1. Canonical por periodo
    s_prev = load_canonical(f_prev, iso_prev, data_dir=data_dir)
    s_curr = load_canonical(f_curr, iso_curr, data_dir=data_dir)

    # 2. Identidades P61
    i_prev = build_identities(s_prev, iso_prev, mappings_dir=mp)
    i_curr = build_identities(s_curr, iso_curr, mappings_dir=mp)

    # 3. Units
    u_prev = compute_reported_position_units(
        s_prev["INFOTABLE"], s_prev["SUBMISSION"],
        report_period=iso_prev, identity_results=i_prev)
    u_curr = compute_reported_position_units(
        s_curr["INFOTABLE"], s_curr["SUBMISSION"],
        report_period=iso_curr, identity_results=i_curr)

    # 4. Delta full
    delta = compute_delta_shares(u_curr, u_prev)

    # 5. Delta radar
    cat = pd.read_csv(mp / "radar_target_catalog.csv", dtype=str)
    radar_tickers = set(cat["radar_ticker"].dropna().astype(str).str.strip())
    d = delta.copy()
    d["_ticker"] = ticker_of(d["canonical_security"])
    radar = d[d["_ticker"].isin(radar_tickers)].copy()

    # 6. Universo catalogo + adapter
    mem = pd.read_csv(mp / "catalog_membership.csv", dtype=str)
    asg = pd.read_csv(mp / "catalog_assignments.csv", dtype=str)
    version_id = str(mem["version_id"].iloc[0])
    u_full = build_target(cat, mem, asg, version_id=version_id,
                          period_end=iso_curr,
                          catalog_version_id="cat", catalog_sha256="a" * 64)
    figi_by_ticker = dict(zip(cat["radar_ticker"], cat["share_class_figi"]))
    cw_csv = pd.read_csv(mp / "cusip_radar_crosswalk.csv", dtype=str)
    cusip_to_figi = {}
    for _, r in cw_csv.iterrows():
        tk = str(r["ticker"]).strip()
        f = figi_by_ticker.get(tk)
        if pd.notna(f) and f:
            cusip_to_figi[str(r["CUSIP"]).strip()] = f
    figi_to_key = {u_full.figi_by_key[k]: k for k in u_full.declared_keys}

    # Sub-universos por periodo
    per_period = {}
    for folder, iso, idmap in ((f_prev, iso_prev, i_prev),
                               (f_curr, iso_curr, i_curr)):
        sc = load_canonical(folder, iso, data_dir=data_dir)
        off = sl.load_official_list(off_dir / ("13flist_" + folder + ".txt"))
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

    u_q_prev = _subset_universe(u_full, per_period[f_prev]["keys"])
    u_q_curr = _subset_universe(u_full, per_period[f_curr]["keys"])
    pairwise = per_period[f_prev]["keys"] & per_period[f_curr]["keys"]

    st_prev = ps.build_period_state(
        u_q_prev,
        operational_evidence={k: "VERIFIED" for k in u_q_prev.declared_keys},
        sshprnamt_evidence=per_period[f_prev]["sh_by_key"])
    st_curr = ps.build_period_state(
        u_q_curr,
        operational_evidence={k: "VERIFIED" for k in u_q_curr.declared_keys},
        sshprnamt_evidence=per_period[f_curr]["sh_by_key"])

    t_prev, t_curr, r_prev, r_curr, feas = ca.catalog_to_p38_targets(
        u_q_prev, u_q_curr, state_q4=st_prev, state_q1=st_curr,
        pairwise_keys=pairwise)

    # 7. compute_nipc_contractual
    result = compute_nipc_contractual(
        radar, target_q4=t_prev, target_q1=t_curr,
        records_q4=r_prev, records_q1=r_curr)

    out = dict(result)
    out["periods"] = {"previous": f_prev, "current": f_curr,
                      "iso_previous": iso_prev, "iso_current": iso_curr}
    out["delta_full_rows"] = int(len(delta))
    out["delta_radar_rows"] = int(len(radar))
    out["target_q4_size"] = int(len(t_prev))
    out["target_q1_size"] = int(len(t_curr))
    out["feasibility"] = str(feas)
    return out