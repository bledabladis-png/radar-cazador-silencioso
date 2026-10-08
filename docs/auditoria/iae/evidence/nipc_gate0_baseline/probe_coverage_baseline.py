"""Probe coverage baseline NIPC - Q4 2025 -> Q1 2026.

Fase: coverage baseline (Q-T50-5, post dictamen TOP 50).
Dictamen: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md.
Metodologia: README.md (congelado antes de ejecutar).

Construye 4 universos anidados y calcula las 6 metricas pairwise
obligatorias (NIPC_COVERAGE_POLICY.md seccion 2) sobre cada uno:

    RAW_13F_SH_NULL      SH + PUTCALL NULL
       v
    ELIGIBLE_SEC         + SEC Official List {ACTIVE, ADDED}
       v
    TECHNICAL_RADAR      + ticker in radar_equities
       v  (universo tecnico aprobado cierra aqui)
    OPERATIONAL_EQUITY   + get_instrument_class(ticker) == EQUITY
                         (universo operativo de policy seccion 10)

Uso:
    py docs/auditoria/evidence/nipc_gate0_baseline/probe_coverage_baseline.py

Deterministico. Sin datetime.now(). Solo stdout.
NO escribe parquet. NO modifica el motor. NO fija thresholds.
NO llama a OpenFIGI.
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
    load_official_list, resolve_eligibility,
)
from src.institutional_accumulation.sec_13f.identity.security_identity import (
    load_crosswalk_internal, load_cusip_equivalence, resolve_batch_identities,
)
from src.institutional_accumulation.aggregation.delta_shares import (
    compute_reported_position_units, compute_delta_shares,
)
from src.institutional_accumulation.aggregation.nipc import (
    compute_nipc_and_coverage,
)
from src.instrument_registry import get_instrument_class

PROBE_DIR = Path(r"D:\13f_probe\processed")
OFFICIAL_DIR = Path(r"D:\13f_probe\official_list_13f")
STOCK_PRICES = ROOT / "data" / "stock_prices.parquet"
EU_SUFFIXES = (".L", ".DE", ".MC", ".PA", ".AS", ".MI",
               ".BR", ".ST", ".HE", ".CO", ".OL", ".VI", ".LS", ".IR")

PERIODS = {
    "Q4_2025": {"dir": PROBE_DIR / "2025Q4", "period": "2025-12-31",
                "list": OFFICIAL_DIR / "13flist_2025Q4.txt"},
    "Q1_2026": {"dir": PROBE_DIR / "2026Q1", "period": "2026-03-31",
                "list": OFFICIAL_DIR / "13flist_2026Q1.txt"},
}

TSVS = ["SUBMISSION", "COVERPAGE", "SUMMARYPAGE", "OTHERMANAGER",
        "OTHERMANAGER2", "SIGNATURE", "INFOTABLE"]


def _sep(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_radar_equities():
    """radar_equities = tickers no-EU en stock_prices.parquet.

    Mismo criterio que build_sample.py del Gate 0 OpenFIGI.
    Devuelve (radar_set, excluidos_por_sufijo_eu).
    """
    sp = pd.read_parquet(STOCK_PRICES)
    tickers = set(sp.columns.get_level_values(1).unique())
    tickers_str = {t for t in tickers if isinstance(t, str)}
    excluidos = {t for t in tickers_str if t.endswith(EU_SUFFIXES)}
    radar = tickers_str - excluidos
    return radar, excluidos

def _classify_fallout(units):
    # Clasifica cada fila de units segun por que entra o no en TECHNICAL_RADAR.
    def _row_reason(r):
        if r["in_radar"]:
            return "IN_RADAR"
        status = r["security_resolution_status"]
        canon = r["canonical_security"]
        if status == "CANONICAL":
            if isinstance(canon, str) and canon.startswith("equity:"):
                return "CANONICAL_TICKER_OUTSIDE_RADAR"
            if isinstance(canon, str) and canon.startswith("figi:"):
                return "FIGI_CANONICAL_NO_TICKER"
            return "OTHER"
        if status == "OBSERVED_ONLY":
            return "OBSERVED_ONLY_NO_CANONICAL"
        if status == "UNRESOLVED":
            return "UNRESOLVED_NO_CANONICAL"
        if status == "AMBIGUOUS":
            return "AMBIGUOUS_NO_CANONICAL"
        if status == "CONFLICT":
            return "CONFLICT_NO_CANONICAL"
        return "OTHER"
    return units.apply(_row_reason, axis=1)


FALLOUT_ORDER = (
    "IN_RADAR",
    "CANONICAL_TICKER_OUTSIDE_RADAR",
    "FIGI_CANONICAL_NO_TICKER",
    "OBSERVED_ONLY_NO_CANONICAL",
    "UNRESOLVED_NO_CANONICAL",
    "AMBIGUOUS_NO_CANONICAL",
    "CONFLICT_NO_CANONICAL",
    "OTHER",
)


def print_fallout_summary(units_elig, label):
    # Reporta por que las units de ELIGIBLE_SEC caen (o no) en TECHNICAL_RADAR.
    df = units_elig
    if df.empty:
        print(f"  [fallout {label}] units_elig vacio")
        return
    counts = df["fallout_reason"].value_counts().to_dict()
    weights = (df.groupby("fallout_reason")["sshprnamt_total"]
               .apply(lambda s: float(pd.to_numeric(s, errors="coerce").sum()))
               .to_dict())
    total_units = len(df)
    total_weight = float(pd.to_numeric(df["sshprnamt_total"], errors="coerce").sum())
    print(f"  --- fallout {label} (ELIGIBLE_SEC -> TECHNICAL_RADAR) ---")
    print("  {:<32} {:>10} {:>8} {:>20} {:>7}".format("categoria", "units", "%units", "peso_ssh", "%peso"))
    for cat in FALLOUT_ORDER:
        n = counts.get(cat, 0)
        w = weights.get(cat, 0.0)
        pu = (n / total_units * 100) if total_units else 0.0
        pw = (w / total_weight * 100) if total_weight else 0.0
        print("  {:<32} {:>10d} {:>7.2f}% {:>20,.0f} {:>6.2f}%".format(cat, n, pu, w, pw))

    fuera = df[df["fallout_reason"] == "CANONICAL_TICKER_OUTSIDE_RADAR"]
    if not fuera.empty:
        n_tick = fuera["ticker"].nunique()
        print(f"  detalle CANONICAL_TICKER_OUTSIDE_RADAR: {n_tick} tickers unicos")
        agg = (fuera.groupby("ticker")
               .agg(units=("ticker", "size"),
                    peso=("sshprnamt_total",
                          lambda s: float(pd.to_numeric(s, errors="coerce").sum())))
               .sort_values("peso", ascending=False)
               .head(10))
        for tick, row in agg.iterrows():
            print("    {:<12} units={:>6}  peso={:>18,.0f}".format(tick, int(row["units"]), row["peso"]))

    nocanon = df[df["fallout_reason"].str.endswith("_NO_CANONICAL", na=False)]
    if not nocanon.empty:
        n_cusips = nocanon["cusip"].nunique()
        print(f"  detalle NO_CANONICAL: {n_cusips} CUSIPs unicos sin canonical_security")
        agg = (nocanon.groupby("cusip")
               .agg(units=("cusip", "size"),
                    peso=("sshprnamt_total",
                          lambda s: float(pd.to_numeric(s, errors="coerce").sum())),
                    status=("security_resolution_status", "first"))
               .sort_values("peso", ascending=False)
               .head(10))
        for c, row in agg.iterrows():
            print("    {:<12} units={:>6}  peso={:>18,.0f}  status={}".format(c, int(row["units"]), row["peso"], row["status"]))
def load_period_dfs(q):
    d = PERIODS[q]["dir"]
    return {name: pd.read_parquet(d / (name + ".parquet")) for name in TSVS}


def _extract_ticker(canonical_security):
    """canonical_security es 'equity:<X>', 'figi:<X>' o None.
    Solo 'equity:' produce ticker. 'figi:' NO es ticker."""
    if not isinstance(canonical_security, str):
        return None
    if canonical_security.startswith("equity:"):
        return canonical_security.split(":", 1)[1]
    return None


def build_period(q, radar_equities, cw_df, eq_df):
    meta = PERIODS[q]
    period = meta["period"]
    dfs = load_period_dfs(q)
    filtered = filter_by_period(dfs, period)
    am = apply_amendments(filtered, period=period)
    snap = am["canonical_snapshot"]
    info = snap["INFOTABLE"]
    sub = snap["SUBMISSION"]

    mask = ((info["SSHPRNAMTTYPE"].astype(str).str.strip() == "SH")
            & info["PUTCALL"].isna())
    info_sh = info[mask].copy()
    cusips = info_sh["CUSIP"].astype(str).str.strip().unique().tolist()
    sec_list = load_official_list(meta["list"])
    elig_map = resolve_eligibility(cusips, sec_list)
    elig_set = {c for c, v in elig_map.items() if v["eligible"]}

    identity = resolve_batch_identities(
        cusips, period,
        equivalence_df=eq_df, crosswalk_internal_df=cw_df,
    )

    units = compute_reported_position_units(
        info_sh, sub, snap.get("COVERPAGE"),
        report_period=period, identity_results=identity,
    )
    units = units.copy()
    units["cusip"] = (units["observed_security_key"]
                      .str.replace("cusip:", "", regex=False))
    units["in_elig"] = units["cusip"].isin(elig_set)
    units["ticker"] = units["canonical_security"].apply(_extract_ticker)
    units["in_radar"] = units["ticker"].isin(radar_equities)
    units["instrument_class"] = units["ticker"].apply(
        lambda t: get_instrument_class(t) if isinstance(t, str) else "UNKNOWN"
    )
    units["is_equity"] = units["instrument_class"] == "EQUITY"
    units["fallout_reason"] = _classify_fallout(units)
    return {
        "period": period,
        "units_raw": units,
        "units_elig": units[units["in_elig"]].copy(),
        "units_tech_radar": units[
            units["in_elig"] & units["in_radar"]
        ].copy(),
        "units_op_equity": units[
            units["in_elig"] & units["in_radar"] & units["is_equity"]
        ].copy(),
        "n_cusips": len(cusips),
        "n_elig": len(elig_set),
    }


def run_pairwise(u_prev, u_curr, label):
    delta = compute_delta_shares(u_curr, u_prev)
    cov = compute_nipc_and_coverage(delta, u_curr, u_prev)
    print()
    print(f"--- {label} ---")
    print(f"  Q4 units: {len(u_prev)}   Q1 units: {len(u_curr)}")
    print(f"  delta filas: {len(delta)}")
    print(f"  match_status: {delta['match_status'].value_counts().to_dict()}")
    print(f"  NIPC total (observable): {cov['nipc_total']:.0f}")
    print(f"  coverage_previous={cov['coverage_previous']:.4f}"
          f"  coverage_current={cov['coverage_current']:.4f}")
    print(f"  paired_security_coverage={cov['paired_security_coverage']:.4f}"
          f"  paired_weighted_share_coverage={cov['paired_weighted_share_coverage']:.4f}")
    print(f"  unmapped_weight_previous={cov['unmapped_weight_previous']:.4f}"
          f"  unmapped_weight_current={cov['unmapped_weight_current']:.4f}")
    print(f"  STATUS: {cov['status']}")


def main():
    _sep("COVERAGE BASELINE NIPC - Q4 2025 -> Q1 2026")
    radar, excluidos = load_radar_equities()
    print(f"radar_equities = {len(radar)}")
    print(f"excluidos sufijo EU = {len(excluidos)}")
    print(f"  tickers EU: {sorted(excluidos)[:40]}")
    if len(excluidos) > 40:
        print(f"  ... y {len(excluidos) - 40} mas")

    cw = load_crosswalk_internal()
    eq = load_cusip_equivalence()
    print(f"crosswalk_internal rows: {len(cw)}")
    print(f"cusip_equivalence rows:  {len(eq)}")
    data = {}
    for q in ("Q4_2025", "Q1_2026"):
        _sep(f"PERIODO {q}")
        d = build_period(q, radar, cw, eq)
        print(f"period: {d['period']}")
        print(f"CUSIPs SH+null:        {d['n_cusips']}")
        print(f"CUSIPs elegibles SEC:  {d['n_elig']}")
        print(f"units RAW_13F_SH_NULL: {len(d['units_raw'])}")
        print(f"units ELIGIBLE_SEC:    {len(d['units_elig'])}")
        print(f"units TECHNICAL_RADAR: {len(d['units_tech_radar'])}")
        print(f"units OPERATIONAL_EQUITY: {len(d['units_op_equity'])}")
        print_fallout_summary(d["units_elig"], q)
        data[q] = d

    _sep("PAIRWISE COVERAGE - 4 UNIVERSOS ANIDADOS")
    for label, key in (
        ("RAW_13F_SH_NULL", "units_raw"),
        ("ELIGIBLE_SEC", "units_elig"),
        ("TECHNICAL_RADAR", "units_tech_radar"),
        ("OPERATIONAL_EQUITY", "units_op_equity"),
    ):
        run_pairwise(data["Q4_2025"][key], data["Q1_2026"][key], label)

    _sep("FIN")


if __name__ == "__main__":
    main()