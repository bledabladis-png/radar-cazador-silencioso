#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compara 3 candidatas SOW sobre el historico extendido.

Candidatas:
  C1: N=40 M=5  X_ATR=1.00 Y_VOL=1.10  (ganadora TRAIN 2015-2020)
  C2: N=60 M=5  X_ATR=0.50 Y_VOL=1.10  (top TEST 2021-2026)
  C0: N=60 M=30 X_ATR=0.25 Y_VOL=1.10  (congelada v1.9, referencia)

Evaluacion:
  - TRAIN 2015-2020 (calibracion)
  - TEST  2021-2026 (validacion OOS limpia)
  - Bloques anuales 2015-2026

Uso:
    py scripts/compare_sow_candidates.py --sims 2000

Salida:
    outputs/audit/wyckoff_compare_candidates.csv
    outputs/audit/wyckoff_compare_candidates_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.calibrate_wyckoff_sow_5b4bis as calib

_EXT = ROOT / "data" / "stock_prices_extended.parquet"
if _EXT.exists():
    calib.DATA_PARQUET = _EXT
    print(f"[override] DATA_PARQUET -> {_EXT.name}")

CANDIDATES = {
    "C1_winner_train": {"N": 40, "M": 5,  "X_ATR": 1.00, "Y_VOL": 1.10},
    "C2_top_test":     {"N": 60, "M": 5,  "X_ATR": 0.50, "Y_VOL": 1.10},
    "C0_frozen_v19":   {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10},
}

TRAIN_START = pd.Timestamp("2015-01-02")
TRAIN_END   = pd.Timestamp("2020-12-31")
TEST_START  = pd.Timestamp("2021-01-01")
TEST_END    = pd.Timestamp("2026-10-01")

BLOQUES_ANUALES = (
    ("2015", pd.Timestamp("2015-01-02"), pd.Timestamp("2015-12-31")),
    ("2016", pd.Timestamp("2016-01-01"), pd.Timestamp("2016-12-31")),
    ("2017", pd.Timestamp("2017-01-01"), pd.Timestamp("2017-12-31")),
    ("2018", pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")),
    ("2019", pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")),
    ("2020", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
    ("2021", pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
    ("2022", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    ("2023", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
    ("2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
    ("2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    ("2026", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-10-01")),
)

OUT_CSV = ROOT / "outputs" / "audit" / "wyckoff_compare_candidates.csv"
OUT_JSON = ROOT / "outputs" / "audit" / "wyckoff_compare_candidates_summary.json"

def _eval_periodo(feats, starts_cache, cand, start, end, sims, seed):
    """Evalua una candidata sobre [start, end] por block_L."""
    calib.BLOCKS = (("PERIOD", start, end),)
    N, M, X, Y = cand["N"], cand["M"], cand["X_ATR"], cand["Y_VOL"]

    sow_cache = calib.compute_sow_cache(feats, N, X, Y)
    episodes = calib.build_episodes_landmark(feats, sow_cache, starts_cache, N, M)
    eps = [e for e in episodes if e["block_L"] == "PERIOD"]

    if not eps:
        return {"n_conf": 0, "n_base": 0, "lift": None,
                "lower_ci": None, "upper_ci": None, "D2": False}

    agg = calib.aggregate_episodes(eps, feats, N, 20)
    b = agg["by_block"]["PERIOD"]
    if b["n_conf"] == 0 or b["n_base"] == 0:
        return {"n_conf": b["n_conf"], "n_base": b["n_base"], "lift": None,
                "lower_ci": None, "upper_ci": None, "D2": False}

    boot = calib.bootstrap_lift_H20(b["rows"], B=sims, seed=seed)
    lc = boot.get("lower_ci")
    return {
        "n_conf": b["n_conf"],
        "n_base": b["n_base"],
        "lift": boot.get("lift_point"),
        "lower_ci": lc,
        "upper_ci": boot.get("upper_ci"),
        "D2": bool(lc is not None and lc > 0),
    }


def _eval_anual(feats, starts_cache, cand, sims, seed):
    """Evalua una candidata por bloque anual."""
    out = {}
    for label, a, b in BLOQUES_ANUALES:
        r = _eval_periodo(feats, starts_cache, cand, a, b, sims, seed)
        out[label] = r
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    print("== Comparacion de candidatas SOW ==")
    print(f"TRAIN: {TRAIN_START.date()} -> {TRAIN_END.date()}")
    print(f"TEST:  {TEST_START.date()} -> {TEST_END.date()}")
    print(f"Bootstrap: B={args.sims}")
    print()
    for name, c in CANDIDATES.items():
        print(f"  {name}: N={c['N']} M={c['M']} X_ATR={c['X_ATR']} Y_VOL={c['Y_VOL']}")
    print()

    print("== Cargando dataset ==")
    t0 = time.time()
    df, tickers = calib.load_dataset()
    print(f"Tickers: {len(tickers)} ({time.time()-t0:.1f}s)")

    print("== Precomputando features ==")
    t0 = time.time()
    feats = calib.precompute_features(df, tickers)
    print(f"Features: {len(feats)} ({time.time()-t0:.1f}s)")

    starts_cache = {
        tk: calib.find_candidate_starts(feat["candidate"])
        for tk, feat in feats.items()
    }
    print()

    results = {}
    for name, cand in CANDIDATES.items():
        print(f"== {name} ==")
        print("  TRAIN...")
        train = _eval_periodo(feats, starts_cache, cand,
                              TRAIN_START, TRAIN_END, args.sims, args.seed)
        print(f"    n_conf={train['n_conf']} n_base={train['n_base']} "
              f"lift={train['lift']} IC=[{train['lower_ci']}, {train['upper_ci']}]")
        print("  TEST...")
        test = _eval_periodo(feats, starts_cache, cand,
                             TEST_START, TEST_END, args.sims, args.seed)
        print(f"    n_conf={test['n_conf']} n_base={test['n_base']} "
              f"lift={test['lift']} IC=[{test['lower_ci']}, {test['upper_ci']}]")
        print("  Bloques anuales...")
        anual = _eval_anual(feats, starts_cache, cand, args.sims, args.seed)
        for label, r in anual.items():
            print(f"    {label}: n_conf={r['n_conf']:4d} "
                  f"lift={r['lift'] if r['lift'] is None else round(r['lift'], 4)} "
                  f"IC=[{r['lower_ci']}, {r['upper_ci']}] D2={r['D2']}")
        results[name] = {"train": train, "test": test, "anual": anual}
        print()

    # --- Tabla resumen ---
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)
    print(f"{'Candidata':<22} {'TRAIN_lift':>12} {'TEST_lift':>12} "
          f"{'TEST_IC_lo':>12} {'TEST_D2':>8}")
    for name, r in results.items():
        tl = r["train"]["lift"]
        te = r["test"]["lift"]
        lc = r["test"]["lower_ci"]
        d2 = r["test"]["D2"]
        print(f"{name:<22} {tl if tl is not None else '--':>12} "
              f"{te if te is not None else '--':>12} "
              f"{lc if lc is not None else '--':>12} "
              f"{str(d2):>8}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, r in results.items():
        for period in ("train", "test"):
            row = {"candidate": name, "period": period}
            row.update(r[period])
            rows.append(row)
        for label, val in r["anual"].items():
            row = {"candidate": name, "period": f"anual_{label}"}
            row.update(val)
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "candidates": CANDIDATES,
            "bootstrap": {"B": args.sims, "seed": args.seed},
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"Escrito: {OUT_CSV}")
    print(f"Escrito: {OUT_JSON}")


if __name__ == "__main__":
    main()