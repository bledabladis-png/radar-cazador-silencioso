#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fase 3 del walk-forward de calibracion: evalua TODAS las 240
combinaciones del grid sobre TEST y calcula la distribucion.

Responde: de las 240 combinaciones, ¿cuantas generalizan a TEST?
¿La "ganadora TRAIN" es especial o es una entre muchas?

Uso:
    py scripts/calibrate_sow_wf_phase3.py --sims 500
    py scripts/calibrate_sow_wf_phase3.py --sims 2000   # oficial

Salida:
    outputs/audit/wyckoff_calibrate_wf_phase3.csv
    outputs/audit/wyckoff_calibrate_wf_phase3_summary.json
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


# --- Override DATA_PARQUET al extendido ---
_EXT = ROOT / "data" / "stock_prices_extended.parquet"
if _EXT.exists():
    calib.DATA_PARQUET = _EXT
    print(f"[override] DATA_PARQUET -> {_EXT.name}")

# --- Split walk-forward ---
TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2026-10-01")

# --- Candidata ganadora TRAIN (identificada en fase 1) ---
WINNER_TRAIN = {"N": 40, "M": 5, "X_ATR": 1.0, "Y_VOL": 1.10}
# --- Candidata congelada v1.9 (referencia) ---
FROZEN = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_CSV = OUT_DIR / "wyckoff_calibrate_wf_phase3.csv"
OUT_JSON = OUT_DIR / "wyckoff_calibrate_wf_phase3_summary.json"

def _set_test_blocks():
    calib.BLOCKS = (("TEST", TEST_START, TEST_END),)


def _evaluate_combo(feats, starts_cache, N, M, X, Y, sims, seed):
    """Evalua una combinacion sobre TEST. Devuelve dict con metricas."""
    sow_cache = calib.compute_sow_cache(feats, N, X, Y)
    episodes = calib.build_episodes_landmark(feats, sow_cache, starts_cache, N, M)
    eps_test = [e for e in episodes if e["block_L"] == "TEST"]

    if not eps_test:
        return {
            "n_conf": 0, "n_base": 0,
            "lift_point": None, "lower_ci": None, "upper_ci": None,
            "D2": False,
        }

    agg = calib.aggregate_episodes(eps_test, feats, N, 20)
    b_raw = agg["by_block"]["TEST"]

    if b_raw["n_conf"] == 0 or b_raw["n_base"] == 0:
        return {
            "n_conf": b_raw["n_conf"], "n_base": b_raw["n_base"],
            "lift_point": None, "lower_ci": None, "upper_ci": None,
            "D2": False,
        }

    boot = calib.bootstrap_lift_H20(b_raw["rows"], B=sims, seed=seed)
    lc = boot.get("lower_ci")
    return {
        "n_conf": b_raw["n_conf"],
        "n_base": b_raw["n_base"],
        "lift_point": boot.get("lift_point"),
        "lower_ci": lc,
        "upper_ci": boot.get("upper_ci"),
        "D2": bool(lc is not None and lc > 0),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500,
                        help="Bootstrap por combo (default 500)")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    _set_test_blocks()

    print("== Fase 3: evaluacion completa del grid sobre TEST ==")
    print(f"TEST: {TEST_START.date()} -> {TEST_END.date()}")
    print(f"Grid: {len(calib.GRID_N)}x{len(calib.GRID_M)}x{len(calib.GRID_X)}x{len(calib.GRID_Y)} = "
          f"{len(calib.GRID_N)*len(calib.GRID_M)*len(calib.GRID_X)*len(calib.GRID_Y)} combos")
    print(f"Bootstrap: B={args.sims}")
    print()

    print("== Cargando dataset ==")
    t0 = time.time()
    df, tickers = calib.load_dataset()
    print(f"Tickers: {len(tickers)} ({time.time()-t0:.1f}s)")
    print(f"Rango: {df.index.min()} -> {df.index.max()}")

    print("== Precomputando features ==")
    t0 = time.time()
    feats = calib.precompute_features(df, tickers)
    print(f"Features: {len(feats)} ({time.time()-t0:.1f}s)")

    starts_cache = {
        tk: calib.find_candidate_starts(feat["candidate"])
        for tk, feat in feats.items()
    }
    print()

    # --- Recorrido completo del grid ---
    print("== Evaluando 240 combos sobre TEST ==")
    rows = []
    total = len(calib.GRID_N) * len(calib.GRID_M) * len(calib.GRID_X) * len(calib.GRID_Y)
    n_done = 0
    t0 = time.time()
    for N in calib.GRID_N:
        for M in calib.GRID_M:
            for X in calib.GRID_X:
                for Y in calib.GRID_Y:
                    r = _evaluate_combo(feats, starts_cache, N, M, X, Y,
                                        args.sims, args.seed)
                    r.update({"N": N, "M": M, "X_ATR": X, "Y_VOL": Y})
                    rows.append(r)
                    n_done += 1
                    if n_done % 20 == 0:
                        el = time.time() - t0
                        print(f"  [{n_done}/{total}] {el:.1f}s", flush=True)
    el = time.time() - t0
    print(f"  completado: {n_done} combos en {el:.1f}s")
    print()

    # --- Analisis de la distribucion ---
    df_out = pd.DataFrame(rows)
    n_total = len(df_out)
    n_d2_pass = int(df_out["D2"].sum())
    print("== Distribucion de D2 (lower_ci > 0) sobre TEST ==")
    print(f"  combos que pasan D2: {n_d2_pass}/{n_total} "
          f"({100*n_d2_pass/n_total:.1f}%)")

    # Lifts y posicion de winner/frozen
    df_lift = df_out.dropna(subset=["lift_point"]).sort_values(
        "lift_point", ascending=False,
    )
    print()
    print("== Top 5 combinaciones por lift (TEST) ==")
    for _, r in df_lift.head(5).iterrows():
        print(f"  N={int(r['N'])} M={int(r['M'])} X_ATR={r['X_ATR']:.2f} Y_VOL={r['Y_VOL']:.2f} "
              f"lift={r['lift_point']:+.4f} IC95=[{r['lower_ci']:+.4f}, {r['upper_ci']:+.4f}] "
              f"D2={r['D2']}")

    # Posicion de WINNER_TRAIN
    print()
    print("== Posicion de la ganadora TRAIN (N=40 M=5 X_ATR=1.0 Y_VOL=1.1) ==")
    mask_w = ((df_out["N"] == 40) & (df_out["M"] == 5) &
              (df_out["X_ATR"] == 1.0) & (df_out["Y_VOL"] == 1.1))
    row_w = df_out[mask_w]
    if not row_w.empty:
        rw = row_w.iloc[0]
        rank = (df_lift["lift_point"] > rw["lift_point"]).sum() + 1
        pct = 100 * rank / len(df_lift)
        print(f"  lift={rw['lift_point']:+.4f} IC95=[{rw['lower_ci']:+.4f}, {rw['upper_ci']:+.4f}] "
              f"D2={rw['D2']}")
        print(f"  ranking por lift: {rank}/{len(df_lift)} (percentil {pct:.1f})")

    # Posicion de FROZEN v1.9
    print()
    print("== Posicion de la congelada v1.9 (N=60 M=30 X_ATR=0.25 Y_VOL=1.1) ==")
    mask_f = ((df_out["N"] == 60) & (df_out["M"] == 30) &
              (df_out["X_ATR"] == 0.25) & (df_out["Y_VOL"] == 1.1))
    row_f = df_out[mask_f]
    if not row_f.empty:
        rf = row_f.iloc[0]
        rank = (df_lift["lift_point"] > rf["lift_point"]).sum() + 1
        pct = 100 * rank / len(df_lift)
        print(f"  lift={rf['lift_point']:+.4f} IC95=[{rf['lower_ci']:+.4f}, {rf['upper_ci']:+.4f}] "
              f"D2={rf['D2']}")
        print(f"  ranking por lift: {rank}/{len(df_lift)} (percentil {pct:.1f})")

    # --- Escritura ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    summary = {
        "split_test": [str(TEST_START.date()), str(TEST_END.date())],
        "grid_size": n_total,
        "n_pass_D2": n_d2_pass,
        "pct_pass_D2": 100 * n_d2_pass / n_total,
        "winner_train": WINNER_TRAIN,
        "frozen_v19": FROZEN,
        "bootstrap": {"B": args.sims, "seed": args.seed},
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print()
    print(f"Escrito: {OUT_CSV}")
    print(f"Escrito: {OUT_JSON}")


if __name__ == "__main__":
    main()