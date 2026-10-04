#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibracion walk-forward con split limpio:
    TRAIN 2015-01-02 -> 2020-12-31  (calibracion del grid)
    TEST  2021-01-01 -> 2026-10-01  (validacion OOS limpia)

Reutiliza el calibrador 5b.4-bis. NO modifica produccion.
Override de DATA_PARQUET al extendido y de BLOCKS.

Uso:
    py scripts/calibrate_sow_wf.py --sims 100     # smoke
    py scripts/calibrate_sow_wf.py                # B=500 por combo

Salida:
    outputs/audit/wyckoff_calibrate_wf_grid.csv
    outputs/audit/wyckoff_calibrate_wf_summary.json
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

# --- Split walk-forward (limpio) ---
TRAIN_START = pd.Timestamp("2015-01-02")
TRAIN_END = pd.Timestamp("2020-12-31")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2026-10-01")

# --- Candidata congelada v1.9 (referencia) ---
FROZEN = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_GRID = OUT_DIR / "wyckoff_calibrate_wf_grid.csv"
OUT_JSON = OUT_DIR / "wyckoff_calibrate_wf_summary.json"


def _set_train_blocks():
    calib.BLOCKS = (("TRAIN", TRAIN_START, TRAIN_END),)


def _set_test_blocks():
    calib.BLOCKS = (("TEST", TEST_START, TEST_END),)

def _run_grid_train(feats, starts_cache, sims, seed):
    """Corre el grid sobre TRAIN. Devuelve filas con D1-D3 + bootstrap."""
    _set_train_blocks()
    t0 = time.time()
    rows = calib.run_grid(feats, starts_cache)
    print(f"  grid corrido en {time.time()-t0:.1f}s ({len(rows)} filas)")

    print(f"  bootstrap por fila (B={sims})...")
    t0 = time.time()
    boot_map = {}
    for i, r in enumerate(rows):
        boot_map[i] = calib.bootstrap_lift_H20(r["_rows_H20"], B=sims, seed=seed)
    print(f"  bootstrap OK ({time.time()-t0:.1f}s)")

    # Evaluar D1-D3 sobre TRAIN
    for i, r in enumerate(rows):
        d1, d2, d3, bl_eval, bl_pos = calib.evaluate_row(r, boot_map[i])
        r["D1"] = d1
        r["D2"] = d2
        r["D3"] = d3
        r["_bloques_eval"] = bl_eval
        r["_bloques_lift_pos"] = bl_pos

    return rows, boot_map


def _select_winner(rows, boot_map):
    """Con un solo bloque en TRAIN, select_candidate colapsa al tiebreak
    lexicografico entre las 200+ combinaciones empatadas. Se usa un
    ranking propio: (1) D1-D3 pasa, (2) mayor lower_ci, (3) mayor n_conf,
    (4) tiebreak lexicografico."""
    candidatos = []
    for i, r in enumerate(rows):
        if not (r.get("D1") and r.get("D2") and r.get("D3")):
            continue
        boot = boot_map[i]
        lc = boot.get("lower_ci") or -1e9
        h20 = r["by_horizon"][20]["ALL"]
        n_conf = h20["n_conf"]
        candidatos.append((
            -lc,  # menor = mejor
            -n_conf,
            r["N"], r["M"], r["X_ATR"], r["Y_VOL"],
            i,
        ))
    if not candidatos:
        return None, "ninguna combinacion pasa D1-D3 en TRAIN"
    candidatos.sort()
    _, _, _, _, _, _, idx = candidatos[0]
    return rows[idx], "ranking propio (lower_ci desc, n_conf desc)"


def _summarize_row(r, label):
    """Extrae N, M, X_ATR, Y_VOL, D1-D3, lift, IC."""
    h20 = r["by_horizon"][20]
    allb = h20["ALL"]
    return {
        "block": label,
        "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
        "n_conf": allb["n_conf"], "n_base": allb["n_base"],
        "D1": r.get("D1"), "D2": r.get("D2"), "D3": r.get("D3"),
    }


def _validate_on_test(feats, starts_cache, cand, sims, seed):
    """Corre la candidata sobre TEST. Bootstrap + D1-D3."""
    _set_test_blocks()
    N, M = cand["N"], cand["M"]
    X, Y = cand["X_ATR"], cand["Y_VOL"]

    sow_cache = calib.compute_sow_cache(feats, N, X, Y)
    episodes = calib.build_episodes_landmark(feats, sow_cache, starts_cache, N, M)
    print(f"    episodios TEST: {len(episodes)}")

    # Filtrar a TEST explicitamente por block_L
    eps_test = [e for e in episodes if e["block_L"] == "TEST"]
    print(f"    episodios en bloque TEST: {len(eps_test)}")

    agg = calib.aggregate_episodes(eps_test, feats, N, 20)
    b_raw = agg["by_block"]["TEST"]

    # Enriquecer el bucket con lifts antes de evaluate_row
    b = calib.summarize_block(b_raw)
    # rows no esta en el resumen, pero bootstrap_lift_H20 lo necesita
    b["rows"] = b_raw["rows"]

    boot = calib.bootstrap_lift_H20(b["rows"], B=sims, seed=seed)

    h20 = {"TEST": b, "ALL": b}
    r = {"N": N, "M": M, "X_ATR": X, "Y_VOL": Y, "by_horizon": {20: h20}}
    d1, d2, d3, bl_eval, bl_pos = calib.evaluate_row(r, boot)

    return {
        "N": N, "M": M, "X_ATR": X, "Y_VOL": Y,
        "n_conf": b["n_conf"], "n_base": b["n_base"],
        "lift_point": boot["lift_point"],
        "lower_ci": boot["lower_ci"], "upper_ci": boot["upper_ci"],
        "D1": d1, "D2": d2, "D3": d3,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500,
                        help="Bootstrap por combo en TRAIN (default 500)")
    parser.add_argument("--sims-test", type=int, default=2000,
                        help="Bootstrap en TEST (default 2000)")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    print("== Walk-forward calibracion ==")
    print(f"TRAIN: {TRAIN_START.date()} -> {TRAIN_END.date()}")
    print(f"TEST:  {TEST_START.date()} -> {TEST_END.date()}")
    print(f"Grid: {len(calib.GRID_N)}x{len(calib.GRID_M)}x{len(calib.GRID_X)}x{len(calib.GRID_Y)} = "
          f"{len(calib.GRID_N)*len(calib.GRID_M)*len(calib.GRID_X)*len(calib.GRID_Y)} combos")
    print(f"Bootstrap TRAIN: B={args.sims} por combo")
    print(f"Bootstrap TEST:  B={args.sims_test}")
    print()

    print("== Cargando dataset extendido ==")
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
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Candidate starts totales: {total_starts}")
    print()

    # ---- Fase 1: calibracion sobre TRAIN ----
    print("== FASE 1: calibracion sobre TRAIN ==")
    rows, boot_map = _run_grid_train(feats, starts_cache, args.sims, args.seed)

    pasan = [r for r in rows if r.get("D1") and r.get("D2") and r.get("D3")]
    print(f"  combos que pasan D1+D2+D3: {len(pasan)}/{len(rows)}")

    elegido, motivo = _select_winner(rows, boot_map)
    if elegido is None:
        print(f"  [WARN] ninguna combinacion pasa D1-D3 en TRAIN. Motivo: {motivo}")
        print("  Se usa la candidata congelada v1.9 como referencia.")
        winner = dict(FROZEN)
        winner_src = "frozen (no winner in TRAIN)"
    else:
        winner = {
            "N": elegido["N"], "M": elegido["M"],
            "X_ATR": elegido["X_ATR"], "Y_VOL": elegido["Y_VOL"],
        }
        winner_src = "TRAIN winner"
        print(f"  Ganadora TRAIN: N={winner['N']} M={winner['M']} "
              f"X_ATR={winner['X_ATR']} Y_VOL={winner['Y_VOL']}")

    print()

    # Escribir grid CSV inmediatamente (cache por si falla la fase 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_rows_now = []
    for r in rows:
        h20 = r["by_horizon"][20]["ALL"]
        boot = boot_map[rows.index(r)]
        grid_rows_now.append({
            "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
            "n_conf": h20["n_conf"], "n_base": h20["n_base"],
            "lift_point": boot.get("lift_point"),
            "lower_ci": boot.get("lower_ci"),
            "upper_ci": boot.get("upper_ci"),
            "D1": r.get("D1"), "D2": r.get("D2"), "D3": r.get("D3"),
        })
    pd.DataFrame(grid_rows_now).to_csv(OUT_GRID, index=False)
    print(f"  [cache] grid TRAIN escrito en {OUT_GRID}")

    # ---- Fase 2: validacion sobre TEST ----
    print("== FASE 2: validacion sobre TEST ==")
    print(f"  [a] Candidata ganadora TRAIN: {winner}")
    test_winner = _validate_on_test(
        feats, starts_cache, winner, args.sims_test, args.seed,
    )
    print(f"      n_conf={test_winner['n_conf']} n_base={test_winner['n_base']}")
    print(f"      lift={test_winner['lift_point']} "
          f"IC95=[{test_winner['lower_ci']}, {test_winner['upper_ci']}]")
    print(f"      D1={test_winner['D1']} D2={test_winner['D2']} D3={test_winner['D3']}")

    print()
    print(f"  [b] Candidata congelada v1.9: {FROZEN}")
    test_frozen = _validate_on_test(
        feats, starts_cache, FROZEN, args.sims_test, args.seed,
    )
    print(f"      n_conf={test_frozen['n_conf']} n_base={test_frozen['n_base']}")
    print(f"      lift={test_frozen['lift_point']} "
          f"IC95=[{test_frozen['lower_ci']}, {test_frozen['upper_ci']}]")
    print(f"      D1={test_frozen['D1']} D2={test_frozen['D2']} D3={test_frozen['D3']}")

    # ---- Verdicto ----
    print()
    if test_winner["D2"] and test_frozen["D2"]:
        verdict = "PASS_AMBAS"
    elif test_winner["D2"] or test_frozen["D2"]:
        verdict = "PASS_PARCIAL"
    else:
        verdict = "FAIL"
    print(f"VEREDICTO: {verdict}")

    # ---- Escritura ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Grid TRAIN
    grid_rows = []
    for r in rows:
        h20 = r["by_horizon"][20]["ALL"]
        grid_rows.append({
            "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
            "n_conf": h20["n_conf"], "n_base": h20["n_base"],
            "D1": r.get("D1"), "D2": r.get("D2"), "D3": r.get("D3"),
        })
    pd.DataFrame(grid_rows).to_csv(OUT_GRID, index=False)

    summary = {
        "split": {
            "train": [str(TRAIN_START.date()), str(TRAIN_END.date())],
            "test": [str(TEST_START.date()), str(TEST_END.date())],
        },
        "grid_size": len(rows),
        "n_pass_D1D2D3_train": len(pasan),
        "winner_train": winner,
        "winner_source": winner_src,
        "test_winner": test_winner,
        "test_frozen_v19": test_frozen,
        "verdict": verdict,
        "bootstrap": {"train_per_combo": args.sims, "test": args.sims_test,
                      "seed": args.seed},
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"Escrito: {OUT_GRID}")
    print(f"Escrito: {OUT_JSON}")


if __name__ == "__main__":
    main()