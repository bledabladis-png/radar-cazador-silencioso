#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Power analysis 5b.X - simula potencia bajo la estructura de clusters real.

Requisito del auditor externo (dictamen 2026-10-02): el umbral de muestra
para 5b.X debe derivarse de una simulacion de potencia que respete el
cluster-bootstrap ticker real, la censura H20 y la estructura de episodios
del desarrollo.

Read-only sobre data. NO toca config. NO modifica el calibrador.

Uso:
    py scripts/power_analysis_5bX.py              # N_SIM=500
    py scripts/power_analysis_5bX.py --sims 50    # smoke test

Salida:
    outputs/audit/wyckoff_5bX_power_grid.csv
    outputs/audit/wyckoff_5bX_power_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_wyckoff_sow_5b4bis import (
    load_dataset,
    precompute_features,
    find_candidate_starts,
    compute_sow_cache,
    build_episodes_landmark,
    episode_metrics,
    bootstrap_lift_H20,
    BOOT_SEED,
)

OUT_DIR = ROOT / "outputs" / "audit"
OUT_GRID = OUT_DIR / "wyckoff_5bX_power_grid.csv"
OUT_SUMMARY = OUT_DIR / "wyckoff_5bX_power_summary.json"

# --- Candidata congelada v1.9 ---
CANDIDATE = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

# --- Grid de simulacion ---
LIFT_REAL_GRID = (0.03, 0.05, 0.074, 0.10, 0.15)
N_CONF_GRID = (100, 200, 300, 400, 450, 500, 550, 600, 700, 800)
POWER_TARGET = 0.80

# --- Bootstrap interno de la simulacion (menor que BOOT_B=2000 por velocidad) ---
BOOT_B_SIM = 300
SIM_SEED = 20261003


def build_real_pool():
    """Construye pool de episodios reales con outcome H20."""
    print("== Cargando dataset y construyendo pool real ==")
    df, tickers = load_dataset()
    print(f"Tickers: {len(tickers)}")
    feats = precompute_features(df, tickers)
    print(f"Features: {len(feats)}")

    starts_cache = {tk: find_candidate_starts(feat["candidate"])
                    for tk, feat in feats.items()}
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Candidate starts: {total_starts}")

    N = CANDIDATE["N"]
    M = CANDIDATE["M"]
    X = CANDIDATE["X_ATR"]
    Y = CANDIDATE["Y_VOL"]

    sow_cache = compute_sow_cache(feats, N, X, Y)
    episodes = build_episodes_landmark(feats, sow_cache, starts_cache, N, M)
    print(f"Episodios con landmark: {len(episodes)}")

    pool_by_tk = {}
    n_dropped = 0
    for e in episodes:
        tk = e["ticker"]
        feat = feats[tk]
        m = episode_metrics(feat, e["L_idx"], N, 20)
        if m is None or m.get("struct_deterioration") is None:
            n_dropped += 1
            continue
        v = int(m["struct_deterioration"])
        pool_by_tk.setdefault(tk, []).append((bool(e["is_confirmed"]), v))

    n_conf = sum(1 for tk in pool_by_tk for (c, _) in pool_by_tk[tk] if c)
    n_base = sum(1 for tk in pool_by_tk for (c, _) in pool_by_tk[tk] if not c)
    print(f"Pool elegible: conf={n_conf} base={n_base} "
          f"tickers={len(pool_by_tk)} (descartados por censura: {n_dropped})")

    return pool_by_tk, feats


def empirical_rates(pool_by_tk):
    """Tasa observada de struct_deterioration en conf y base."""
    s_c = n_c = s_b = n_b = 0
    for tk in pool_by_tk:
        for (is_conf, v) in pool_by_tk[tk]:
            if is_conf:
                s_c += v; n_c += 1
            else:
                s_b += v; n_b += 1
    p_conf = s_c / n_c if n_c else None
    p_base = s_b / n_b if n_b else None
    lift = (p_conf - p_base) if (p_conf is not None and p_base is not None) else None
    return p_conf, p_base, lift, n_c, n_b


def simulate_once(pool_by_tk, n_conf_target, lift_real, p_base, rng):
    """Una simulacion: remuestrea tickers, asigna outcomes, corre bootstrap."""
    tickers = list(pool_by_tk.keys())
    n_tk = len(tickers)

    chosen = []
    n_conf_acc = 0
    max_draws = n_tk * 20
    while n_conf_acc < n_conf_target and len(chosen) < max_draws:
        idx = int(rng.integers(0, n_tk))
        tk = tickers[idx]
        chosen.append(tk)
        for (is_conf, _) in pool_by_tk[tk]:
            if is_conf:
                n_conf_acc += 1

    if n_conf_acc < n_conf_target:
        return None

    rows = []
    for tk in chosen:
        for (is_conf, _) in pool_by_tk[tk]:
            p = p_base + (lift_real if is_conf else 0.0)
            p = min(max(p, 0.0), 1.0)
            y = int(rng.random() < p)
            rows.append((tk, is_conf, {"struct_deterioration": y}))

    seed_boot = int(rng.integers(0, 2**31 - 1))
    boot = bootstrap_lift_H20(rows, B=BOOT_B_SIM, seed=seed_boot)
    return boot


def run_power_grid(pool_by_tk, p_base, n_sims):
    """Grid completo lift_real x n_conf_target."""
    results = []
    t_start = time.time()

    for lift_real in LIFT_REAL_GRID:
        for n_conf_target in N_CONF_GRID:
            t0 = time.time()
            hits = 0
            valid = 0
            lowers = []
            for s in range(n_sims):
                rng = np.random.default_rng(SIM_SEED + s * 7919 + int(lift_real * 10000))
                boot = simulate_once(pool_by_tk, n_conf_target, lift_real, p_base, rng)
                if boot is None or boot.get("lower_ci") is None:
                    continue
                valid += 1
                lowers.append(boot["lower_ci"])
                if boot["lower_ci"] > 0:
                    hits += 1
            power = hits / valid if valid else None
            mean_lower = float(np.mean(lowers)) if lowers else None
            if valid > 0 and power is not None:
                se_power = float(np.sqrt(power * (1 - power) / valid))
                ci_lo = max(0.0, power - 1.96 * se_power)
                ci_hi = min(1.0, power + 1.96 * se_power)
            else:
                se_power = None
                ci_lo = ci_hi = None
            results.append({
                "lift_real": lift_real,
                "n_conf_target": n_conf_target,
                "n_sims": n_sims,
                "n_valid": valid,
                "power": power,
                "power_se": se_power,
                "power_ci95_lo": ci_lo,
                "power_ci95_hi": ci_hi,
                "mean_lower_ci": mean_lower,
            })
            el = time.time() - t0
            p_str = f"{power:.3f}" if power is not None else "nan"
            print(f"  lift={lift_real:.3f} n_conf={n_conf_target:>4} "
                  f"power={p_str} valid={valid}/{n_sims} ({el:.1f}s)", flush=True)

    print(f"Grid completado en {time.time()-t_start:.1f}s")
    return results


def find_min_n_for_power(results, lift_real, power_target):
    for r in sorted([x for x in results if x["lift_real"] == lift_real],
                    key=lambda x: x["n_conf_target"]):
        if r["power"] is not None and r["power"] >= power_target:
            return r["n_conf_target"], r["power"]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=500)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_by_tk, _ = build_real_pool()
    p_conf, p_base, lift_dev, n_c, n_b = empirical_rates(pool_by_tk)
    print(f"Tasas observadas: p_conf={p_conf:.4f} p_base={p_base:.4f} "
          f"lift_dev={lift_dev:+.4f}")
    print()
    print(f"== Power grid (N_SIM={args.sims}, BOOT_B_SIM={BOOT_B_SIM}) ==")

    results = run_power_grid(pool_by_tk, p_base, args.sims)

    df = pd.DataFrame(results)
    df.to_csv(OUT_GRID, index=False)
    print(f"\\nGrid CSV: {OUT_GRID}")

    summary = {
        "schema": "wyckoff_5bX_power_summary_v1",
        "candidate": CANDIDATE,
        "pool_real": {
            "n_confirmed": n_c,
            "n_baseline": n_b,
            "n_tickers": len(pool_by_tk),
            "p_conf_obs": p_conf,
            "p_base_obs": p_base,
            "lift_dev_obs": lift_dev,
        },
        "simulation": {
            "n_sims": args.sims,
            "boot_b_sim": BOOT_B_SIM,
            "boot_seed_calibrator": BOOT_SEED,
            "sim_seed": SIM_SEED,
            "lift_real_grid": list(LIFT_REAL_GRID),
            "n_conf_grid": list(N_CONF_GRID),
            "power_target": POWER_TARGET,
        },
        "min_n_conf_for_power": {},
    }

    for lr in LIFT_REAL_GRID:
        n_min, p_at = find_min_n_for_power(results, lr, POWER_TARGET)
        summary["min_n_conf_for_power"][f"lift_{lr}"] = {
            "n_conf_min": n_min, "power_at_min": p_at,
        }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON: {OUT_SUMMARY}")
    print()
    print("== Resumen ==")
    for lr in LIFT_REAL_GRID:
        n_min, p_at = find_min_n_for_power(results, lr, POWER_TARGET)
        if n_min is None:
            print(f"  lift={lr:.3f}: no alcanza {POWER_TARGET:.0%} con el grid actual")
        else:
            print(f"  lift={lr:.3f}: n_conf_min={n_min} (power={p_at:.3f})")


if __name__ == "__main__":
    main()