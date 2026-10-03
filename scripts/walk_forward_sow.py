#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward historico de SOW (plan migracion v3, seccion 15).

Split temporal estricto:
    Periodo A (calibracion): 2021-10-01 -> 2023-12-31
    Periodo B (validacion):  2024-01-01 -> 2026-10-01

Evalua la candidata congelada v1.9 (N=60, M=30, X_ATR=0.25, Y_VOL=1.10)
por separado sobre A y B. PASS si lift_H20 tiene IC95% cuyo limite
inferior > 0 en AMBOS periodos.

Read-only. NO modifica config, contrato ni datos.
Reutiliza el calibrador 5b.4-bis via import.

Uso:
    py scripts/walk_forward_sow.py
    py scripts/walk_forward_sow.py --sims 500   # smoke (bootstrap reducido)

Salida:
    outputs/audit/wyckoff_walk_forward.csv
    outputs/audit/wyckoff_walk_forward_summary.json
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

import scripts.calibrate_wyckoff_sow_5b4bis as calib


# --- Split walk-forward ---
A_START = pd.Timestamp("2021-10-01")
A_END = pd.Timestamp("2023-12-31")
B_START = pd.Timestamp("2024-01-01")
B_END = pd.Timestamp("2026-10-01")

# --- Candidata congelada v1.9 ---
CANDIDATE = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_CSV = OUT_DIR / "wyckoff_walk_forward.csv"
OUT_JSON = OUT_DIR / "wyckoff_walk_forward_summary.json"

def _patch_blocks():
    """Sustituye los BLOCKS del calibrador por A/B del walk-forward."""
    calib.BLOCKS = (
        ("A", A_START, A_END),
        ("B", B_START, B_END),
    )


def _filter_episodes_by_block(episodes, block_label):
    """Filtra episodios por block_L."""
    return [e for e in episodes if e["block_L"] == block_label]


def _run_candidate(feats, starts_cache, N, M, X, Y, boot_b, seed):
    """Construye episodios y metricas para una candidata.

    Devuelve dict con:
        by_block_A: agregado de A (H20)
        by_block_B: agregado de B (H20)
        by_block_ALL: agregado total
        boot_A, boot_B: bootstrap de lift H20 (struct_deterioration)
    """
    t0 = time.time()

    # 1. Cache SOW
    sow_cache = calib.compute_sow_cache(feats, N, X, Y)
    print(f"  sow_cache OK ({time.time()-t0:.1f}s)")

    # 2. Episodios (todos; block_L ya viene del BLOCKS parcheado)
    episodes = calib.build_episodes_landmark(
        feats, sow_cache, starts_cache, N, M,
    )
    eps_A = _filter_episodes_by_block(episodes, "A")
    eps_B = _filter_episodes_by_block(episodes, "B")
    print(f"  episodios: A={len(eps_A)} B={len(eps_B)} total={len(episodes)}")

    # 3. Metricas por horizonte (solo H20 por rapidez)
    agg_A = calib.aggregate_episodes(eps_A, feats, N, 20)
    agg_B = calib.aggregate_episodes(eps_B, feats, N, 20)
    agg_ALL = calib.aggregate_episodes(episodes, feats, N, 20)

    # 4. Bootstrap sobre struct_deterioration H20
    def _boot(agg):
        # aggregate_episodes devuelve {'by_block': {'A': {...}, 'ALL': {...}}}
        # Necesitamos las rows del bloque ALL (o el unico bloque relevante).
        by_block = agg["by_block"]
        # Elegir el bloque con mas filas
        best = None
        best_n = 0
        for label, bucket in by_block.items():
            n = len(bucket["rows"])
            if n > best_n:
                best = bucket
                best_n = n
        if best is None or best_n == 0:
            return {"lift_point": None, "lower_ci": None, "upper_ci": None,
                    "n_boot_ok": 0}
        return calib.bootstrap_lift_H20(best["rows"], B=boot_b, seed=seed)

    boot_A = _boot(agg_A)
    boot_B = _boot(agg_B)
    boot_ALL = _boot(agg_ALL)

    return {
        "N": N, "M": M, "X_ATR": X, "Y_VOL": Y,
        "n_eps_A": len(eps_A),
        "n_eps_B": len(eps_B),
        "agg_A": agg_A, "agg_B": agg_B, "agg_ALL": agg_ALL,
        "boot_A": boot_A, "boot_B": boot_B, "boot_ALL": boot_ALL,
    }

def _block_summary(agg, block_label, N):
    """Extrae resumen de un bloque del agregado."""
    by_block = agg["by_block"]
    # Buscar bloque especifico o ALL
    if block_label in by_block:
        b = by_block[block_label]
    elif "ALL" in by_block:
        b = by_block["ALL"]
    else:
        return None
    n_conf = b["n_conf"]
    n_base = b["n_base"]
    if n_conf == 0 or n_base == 0:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None}
    vals_conf = b["vals_conf"]["struct_deterioration"]
    vals_base = b["vals_base"]["struct_deterioration"]
    if not vals_conf or not vals_base:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None}
    p_conf = float(np.mean(vals_conf))
    p_base = float(np.mean(vals_base))
    return {"n_conf": n_conf, "n_base": n_base, "lift": p_conf - p_base}


def _classify(boot_A, boot_B):
    """Devuelve PASS / FAIL / INSUFFICIENT.

    PASS: lower_ci > 0 en A y en B.
    FAIL: IC valido pero lower_ci <= 0 en alguno.
    INSUFFICIENT: IC no calculable (menos de 50 boots validos).
    """
    la = boot_A.get("lower_ci")
    lb = boot_B.get("lower_ci")
    if la is None or lb is None:
        return "INSUFFICIENT"
    if la > 0 and lb > 0:
        return "PASS"
    return "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=2000,
                        help="Iteraciones de bootstrap (default 2000)")
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    print("== Walk-forward SOW ==")
    print(f"Candidata: N={CANDIDATE['N']} M={CANDIDATE['M']} "
          f"X_ATR={CANDIDATE['X_ATR']} Y_VOL={CANDIDATE['Y_VOL']}")
    print(f"Periodo A: {A_START.date()} -> {A_END.date()}")
    print(f"Periodo B: {B_START.date()} -> {B_END.date()}")
    print(f"Bootstrap: B={args.sims} seed={args.seed}")
    print()

    _patch_blocks()

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
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Candidate starts totales: {total_starts}")
    print()

    print("== Evaluando candidata congelada ==")
    r = _run_candidate(
        feats, starts_cache,
        CANDIDATE["N"], CANDIDATE["M"],
        CANDIDATE["X_ATR"], CANDIDATE["Y_VOL"],
        boot_b=args.sims, seed=args.seed,
    )
    print()
    print("== Resultados ==")
    s_A = _block_summary(r["agg_A"], "A", CANDIDATE["N"])
    s_B = _block_summary(r["agg_B"], "B", CANDIDATE["N"])
    s_ALL = _block_summary(r["agg_ALL"], "ALL", CANDIDATE["N"])
    print(f"Periodo A: n_conf={s_A['n_conf'] if s_A else 'NA'} "
          f"n_base={s_A['n_base'] if s_A else 'NA'} "
          f"lift={s_A['lift'] if s_A else 'NA'}")
    print(f"  bootstrap: lift={r['boot_A']['lift_point']} "
          f"IC95=[{r['boot_A']['lower_ci']}, {r['boot_A']['upper_ci']}]")
    print(f"Periodo B: n_conf={s_B['n_conf'] if s_B else 'NA'} "
          f"n_base={s_B['n_base'] if s_B else 'NA'} "
          f"lift={s_B['lift'] if s_B else 'NA'}")
    print(f"  bootstrap: lift={r['boot_B']['lift_point']} "
          f"IC95=[{r['boot_B']['lower_ci']}, {r['boot_B']['upper_ci']}]")
    print(f"TOTAL:     n_conf={s_ALL['n_conf'] if s_ALL else 'NA'} "
          f"n_base={s_ALL['n_base'] if s_ALL else 'NA'} "
          f"lift={s_ALL['lift'] if s_ALL else 'NA'}")
    print(f"  bootstrap: lift={r['boot_ALL']['lift_point']} "
          f"IC95=[{r['boot_ALL']['lower_ci']}, {r['boot_ALL']['upper_ci']}]")

    verdict = _classify(r["boot_A"], r["boot_B"])
    print()
    print(f"VEREDICTO: {verdict}")

    # --- Escritura de resultados ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, s, b in (("A", s_A, r["boot_A"]), ("B", s_B, r["boot_B"]),
                        ("ALL", s_ALL, r["boot_ALL"])):
        if s is None:
            continue
        rows.append({
            "period": label,
            "n_conf": s["n_conf"],
            "n_base": s["n_base"],
            "lift_point": b["lift_point"],
            "lower_ci": b["lower_ci"],
            "upper_ci": b["upper_ci"],
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    summary = {
        "candidate": CANDIDATE,
        "split": {
            "A": [str(A_START.date()), str(A_END.date())],
            "B": [str(B_START.date()), str(B_END.date())],
        },
        "bootstrap": {"B": args.sims, "seed": args.seed},
        "results": rows,
        "verdict": verdict,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print()
    print(f"Escrito: {OUT_CSV}")
    print(f"Escrito: {OUT_JSON}")


if __name__ == "__main__":
    main()