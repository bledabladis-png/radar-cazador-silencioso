#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward anual de SOW (opcion 2, investigacion temporal).

Divide el historico en bloques anuales y evalua la candidata congelada
v1.9 (N=60, M=30, X_ATR=0.25, Y_VOL=1.10) por separado sobre cada uno.
Objetivo: ver si el efecto SOW es estable o concentrado en un periodo.

Read-only. NO modifica config ni datos.

Uso:
    py scripts/walk_forward_sow_anual.py
    py scripts/walk_forward_sow_anual.py --sims 100   # smoke

Salida:
    outputs/audit/wyckoff_walk_forward_anual.csv
    outputs/audit/wyckoff_walk_forward_anual_summary.json
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


# Override: usar el parquet extendido para mas regimenes historicos.
_EXT = ROOT / "data" / "stock_prices_extended.parquet"
if _EXT.exists():
    calib.DATA_PARQUET = _EXT
    print(f"[override] DATA_PARQUET -> {_EXT.name}")

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

CANDIDATE = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_CSV = OUT_DIR / "wyckoff_walk_forward_anual_ext.csv"
OUT_JSON = OUT_DIR / "wyckoff_walk_forward_anual_ext_summary.json"


def _patch_blocks(blocks):
    calib.BLOCKS = blocks

def _summarize_agg(agg, block_label):
    """Extrae n_conf, n_base, lift_H20 del bloque pedido."""
    by_block = agg["by_block"]
    if block_label not in by_block:
        return None
    b = by_block[block_label]
    n_conf = b["n_conf"]
    n_base = b["n_base"]
    if n_conf == 0 or n_base == 0:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None}
    vc = b["vals_conf"]["struct_deterioration"]
    vb = b["vals_base"]["struct_deterioration"]
    if not vc or not vb:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None}
    return {
        "n_conf": n_conf, "n_base": n_base,
        "lift": float(np.mean(vc)) - float(np.mean(vb)),
    }


def _boot_from_agg(agg, block_label, sims, seed):
    """Bootstrap H20 struct_deterioration sobre las rows del bloque."""
    b = agg["by_block"].get(block_label)
    if b is None or not b["rows"]:
        return {"lift_point": None, "lower_ci": None, "upper_ci": None}
    return calib.bootstrap_lift_H20(b["rows"], B=sims, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    print("== Walk-forward ANUAL SOW ==")
    print(f"Candidata: N={CANDIDATE['N']} M={CANDIDATE['M']} "
          f"X_ATR={CANDIDATE['X_ATR']} Y_VOL={CANDIDATE['Y_VOL']}")
    print(f"Bloques: {[b[0] for b in BLOQUES_ANUALES]}")
    print(f"Bootstrap: B={args.sims} seed={args.seed}")
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
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Candidate starts totales: {total_starts}")
    print()

    N = CANDIDATE["N"]
    M = CANDIDATE["M"]
    X = CANDIDATE["X_ATR"]
    Y = CANDIDATE["Y_VOL"]

    # Cache SOW una sola vez (independiente del bloque)
    print("== Cache SOW ==")
    t0 = time.time()
    sow_cache = calib.compute_sow_cache(feats, N, X, Y)
    print(f"OK ({time.time()-t0:.1f}s)")
    print()

    rows = []
    for label, a, b in BLOQUES_ANUALES:
        print(f"== Bloque {label} ({a.date()} -> {b.date()}) ==")
        _patch_blocks(((label, a, b),))

        episodes = calib.build_episodes_landmark(
            feats, sow_cache, starts_cache, N, M,
        )
        print(f"  episodios: {len(episodes)}")

        agg = calib.aggregate_episodes(episodes, feats, N, 20)
        s = _summarize_agg(agg, label)
        boot = _boot_from_agg(agg, label, args.sims, args.seed)

        if s is None:
            print("  sin datos")
            rows.append({
                "period": label, "n_conf": 0, "n_base": 0,
                "lift_point": None, "lower_ci": None, "upper_ci": None,
            })
            continue

        print(f"  n_conf={s['n_conf']} n_base={s['n_base']} "
              f"lift={s['lift']}")
        print(f"  bootstrap: lift={boot['lift_point']} "
              f"IC95=[{boot['lower_ci']}, {boot['upper_ci']}]")

        rows.append({
            "period": label,
            "n_conf": s["n_conf"],
            "n_base": s["n_base"],
            "lift_point": boot["lift_point"],
            "lower_ci": boot["lower_ci"],
            "upper_ci": boot["upper_ci"],
        })

    # Verdicto global simplificado
    concl = "MIXED"
    positivos = sum(1 for r in rows if r["lower_ci"] is not None and r["lower_ci"] > 0)
    negativos = sum(1 for r in rows if r["upper_ci"] is not None and r["upper_ci"] < 0)
    if positivos == len(rows):
        concl = "TODOS_POSITIVOS"
    elif negativos == len(rows):
        concl = "TODOS_NEGATIVOS"
    elif positivos + negativos == 0:
        concl = "NINGUNO_SIGNIFICATIVO"
    print()
    print(f"Veredicto global: {concl}")
    print(f"  bloques con IC95 totalmente positivo: {positivos}/{len(rows)}")
    print(f"  bloques con IC95 totalmente negativo: {negativos}/{len(rows)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "candidate": CANDIDATE,
            "blocks": [
                {"label": l, "start": str(a.date()), "end": str(b.date())}
                for l, a, b in BLOQUES_ANUALES
            ],
            "bootstrap": {"B": args.sims, "seed": args.seed},
            "results": rows,
            "verdict_global": concl,
        }, f, indent=2, ensure_ascii=False, default=str)
    print()
    print(f"Escrito: {OUT_CSV}")
    print(f"Escrito: {OUT_JSON}")


if __name__ == "__main__":
    main()