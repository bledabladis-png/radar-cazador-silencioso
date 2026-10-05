"""Bootstrap panel + maxT - Protocolo v3 FIRMADO 2026-10-05.

Uso:
    py scripts/sow_v19_bootstrap.py --phase 100 --workers 12
    py scripts/sow_v19_bootstrap.py --phase 500 --workers 12
    py scripts/sow_v19_bootstrap.py --phase 2000 --workers 12
    py scripts/sow_v19_bootstrap.py --phase 20000 --workers 12

Implementa:
- panel_calendar_block bootstrap: bloques de 120 sesiones + burn_in 200.
- una recomputacion por panel via reindex de features y SOW precomputados.
- equivalence: en region activa, burn_in 200 > max rolling window (200).
- fixed-SE studentized maxT: T_j = (lift_j - 0.05) / SE_j.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_wyckoff_sow_5b4bis import (
    load_dataset, precompute_features, find_candidate_starts,
    compute_sow_cache, build_episodes_landmark, build_analysis_rows,
    GRID_N, GRID_M, GRID_X, GRID_Y, MDE_LIFT,
    STRUCT_DETERIORO, T_NORM_STRONG, PREC_STRUCT_STRONG,
)

BLOCK_LEN = 120
BURN_IN = 200
H = 20
SEED = 20261005
MIN_GAP = 50
OUT_DIR = ROOT / "outputs" / "audit" / "sow_v19"
COMBOS = [(N, M, X, Y) for N in GRID_N for X in GRID_X for Y in GRID_Y for M in GRID_M]

_G = {}


def precompute_original():
    df, tickers = load_dataset()
    feats = precompute_features(df, tickers)
    s = set()
    for f in feats.values():
        s.update(f["dates"])
    session_dates = pd.DatetimeIndex(sorted(s))
    return df, tickers, feats, session_dates


def precompute_sow_all(feats):
    sow_all = {}
    for N in GRID_N:
        for X in GRID_X:
            for Y in GRID_Y:
                cache = compute_sow_cache(feats, N, X, Y)
                key = (N, X, Y)
                sow_all[key] = {}
                for tk, arr in cache.items():
                    if arr is None:
                        continue
                    sow_all[key][tk] = pd.Series(arr, index=feats[tk]["dates"])
    return sow_all


def sample_panel_indices(session_dates, block_len, burn_in, rng):
    T = len(session_dates)
    n_blocks = int(np.ceil(T / block_len))
    parts, active, bids = [], [], []
    for bid in range(n_blocks):
        s = int(rng.integers(burn_in, T - block_len))
        parts.append(session_dates[s - burn_in : s + block_len])
        active.append(np.array([False] * burn_in + [True] * block_len))
        bids.append(np.full(burn_in + block_len, bid))
    return (pd.DatetimeIndex(np.concatenate(parts)),
            np.concatenate(active),
            np.concatenate(bids))


def build_panel(feats_orig, sow_all, panel_idx):
    feats_panel = {}
    for tk, f in feats_orig.items():
        struct_p = f["struct"].reindex(panel_idx)
        t_norm_p = f["t_norm"].reindex(panel_idx)
        struct_max = struct_p.rolling(60, min_periods=60).max().shift(1)
        candidate_p = (
            (struct_p < STRUCT_DETERIORO)
            & (t_norm_p > -T_NORM_STRONG)
            & (struct_max > PREC_STRUCT_STRONG)
        ).fillna(False).astype(bool)
        feats_panel[tk] = {
            "tdf": f["tdf"].reindex(panel_idx),
            "dates": panel_idx,
            "struct": struct_p,
            "t_norm": t_norm_p,
            "close": f["close"].reindex(panel_idx),
            "low": f["low"].reindex(panel_idx),
            "high": f["high"].reindex(panel_idx),
            "volume": f["volume"].reindex(panel_idx),
            "candidate": candidate_p,
        }
    sow_panel = {}
    for key, cache in sow_all.items():
        sow_panel[key] = {}
        for tk, ser in cache.items():
            sow_panel[key][tk] = ser.reindex(panel_idx).fillna(0).astype(int).values
    return feats_panel, sow_panel


def compute_lifts(feats_panel, sow_panel, active_mask, block_ids, M_max):
    starts_cache = {tk: find_candidate_starts(f["candidate"])
                    for tk, f in feats_panel.items()}
    lifts = np.full(len(COMBOS), np.nan)
    n_pos = len(active_mask)
    for j, (N, M, X, Y) in enumerate(COMBOS):
        sow_key = (N, X, Y)
        if sow_key not in sow_panel:
            continue
        eps = build_episodes_landmark(
            feats_panel, sow_panel[sow_key], starts_cache, N, M,
            min_gap_sessions=MIN_GAP,
        )
        # Filtro activo: t0 en activo Y L+H en mismo bloque
        eps_f = []
        for e in eps:
            ti = e["t0_idx"]
            end = ti + M + H
            if end >= n_pos:
                continue
            if not active_mask[ti]:
                continue
            if block_ids[ti] != block_ids[end]:
                continue
            eps_f.append(e)
        analysis = build_analysis_rows(eps_f, feats_panel, N, h=H)
        rows = analysis["rows"]
        if not rows:
            continue
        conf = [m["struct_deterioration"] for _, c, m in rows if c
                and m.get("struct_deterioration") is not None]
        base = [m["struct_deterioration"] for _, c, m in rows if not c
                and m.get("struct_deterioration") is not None]
        if not conf or not base:
            continue
        lifts[j] = float(np.mean(conf) - np.mean(base))
    return lifts


def _worker(seed):
    rng = np.random.default_rng(seed)
    panel_idx, active, bids = sample_panel_indices(
        _G["session_dates"], BLOCK_LEN, BURN_IN, rng)
    feats_p, sow_p = build_panel(_G["feats"], _G["sow_all"], panel_idx)
    return compute_lifts(feats_p, sow_p, active, bids, 30)


def _init_worker(feats, sow_all, session_dates):
    _G["feats"] = feats
    _G["sow_all"] = sow_all
    _G["session_dates"] = session_dates


def compute_obs_lifts(feats, sow_all):
    starts_cache = {tk: find_candidate_starts(f["candidate"])
                    for tk, f in feats.items()}
    obs = np.full(len(COMBOS), np.nan)
    for j, (N, M, X, Y) in enumerate(COMBOS):
        key = (N, X, Y)
        eps = build_episodes_landmark(feats, sow_all[key], starts_cache, N, M,
                                      min_gap_sessions=MIN_GAP)
        analysis = build_analysis_rows(eps, feats, N, h=H)
        rows = analysis["rows"]
        conf = [m["struct_deterioration"] for _, c, m in rows if c
                and m.get("struct_deterioration") is not None]
        base = [m["struct_deterioration"] for _, c, m in rows if not c
                and m.get("struct_deterioration") is not None]
        if conf and base:
            obs[j] = float(np.mean(conf) - np.mean(base))
    return obs


def run_phase(B, workers, obs_lifts):
    print(f"[PHASE B={B}] workers={workers}", flush=True)
    t0 = time.time()
    seeds = [SEED + b for b in range(B)]
    results = []
    with Pool(processes=workers,
              initializer=_init_worker,
              initargs=(_G["feats"], _G["sow_all"], _G["session_dates"])) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, seeds, chunksize=1)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{B}] {time.time()-t0:.1f}s", flush=True)
    lift_matrix = np.vstack(results)
    print(f"[PHASE B={B}] done in {time.time()-t0:.1f}s", flush=True)

    SE_j = np.nanstd(lift_matrix, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        T_obs = (obs_lifts - MDE_LIFT) / SE_j
        T_star = (lift_matrix - obs_lifts) / SE_j
    maxT_b = np.nanmax(T_star, axis=1)
    p_maxT = np.array([
        (1.0 + np.sum(maxT_b >= T_obs[j])) / (1.0 + B)
        for j in range(len(COMBOS))
    ])
    return lift_matrix, SE_j, T_obs, maxT_b, p_maxT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True,
                    choices=[100, 500, 2000, 20000])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] precomputo original", flush=True)
    df, tickers, feats, session_dates = precompute_original()
    print(f"      tickers={len(tickers)} sessions={len(session_dates)}", flush=True)
    _G["feats"] = feats
    _G["session_dates"] = session_dates

    print("[2/3] precomputo SOW (15 combos NXY)", flush=True)
    t0 = time.time()
    sow_all = precompute_sow_all(feats)
    _G["sow_all"] = sow_all
    print(f"      {time.time()-t0:.1f}s", flush=True)

    print("[3/3] lift observado por combo", flush=True)
    obs_lifts = compute_obs_lifts(feats, sow_all)

    lift_matrix, SE_j, T_obs, maxT_b, p_maxT = run_phase(args.phase, args.workers, obs_lifts)

    summary = {
        "protocol": "SOW_v19_PROTOCOL.json",
        "protocol_sha256": "c6dae718a258ac47e4c75326ec5278397a97f116a56f5070a22f8c8778f3a1f2",
        "phase_B": args.phase,
        "workers": args.workers,
        "seed": SEED,
        "block_len": BLOCK_LEN,
        "burn_in": BURN_IN,
        "H": H,
        "min_gap_sessions": MIN_GAP,
        "combos": [{"N": int(N), "M": int(M), "X": float(X), "Y": float(Y)}
                   for N, M, X, Y in COMBOS],
        "obs_lifts": obs_lifts.tolist(),
        "SE_j": SE_j.tolist(),
        "T_obs": T_obs.tolist(),
        "p_maxT": p_maxT.tolist(),
    }
    out = OUT_DIR / f"bootstrap_B{args.phase}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] {out}", flush=True)


if __name__ == "__main__":
    main()
