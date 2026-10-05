"""Bootstrap panel + maxT - v2 CORREGIDO (multiplicidad de episodios).

Corrige bug v2: los paneles se construyen concatenando ventanas
solapadas; la misma fecha puede aparecer en multiples bloques. v1
(no optimizada) iteraba posiciones del panel y contaba cada aparicion
por separado. v2 sin corregir mapeaba cada fecha a una sola posicion.

Fix: multiplicidad = numero de bloques donde t0 y end son ambos
activos. El lift pondera cada episodio por su multiplicidad.
Equivalente bit-a-bit a v1.
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
    compute_sow_cache, build_episodes_landmark, episode_metrics,
    GRID_N, GRID_M, GRID_X, GRID_Y, MDE_LIFT,
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
                    sow_all[key][tk] = arr
    return sow_all


def precompute_episodes_by_combo(feats, sow_all, session_dates):
    starts_cache = {tk: find_candidate_starts(f["candidate"])
                    for tk, f in feats.items()}
    pos_of_date = {d: i for i, d in enumerate(session_dates)}
    precomp = []
    for j, (N, M, X, Y) in enumerate(COMBOS):
        eps = build_episodes_landmark(
            feats, sow_all[(N, X, Y)], starts_cache, N, M,
            min_gap_sessions=MIN_GAP,
        )
        rows = []
        for e in eps:
            tk = e["ticker"]
            feat = feats[tk]
            L_idx = e["L_idx"]
            Lh_idx = L_idx + H
            if Lh_idx >= len(feat["struct"]):
                continue
            m = episode_metrics(feat, L_idx, N, H)
            if m is None:
                continue
            det = m.get("struct_deterioration")
            if det is None:
                continue
            t0_orig = pos_of_date[e["t0_date"]]
            rows.append((t0_orig, Lh_idx, int(e["is_confirmed"]), int(det)))
        arr = np.array(rows, dtype=np.int64) if rows else np.zeros((0, 4), dtype=np.int64)
        precomp.append(arr)
    return precomp


def sample_panel(session_dates, block_len, burn_in, rng):
    T = len(session_dates)
    n_blocks = int(np.ceil(T / block_len))
    parts, active, bids = [], [], []
    for bid in range(n_blocks):
        s = int(rng.integers(burn_in, T - block_len))
        parts.append(session_dates[s - burn_in : s + block_len])
        active.append(np.array([False] * burn_in + [True] * block_len))
        bids.append(np.full(burn_in + block_len, bid))
    panel_dates = pd.DatetimeIndex(np.concatenate(parts))
    panel_active = np.concatenate(active)
    panel_bid = np.concatenate(bids)
    return panel_dates, panel_active, panel_bid, n_blocks


def build_active_bits(panel_dates, panel_active, panel_bid, pos_of_date, T_orig, n_blocks):
    bits = np.zeros((T_orig, n_blocks), dtype=bool)
    for pi in range(len(panel_dates)):
        if not panel_active[pi]:
            continue
        oi = pos_of_date.get(panel_dates[pi], -1)
        if oi < 0:
            continue
        bits[oi, panel_bid[pi]] = True
    return bits


def filter_lift_for_combo(arr, active_bits, H):
    if arr.shape[0] == 0:
        return np.nan
    t0_orig = arr[:, 0]
    end_orig = arr[:, 1] + H
    is_conf = arr[:, 2].astype(bool)
    det = arr[:, 3].astype(float)

    valid = end_orig < active_bits.shape[0]
    if not valid.any():
        return np.nan
    t0_v = t0_orig[valid]
    end_v = end_orig[valid]
    is_conf_v = is_conf[valid]
    det_v = det[valid]

    mult = (active_bits[t0_v] & active_bits[end_v]).sum(axis=1).astype(float)
    keep = mult > 0
    if not keep.any():
        return np.nan

    is_conf_k = is_conf_v[keep]
    det_k = det_v[keep]
    mult_k = mult[keep]

    conf_w = mult_k[is_conf_k]
    base_w = mult_k[~is_conf_k]
    if conf_w.sum() == 0 or base_w.sum() == 0:
        return np.nan
    conf_det = det_k[is_conf_k]
    base_det = det_k[~is_conf_k]
    p_c = (conf_w * conf_det).sum() / conf_w.sum()
    p_b = (base_w * base_det).sum() / base_w.sum()
    return float(p_c - p_b)


def _worker(seed):
    rng = np.random.default_rng(seed)
    panel_dates, panel_active, panel_bid, n_blocks = sample_panel(
        _G["session_dates"], BLOCK_LEN, BURN_IN, rng)
    active_bits = build_active_bits(
        panel_dates, panel_active, panel_bid, _G["pos_of_date"],
        _G["T_orig"], n_blocks)
    lifts = np.full(len(COMBOS), np.nan)
    for j in range(len(COMBOS)):
        lifts[j] = filter_lift_for_combo(_G["episodes_precomp"][j], active_bits, H)
    return lifts


def _init_worker(session_dates, pos_of_date, T_orig, episodes_precomp):
    _G["session_dates"] = session_dates
    _G["pos_of_date"] = pos_of_date
    _G["T_orig"] = T_orig
    _G["episodes_precomp"] = episodes_precomp


def compute_obs_lifts(episodes_precomp):
    obs = np.full(len(COMBOS), np.nan)
    for j, arr in enumerate(episodes_precomp):
        if arr.shape[0] == 0:
            continue
        is_conf = arr[:, 2].astype(bool)
        det = arr[:, 3].astype(float)
        n_c = int(is_conf.sum())
        n_b = int((~is_conf).sum())
        if n_c == 0 or n_b == 0:
            continue
        obs[j] = float(det[is_conf].mean() - det[~is_conf].mean())
    return obs


def run_phase(B, workers, obs_lifts, session_dates, pos_of_date, T_orig, episodes_precomp):
    print(f"[PHASE B={B}] workers={workers}", flush=True)
    t0 = time.time()
    seeds = [SEED + b for b in range(B)]
    results = []
    with Pool(processes=workers,
              initializer=_init_worker,
              initargs=(session_dates, pos_of_date, T_orig, episodes_precomp)) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, seeds, chunksize=1)):
            results.append(r)
            if (i + 1) % 100 == 0:
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

    print("[1/4] precomputo original", flush=True)
    df, tickers, feats, session_dates = precompute_original()
    pos_of_date = {d: i for i, d in enumerate(session_dates)}
    T_orig = len(session_dates)
    print(f"      tickers={len(tickers)} sessions={T_orig}", flush=True)

    print("[2/4] precomputo SOW", flush=True)
    t0 = time.time()
    sow_all = precompute_sow_all(feats)
    print(f"      {time.time()-t0:.1f}s", flush=True)

    print("[3/4] precomputo episodios (240 combos)", flush=True)
    t0 = time.time()
    episodes_precomp = precompute_episodes_by_combo(feats, sow_all, session_dates)
    print(f"      {time.time()-t0:.1f}s", flush=True)

    print("[4/4] lift observado + bootstrap", flush=True)
    obs_lifts = compute_obs_lifts(episodes_precomp)

    lift_matrix, SE_j, T_obs, maxT_b, p_maxT = run_phase(
        args.phase, args.workers, obs_lifts, session_dates, pos_of_date,
        T_orig, episodes_precomp)

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
    out = OUT_DIR / f"bootstrap_v2fix_B{args.phase}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] {out}", flush=True)


if __name__ == "__main__":
    main()
