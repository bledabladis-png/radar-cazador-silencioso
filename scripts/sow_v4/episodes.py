"""Construccion de episodios y covariables X_0 (seccion 4 del protocolo v4.1).

Reutiliza piezas de scripts.calibrate_wyckoff_sow_5b4bis:
- precompute_features
- find_candidate_starts
- build_episodes_landmark
- episode_metrics

Anade:
- covariables X_0 en t0: D1_prev, D2_prev, D3_prev (seccion 4.3)
- filtro por R_episode in {stress, normal} (secciones 3.6, 4.4)
- filtro de censura completa (seccion 4.4)
"""
from __future__ import annotations

import pandas as pd

import scripts.calibrate_wyckoff_sow_5b4bis as calib
from scripts.sow_v4 import config


def precompute_base(df: pd.DataFrame, tickers: list[str]):
    """feats y starts_cache: independientes de (N, M, X, Y)."""
    feats = calib.precompute_features(df, tickers)
    starts_cache = {
        tk: calib.find_candidate_starts(f["candidate"]) for tk, f in feats.items()
    }
    return feats, starts_cache


def build_episodes_for_combo(
    feats: dict,
    starts_cache: dict,
    N: int,
    M: int,
    X_ATR: float,
    Y_VOL: float,
    regime_labels,
    sow_cache_map: dict | None = None,
    landmark_cache: dict | None = None,
    metrics_cache: dict | None = None,
) -> list[dict]:
    """Episodios filtrados para un combo (N, M, X_ATR, Y_VOL).

    Salida: lista de dicts con claves:
        ticker, t0_idx, t0_date, L_idx, L_date, sow_idx, is_confirmed,
        D1_prev, D2_prev, D3_prev, Y, R_episode
    """
    # 1. SOW cache (funcion pura de N, X_ATR, Y_VOL)
    key_sow = (N, X_ATR, Y_VOL)
    if sow_cache_map is not None and key_sow in sow_cache_map:
        sow_cache = sow_cache_map[key_sow]
    else:
        sow_cache = calib.compute_sow_cache(feats, N, X_ATR, Y_VOL)
        if sow_cache_map is not None:
            sow_cache_map[key_sow] = sow_cache

    # 2. Landmark cache (funcion pura de N, M, X_ATR, Y_VOL)
    #    (depende del sow_cache, que depende de N, X_ATR, Y_VOL)
    key_lm = (N, M, X_ATR, Y_VOL)
    if landmark_cache is not None and key_lm in landmark_cache:
        episodes = landmark_cache[key_lm]
    else:
        episodes = calib.build_episodes_landmark(
            feats, sow_cache, starts_cache, N, M
        )
        if landmark_cache is not None:
            landmark_cache[key_lm] = episodes

    delta_w = config.BASELINE_DELTA_WINDOW
    out: list[dict] = []

    # Convertir labels a dict para O(1) lookup
    if isinstance(regime_labels, dict):
        labels_dict = regime_labels
    else:
        labels_dict = {d: v for d, v in regime_labels.items() if pd.notna(v)}

    for e in episodes:
        ticker = e["ticker"]
        feat = feats[ticker]
        t0_idx = int(e["t0_idx"])

        # Censura rapida
        if t0_idx < delta_w:
            continue

        # Regimen R(t0) ANTES de computar outcome
        r = labels_dict.get(e["t0_date"])
        if r not in ("stress", "normal"):
            continue

        struct = feat["struct"]
        t_norm = feat["t_norm"]

        d1_prev = struct.iloc[t0_idx]
        d2_prev = struct.iloc[t0_idx] - struct.iloc[t0_idx - delta_w]
        d3_prev = t_norm.iloc[t0_idx]

        if pd.isna(d1_prev) or pd.isna(d2_prev) or pd.isna(d3_prev):
            continue

        # Outcome H20 (con cache)
        L_idx = int(e["L_idx"])
        if metrics_cache is not None:
            key_m = (ticker, N, L_idx)
            if key_m in metrics_cache:
                m = metrics_cache[key_m]
            else:
                m = calib.episode_metrics(feat, L_idx, N, config.HORIZON)
                metrics_cache[key_m] = m
        else:
            m = calib.episode_metrics(feat, L_idx, N, config.HORIZON)
        if m is None:
            continue
        y = m.get("struct_deterioration")
        if y is None:
            continue

        out.append(
            {
                "ticker": e["ticker"],
                "t0_idx": t0_idx,
                "t0_date": e["t0_date"],
                "L_idx": int(e["L_idx"]),
                "L_date": e["L_date"],
                "sow_idx": e["sow_idx"],
                "is_confirmed": bool(e["is_confirmed"]),
                "D1_prev": float(d1_prev),
                "D2_prev": float(d2_prev),
                "D3_prev": float(d3_prev),
                "Y": int(y),
                "R_episode": r,
            }
        )

    return out


def purge_train_frontier(
    episodes: list[dict],
    M: int,
    last_train_idx: int,
) -> list[dict]:
    """Seccion 8.2: elimina episodios con t0_idx + M + 20 > last_train_idx."""
    limit = last_train_idx
    return [
        e for e in episodes if (e["t0_idx"] + M + config.HORIZON) <= limit
    ]