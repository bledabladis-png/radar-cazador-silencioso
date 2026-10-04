"""Seleccion INNER (seccion 9 del protocolo v4.1).

Estructura por outer fold:
- OUTER_TRAIN dividido en 4 segmentos de igual duracion en sesiones.
- 3 inner folds expanding: (S1->S2), (S1+S2->S3), (S1+S2+S3->S4).
- Umbrales de regimen recalculados dentro de cada INNER_TRAIN.
- Purga por M en cada INNER_TRAIN.
- Elegibilidad obligatoria en AMBOS regimenes (stress y normal) en
  TRAIN y VALID (seccion 6.3 / 9.1).
- Criterio de paso: RD_pool > 0.
- Regla 2 de 3 para candidatura.
- Score: median(RD_pool) - 0.5 * IQR(RD_pool), sin imputar ceros.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.sow_v4 import config, regime
from scripts.sow_v4.episodes import build_episodes_for_combo, purge_train_frontier
from scripts.sow_v4.model_primary import (
    fit_primary_pooled,
    predict_rd_pool,
)


def split_into_segments(session_dates: list, n_segments: int = 4) -> list[list]:
    """Divide la lista de fechas en n segmentos de igual duracion en sesiones.

    Devuelve lista de listas de fechas. El ultimo segmento absorbe el
    remanente si la division no es exacta.
    """
    T = len(session_dates)
    if T < n_segments:
        raise ValueError(f"T={T} < n_segments={n_segments}")
    base = T // n_segments
    out = []
    start = 0
    for i in range(n_segments - 1):
        out.append(session_dates[start : start + base])
        start += base
    out.append(session_dates[start:])
    return out


def inner_folds(dates: list, n_segments: int = 4) -> list[dict]:
    """3 inner expanding folds.

    Devuelve lista de dicts:
      {train_dates: [...], valid_dates: [...], j: 0|1|2}
    """
    segs = split_into_segments(dates, n_segments)
    out = []
    for j in range(n_segments - 1):
        train = [d for s in segs[: j + 1] for d in s]
        valid = segs[j + 1]
        out.append({"train_dates": train, "valid_dates": valid, "j": j})
    return out


def _count_regime_support(episodes: list[dict], regime_name: str) -> dict:
    sub = [e for e in episodes if e["R_episode"] == regime_name]
    tickers_conf = {e["ticker"] for e in sub if e["is_confirmed"]}
    tickers_base = {e["ticker"] for e in sub if not e["is_confirmed"]}
    n_conf = sum(1 for e in sub if e["is_confirmed"])
    n_base = sum(1 for e in sub if not e["is_confirmed"])
    return {
        "n_conf": n_conf,
        "n_base": n_base,
        "n_tickers_conf": len(tickers_conf),
        "n_tickers_base": len(tickers_base),
    }


def support_ok(stats: dict) -> bool:
    return (
        stats["n_conf"] >= config.SUPPORT_MIN_N_CONF
        and stats["n_base"] >= config.SUPPORT_MIN_N_BASE
        and stats["n_tickers_conf"] >= config.SUPPORT_MIN_TICKERS_CONF
        and stats["n_tickers_base"] >= config.SUPPORT_MIN_TICKERS_BASE
    )


def global_support_ok(episodes: list[dict]) -> bool:
    """Elegibilidad pooled (seccion 9.1 v4.1 revisada).

    No exige soporte por regimen. Solo:
      n_conf >= 20
      n_base >= 20
      >= 5 tickers con SOW=1
      >= 5 tickers con SOW=0
    """
    n_conf = sum(1 for e in episodes if e["is_confirmed"])
    n_base = sum(1 for e in episodes if not e["is_confirmed"])
    tk_conf = len({e["ticker"] for e in episodes if e["is_confirmed"]})
    tk_base = len({e["ticker"] for e in episodes if not e["is_confirmed"]})
    return (
        n_conf >= config.SUPPORT_MIN_N_CONF
        and n_base >= config.SUPPORT_MIN_N_BASE
        and tk_conf >= config.SUPPORT_MIN_TICKERS_CONF
        and tk_base >= config.SUPPORT_MIN_TICKERS_BASE
    )


def evaluate_combo_on_inner(
    combo: tuple,
    feats: dict,
    starts_cache: dict,
    sector_returns: pd.DataFrame,
    fold_idx: int,
    inner_folds_list: list[dict],
    inner_labels: list,
    sow_cache_map: dict | None = None,
    landmark_cache: dict | None = None,
    metrics_cache: dict | None = None,
) -> dict:
    """Evalua un combo (N, M, X, Y) en los 3 inner folds del outer fold.

    Devuelve dict con:
      n_elegibles, rds, score, ok
    """
    N, M, X, Y = combo
    rds: list[float | None] = []
    eligibles = 0

    for j, inner in enumerate(inner_folds_list):
        train_dates = inner["train_dates"]
        valid_dates = inner["valid_dates"]
        if not train_dates or not valid_dates:
            rds.append(None)
            continue

        labels = inner_labels[j]
        if labels is None:
            rds.append(None)
            continue

        train_start = pd.Timestamp(train_dates[0])
        train_end = pd.Timestamp(train_dates[-1])
        valid_start = pd.Timestamp(valid_dates[0])
        valid_end = pd.Timestamp(valid_dates[-1])

        train_all = build_episodes_for_combo(
            feats, starts_cache, N, M, X, Y, labels,
            sow_cache_map=sow_cache_map,
            landmark_cache=landmark_cache,
            metrics_cache=metrics_cache,
        )
        train_eps = [
            e for e in train_all
            if train_start <= e["t0_date"] <= train_end
        ]
        last_train_idx = _last_idx_in_dates(feats, train_end, train_dates)
        train_eps = purge_train_frontier(train_eps, M, last_train_idx)

        valid_eps = [
            e for e in train_all
            if valid_start <= e["t0_date"] <= valid_end
        ]
        last_valid_idx = _last_idx_in_dates(feats, valid_end, valid_dates)
        valid_eps = purge_train_frontier(valid_eps, M, last_valid_idx)

        # Elegibilidad pooled (v4.1 revisada): sin soporte por regimen
        if not (global_support_ok(train_eps) and global_support_ok(valid_eps)):
            rds.append(None)
            continue

        # Modelo pooled para seleccion INNER
        fit = fit_primary_pooled(train_eps)
        if not fit.ok:
            rds.append(None)
            continue

        rd = predict_rd_pool(fit, valid_eps)
        if not rd["ok"] or rd["RD_pool"] is None:
            rds.append(None)
            continue

        eligibles += 1
        rds.append(rd["RD_pool"])

    valid_rds = [r for r in rds if r is not None]
    passes = sum(1 for r in valid_rds if r > 0)

    if eligibles < config.INNER_MIN_FOLDS_ELEGIBLES:
        return {"ok": False, "reason": "pocos_elegibles", "n_elegibles": eligibles}
    if passes < config.INNER_MIN_FOLDS_PASAN:
        return {"ok": False, "reason": "pocos_pasan", "n_elegibles": eligibles, "n_pasan": passes}

    arr = np.array(valid_rds, dtype=float)
    med = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    score = med - config.INNER_LAMBDA_IQR * iqr

    return {
        "ok": True,
        "reason": "ok",
        "n_elegibles": eligibles,
        "n_pasan": passes,
        "median": med,
        "iqr": iqr,
        "score": score,
        "rds": valid_rds,
    }


def _last_idx_in_dates(feats: dict, target_date, dates_list=None) -> int:
    """Devuelve el indice maximo valido en la sesion del ultimo dia util
    del segmento. Usa la union de calendarios de todos los tickers."""
    if not feats:
        return -1
    sessions = config.session_union_dates(feats)
    idx = [i for i, d in enumerate(sessions) if d <= target_date]
    if not idx:
        return -1
    return idx[-1]


def select_candidate_for_outer(
    feats: dict,
    starts_cache: dict,
    sector_returns: pd.DataFrame,
    outer_fold_idx: int,
    outer_train_dates: list,
) -> dict:
    """Itera las 240 combinaciones y devuelve la mejor.

    Desempate (seccion 9.6):
      1. mayor mediana, 2. menor IQR, 3. menor N, 4. menor M,
      5. menor X, 6. menor Y.
    """
    inner_list = inner_folds(outer_train_dates, config.INNER_N_SEGMENTOS)

    # Precomputar regime_labels por inner fold (independiente del combo).
    log_vol = regime.compute_log_vol_lag1(
        regime.compute_vol_median(sector_returns)
    )
    inner_labels: list = []
    for inner in inner_list:
        tr = inner["train_dates"]
        if not tr:
            inner_labels.append(None)
            continue
        try:
            thr = regime.fit_thresholds(log_vol, tr[0], tr[-1])
            labels = regime.classify_regime(log_vol, thr)
        except ValueError:
            labels = None
        inner_labels.append(labels)

    # Caches compartidos para las 240 combinaciones del fold.
    # sow_cache_map: {(N, X, Y): sow_cache}   -> 48 unicas.
    # landmark_cache: {(N, M): episodes_landmark} -> 20 unicas.
    sow_cache_map: dict = {}
    landmark_cache: dict = {}
    metrics_cache: dict = {}

    results = []
    for N in config.GRID_N:
        for M in config.GRID_M:
            for X in config.GRID_X_ATR:
                for Y in config.GRID_Y_VOL:
                    res = evaluate_combo_on_inner(
                        (N, M, X, Y),
                        feats,
                        starts_cache,
                        sector_returns,
                        outer_fold_idx,
                        inner_list,
                        inner_labels,
                        sow_cache_map=sow_cache_map,
                        landmark_cache=landmark_cache,
                        metrics_cache=metrics_cache,
                    )
                    if res["ok"]:
                        results.append({
                            "combo": (N, M, X, Y),
                            "score": res["score"],
                            "median": res["median"],
                            "iqr": res["iqr"],
                            "n_elegibles": res["n_elegibles"],
                        })

    if not results:
        return {"ok": False, "reason": "sin_candidatas"}

    results.sort(key=lambda r: (
        -r["score"],
        -r["median"],
        r["iqr"],
        r["combo"][0],
        r["combo"][1],
        r["combo"][2],
        r["combo"][3],
    ))
    return {"ok": True, "best": results[0], "n_candidatas": len(results)}