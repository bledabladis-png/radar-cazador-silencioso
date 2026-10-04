"""Evaluacion exterior (seccion 10 del protocolo v4.1).

Para cada outer fold, se evalua unicamente la candidata seleccionada
en el INNER del mismo fold. Modelo ajustado una vez sobre OUTER_TRAIN,
predicciones sobre OUTER_TEST, bootstrap MBB B20/B40/B50 con modelo
congelado.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.sow_v4 import bootstrap, config, regime
from scripts.sow_v4.episodes import build_episodes_for_combo, purge_train_frontier
from scripts.sow_v4.model_primary import fit_primary, predict_rd


def evaluate_outer(
    combo: tuple,
    feats: dict,
    starts_cache: dict,
    sector_returns: pd.DataFrame,
    outer_train_dates: list,
    outer_test_dates: list,
    fold_idx: int,
    skip_bootstrap: bool = False,
) -> dict:
    """Evalua el combo sobre el OUTER_TEST del fold."""
    N, M, X, Y = combo

    log_vol = regime.compute_log_vol_lag1(regime.compute_vol_median(sector_returns))
    try:
        thr = regime.fit_thresholds(
            log_vol, outer_train_dates[0], outer_train_dates[-1]
        )
    except ValueError as exc:
        return {"ok": False, "reason": f"regimen_fit:{exc}"}
    labels = regime.classify_regime(log_vol, thr)

    train_start = pd.Timestamp(outer_train_dates[0])
    train_end = pd.Timestamp(outer_train_dates[-1])
    test_start = pd.Timestamp(outer_test_dates[0])
    test_end = pd.Timestamp(outer_test_dates[-1])

    all_eps = build_episodes_for_combo(feats, starts_cache, N, M, X, Y, labels)
    train_eps = [e for e in all_eps if train_start <= e["t0_date"] <= train_end]
    test_eps = [e for e in all_eps if test_start <= e["t0_date"] <= test_end]

    last_train_idx = _last_idx_for_date(feats, train_end)
    train_eps = purge_train_frontier(train_eps, M, last_train_idx)

    fit = fit_primary(train_eps)
    if not fit.ok:
        return {"ok": False, "reason": f"fit:{fit.reason}"}

    rd_point = predict_rd(fit, test_eps)
    if not rd_point["ok"]:
        return {"ok": False, "reason": f"predict:{rd_point['reason']}"}

    session_dates = _session_dates_for(feats)
    results = {"rd_point": rd_point, "bootstraps": {}, "fold_idx": fold_idx}

    if skip_bootstrap:
        return {"ok": True, **results}

    for B_name, B in (
        ("B20", config.BOOT_B_PRIMARY),
        ("B40", config.BOOT_B_SENSITIVITY),
        ("B50", config.BOOT_B_DIAGNOSTIC),
    ):
        bs = bootstrap.bootstrap_rd(
            fit, test_eps, session_dates, B,
            config.BOOT_N_REPLICAS,
            config.BOOT_SEED + fold_idx * 1000 + B,
        )
        if bs is None:
            results["bootstraps"][B_name] = None
            continue
        ci95_s = bootstrap.ci95(bs["rd_stress"])
        ci95_n = bootstrap.ci95(bs["rd_normal"])
        ci95_p = bootstrap.ci95(bs["rd_pool"])
        ci90_s = bootstrap.ci90(bs["rd_stress"])
        ci90_n = bootstrap.ci90(bs["rd_normal"])
        results["bootstraps"][B_name] = {
            "B": B,
            "ci95_stress": ci95_s,
            "ci95_normal": ci95_n,
            "ci95_pool": ci95_p,
            "ci90_stress": ci90_s,
            "ci90_normal": ci90_n,
        }

    return {"ok": True, **results}


def _last_idx_for_date(feats: dict, target_date) -> int:
    if not feats:
        return -1
    first_key = next(iter(feats))
    session_dates = feats[first_key]["dates"]
    mask = session_dates <= target_date
    if not mask.any():
        return -1
    return int(np.where(mask)[0].max())


def _session_dates_for(feats: dict) -> list:
    if not feats:
        return []
    first_key = next(iter(feats))
    return list(feats[first_key]["dates"])