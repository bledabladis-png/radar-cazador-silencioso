#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Purged K-Fold temporal de SOW v1.9 (C0).

v2: lift WITHIN-TICKER (elimina efecto tickers volatiles).
9 folds, TEST = anio natural 2017..2025. TRAIN = todo lo anterior
con embargo 50. Purga H=20 al final del TEST.

Criterio PRE-REGISTRADO:
  FUNCIONA:     mean(lift) > 0.05 AND CI95_lower(mean) > 0 AND folds_pos >= 7
  NO_FUNCIONA:  mean(lift) <= 0 OR CI95_upper(mean) < 0.05
  INCONCLUSO:   resto
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
calib.DATA_PARQUET = ROOT / "data" / "stock_prices_extended.parquet"
calib.BLOCKS = ()

CANDIDATE = {"N": 40, "M": 10, "X_ATR": 1.00, "Y_VOL": 1.10}  # C1-M10
H = 20
EMBARGO = 50
FOLDS = list(range(2017, 2026))
OUT_DIR = ROOT / "outputs" / "audit" / "sow_purged_kfold_c1_m10_20261024"


def classify(mean_lift, ci_lower, ci_upper, folds_positive):
    if mean_lift is not None and mean_lift > 0.05 and ci_lower is not None and ci_lower > 0 and folds_positive >= 7:
        return "FUNCIONA"
    if mean_lift is not None and mean_lift <= 0:
        return "NO_FUNCIONA"
    if ci_upper is not None and ci_upper < 0.05:
        return "NO_FUNCIONA"
    return "INCONCLUSO"


def fold_bounds(dates, year):
    ys = pd.Timestamp(str(year) + "-01-01")
    ye = pd.Timestamp(str(year) + "-12-31")
    ms = dates >= ys
    me = dates <= ye
    if not ms.any() or not me.any():
        return None
    return int(np.argmax(ms)), int(np.where(me)[0][-1])


def in_test(e, ts, te):
    L = e["L_idx"]
    return ts <= L and L + H <= te


def in_train(e, ts):
    return e["L_idx"] + H + EMBARGO <= ts


def evaluate_fold(episodes, feats, ts, te, N, boot_b, seed):
    test_eps = [e for e in episodes if in_test(e, ts, te)]
    train_eps = [e for e in episodes if in_train(e, ts)]
    rows = []
    for e in test_eps:
        m = calib.episode_metrics(feats[e["ticker"]], e["L_idx"], N, H)
        if m is None or m["struct_deterioration"] is None:
            continue
        rows.append({"ticker": e["ticker"], "is_confirmed": bool(e["is_confirmed"]), "y": int(m["struct_deterioration"])})
    if not rows:
        return {"n_conf": 0, "n_base": 0, "lift": None, "ci_lower": None, "ci_upper": None, "placebo_a": None, "placebo_b": None, "n_test": 0, "n_train": len(train_eps)}
    df = pd.DataFrame(rows)
    n_conf = int(df["is_confirmed"].sum())
    n_base = int(len(df) - n_conf)
    if n_conf == 0 or n_base == 0:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None, "ci_lower": None, "ci_upper": None, "placebo_a": None, "placebo_b": None, "n_test": len(df), "n_train": len(train_eps)}

    def lift(d):
        # v2 within-ticker: elimina el sesgo de tickers volatiles.
        # El placebo B permuta SOW dentro de ticker. Si el lift within-ticker
        # sigue > 0 y el placebo B ~ 0, la senal es temporal, no de ticker.
        tk_u = d["ticker"].unique()
        ws = 0.0
        tw = 0.0
        for _t in tk_u:
            sub = d[d["ticker"] == _t]
            c = sub[sub["is_confirmed"]]
            b = sub[~sub["is_confirmed"]]
            if len(c) == 0 or len(b) == 0:
                continue
            lk = float(c["y"].mean() - b["y"].mean())
            ws += len(c) * lk
            tw += len(c)
        if tw == 0:
            return None
        return float(ws / tw)

    point = lift(df)
    if point is None:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None, "ci_lower": None, "ci_upper": None, "placebo_a": None, "placebo_b": None, "n_test": len(df), "n_train": len(train_eps)}
    rng = np.random.default_rng(seed)
    tks = df["ticker"].unique()
    boots = []
    for _ in range(boot_b):
        samp = rng.choice(tks, size=len(tks), replace=True)
        bdf = pd.concat([df[df["ticker"] == t] for t in samp], ignore_index=True)
        lv = lift(bdf)
        if lv is not None:
            boots.append(lv)
    if boots:
        ci_l, ci_u = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    else:
        ci_l = ci_u = None

    pa = []
    for _ in range(500):
        p = df.copy()
        p["is_confirmed"] = rng.permutation(p["is_confirmed"].values)
        if p["is_confirmed"].any() and (~p["is_confirmed"]).any():
            pa.append(lift(p))
    pb = []
    for _ in range(500):
        p = df.copy()
        for tk in p["ticker"].unique():
            mask = p["ticker"] == tk
            p.loc[mask, "is_confirmed"] = rng.permutation(p.loc[mask, "is_confirmed"].values)
        if p["is_confirmed"].any() and (~p["is_confirmed"]).any():
            pb.append(lift(p))

    return {"n_conf": n_conf, "n_base": n_base, "lift": point, "ci_lower": ci_l, "ci_upper": ci_u, "placebo_a": float(np.mean(pa)) if pa else None, "placebo_b": float(np.mean(pb)) if pb else None, "n_test": len(df), "n_train": len(train_eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    print("== Purged k-fold SOW v1.9 ==")
    print("Candidata:", CANDIDATE)
    print("Folds:", FOLDS, "H:", H, "EMBARGO:", EMBARGO, "B:", args.boot_b)

    print("\\n== Cargando dataset extendido ==")
    t0 = time.time()
    df, tickers = calib.load_dataset()
    print("Tickers:", len(tickers), "t=", round(time.time()-t0, 1), "s")
    dates = pd.to_datetime(df.index).sort_values()
    print("Rango:", dates.min().date(), "->", dates.max().date(), "(", len(dates), "sesiones)")

    print("\\n== Precomputando features ==")
    t0 = time.time()
    feats = calib.precompute_features(df, tickers)
    print("Features:", len(feats), "tickers (", round(time.time()-t0, 1), "s)")

    print("\\n== Starts + SOW cache ==")
    t0 = time.time()
    starts_cache = {tk: calib.find_candidate_starts(f["candidate"]) for tk, f in feats.items()}
    print("Starts:", sum(len(v) for v in starts_cache.values()), "(", round(time.time()-t0, 1), "s)")

    t0 = time.time()
    sow_cache = calib.compute_sow_cache(feats, CANDIDATE["N"], CANDIDATE["X_ATR"], CANDIDATE["Y_VOL"])
    print("SOW cache (", round(time.time()-t0, 1), "s)")

    print("\\n== Episodios ==")
    t0 = time.time()
    episodes = calib.build_episodes_landmark(feats, sow_cache, starts_cache, CANDIDATE["N"], CANDIDATE["M"])
    print("Episodios:", len(episodes), "(", round(time.time()-t0, 1), "s)")

    print("\\n== Evaluando folds ==")
    fold_results = []
    for year in FOLDS:
        b = fold_bounds(dates, year)
        if b is None:
            continue
        ts, te = b
        t0 = time.time()
        res = evaluate_fold(episodes, feats, ts, te, CANDIDATE["N"], args.boot_b, args.seed + year)
        res["fold"] = year
        res["test_start"] = str(dates[ts].date())
        res["test_end"] = str(dates[te].date())
        fold_results.append(res)
        print("  ", year, ": n_test=", res["n_test"], " n_conf=", res["n_conf"], " n_base=", res["n_base"], " lift=", res["lift"], " IC=[", res["ci_lower"], ",", res["ci_upper"], "] (", round(time.time()-t0), "s)")

    valid = [r for r in fold_results if r["lift"] is not None]
    if not valid:
        print("ERROR: ningun fold evaluable")
        return 1

    lifts = np.array([r["lift"] for r in valid])
    mean_lift = float(lifts.mean())
    folds_pos = int((lifts > 0).sum())
    rng = np.random.default_rng(args.seed + 999)
    boot_m = [float(rng.choice(lifts, size=len(lifts), replace=True).mean()) for _ in range(5000)]
    ci_l = float(np.percentile(boot_m, 2.5))
    ci_u = float(np.percentile(boot_m, 97.5))
    verdict = classify(mean_lift, ci_l, ci_u, folds_pos)

    summary = {"candidate": CANDIDATE, "folds": FOLDS, "embargo": EMBARGO, "H": H, "bootstrap_B": args.boot_b, "seed": args.seed, "n_folds_evaluable": len(valid), "mean_lift": mean_lift, "mean_lift_ci_lower": ci_l, "mean_lift_ci_upper": ci_u, "folds_positive": folds_pos, "fold_results": fold_results, "verdict": verdict}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(OUT_DIR / "folds.csv", index=False)
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    dec = "SOW v1.9 Purged K-Fold - Decision\\nFecha: 2026-10-23\\n\\nFolds evaluables: " + str(len(valid)) + "/9\\nMean lift: " + str(round(mean_lift, 4)) + "\\nCI95 (media entre folds): [" + str(round(ci_l, 4)) + ", " + str(round(ci_u, 4)) + "]\\nFolds con lift > 0: " + str(folds_pos) + "/" + str(len(valid)) + "\\n\\nVEREDICTO: " + verdict + "\\n\\nCriterio:\\n  FUNCIONA: mean_lift > 0.05 AND ci_lower > 0 AND folds_positive >= 7\\n  NO FUNCIONA: mean_lift <= 0 OR ci_upper < 0.05\\n  INCONCLUSO: resto\\n"
    (OUT_DIR / "decision.txt").write_text(dec, encoding="utf-8")

    print("\\n== RESULTADO ==")
    print("Folds evaluables:", len(valid), "/9")
    print("Mean lift:", round(mean_lift, 4), " IC95: [", round(ci_l, 4), ",", round(ci_u, 4), "]")
    print("Folds positivos:", folds_pos, "/", len(valid))
    print("VEREDICTO:", verdict)
    print("Escrito:", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
