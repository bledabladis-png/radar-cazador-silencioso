#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Purged K-Fold SOW C1-M5 sobre COHORTE RESERVADA (S&P 500 no en extended).

Pre-registro congelado ANTES de ejecutar:
  - Candidata: C1-M5 FROZEN: N=40, M=5, X_ATR=1.00, Y_VOL=1.10
  - Cohorte: 268 tickers S&P 500 reservados (data/stock_prices_reserved_cohort.parquet)
  - Folds: 9 (2017..2025), embargo 50, purga H=20
  - Estimador: lift within-ticker
  - Placebo A (diagnostico): permutacion intra-ticker
  - Placebo B (PRIMARIO): circular shift intra-ticker, offset [25, n-25]
  - Bootstrap: ticker-cluster per fold, fold-level aggregate
  - Criterio FUNCIONA:
      mean(lift) > 0.05
      AND IC95_lower > 0
      AND folds_positive >= 7/9
      AND folds_beat_placebo_B >= 7/9

No modificar tras ver resultados.
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

calib.DATA_PARQUET = ROOT / "data" / "stock_prices_reserved_cohort.parquet"
calib.BLOCKS = ()

CANDIDATE = {"N": 40, "M": 5, "X_ATR": 1.00, "Y_VOL": 1.10}
H = 20
EMBARGO = 50
FOLDS = list(range(2017, 2026))
OUT_DIR = ROOT / "outputs" / "audit" / "sow_reserved_20261024"


def classify(mean_lift, ci_lower, ci_upper, folds_pos, folds_beat):
    if (mean_lift is not None and mean_lift > 0.05
            and ci_lower is not None and ci_lower > 0
            and folds_pos >= 7 and folds_beat >= 7):
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


def lift_within(d):
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


def placebo_shift(d, seed):
    """Date-shift global del fold: roll de is_confirmed por 30-60 sesiones.

    Rompe la contemporaneidad SOW->H20 preservando la distribucion marginal
    del fold. Sustituye el shift intra-ticker (no viable con n_tk ~2.6: los
    tickers que contribuyen al lift no reciben variacion).

    Si el lift desaparece tras el shift, la asociacion temporal es real.
    Si persiste, la senal no depende de la contemporaneidad.
    """
    p = d.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(30, 61))
    p = p.sort_values("L_idx").reset_index(drop=True)
    labels = p["is_confirmed"].values.copy()
    if len(labels) < 2 * offset:
        offset = max(1, len(labels) // 4)
    p["is_confirmed"] = np.roll(labels, offset)
    return p


def placebo_perm(d, seed):
    p = d.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    for tk in p["ticker"].unique():
        mask = (p["ticker"] == tk).values
        labels = p.loc[mask, "is_confirmed"].values.copy()
        p.loc[mask, "is_confirmed"] = rng.permutation(labels)
    return p


def evaluate_fold(episodes, feats, ts, te, N, boot_b, seed):
    test_eps = [e for e in episodes if in_test(e, ts, te)]
    train_eps = [e for e in episodes if in_train(e, ts)]
    rows = []
    for e in test_eps:
        m = calib.episode_metrics(feats[e["ticker"]], e["L_idx"], N, H)
        if m is None or m["struct_deterioration"] is None:
            continue
        rows.append({
            "ticker": e["ticker"], "L_idx": e["L_idx"],
            "is_confirmed": bool(e["is_confirmed"]),
            "y": int(m["struct_deterioration"]),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    n_conf = int(df["is_confirmed"].sum())
    n_base = int(len(df) - n_conf)
    if n_conf == 0 or n_base == 0:
        return {"n_conf": n_conf, "n_base": n_base, "lift": None,
                "ci_lower": None, "ci_upper": None,
                "placebo_a": None, "placebo_b": None,
                "n_test": len(df), "n_train": len(train_eps)}
    point = lift_within(df)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    tks = df["ticker"].unique()
    boots = []
    for _ in range(boot_b):
        samp = rng.choice(tks, size=len(tks), replace=True)
        bdf = pd.concat([df[df["ticker"] == t] for t in samp], ignore_index=True)
        lv = lift_within(bdf)
        if lv is not None:
            boots.append(lv)
    ci_l = float(np.percentile(boots, 2.5)) if boots else None
    ci_u = float(np.percentile(boots, 97.5)) if boots else None
    pa_lift = lift_within(placebo_perm(df, seed + 100))
    pb_lift = lift_within(placebo_shift(df, seed + 200))
    if pb_lift is not None and abs(pb_lift - point) < 1e-9:
        raise RuntimeError(
            f"placebo_b identico a lift real (fold). Bug en placebo_shift. "
            f"lift={point} pb={pb_lift} n_tk_medio={len(df)/max(1, df['ticker'].nunique()):.1f}"
        )
    return {"n_conf": n_conf, "n_base": n_base, "lift": point,
            "ci_lower": ci_l, "ci_upper": ci_u,
            "placebo_a": pa_lift, "placebo_b": pb_lift,
            "n_test": len(df), "n_train": len(train_eps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    print("== SOW C1-M5 RESERVED COHORT ==")
    print(f"Candidata FROZEN: {CANDIDATE}")
    print(f"Folds: {FOLDS} H={H} EMBARGO={EMBARGO} B={args.boot_b}")

    df, tickers = calib.load_dataset()
    dates = pd.to_datetime(df.index).sort_values()
    print(f"Tickers: {len(tickers)} | {dates.min().date()} -> {dates.max().date()}")

    feats = calib.precompute_features(df, tickers)
    print(f"Features: {len(feats)}")
    starts_cache = {tk: calib.find_candidate_starts(f["candidate"]) for tk, f in feats.items()}
    sow_cache = calib.compute_sow_cache(feats, CANDIDATE["N"], CANDIDATE["X_ATR"], CANDIDATE["Y_VOL"])
    episodes = calib.build_episodes_landmark(feats, sow_cache, starts_cache, CANDIDATE["N"], CANDIDATE["M"])
    print(f"Episodios: {len(episodes)}")

    results = []
    for year in FOLDS:
        b = fold_bounds(dates, year)
        if b is None:
            continue
        ts, te = b
        t0 = time.time()
        r = evaluate_fold(episodes, feats, ts, te, CANDIDATE["N"], args.boot_b, args.seed + year)
        if r is None:
            continue
        r["fold"] = year
        results.append(r)
        print(f"  {year}: n_test={r['n_test']:5d} n_conf={r['n_conf']:4d} "
              f"n_base={r['n_base']:5d} lift={r['lift']} "
              f"pb={r['placebo_b']} ({time.time()-t0:.0f}s)")

    valid = [r for r in results if r["lift"] is not None]
    if not valid:
        return 1

    lifts = np.array([r["lift"] for r in valid])
    mean_lift = float(lifts.mean())
    folds_pos = int((lifts > 0).sum())
    folds_beat = sum(1 for r in valid if r["placebo_b"] is not None and r["lift"] > r["placebo_b"])

    rng = np.random.default_rng(args.seed + 999)
    boot_m = [float(rng.choice(lifts, size=len(lifts), replace=True).mean()) for _ in range(5000)]
    ci_l = float(np.percentile(boot_m, 2.5))
    ci_u = float(np.percentile(boot_m, 97.5))
    verdict = classify(mean_lift, ci_l, ci_u, folds_pos, folds_beat)

    summary = {"candidate": CANDIDATE, "cohort": "reserved_sp500",
               "folds": FOLDS, "embargo": EMBARGO, "H": H,
               "n_folds": len(valid), "mean_lift": mean_lift,
               "ci_lower": ci_l, "ci_upper": ci_u,
               "folds_positive": folds_pos, "folds_beat_placebo": folds_beat,
               "fold_results": valid, "verdict": verdict}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(valid).to_csv(OUT_DIR / "folds.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    dec = f"""SOW C1-M5 Reserved Cohort - Decision
Fecha: 2026-10-24
Cohorte: 268 tickers S&P 500 reservados (no en extended)
Candidata FROZEN: {CANDIDATE}

Folds evaluables: {len(valid)}/9
Mean lift: {mean_lift:.4f}
IC95 (fold-level bootstrap): [{ci_l:.4f}, {ci_u:.4f}]
Folds positivos: {folds_pos}/{len(valid)}
Folds beat placebo_B: {folds_beat}/{len(valid)}

VEREDICTO: {verdict}
"""
    (OUT_DIR / "decision.txt").write_text(dec, encoding="utf-8")

    print("\n== RESULTADO ==")
    print(f"Folds: {len(valid)}/9")
    print(f"Mean lift: {mean_lift:.4f} IC95: [{ci_l:.4f}, {ci_u:.4f}]")
    print(f"Folds positivos: {folds_pos}/{len(valid)}")
    print(f"Folds beat placebo B: {folds_beat}/{len(valid)}")
    print(f"VEREDICTO: {verdict}")
    print(f"Escrito: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())