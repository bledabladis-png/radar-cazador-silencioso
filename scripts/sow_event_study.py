#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event study SOW: mide el SOW como evento, no como ventana.

Motivacion: purged_kfold_sow_* reduce a binario "hay algun SOW en [t0+1, t0+5]".
Con 12 SOW/ano/ticker (v1) o 4 (v2), casi toda ventana contiene al menos uno.
Por eso v1 y v2 dan identicos n_conf. No puede distinguir detectores.

Este script evalua cada SOW individual: y = struct[t+H] < struct[t], con
control = toda fecha sin SOW con struct valido. Lift within-ticker. Placebo
con distribucion (N=100 shifts). Bootstrap block por fecha (no por ticker).

Criterio PRE-REGISTRADO:
  FUNCIONA_V2 si:
    mean_lift_v2 > 0.05
    AND CI95_lower_v2 > 0
    AND lift_v2 > lift_v1
    AND lift_v2 > mean_placebo_v2 + 2*sd_placebo_v2
    AND folds_positivos_v2 >= 7/9
  NO_FUNCIONA si:
    mean_lift_v2 <= 0 OR lift_v2 <= lift_v1
  INCONCLUSO: resto

No se modifica el criterio tras ver resultados.
Read-only: solo lee stock_prices_extended.parquet, escribe en outputs/audit/.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators import wyckoff_v1 as w1

PARQUET = ROOT / "data" / "stock_prices_extended.parquet"
OUT_DIR = ROOT / "outputs" / "audit" / "sow_event_study_20261024"

N = 40
X_ATR = 1.00
Y_VOL = 1.10
H = 20
MIN_OBS = 200
FOLDS = list(range(2017, 2026))
N_PLACEBO_SHIFTS = 100
N_BOOTSTRAP = 2000
BLOCK_SESSIONS = 20


def build_events(tdf, ticker, sow_arr, struct_arr, h):
    """Para cada t con struct[t] y struct[t+h] validos, registra is_sow y y."""
    n = len(tdf)
    dates = tdf.index
    out = []
    for i in range(n - h):
        s_t = struct_arr[i]
        s_h = struct_arr[i + h]
        if pd.isna(s_t) or pd.isna(s_h):
            continue
        out.append({
            "date": dates[i],
            "ticker": ticker,
            "is_sow": bool(int(sow_arr[i]) == 1),
            "y": int(float(s_h) < float(s_t)),
        })
    return out


def lift_within(events_df):
    if events_df.empty:
        return None
    conf = events_df[events_df["is_sow"]].groupby("ticker")["y"].agg(["mean", "count"])
    base = events_df[~events_df["is_sow"]].groupby("ticker")["y"].agg(["mean", "count"])
    valid = conf.index.intersection(base.index)
    if len(valid) == 0:
        return None
    lift_tk = conf.loc[valid, "mean"] - base.loc[valid, "mean"]
    w = conf.loc[valid, "count"].astype(float)
    tw = w.sum()
    if tw == 0:
        return None
    return float((lift_tk * w).sum() / tw)


def block_bootstrap_by_date(events_df, B=N_BOOTSTRAP, block=BLOCK_SESSIONS, seed=42):
    """Bootstrap por bloques de sesiones (fechas). Captura dependencia temporal."""
    if events_df.empty:
        return None, None
    unique_dates = np.array(sorted(events_df["date"].unique()))
    n_dates = len(unique_dates)
    if n_dates == 0:
        return None, None
    n_blocks = max(1, n_dates // block + 1)
    rng = np.random.default_rng(seed)
    lifts = []
    for _ in range(B):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        sampled_dates = []
        for b in idx:
            s = b * block
            e = min(s + block, n_dates)
            sampled_dates.extend(unique_dates[s:e])
        sub = events_df[events_df["date"].isin(sampled_dates)]
        if len(sub) < 10:
            continue
        lv = lift_within(sub)
        if lv is not None:
            lifts.append(lv)
    if not lifts:
        return None, None
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def placebo_lift_series(per_ticker_data, h, n_shifts=N_PLACEBO_SHIFTS, seed=43):
    """N shifts globales por ticker. Devuelve lista de lifts placebo."""
    rng = np.random.default_rng(seed)
    lifts = []
    for k in range(n_shifts):
        all_events = []
        for tk, data in per_ticker_data.items():
            tdf = data["tdf"]
            struct_arr = data["struct"]
            sow_arr = data["sow_v1"]  # placebo sobre v1, mismo esquema
            n = len(sow_arr)
            if n < 61:
                continue
            offset = int(rng.integers(30, 61))
            shifted = np.roll(sow_arr, offset)
            all_events.extend(build_events(tdf, tk, shifted, struct_arr, h))
        if not all_events:
            continue
        df = pd.DataFrame(all_events)
        lv = lift_within(df)
        if lv is not None:
            lifts.append(lv)
    return lifts


def fold_metrics(events_df, year):
    dates = pd.to_datetime(events_df["date"])
    ys = pd.Timestamp(str(year) + "-01-01")
    ye = pd.Timestamp(str(year) + "-12-31")
    sub = events_df[(dates >= ys) & (dates <= ye)].copy()
    if sub.empty:
        return None
    n_conf = int(sub["is_sow"].sum())
    n_base = int(len(sub) - n_conf)
    if n_conf < 5 or n_base < 50:
        return {"year": year, "n_conf": n_conf, "n_base": n_base,
                "lift": None, "ci_lower": None, "ci_upper": None}
    lift = lift_within(sub)
    ci_l, ci_u = block_bootstrap_by_date(sub)
    return {"year": year, "n_conf": n_conf, "n_base": n_base,
            "lift": lift, "ci_lower": ci_l, "ci_upper": ci_u}


def classify_v2(mean_lift_v2, ci_low_v2, lift_v2, lift_v1, folds_pos, placebo_mean, placebo_sd):
    if mean_lift_v2 is None or lift_v1 is None:
        return "INCONCLUSO"
    if mean_lift_v2 <= 0:
        return "NO_FUNCIONA"
    if lift_v2 <= lift_v1:
        return "NO_FUNCIONA"
    if placebo_mean is None or placebo_sd is None:
        return "INCONCLUSO"
    threshold = placebo_mean + 2 * placebo_sd
    if (mean_lift_v2 > 0.05 and ci_low_v2 is not None and ci_low_v2 > 0
            and lift_v2 > threshold and folds_pos >= 7):
        return "FUNCIONA_V2"
    return "INCONCLUSO"


def main() -> int:
    print("== SOW Event Study (v1 vs v2) ==")
    print(f"Candidata: N={N} X_ATR={X_ATR} Y_VOL={Y_VOL} H={H}")
    print(f"Placebos: {N_PLACEBO_SHIFTS} shifts | Bootstrap: B={N_BOOTSTRAP} block={BLOCK_SESSIONS}")

    if not PARQUET.exists():
        print(f"ERROR: no existe {PARQUET}")
        return 1

    df = pd.read_parquet(PARQUET)
    close = df.xs("Close", axis=1, level=0)
    tickers = sorted(close.columns.tolist())
    print(f"Tickers: {len(tickers)}")

    per_ticker = {}
    t0 = time.time()
    for tk in tickers:
        try:
            tdf = w1.build_ticker_df(df, tk)
            if tdf.empty or len(tdf) < MIN_OBS:
                continue
            _, struct_s, _, _, _, _, _ = w1.wyckoff_score(tdf, tk)
            struct_arr = np.asarray(struct_s).ravel()
            sow_v1 = w1.detect_sow(tdf, tk, window=N, x_atr=X_ATR, y_vol=Y_VOL)
            sow_v2 = w1.detect_sow_v2(tdf, tk, window=N, x_atr=X_ATR, y_vol=Y_VOL, candidate_lookback=5)
            per_ticker[tk] = {
                "tdf": tdf,
                "struct": struct_arr,
                "sow_v1": np.asarray(sow_v1).astype(int).ravel(),
                "sow_v2": np.asarray(sow_v2).astype(int).ravel(),
            }
        except Exception as e:
            print(f"  {tk}: {e}")
            continue
    print(f"Tickers procesados: {len(per_ticker)} ({time.time()-t0:.1f}s)")

    print("== Construyendo eventos ==")
    t0 = time.time()
    events_v1 = []
    events_v2 = []
    for tk, d in per_ticker.items():
        events_v1.extend(build_events(d["tdf"], tk, d["sow_v1"], d["struct"], H))
        events_v2.extend(build_events(d["tdf"], tk, d["sow_v2"], d["struct"], H))
    df_v1 = pd.DataFrame(events_v1)
    df_v2 = pd.DataFrame(events_v2)
    print(f"v1: {len(df_v1)} obs, {int(df_v1['is_sow'].sum())} SOW")
    print(f"v2: {len(df_v2)} obs, {int(df_v2['is_sow'].sum())} SOW")
    print(f"({time.time()-t0:.1f}s)")

    print("== Lift por fold (v1, v2) ==")
    folds_v1 = []
    folds_v2 = []
    for year in FOLDS:
        r1 = fold_metrics(df_v1, year)
        r2 = fold_metrics(df_v2, year)
        if r1 is not None:
            r1["detector"] = "v1"
            folds_v1.append(r1)
        if r2 is not None:
            r2["detector"] = "v2"
            folds_v2.append(r2)
        print(f"  {year}: v1 lift={r1['lift'] if r1 else None} n_conf={r1['n_conf'] if r1 else 0} | "
              f"v2 lift={r2['lift'] if r2 else None} n_conf={r2['n_conf'] if r2 else 0}")

    valid_v1 = [r for r in folds_v1 if r.get("lift") is not None]
    valid_v2 = [r for r in folds_v2 if r.get("lift") is not None]

    if not valid_v1 or not valid_v2:
        print("ERROR: no hay folds evaluables")
        return 1

    lifts_v1 = np.array([r["lift"] for r in valid_v1])
    lifts_v2 = np.array([r["lift"] for r in valid_v2])
    mean_v1 = float(lifts_v1.mean())
    mean_v2 = float(lifts_v2.mean())
    folds_pos_v2 = int((lifts_v2 > 0).sum())

    print("\n== Placebo con distribucion (N=100 shifts) ==")
    t0 = time.time()
    placebo_lifts = placebo_lift_series(per_ticker, H)
    if placebo_lifts:
        placebo_mean = float(np.mean(placebo_lifts))
        placebo_sd = float(np.std(placebo_lifts))
    else:
        placebo_mean = None
        placebo_sd = None
    print(f"placebo lifts: n={len(placebo_lifts)} mean={placebo_mean} sd={placebo_sd} ({time.time()-t0:.1f}s)")

    # CI de v2 con block bootstrap sobre el agregado
    print("\n== Bootstrap block por fecha (v2 agregado) ==")
    t0 = time.time()
    ci_l_v2, ci_u_v2 = block_bootstrap_by_date(df_v2)
    print(f"CI95 v2: [{ci_l_v2}, {ci_u_v2}] ({time.time()-t0:.1f}s)")

    verdict = classify_v2(mean_v2, ci_l_v2, mean_v2, mean_v1, folds_pos_v2, placebo_mean, placebo_sd)

    summary = {
        "candidate": {"N": N, "X_ATR": X_ATR, "Y_VOL": Y_VOL, "H": H},
        "folds": FOLDS,
        "v1": {"mean_lift": mean_v1, "folds_positive": int((lifts_v1 > 0).sum()),
               "n_folds": len(valid_v1), "folds": valid_v1},
        "v2": {"mean_lift": mean_v2, "ci_lower": ci_l_v2, "ci_upper": ci_u_v2,
               "folds_positive": folds_pos_v2, "n_folds": len(valid_v2), "folds": valid_v2},
        "placebo": {"n_shifts": len(placebo_lifts), "mean": placebo_mean,
                    "sd": placebo_sd, "raw": placebo_lifts},
        "verdict": verdict,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(folds_v1).to_csv(OUT_DIR / "folds_v1.csv", index=False)
    pd.DataFrame(folds_v2).to_csv(OUT_DIR / "folds_v2.csv", index=False)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n== RESULTADO ==")
    print(f"v1 mean_lift: {mean_v1:.4f} ({len(valid_v1)} folds)")
    print(f"v2 mean_lift: {mean_v2:.4f} (IC95 [{ci_l_v2}, {ci_u_v2}], {folds_pos_v2}/{len(valid_v2)} pos)")
    print(f"placebo: mean={placebo_mean} sd={placebo_sd}")
    print(f"lift_v2 - lift_v1: {mean_v2 - mean_v1:+.4f}")
    print(f"lift_v2 - placebo_threshold: {mean_v2 - (placebo_mean + 2*placebo_sd) if placebo_mean is not None else 'N/A'}")
    print(f"VEREDICTO: {verdict}")
    print(f"Escrito: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())