#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QA de precondiciones 5b.X.

Read-only. Justifica o corrige los umbrales del protocolo 5b.X seccion
2.1 (12 meses / 50 episodios / 20 confirmados) midiendo la tasa
historica de episodios y la tasa candidate->confirmed.

Entrada: data/stock_prices.parquet (historico hasta 2026-10-01).
Salida:  outputs/audit/wyckoff_5bX_precond_qa.json

Uso:
    py scripts/qa_precond_5bX.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_wyckoff_sow_5b4bis import (
    load_dataset, precompute_features, compute_sow_cache,
    find_candidate_starts, build_episodes_landmark,
)

CAND = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_JSON = OUT_DIR / "wyckoff_5bX_precond_qa.json"

HIST_START = pd.Timestamp("2021-10-01")
HIST_END = pd.Timestamp("2026-10-01")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== QA precondiciones 5b.X ==")
    print(f"Candidata: N={CAND['N']} M={CAND['M']} "
          f"X_ATR={CAND['X_ATR']} Y_VOL={CAND['Y_VOL']}")
    print()

    df, tickers = load_dataset()
    feats = precompute_features(df, tickers)
    print(f"Tickers: {len(feats)}")

    idx = df.index
    idx_hist = idx[(idx >= HIST_START) & (idx <= HIST_END)]
    n_sessions_total = len(idx_hist)
    months_hist = (HIST_END - HIST_START).days / 30.44
    sessions_per_month = n_sessions_total / months_hist if months_hist > 0 else 0
    print(f"Historico: {idx_hist.min().date()} -> {idx_hist.max().date()}")
    print(f"  sesiones: {n_sessions_total}")
    print(f"  meses: {months_hist:.1f}")
    print(f"  sesiones/mes: {sessions_per_month:.1f}")

    starts_cache = {tk: find_candidate_starts(feat["candidate"])
                    for tk, feat in feats.items()}
    total_starts = sum(len(v) for v in starts_cache.values())
    starts_per_month = total_starts / months_hist if months_hist > 0 else 0
    print(f"\nCandidate starts totales: {total_starts}")
    print(f"  starts/mes: {starts_per_month:.1f}")

    sow_cache = compute_sow_cache(feats, CAND["N"], CAND["X_ATR"], CAND["Y_VOL"])
    episodes = build_episodes_landmark(feats, sow_cache, starts_cache,
                                        CAND["N"], CAND["M"])
    n_ep = len(episodes)
    n_conf = sum(1 for e in episodes if e["is_confirmed"])
    tasa_conf = n_conf / n_ep if n_ep > 0 else 0.0
    print("\nEpisodios con landmark L=t0+M:")
    print(f"  total: {n_ep}")
    print(f"  confirmed: {n_conf}")
    print(f"  tasa confirmed: {tasa_conf:.3f}")

    print("\n== Estimacion a X meses post-cutoff ==")
    est = {}
    for months in (6, 12, 18, 24):
        ep_new = starts_per_month * months
        conf_new = ep_new * tasa_conf
        est[months] = {
            "months": months,
            "n_episodes_estimated": round(ep_new, 1),
            "n_confirmed_estimated": round(conf_new, 1),
        }
        print(f"  {months:2d} meses: episodios ~{ep_new:.0f}, "
              f"confirmed ~{conf_new:.0f}")

    print("\n== Comparativa con umbrales propuestos ==")
    MIN_MONTHS = 12
    MIN_N_EP = 50
    MIN_N_CONF = 20

    for months in (6, 12, 18, 24):
        e = est[months]
        cumple_ep = e["n_episodes_estimated"] >= MIN_N_EP
        cumple_conf = e["n_confirmed_estimated"] >= MIN_N_CONF
        print(f"  {months:2d} meses: "
              f"ep>={MIN_N_EP}? {cumple_ep}  conf>={MIN_N_CONF}? {cumple_conf}")

    print("\n== Variabilidad de la tasa de episodios por sub-bloques de 6m ==")
    sub_blocks = pd.date_range(start=HIST_START, end=HIST_END, freq="6ME")
    rates = []
    prev = HIST_START
    for b in sub_blocks:
        if b <= prev:
            continue
        n_start = sum(
            1
            for tk, feat in feats.items()
            for s in starts_cache[tk]
            if prev <= feat["dates"][int(s)] < b
        )
        days = (b - prev).days
        months = days / 30.44
        rate = n_start / months if months > 0 else 0
        rates.append({
            "period": f"{prev.date()} -> {b.date()}",
            "n_starts": n_start,
            "months": round(months, 1),
            "starts_per_month": round(rate, 1),
        })
        prev = b

    for r in rates:
        print(f"  {r['period']}: starts={r['n_starts']} "
              f"({r['starts_per_month']}/mes)")

    summary = {
        "schema": "wyckoff_5bX_precond_qa_v1",
        "candidata": CAND,
        "historico": {
            "start": str(idx_hist.min().date()),
            "end": str(idx_hist.max().date()),
            "n_sessions": n_sessions_total,
            "months": round(months_hist, 1),
            "sessions_per_month": round(sessions_per_month, 1),
        },
        "tasas": {
            "total_candidate_starts": total_starts,
            "starts_per_month": round(starts_per_month, 1),
            "tasa_confirmed": round(tasa_conf, 4),
        },
        "estimaciones": est,
        "umbrales_propuestos": {
            "MIN_MONTHS": MIN_MONTHS,
            "MIN_N_EPISODES": MIN_N_EP,
            "MIN_N_CONFIRMED": MIN_N_CONF,
        },
        "sub_bloques_6m": rates,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nQA JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
