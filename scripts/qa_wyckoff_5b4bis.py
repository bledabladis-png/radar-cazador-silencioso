#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""QA 5b.4-bis - IC por bloque + concentracion por ticker.

Protocolo QA solicitado por dictamen 33 seccion 19.
Read-only. NO modifica datos, config ni contrato.

Uso:
    py scripts/qa_wyckoff_5b4bis.py

Salidas:
    outputs/audit/wyckoff_5b4bis_qa_blocks.csv
    outputs/audit/wyckoff_5b4bis_qa_tickers.csv
    outputs/audit/wyckoff_5b4bis_qa.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.calibrate_wyckoff_sow_5b4bis import (
    load_dataset, precompute_features, compute_sow_cache,
    find_candidate_starts, build_episodes_landmark,
    episode_metrics, BLOCKS,
    BOOT_B, BOOT_SEED,
)

CAND = {"N": 60, "M": 30, "X_ATR": 0.25, "Y_VOL": 1.10}

OUT_DIR = ROOT / "outputs" / "audit"
OUT_BLOCKS = OUT_DIR / "wyckoff_5b4bis_qa_blocks.csv"
OUT_TICKERS = OUT_DIR / "wyckoff_5b4bis_qa_tickers.csv"
OUT_JSON = OUT_DIR / "wyckoff_5b4bis_qa.json"


def bootstrap_lift_by_block(rows_by_block, B=BOOT_B, seed=BOOT_SEED):
    """Bootstrap ticker-cluster del lift en subconjuntos de episodios."""
    out = {}
    for label, rows in rows_by_block.items():
        if not rows:
            out[label] = {"lift_point": None, "lower_ci": None,
                          "upper_ci": None, "n_conf": 0, "n_base": 0,
                          "n_tickers": 0, "n_boot": 0}
            continue
        by_tk = {}
        for tk, is_conf, m in rows:
            v = m.get("struct_deterioration")
            if v is None:
                continue
            by_tk.setdefault(tk, []).append((is_conf, v))
        tickers = list(by_tk.keys())

        def _lift(tks):
            nc = nb = sc = sb = 0
            for tk in tks:
                for is_conf, v in by_tk[tk]:
                    if is_conf:
                        nc += 1; sc += v
                    else:
                        nb += 1; sb += v
            if nc == 0 or nb == 0:
                return None
            return (sc / nc) - (sb / nb)

        lift_point = _lift(tickers)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(B):
            sample = rng.choice(tickers, size=len(tickers), replace=True)
            v = _lift(list(sample))
            if v is not None:
                boots.append(v)
        if len(boots) >= 50:
            lower = float(np.percentile(boots, 2.5))
            upper = float(np.percentile(boots, 97.5))
        else:
            lower = upper = None
        nc = sum(1 for tk in tickers for (c, _) in by_tk[tk] if c)
        nb = sum(1 for tk in tickers for (c, _) in by_tk[tk] if not c)
        out[label] = {
            "lift_point": lift_point,
            "lower_ci": lower,
            "upper_ci": upper,
            "n_conf": nc,
            "n_base": nb,
            "n_tickers": len(tickers),
            "n_boot": len(boots),
        }
    return out


def ticker_distribution(rows_all):
    """Distribucion de episodios por ticker."""
    df = pd.DataFrame(rows_all, columns=["ticker", "is_confirmed", "m"])
    df["has_struct"] = df["m"].apply(lambda m: m.get("struct_deterioration") is not None)
    df = df[df["has_struct"]]
    rows = []
    for tk, sub in df.groupby("ticker"):
        n_tot = len(sub)
        n_conf = int(sub["is_confirmed"].sum())
        n_base = n_tot - n_conf
        rows.append({
            "ticker": tk,
            "n_episodes": n_tot,
            "n_confirmed": n_conf,
            "n_baseline": n_base,
        })
    return pd.DataFrame(rows).sort_values("n_episodes", ascending=False)

def collect_rows_by_block(episodes, feats, N):
    """Construye dict {block_label: [(ticker, is_confirmed, metrics_H20)]}."""
    rows = {label: [] for label in [b[0] for b in BLOCKS] + ["ALL"]}
    for e in episodes:
        tk = e["ticker"]
        m = episode_metrics(feats[tk], e["L_idx"], N, 20)
        if m is None:
            continue
        entry = (tk, e["is_confirmed"], m)
        rows["ALL"].append(entry)
        if e["block_L"] is not None:
            rows[e["block_L"]].append(entry)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== QA 5b.4-bis ==")
    print(f"Candidata: N={CAND['N']} M={CAND['M']} X={CAND['X_ATR']} Y={CAND['Y_VOL']}")

    df, tickers = load_dataset()
    feats = precompute_features(df, tickers)
    starts_cache = {tk: find_candidate_starts(feat["candidate"])
                    for tk, feat in feats.items()}

    sow_cache = compute_sow_cache(feats, CAND["N"], CAND["X_ATR"], CAND["Y_VOL"])
    episodes = build_episodes_landmark(feats, sow_cache, starts_cache,
                                        CAND["N"], CAND["M"])
    print(f"Episodios: {len(episodes)} "
          f"(conf={sum(1 for e in episodes if e['is_confirmed'])}, "
          f"base={sum(1 for e in episodes if not e['is_confirmed'])})")

    rows_by_block = collect_rows_by_block(episodes, feats, CAND["N"])

    print()
    print("== Bootstrap IC por bloque (B=2000, seed=20261002) ==")
    block_res = bootstrap_lift_by_block(rows_by_block)
    for label in [b[0] for b in BLOCKS] + ["ALL"]:
        r = block_res[label]
        print(f"  {label}: lift={r['lift_point']}  "
              f"IC95=[{r['lower_ci']}, {r['upper_ci']}]  "
              f"n_conf={r['n_conf']} n_base={r['n_base']} n_tickers={r['n_tickers']}")

    # CSV por bloque
    rows = []
    for label in [b[0] for b in BLOCKS] + ["ALL"]:
        r = block_res[label]
        rows.append({
            "bloque": label,
            "lift_point": r["lift_point"],
            "lower_ci": r["lower_ci"],
            "upper_ci": r["upper_ci"],
            "n_conf": r["n_conf"],
            "n_base": r["n_base"],
            "n_tickers": r["n_tickers"],
            "n_boot_validos": r["n_boot"],
            "ic_cruza_0": (r["lower_ci"] is not None and r["lower_ci"] <= 0),
        })
    pd.DataFrame(rows).to_csv(OUT_BLOCKS, index=False)
    print(f"QA blocks CSV: {OUT_BLOCKS}")

    # Distribucion por ticker
    print()
    print("== Concentracion por ticker (bloque ALL) ==")
    tick_df = ticker_distribution(rows_by_block["ALL"])
    tick_df.to_csv(OUT_TICKERS, index=False)
    print(f"QA tickers CSV: {OUT_TICKERS} ({len(tick_df)} tickers)")

    n_tot = int(tick_df["n_episodes"].sum())
    n_conf = int(tick_df["n_confirmed"].sum())
    n_base = int(tick_df["n_baseline"].sum())
    top5_share = float(tick_df.head(5)["n_episodes"].sum() / n_tot) if n_tot else 0.0
    top10_share = float(tick_df.head(10)["n_episodes"].sum() / n_tot) if n_tot else 0.0
    print(f"  n_tickers={len(tick_df)} n_episodes={n_tot} n_conf={n_conf} n_base={n_base}")
    print(f"  max_episodios_por_ticker={int(tick_df['n_episodes'].max())}")
    print(f"  mediana_episodios_por_ticker={float(tick_df['n_episodes'].median()):.1f}")
    print(f"  top5_share={top5_share:.3f} top10_share={top10_share:.3f}")

    # JSON resumen
    summary = {
        "schema": "wyckoff_5b4bis_qa_v1",
        "dictamen_base": "33_dictamen_5b4bis_v2.md",
        "candidata": CAND,
        "n_episodes_total": len(episodes),
        "n_episodes_conf": sum(1 for e in episodes if e["is_confirmed"]),
        "n_episodes_base": sum(1 for e in episodes if not e["is_confirmed"]),
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED},
        "por_bloque": {label: block_res[label]
                       for label in [b[0] for b in BLOCKS] + ["ALL"]},
        "ticker_distribution": {
            "n_tickers": len(tick_df),
            "n_episodes": n_tot,
            "n_confirmed": n_conf,
            "n_baseline": n_base,
            "max_episodios_por_ticker": int(tick_df["n_episodes"].max()),
            "mediana_episodios_por_ticker": float(tick_df["n_episodes"].median()),
            "top5_share": top5_share,
            "top10_share": top10_share,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"QA JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
