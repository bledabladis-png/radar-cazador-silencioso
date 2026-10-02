#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validacion 5b.X - Out-of-sample de la candidata SOW congelada.

Protocolo: docs/auditoria/wyckoff/35_protocolo_5bX.md
Contrato:  docs/auditoria/wyckoff/01_contrato_semantico_v1_9.md
Dictamen:  33_dictamen_5b4bis_v2.md

Candidata congelada (v1.9, FROZEN_FOR_VALIDATION):
    N=60, M=30, X_ATR=0.25, Y_VOL=1.10

Read-only. NO modifica datos, config ni contrato. NO recalibra. NO
explora grid. UNA sola configuracion, un solo bloque.

Entrada:   data/stock_prices.parquet (posterior al cutoff)
Salida:
    outputs/audit/wyckoff_5bX_summary.json
    outputs/audit/wyckoff_5bX_bootstrap.csv
    outputs/audit/wyckoff_5bX_run.log

Uso:
    py scripts/validate_wyckoff_sow_5bX.py

Si no hay suficientes datos posteriores al cutoff, el script NO
ejecuta y reporta el estado de bloqueo.
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

from scripts.calibrate_wyckoff_sow_5b4bis import (
    load_dataset, precompute_features, compute_sow_cache,
    find_candidate_starts, build_episodes_landmark,
    episode_metrics,
    BOOT_B, BOOT_SEED,
)

# --- Candidata congelada (contrato v1.9, seccion 13.2) ---
CANDIDATA = {
    "N": 60,
    "M": 30,
    "X_ATR": 0.25,
    "Y_VOL": 1.10,
}

# --- Cutoff: todo dato posterior a esta fecha es candidato a validacion ---
CUTOFF_DATE = pd.Timestamp("2026-10-01")

# --- Condiciones previas (protocolo 5b.X seccion 2.1) ---
# Propuesta sujeta a dictamen. Si el auditor cambia los umbrales, se
# actualizan aqui antes de ejecutar.
MIN_MONTHS = 12
MIN_N_EPISODES = 50
MIN_N_CONFIRMED = 20

# --- Rutas de salida ---
OUT_DIR = ROOT / "outputs" / "audit"
OUT_SUMMARY = OUT_DIR / "wyckoff_5bX_summary.json"
OUT_BOOT = OUT_DIR / "wyckoff_5bX_bootstrap.csv"
OUT_LOG = OUT_DIR / "wyckoff_5bX_run.log"


def check_preconditions(df, tickers, feats):
    """Verifica condiciones previas antes de ejecutar la validacion.

    Devuelve dict con:
        cutoff_date, n_sessions_after, n_months_after,
        n_tickers_with_data, n_candidate_starts_after, ok (bool),
        reasons (list).
    """
    reasons = []
    idx = df.index
    idx_after = idx[idx > CUTOFF_DATE]
    n_sessions = len(idx_after)
    # Meses aproximados (20 sesiones/mes)
    n_months = n_sessions / 20.0

    if n_months < MIN_MONTHS:
        reasons.append(
            f"meses posteriores al cutoff: {n_months:.1f} < {MIN_MONTHS}"
        )
    if n_sessions == 0:
        reasons.append("no hay sesiones posteriores al cutoff")

    # Contar candidate starts con L > cutoff
    n_starts_after = 0
    for tk, feat in feats.items():
        starts = find_candidate_starts(feat["candidate"])
        dates = feat["dates"]
        for s in starts:
            L_idx = int(s) + CANDIDATA["M"]
            if L_idx < len(dates) and dates[L_idx] > CUTOFF_DATE:
                n_starts_after += 1

    if n_starts_after < MIN_N_EPISODES:
        reasons.append(
            f"episodios con L > cutoff: {n_starts_after} < {MIN_N_EPISODES}"
        )

    return {
        "cutoff_date": str(CUTOFF_DATE.date()),
        "last_data_date": str(idx.max().date()),
        "n_sessions_after_cutoff": n_sessions,
        "n_months_after_cutoff": round(n_months, 1),
        "n_candidate_starts_with_L_after_cutoff": n_starts_after,
        "min_months_required": MIN_MONTHS,
        "min_episodes_required": MIN_N_EPISODES,
        "min_confirmed_required": MIN_N_CONFIRMED,
        "ok": len(reasons) == 0,
        "reasons": reasons,
    }

def bootstrap_lift(rows, B=BOOT_B, seed=BOOT_SEED):
    """Bootstrap ticker-cluster del lift_H20 sobre un bloque de validacion.

    rows: lista de (ticker, is_confirmed, m) con m dict de metricas H20.
    """
    if not rows:
        return {
            "lift_point": None, "lower_ci": None, "upper_ci": None,
            "n_tickers": 0, "n_episodes": 0,
            "n_confirmed": 0, "n_baseline": 0, "n_boot": 0,
        }
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
    return {
        "lift_point": lift_point,
        "lower_ci": lower,
        "upper_ci": upper,
        "n_tickers": len(tickers),
        "n_episodes": len(rows),
        "n_confirmed": nc,
        "n_baseline": nb,
        "n_boot": len(boots),
    }


def classify_scenario(boot):
    """Clasifica el resultado en uno de los tres escenarios ex-ante
    (protocolo 5b.X seccion 6.1).

    A: IC95% lower > 0 -> confirma.
    B: IC95% cruza 0 pero lift_point > 0 -> neutro.
    C: IC95% cruza 0 y lift_point <= 0 -> refuta.
    """
    lift = boot["lift_point"]
    lower = boot["lower_ci"]
    if lift is None or lower is None:
        return "INCONCLUSO", "sin muestra suficiente"
    if lower > 0:
        return "A", "IC95% lower > 0: validacion confirma"
    if lift > 0:
        return "B", "IC95% cruza 0, lift_point > 0: neutro"
    return "C", "IC95% cruza 0 y lift_point <= 0: refuta"


def collect_validation_rows(episodes, feats, N, cutoff):
    """Filtra episodios con L > cutoff y calcula metricas H20.

    Devuelve lista de (ticker, is_confirmed, m_H20).
    """
    rows = []
    for e in episodes:
        if e["L_date"] <= cutoff:
            continue
        m = episode_metrics(feats[e["ticker"]], e["L_idx"], N, 20)
        if m is None:
            continue
        rows.append((e["ticker"], e["is_confirmed"], m))
    return rows

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 5b.X - Validacion out-of-sample de la candidata SOW ==")
    print("Protocolo: 35_protocolo_5bX.md")
    print("Contrato:  01_contrato_semantico_v1_9.md")
    print(f"Candidata: N={CANDIDATA['N']} M={CANDIDATA['M']} "
          f"X_ATR={CANDIDATA['X_ATR']} Y_VOL={CANDIDATA['Y_VOL']}")
    print(f"Cutoff:    {CUTOFF_DATE.date()}")
    print()

    df, tickers = load_dataset()
    print(f"Tickers en dataset: {len(tickers)}")
    feats = precompute_features(df, tickers)
    print(f"Features construidas: {len(feats)}")

    # Verificar condiciones previas
    print()
    print("== Verificando condiciones previas ==")
    precond = check_preconditions(df, tickers, feats)
    for k, v in precond.items():
        print(f"  {k}: {v}")

    if not precond["ok"]:
        print()
        print("BLOQUEADO. La validacion no puede ejecutarse todavia.")
        print("Razones:")
        for r in precond["reasons"]:
            print(f"  - {r}")
        summary = {
            "schema": "wyckoff_5bX_summary_v1",
            "status": "BLOCKED",
            "protocolo": "35_protocolo_5bX.md",
            "contrato": "01_contrato_semantico_v1_9.md",
            "candidata": CANDIDATA,
            "preconditions": precond,
        }
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"Summary JSON: {OUT_SUMMARY}")
        return

    # Construir episodios con la candidata congelada
    print()
    print("== Construyendo episodios ==")
    t0 = time.time()
    starts_cache = {tk: find_candidate_starts(feat["candidate"])
                    for tk, feat in feats.items()}
    sow_cache = compute_sow_cache(feats, CANDIDATA["N"],
                                   CANDIDATA["X_ATR"], CANDIDATA["Y_VOL"])
    episodes = build_episodes_landmark(feats, sow_cache, starts_cache,
                                        CANDIDATA["N"], CANDIDATA["M"])
    print(f"Episodios totales: {len(episodes)} "
          f"({time.time()-t0:.1f}s)")

    # Filtrar al bloque de validacion (L > cutoff)
    rows = collect_validation_rows(episodes, feats, CANDIDATA["N"],
                                    CUTOFF_DATE)
    n_conf = sum(1 for _, c, _ in rows if c)
    n_base = len(rows) - n_conf
    print(f"Episodios de validacion (L > cutoff): {len(rows)} "
          f"(conf={n_conf}, base={n_base})")

    if n_conf < MIN_N_CONFIRMED:
        print()
        print(f"BLOQUEADO. n_confirmed={n_conf} < {MIN_N_CONFIRMED}.")
        summary = {
            "schema": "wyckoff_5bX_summary_v1",
            "status": "BLOCKED_INSUFFICIENT_CONFIRMED",
            "protocolo": "35_protocolo_5bX.md",
            "contrato": "01_contrato_semantico_v1_9.md",
            "candidata": CANDIDATA,
            "preconditions": precond,
            "validation": {
                "n_episodes": len(rows),
                "n_confirmed": n_conf,
                "n_baseline": n_base,
            },
        }
        with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"Summary JSON: {OUT_SUMMARY}")
        return

    # Bootstrap
    print()
    print(f"== Bootstrap B={BOOT_B} seed={BOOT_SEED} ==")
    t0 = time.time()
    boot = bootstrap_lift(rows, B=BOOT_B, seed=BOOT_SEED)
    print(f"Bootstrap completado en {time.time()-t0:.1f}s")
    print(f"  lift_point = {boot['lift_point']}")
    print(f"  IC95% = [{boot['lower_ci']}, {boot['upper_ci']}]")
    print(f"  n_confirmed = {boot['n_confirmed']}")
    print(f"  n_baseline  = {boot['n_baseline']}")
    print(f"  n_tickers   = {boot['n_tickers']}")

    # Clasificar escenario
    esc, motivo = classify_scenario(boot)
    print()
    print(f"== Escenario: {esc} ==")
    print(f"  {motivo}")

    # CSV bootstrap
    pd.DataFrame([{
        "N": CANDIDATA["N"], "M": CANDIDATA["M"],
        "X_ATR": CANDIDATA["X_ATR"], "Y_VOL": CANDIDATA["Y_VOL"],
        "lift_point": boot["lift_point"],
        "lower_ci": boot["lower_ci"],
        "upper_ci": boot["upper_ci"],
        "n_tickers": boot["n_tickers"],
        "n_episodes": boot["n_episodes"],
        "n_confirmed": boot["n_confirmed"],
        "n_baseline": boot["n_baseline"],
        "n_boot": boot["n_boot"],
        "escenario": esc,
    }]).to_csv(OUT_BOOT, index=False)
    print(f"Bootstrap CSV: {OUT_BOOT}")

    # Summary JSON
    summary = {
        "schema": "wyckoff_5bX_summary_v1",
        "status": "EXECUTED",
        "protocolo": "35_protocolo_5bX.md",
        "contrato": "01_contrato_semantico_v1_9.md",
        "candidata": CANDIDATA,
        "cutoff_date": str(CUTOFF_DATE.date()),
        "preconditions": precond,
        "validation": {
            "n_episodes": len(rows),
            "n_confirmed": boot["n_confirmed"],
            "n_baseline": boot["n_baseline"],
            "n_tickers": boot["n_tickers"],
        },
        "bootstrap": {
            "B": BOOT_B,
            "seed": BOOT_SEED,
            "lift_point": boot["lift_point"],
            "lower_ci": boot["lower_ci"],
            "upper_ci": boot["upper_ci"],
        },
        "scenario": {
            "code": esc,
            "motivo": motivo,
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Summary JSON: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
