#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validacion 5b.X - Out-of-sample de la candidata SOW congelada.

Protocolo: docs/auditoria/wyckoff/35_protocolo_5bX_v3.md
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

# --- Cutoff del desarrollo: ultima fecha del historico ---
CUTOFF_DATE = pd.Timestamp("2026-10-01")

# --- Universo OOS (protocolo 5b.X v3, seccion 3.1) ---
# Episodio OOS si y solo si t0 >= OOS_T0_MIN. No: L >= OOS_T0_MIN.
OOS_T0_MIN = pd.Timestamp("2026-10-02")

# --- Precondiciones (protocolo 5b.X v3, seccion 2) ---
# Unico umbral normativo (seccion 2.1):
MIN_N_CONFIRMED_H20 = 550

# Guardrails informativos (seccion 2.3). NO bloquean la ejecucion:
GUARDRAIL_MIN_MONTHS = 12
GUARDRAIL_MIN_STARTS_OOS = 50

# --- Rutas de salida ---
OUT_DIR = ROOT / "outputs" / "audit"
OUT_SUMMARY = OUT_DIR / "wyckoff_5bX_summary.json"
OUT_BOOT = OUT_DIR / "wyckoff_5bX_bootstrap.csv"
OUT_LOG = OUT_DIR / "wyckoff_5bX_run.log"


def check_preconditions(df, tickers, feats):
    """Verifica condiciones previas (protocolo 5b.X v3, seccion 2).

    Unico umbral bloqueante: existencia de datos OOS. El umbral 550
    H20-complete se verifica DESPUES, sobre episodios construidos.

    Guardrails informativos (NO bloquean): meses OOS >=12, candidate
    starts OOS >=50. Se reportan para contexto (seccion 2.3).

    Devuelve dict con:
        cutoff_date, oos_t0_min, last_data_date,
        n_sessions_after_cutoff, n_months_after_cutoff,
        n_candidate_starts_oos, guardrails_ok,
        min_n_confirmed_h20_required, ok, reasons.
    """
    reasons = []
    idx = df.index
    idx_after = idx[idx > CUTOFF_DATE]
    n_sessions = len(idx_after)
    n_months = n_sessions / 20.0

    if n_sessions == 0:
        reasons.append("no hay sesiones posteriores al cutoff")

    # Contar candidate starts OOS: t0 >= OOS_T0_MIN
    n_starts_oos = 0
    for tk, feat in feats.items():
        starts = find_candidate_starts(feat["candidate"])
        dates = feat["dates"]
        for s in starts:
            t0_idx = int(s)
            if t0_idx < len(dates) and dates[t0_idx] >= OOS_T0_MIN:
                n_starts_oos += 1

    guardrails_ok = (
        n_months >= GUARDRAIL_MIN_MONTHS
        and n_starts_oos >= GUARDRAIL_MIN_STARTS_OOS
    )

    return {
        "cutoff_date": str(CUTOFF_DATE.date()),
        "oos_t0_min": str(OOS_T0_MIN.date()),
        "last_data_date": str(idx.max().date()),
        "n_sessions_after_cutoff": n_sessions,
        "n_months_after_cutoff": round(n_months, 1),
        "n_candidate_starts_oos": n_starts_oos,
        "guardrail_min_months": GUARDRAIL_MIN_MONTHS,
        "guardrail_min_starts_oos": GUARDRAIL_MIN_STARTS_OOS,
        "guardrails_ok": guardrails_ok,
        "min_n_confirmed_h20_required": MIN_N_CONFIRMED_H20,
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
    """Clasifica el resultado por IC95% (protocolo 5b.X v3, seccion 6.3).

    Orden exacto:
        lower_ci > 0  -> A (validacion confirma)
        upper_ci < 0  -> C (evidencia contraria)
        resto (IC cruza 0) -> B (inconcluyente)

    PROHIBIDO usar lift_point como frontera entre B y C (D-06.4).
    Si el IC no es calculable (boots validos insuficientes), D.
    """
    lower = boot["lower_ci"]
    upper = boot["upper_ci"]
    if lower is None or upper is None:
        return "D", "IC no calculable (boots validos insuficientes)"
    if lower > 0:
        return "A", "IC95% lower > 0: validacion confirma"
    if upper < 0:
        return "C", "IC95% upper < 0: evidencia contraria"
    return "B", "IC95% cruza 0: inconcluyente"


def collect_validation_rows(episodes, feats, N, t0_min):
    """Filtra episodios OOS (t0 >= t0_min) con H20_complete.

    Protocolo 5b.X v3 secciones 2.1 y 3.1:
      - OOS: t0_date >= t0_min. No: L_date >= t0_min (D-06.2).
      - H20_complete: L_idx + 20 < len(dates). No basta con que exista
        el landmark (D-06.5).

    Devuelve (rows, n_dropped_oos, n_dropped_h20) donde rows es lista
    de (ticker, is_confirmed, m_H20).
    """
    rows = []
    n_dropped_oos = 0
    n_dropped_h20 = 0
    for e in episodes:
        if e["t0_date"] < t0_min:
            n_dropped_oos += 1
            continue
        feat = feats[e["ticker"]]
        if e["L_idx"] + 20 >= len(feat["struct"]):
            n_dropped_h20 += 1
            continue
        m = episode_metrics(feat, e["L_idx"], N, 20)
        if m is None:
            n_dropped_h20 += 1
            continue
        rows.append((e["ticker"], e["is_confirmed"], m))
    return rows, n_dropped_oos, n_dropped_h20

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 5b.X - Validacion out-of-sample de la candidata SOW ==")
    print("Protocolo: 35_protocolo_5bX_v3.md")
    print("Contrato:  01_contrato_semantico_v1_9.md")
    print(f"Candidata: N={CANDIDATA['N']} M={CANDIDATA['M']} "
          f"X_ATR={CANDIDATA['X_ATR']} Y_VOL={CANDIDATA['Y_VOL']}")
    print(f"Cutoff desarrollo: {CUTOFF_DATE.date()}")
    print(f"Universo OOS:      t0 >= {OOS_T0_MIN.date()}")
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
            "protocolo": "35_protocolo_5bX_v3.md",
            "contrato": "01_contrato_semantico_v1_9.md",
            "candidata": CANDIDATA,
            "cutoff_date": str(CUTOFF_DATE.date()),
            "oos_rule": f"t0 >= {OOS_T0_MIN.date()}",
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

    # Filtrar al bloque de validacion (t0 >= OOS_T0_MIN) con H20_complete
    rows, n_dropped_oos, n_dropped_h20 = collect_validation_rows(
        episodes, feats, CANDIDATA["N"], OOS_T0_MIN)
    n_conf = sum(1 for _, c, _ in rows if c)
    n_base = len(rows) - n_conf
    print(f"Episodios OOS con H20_complete: {len(rows)} "
          f"(conf={n_conf}, base={n_base})")
    print(f"  descartados por OOS (t0 < {OOS_T0_MIN.date()}): {n_dropped_oos}")
    print(f"  descartados por H20 censurado: {n_dropped_h20}")

    if n_conf < MIN_N_CONFIRMED_H20:
        print()
        print(f"BLOQUEADO. n_confirmed_H20_complete={n_conf} < "
              f"{MIN_N_CONFIRMED_H20}.")
        summary = {
            "schema": "wyckoff_5bX_summary_v1",
            "status": "BLOCKED_INSUFFICIENT_H20",
            "protocolo": "35_protocolo_5bX_v3.md",
            "contrato": "01_contrato_semantico_v1_9.md",
            "candidata": CANDIDATA,
            "cutoff_date": str(CUTOFF_DATE.date()),
            "oos_rule": f"t0 >= {OOS_T0_MIN.date()}",
            "preconditions": precond,
            "validation": {
                "n_episodes": len(rows),
                "n_confirmed_H20_complete": n_conf,
                "n_baseline_H20_complete": n_base,
                "n_dropped_oos": n_dropped_oos,
                "n_dropped_h20": n_dropped_h20,
            },
            "scenario": {
                "code": "D",
                "motivo": f"n_confirmed_H20_complete < {MIN_N_CONFIRMED_H20}",
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

    # Summary JSON (protocolo v3, seccion 8)
    summary = {
        "schema": "wyckoff_5bX_summary_v1",
        "status": "EXECUTED",
        "protocolo": "35_protocolo_5bX_v3.md",
        "contrato": "01_contrato_semantico_v1_9.md",
        "candidata": CANDIDATA,
        "cutoff_date": str(CUTOFF_DATE.date()),
        "oos_rule": f"t0 >= {OOS_T0_MIN.date()}",
        "preconditions": precond,
        "validation": {
            "n_episodes": len(rows),
            "n_confirmed_H20_complete": boot["n_confirmed"],
            "n_baseline_H20_complete": boot["n_baseline"],
            "n_tickers": boot["n_tickers"],
            "n_dropped_oos": n_dropped_oos,
            "n_dropped_h20": n_dropped_h20,
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
