"""Orquestador de la fase confirmatoria del protocolo SOW v4.1.

Secciones 8-14:
- Para cada uno de los 8 outer folds:
    - Regimen se calcula con datos hasta outer_train_end.
    - Seleccion INNER via inner.select_candidate_for_outer.
    - Evaluacion OUTER via outer.evaluate_outer (B20/B40/B50).
    - Placebos A y B via placebos.evaluate_placebo.
- Agrega resultados por fold.
- Aplica decision.decide con el gate de potencia que se le pasa.
- Congela el modo (seccion 14.2).

Regla bloqueante: aborta si el manifiesto no es valido (arbol sucio o
DATA_SHA256 no coincide).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import scripts.calibrate_wyckoff_sow_5b4bis as calib
from scripts.sow_v4 import config, decision, manifest, outer, placebos, regime
from scripts.sow_v4.episodes import precompute_base
from scripts.sow_v4.inner import select_candidate_for_outer


def _session_list(feats: dict) -> list:
    first_key = next(iter(feats))
    return list(feats[first_key]["dates"])


def _dates_in_range(session_dates: list, start, end) -> list:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return [d for d in session_dates if s <= d <= e]


def _load_sector_map(experiment_tickers) -> dict:
    """Carga el mapping ticker->sector del regimen de referencia.

    Seccion 3 v4.1 (C-1 modificada): el mapping se deriva de
    data/etf_holdings.csv y solo cubre tickers US con sector asignado.
    El resultado se usa para construir el REGIMEN y para Placebo B.
    NO se usa para filtrar el universo del experimento SOW.
    """
    return regime.load_sector_map(experiment_tickers=experiment_tickers)


def run_one_fold(
    fold_idx: int,
    feats: dict,
    starts_cache: dict,
    sector_returns: pd.DataFrame,
    sector_map: dict,
    session_dates: list,
    smoke: bool,
    skip_bootstrap: bool = False,
) -> dict:
    ts, te, vs, ve = config.OUTER_FOLDS[fold_idx]
    print(f"\n[FOLD {fold_idx}] TRAIN {ts}..{te}  TEST {vs}..{ve}")

    train_dates = _dates_in_range(session_dates, ts, te)
    test_dates = _dates_in_range(session_dates, vs, ve)
    if not train_dates or not test_dates:
        return {"ok": False, "reason": "sin_sesiones"}

    t0 = time.time()
    print("  [inner] buscando candidata entre 240 combos...")
    sel = select_candidate_for_outer(
        feats, starts_cache, sector_returns, fold_idx, train_dates,
    )
    if not sel.get("ok"):
        print(f"  [inner] sin candidata: {sel.get('reason')}")
        return {"ok": False, "reason": f"inner:{sel.get('reason')}"}
    best = sel["best"]
    print(f"  [inner] candidata={best['combo']} score={best['score']:.5f} "
          f"n_elegibles={best['n_elegibles']} (t={time.time()-t0:.1f}s)")

    t0 = time.time()
    print("  [outer] evaluando sobre TEST...")
    out = outer.evaluate_outer(
        best["combo"], feats, starts_cache, sector_returns,
        train_dates, test_dates, fold_idx,
        skip_bootstrap=skip_bootstrap,
    )
    if not out.get("ok"):
        print(f"  [outer] fallo: {out.get('reason')}")
        return {"ok": False, "reason": f"outer:{out.get('reason')}"}
    print(f"  [outer] rd_point stress={out['rd_point']['RD_stress']} "
          f"normal={out['rd_point']['RD_normal']} "
          f"pool={out['rd_point']['RD_pool']} (t={time.time()-t0:.1f}s)")

    if smoke:
        return {
            "ok": True,
            "fold_idx": fold_idx,
            "combo": best["combo"],
            "rd_point": out["rd_point"],
            "bootstraps": out["bootstraps"],
            "placebos": [],
            "smoke_evaluable": True,
        }

    # Placebos
    N, M, X, Y = best["combo"]
    log_vol = regime.compute_log_vol_lag1(regime.compute_vol_median(sector_returns))
    thr = regime.fit_thresholds(log_vol, ts, te)
    labels = regime.classify_regime(log_vol, thr)
    from scripts.sow_v4.episodes import build_episodes_for_combo, purge_train_frontier
    all_eps = build_episodes_for_combo(feats, starts_cache, N, M, X, Y, labels)
    train_start = pd.Timestamp(ts)
    train_end = pd.Timestamp(te)
    test_start = pd.Timestamp(vs)
    test_end = pd.Timestamp(ve)
    train_eps = [e for e in all_eps if train_start <= e["t0_date"] <= train_end]
    test_eps = [e for e in all_eps if test_start <= e["t0_date"] <= test_end]
    last_train_idx = int(np.where(np.array(session_dates) <= train_end)[0].max())
    train_eps = purge_train_frontier(train_eps, M, last_train_idx)

    t0 = time.time()
    print(f"  [placebos] P1 y P2 sobre fold {fold_idx}...")
    plcs = []
    for scheme in ("P1", "P2"):
        res = placebos.evaluate_placebo(
            train_eps, test_eps, scheme, fold_idx,
            None,
        )
        if res is None:
            print(f"    [placebo {scheme}] no evaluable")
            continue
        plcs.append(res)
        print(f"    [placebo {scheme}] q5={res['RD_placebo_q5']:.4f} "
              f"q95={res['RD_placebo_q95']:.4f} "
              f"p_rand={res['p_rand']:.4f} b2c_pass={res['b2c_pass']}")
    print(f"  [placebos] t={time.time()-t0:.1f}s")

    return {
        "ok": True,
        "fold_idx": fold_idx,
        "combo": best["combo"],
        "rd_point": out["rd_point"],
        "bootstraps": out["bootstraps"],
        "placebos": plcs,
    }


def run_all(smoke: bool, gate_universal: bool, gate_conditional: bool,
            gate_muestra_insuficiente: bool, allow_dirty: bool = False) -> dict:
    print("[1/4] Cargando dataset y precomputando feats...")
    df = pd.read_parquet(config.DATA_PARQUET)
    close = df.xs("Close", axis=1, level=0)
    cov = close.notna().sum()
    tickers = sorted(cov[cov >= config.MIN_OBS].index.tolist())
    calib.DATA_PARQUET = config.DATA_PARQUET
    feats, starts_cache = precompute_base(df, tickers)
    session_dates = _session_list(feats)
    print(f"      tickers={len(tickers)} feats={len(feats)} "
          f"sesiones={len(session_dates)}")

    print("[2/4] Cargando mapping sectorial del regimen de referencia...")
    sector_map = _load_sector_map(tickers)
    print(f"      regime_reference_count={len(sector_map)} "
          f"(de {len(tickers)} experimentales)")

    print("[3/4] Construyendo y validando manifiesto...")
    regime_tickers = list(sector_map.keys())
    m = manifest.build_manifest(
        experiment_tickers=tickers,
        regime_tickers=regime_tickers,
    )
    if allow_dirty and smoke:
        print("      [WARN] --allow-dirty con --smoke: saltando validacion de arbol")
    else:
        manifest.assert_manifest_valid(m)
    print(f"      DATA_SHA256={m['data_sha256'][:16]}... "
          f"tree_clean={m['code_tree_clean']}")
    if 'experiment_universe_count' in m:
        print(f"      experiment_universe={m['experiment_universe_count']} "
              f"regime_reference={m.get('regime_reference_count', 'N/A')}")

    print("[4/4] Regimen sectorial + outer folds...")
    sector_returns = regime.compute_sector_returns(df, sector_map)
    print(f"      sector_returns shape={sector_returns.shape}")

    fold_results = []
    placebo_results = []
    n_folds = len(config.OUTER_FOLDS)
    if smoke:
        print(f"      [smoke] procesando {n_folds} folds en modo ligero "
              f"(sin bootstrap, sin placebos)")
    for k in range(n_folds):
        res = run_one_fold(
            k, feats, starts_cache, sector_returns, sector_map,
            session_dates, smoke,
            skip_bootstrap=smoke,
        )
        if res.get("ok"):
            fold_results.append(res)
            placebo_results.extend(res.get("placebos", []))
        else:
            print(f"    fold {k} no evaluable: {res.get('reason')}")

    print(f"\n[evaluabilidad] N_eval = {len(fold_results)} / {n_folds} "
          f"(minimo protocolo: {config.DECISION_MIN_EVAL})")

    if smoke:
        # En smoke no aplicamos decision completa (faltan bootstraps y placebos).
        # Solo reportamos si N_eval cumple el minimo del protocolo.
        cumple = len(fold_results) >= config.DECISION_MIN_EVAL
        print(f"[smoke] N_eval cumple minimo: {cumple}")
        return {
            "manifest": m,
            "fold_results": fold_results,
            "placebo_results": placebo_results,
            "smoke": True,
            "n_eval": len(fold_results),
            "n_folds": n_folds,
            "cumple_minimo": cumple,
        }

    print("\n[decision]")
    dec = decision.decide(
        fold_results=fold_results,
        placebo_results=placebo_results,
        gate_universal=gate_universal,
        gate_conditional=gate_conditional,
        gate_muestra_insuficiente=gate_muestra_insuficiente,
    )
    print(f"  estado={dec['estado']}  motivo={dec['motivo']}")

    return {
        "manifest": m,
        "fold_results": fold_results,
        "placebo_results": placebo_results,
        "decision": dec,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="solo fold 0, sin placebos")
    ap.add_argument("--gate-universal", action="store_true")
    ap.add_argument("--gate-conditional", action="store_true")
    ap.add_argument("--gate-muestra-insuficiente", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="solo con --smoke: permite arbol sucio en desarrollo")
    args = ap.parse_args()

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = run_all(
        smoke=args.smoke,
        gate_universal=args.gate_universal,
        gate_conditional=args.gate_conditional,
        gate_muestra_insuficiente=args.gate_muestra_insuficiente,
        allow_dirty=args.allow_dirty,
    )

    path = config.OUT_DIR / ("validation_smoke.json" if args.smoke else "validation.json")
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\n[OK] escrito {path}")


if __name__ == "__main__":
    main()