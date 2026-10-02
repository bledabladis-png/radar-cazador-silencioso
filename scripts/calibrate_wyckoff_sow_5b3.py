#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibracion 5b.3 - Grid 240 de parametros SOW.

Protocolo: docs/auditoria/wyckoff/20_protocolo_fase_5b3.md (v3)
Contrato:  docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md
Dictamen:  2026-10-02 (respuesta a 21_expediente_5b3_aclaraciones.md)

Read-only. No modifica datos, config ni contrato.
Entrada:   data/stock_prices.parquet
Salida:
    outputs/audit/wyckoff_5b3_grid.csv
    outputs/audit/wyckoff_5b3_summary.json

Uso:
    py scripts/calibrate_wyckoff_sow_5b3.py
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


# --- Rutas ---
DATA_PARQUET = ROOT / "data" / "stock_prices.parquet"
OUT_DIR = ROOT / "outputs" / "audit"
OUT_GRID = OUT_DIR / "wyckoff_5b3_grid.csv"
OUT_SUMMARY = OUT_DIR / "wyckoff_5b3_summary.json"

# --- Universo / split (congelados) ---
MIN_OBS = 200
SPLIT_FRAC = 0.70

# --- Grid y horizontes (congelados) ---
HORIZONS = (10, 20, 40)
GRID_N = (20, 30, 40, 60)
GRID_M = (5, 10, 15, 20, 30)
GRID_X = (0.25, 0.50, 0.75, 1.00)
GRID_Y = (1.10, 1.20, 1.50)

# --- Constantes de control (fijas, ver contrato v1.8 §4.1) ---
PRECEDENT_WINDOW = 60
ATR_WINDOW = 20
STRUCT_DETERIORO = -0.10
T_NORM_STRONG = 0.30
PREC_STRUCT_STRONG = 0.30

# --- Guardrails y criterios ex-ante (protocolo §6) ---
D1_MIN_SOW = 100
D2_MIN_CONFIRMED = 20
D3_MIN_PCT_H20 = 0.50
D4_MAX_DELTA_PP = 30.0

# --- Metricas del outcome (protocolo §1-C2) ---
KEYS = ("struct_deterioration", "price_weakness", "below_support", "lower_low")


def load_dataset():
    if not DATA_PARQUET.exists():
        raise SystemExit(f"NO existe {DATA_PARQUET}")
    df = pd.read_parquet(DATA_PARQUET)
    close = df.xs("Close", axis=1, level=0)
    cov = close.notna().sum()
    tickers = sorted(cov[cov >= MIN_OBS].index.tolist())
    idx = df.index
    cutoff_pos = int(len(idx) * SPLIT_FRAC)
    date_cut = idx[cutoff_pos]
    return df, tickers, date_cut


def precompute_features(df, tickers):
    """Precalcula struct, t_norm, close/low/high/volume, candidate por ticker.

    Independiente del grid. Se calcula una vez.
    """
    feats = {}
    n_ok = 0
    for tk in tickers:
        try:
            tdf = w1.build_ticker_df(df, tk)
        except Exception:
            continue
        if tdf.empty or len(tdf) < MIN_OBS:
            continue
        try:
            _, struct_s, _, t_norm, _, _, _ = w1.wyckoff_score(tdf, tk)
        except Exception:
            continue
        struct_clean = struct_s.dropna()
        if struct_clean.empty:
            continue

        # struct_max en [t-W, t-1] con W=PRECEDENT_WINDOW
        struct_max = (
            struct_s.rolling(PRECEDENT_WINDOW, min_periods=PRECEDENT_WINDOW)
            .max()
            .shift(1)
        )
        candidate = (
            (struct_s < STRUCT_DETERIORO)
            & (t_norm > -T_NORM_STRONG)
            & (struct_max > PREC_STRUCT_STRONG)
        ).fillna(False).astype(bool)

        # Soporte rodante para below_support / lower_low: se calcula por N en el grid.
        # Precalculamos low_series; el rolling por N se hace bajo demanda.
        feats[tk] = {
            "tdf": tdf,
            "dates": tdf.index,
            "struct": struct_s,
            "t_norm": t_norm,
            "close": tdf["Close"],
            "low": tdf["Low"],
            "high": tdf["High"],
            "volume": tdf["Volume"],
            "candidate": candidate,
        }
        n_ok += 1
    return feats, n_ok

def find_candidate_starts(candidate: pd.Series):
    """Posiciones donde candidate=True AND candidate_{t-1}=False. Por ticker."""
    c = candidate.fillna(False).astype(bool).values
    if len(c) == 0:
        return np.array([], dtype=int)
    prev = np.concatenate([[False], c[:-1]])
    return np.where(c & ~prev)[0]


def compute_sow_cache(feats, N, X, Y):
    """SOW por ticker para un (N,X,Y). Se reutiliza en los 5 M."""
    cache = {}
    for tk, feat in feats.items():
        try:
            sow = w1.detect_sow(feat["tdf"], tk, window=N, x_atr=X, y_vol=Y)
        except (ValueError, KeyError, TypeError):
            sow = None
        cache[tk] = sow.values if sow is not None else None
    return cache


def first_sow_in_window(sow_arr, t0_idx, max_age):
    """Primer indice >= t0_idx con SOW=1 dentro de [t0_idx, t0_idx + max_age]. None si no hay."""
    if sow_arr is None:
        return None
    a = t0_idx
    b = min(t0_idx + max_age, len(sow_arr) - 1)
    if b < a:
        return None
    window = sow_arr[a:b + 1]
    hits = np.where(window > 0)[0]
    if len(hits) == 0:
        return None
    return a + int(hits[0])


def episode_metrics(feat, t_idx, N):
    """Metricas del outcome en t_idx para H10/H20/H40. None si t_idx+max(H) fuera."""
    n = len(feat["struct"])
    struct_t = feat["struct"].iloc[t_idx]
    if pd.isna(struct_t):
        return None
    close_t = feat["close"].iloc[t_idx]
    if pd.isna(close_t):
        return None

    # Soporte: rolling min de Low en [t-N, t-1] (contrato v1.8 §5.4)
    # base_low: rolling min de Close en [t-N, t-1] (protocolo §1-C2 lower_low)
    lo = max(0, t_idx - N)
    if t_idx - lo < 1:
        return None
    support_t = feat["low"].iloc[lo:t_idx].min()
    base_low = feat["close"].iloc[lo:t_idx].min()

    m = {
        "struct_deterioration": {},
        "price_weakness": {},
        "below_support": {},
        "lower_low": {},
    }
    for h in HORIZONS:
        t_h = t_idx + h
        if t_h >= n:
            for k in KEYS:
                m[k][h] = None
            continue
        struct_h = feat["struct"].iloc[t_h]
        close_h = feat["close"].iloc[t_h]
        # 1. struct_deterioration
        if pd.isna(struct_h):
            m["struct_deterioration"][h] = None
        else:
            m["struct_deterioration"][h] = int(float(struct_h) < float(struct_t))
        # 2. price_weakness
        if pd.isna(close_h):
            m["price_weakness"][h] = None
        else:
            m["price_weakness"][h] = int(float(close_h) < float(close_t) * 0.98)
        # 3. below_support
        if pd.isna(close_h) or pd.isna(support_t):
            m["below_support"][h] = None
        else:
            m["below_support"][h] = int(float(close_h) < float(support_t))
        # 4. lower_low
        seg = feat["close"].iloc[t_idx + 1: t_h + 1]
        if seg.empty or pd.isna(base_low):
            m["lower_low"][h] = None
        else:
            m["lower_low"][h] = int(float(seg.min()) < float(base_low))
    return m


def build_episodes_for_M(feats, sow_cache, starts_cache, N, M):
    """Devuelve lista de dicts por episodio para un (N,M) dado, reutilizando sow_cache.

    Campo 'ticker' y 't0_idx' y 'sow_idx' + 'is_confirmed'.
    No calcula metricas aqui (se hacen por horizonte en aggregate).
    """
    eps = []
    for tk, feat in feats.items():
        sow_arr = sow_cache.get(tk)
        starts = starts_cache.get(tk, np.array([], dtype=int))
        for t0_idx in starts:
            sow_idx = first_sow_in_window(sow_arr, int(t0_idx), M)
            is_conf = sow_idx is not None
            eps.append({
                "ticker": tk,
                "t0_idx": int(t0_idx),
                "t0_date": feat["dates"][int(t0_idx)],
                "sow_idx": int(sow_idx) if sow_idx is not None else None,
                "is_confirmed": is_conf,
            })
    return eps

def split_episodes(episodes, date_cut):
    """Divide episodios por fecha de entrada (t0) vs corte global.

    - calibracion: t0 <= date_cut
    - holdout:     t0 >  date_cut
    """
    cal, hold = [], []
    for e in episodes:
        if e["t0_date"] <= date_cut:
            cal.append(e)
        else:
            hold.append(e)
    return cal, hold


def filter_boundary(episodes, feats, block_end_date):
    """Marca por horizonte si t_anchor + h <= fin_bloque.

    Ancla: t_sow si confirmado, t0 si no. Devuelve dict de listas por horizonte.
    """
    out = {h: [] for h in HORIZONS}
    for e in episodes:
        anchor_idx = e["sow_idx"] if e["is_confirmed"] else e["t0_idx"]
        tk = e["ticker"]
        dates = feats[tk]["dates"]
        n = len(dates)
        for h in HORIZONS:
            t_h = anchor_idx + h
            if t_h >= n:
                continue
            if dates[t_h] <= block_end_date:
                out[h].append(e)
    return out


def aggregate_episodes(episodes, feats, N, block_end_date):
    """Agrega por horizonte, respetando boundary temporal.

    Optimizacion (post-smoke): metricas por episodio calculadas UNA vez
    y reutilizadas en los 3 horizontes.
    """
    n_ep = len(episodes)
    n_conf = sum(1 for e in episodes if e["is_confirmed"])

    result = {
        "n_candidate_events": n_ep,
        "n_confirmed": n_conf,
        "tasa_candidate_confirmed": (n_conf / n_ep) if n_ep > 0 else None,
        "by_horizon": {},
    }

    # Precalcular metricas y datos de boundary por episodio (una sola vez)
    cached = []
    for e in episodes:
        anchor_idx = e["sow_idx"] if e["is_confirmed"] else e["t0_idx"]
        tk = e["ticker"]
        dates = feats[tk]["dates"]
        m = episode_metrics(feats[tk], anchor_idx, N)
        cached.append((e, m, dates, anchor_idx, len(dates)))

    for h in HORIZONS:
        n_out_conf = 0
        n_out_base = 0
        vals_conf = {k: [] for k in KEYS}
        vals_base = {k: [] for k in KEYS}
        for e, m, dates, anchor_idx, n_total in cached:
            if m is None:
                continue
            t_h = anchor_idx + h
            if t_h >= n_total:
                continue
            if dates[t_h] > block_end_date:
                continue
            group = "conf" if e["is_confirmed"] else "base"
            if group == "conf":
                n_out_conf += 1
                for k in KEYS:
                    v = m[k].get(h)
                    if v is not None:
                        vals_conf[k].append(v)
            else:
                n_out_base += 1
                for k in KEYS:
                    v = m[k].get(h)
                    if v is not None:
                        vals_base[k].append(v)

        pct_conf = {k: (float(np.mean(vals_conf[k])) if vals_conf[k] else None) for k in KEYS}
        pct_base = {k: (float(np.mean(vals_base[k])) if vals_base[k] else None) for k in KEYS}
        lift = {}
        for k in KEYS:
            if pct_conf[k] is None or pct_base[k] is None:
                lift[k] = None
            else:
                lift[k] = pct_conf[k] - pct_base[k]

        result["by_horizon"][h] = {
            "n_out_conf": n_out_conf,
            "n_out_base": n_out_base,
            "pct_conf": pct_conf,
            "pct_base": pct_base,
            "lift": lift,
        }
    return result

def count_sow(feats, N, X, Y):
    """n_sow_raw y n_sow_unique (agrupado por ticker, |t_i - t_{i-1}| < N)."""
    n_raw = 0
    n_unique = 0
    for tk, feat in feats.items():
        try:
            sow = w1.detect_sow(feat["tdf"], tk, window=N, x_atr=X, y_vol=Y)
        except (ValueError, KeyError, TypeError):
            continue
        hits = np.where(sow.values > 0)[0]
        n_raw += len(hits)
        if len(hits) == 0:
            continue
        unique = 1
        for i in range(1, len(hits)):
            if hits[i] - hits[i - 1] >= N:
                unique += 1
        n_unique += unique
    return n_raw, n_unique


def evaluate_D1_D4(row):
    """Aplica D1-D4 ex-ante sobre una fila del grid.

    D1: n_sow_total >= 100
    D2: n_confirmed >= 20
    D3: struct_deterioration_H20 >= 50% (calibracion)
    D4: neighbour_confirmation_delta_pp <= 30 pp (se evalua fuera, contra el grid)
    """
    d1 = row["n_sow_raw"] >= D1_MIN_SOW
    d2 = row["cal"]["n_confirmed"] >= D2_MIN_CONFIRMED
    pct_h20 = row["cal"]["by_horizon"][20]["pct_conf"]["struct_deterioration"]
    d3 = (pct_h20 is not None) and (pct_h20 >= D3_MIN_PCT_H20)
    return d1, d2, d3


def run_grid(feats, date_cut):
    """Itera grid. SOW se calcula 1 vez por (N,X,Y). Episodios por M se construyen desde cache."""
    print(f"Universo: {len(feats)} tickers con features")
    print(f"Fecha de corte 70/30: {date_cut.date()}")

    # Cache de candidate starts (independiente del grid)
    starts_cache = {tk: find_candidate_starts(feat["candidate"]) for tk, feat in feats.items()}
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Episodios candidatos totales (universo completo): {total_starts}")

    # Tope temporal para calibracion: date_cut.
    # Tope para holdout: fecha final del dataset.
    last_date = max(feat["dates"].max() for feat in feats.values())

    rows = []
    total_combos = len(GRID_N) * len(GRID_X) * len(GRID_Y) * len(GRID_M)
    n_done = 0
    t0 = time.time()

    for N in GRID_N:
        for X in GRID_X:
            for Y in GRID_Y:
                sow_cache = compute_sow_cache(feats, N, X, Y)
                n_raw, n_unique = count_sow(feats, N, X, Y)
                for M in GRID_M:
                    episodes = build_episodes_for_M(feats, sow_cache, starts_cache, N, M)
                    ep_cal, ep_hold = split_episodes(episodes, date_cut)
                    agg_cal = aggregate_episodes(ep_cal, feats, N, date_cut)
                    agg_hold = aggregate_episodes(ep_hold, feats, N, last_date)
                    rows.append({
                        "N": N, "M": M, "X_ATR": X, "Y_VOL": Y,
                        "n_sow_raw": n_raw,
                        "n_sow_unique": n_unique,
                        "cal": agg_cal,
                        "hold": agg_hold,
                    })
                    n_done += 1
                    if n_done % 20 == 0:
                        el = time.time() - t0
                        print(f"  [{n_done}/{total_combos}] {el:.1f}s", flush=True)
    el = time.time() - t0
    print(f"Grid completo: {n_done} combinaciones en {el:.1f}s")
    return rows


def add_D_flags(rows):
    """Añade D1/D2/D3 por fila y D4 (neighbour_confirmation_delta_pp) contra vecinos inmediatos de la grid.

    Vecinos: N±10, M±5, X±0.25, Y±0.10 si están dentro del grid. Se toma el max de |delta_pp|
    de tasa_candidate_confirmed.
    """
    index = {(r["N"], r["M"], r["X_ATR"], r["Y_VOL"]): r for r in rows}
    for r in rows:
        d1, d2, d3 = evaluate_D1_D4(r)
        r["D1"] = bool(d1)
        r["D2"] = bool(d2)
        r["D3"] = bool(d3)

        # D4: variacion maxima en tasa_candidate_confirmed respecto a vecinos inmediatos
        base = r["cal"]["tasa_candidate_confirmed"]
        deltas = []
        neighbours = []
        for dN in (-10, 10):
            neighbours.append((r["N"] + dN, r["M"], r["X_ATR"], r["Y_VOL"]))
        for dM in (-5, 5):
            neighbours.append((r["N"], r["M"] + dM, r["X_ATR"], r["Y_VOL"]))
        for dX in (-0.25, 0.25):
            neighbours.append((r["N"], r["M"], round(r["X_ATR"] + dX, 2), r["Y_VOL"]))
        for dY in (-0.10, 0.10):
            neighbours.append((r["N"], r["M"], r["X_ATR"], round(r["Y_VOL"] + dY, 2)))
        for key in neighbours:
            if key in index and base is not None:
                nb = index[key]["cal"]["tasa_candidate_confirmed"]
                if nb is not None:
                    deltas.append(abs(nb - base) * 100.0)  # a pp
        if deltas and base is not None:
            max_delta = max(deltas)
            r["neighbour_confirmation_delta_pp"] = float(max_delta)
            r["D4"] = bool(max_delta <= D4_MAX_DELTA_PP)
        else:
            r["neighbour_confirmation_delta_pp"] = None
            r["D4"] = None  # no evaluable (borde del grid o sin vecinos)
    return rows

def _short_key(k):
    return {
        "struct_deterioration": "struct",
        "price_weakness": "price",
        "below_support": "below",
        "lower_low": "lower",
    }[k]


def flatten_row(r):
    """Convierte un row anidado en dict plano para CSV."""
    flat = {
        "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
        "n_sow_raw": r["n_sow_raw"],
        "n_sow_unique": r["n_sow_unique"],
        "D1": r.get("D1"), "D2": r.get("D2"), "D3": r.get("D3"), "D4": r.get("D4"),
        "neighbour_confirmation_delta_pp": r.get("neighbour_confirmation_delta_pp"),
    }
    for scope in ("cal", "hold"):
        agg = r[scope]
        flat[f"{scope}_n_candidate_events"] = agg["n_candidate_events"]
        flat[f"{scope}_n_confirmed"] = agg["n_confirmed"]
        flat[f"{scope}_tasa_candidate_confirmed"] = agg["tasa_candidate_confirmed"]
        for h in HORIZONS:
            bh = agg["by_horizon"][h]
            flat[f"{scope}_n_out_conf_H{h}"] = bh["n_out_conf"]
            flat[f"{scope}_n_out_base_H{h}"] = bh["n_out_base"]
            for k in KEYS:
                sk = _short_key(k)
                flat[f"{scope}_pct_conf_{sk}_H{h}"] = bh["pct_conf"][k]
                flat[f"{scope}_pct_base_{sk}_H{h}"] = bh["pct_base"][k]
                flat[f"{scope}_lift_{sk}_H{h}"] = bh["lift"][k]
    return flat


def select_candidate(rows):
    """Aplica criterios ex-ante del protocolo §6.2.

    Retorna (elegido, motivo) o (None, motivo) si ninguno pasa D1-D4.
    """
    pasan = [r for r in rows if r.get("D1") and r.get("D2") and r.get("D3") and r.get("D4") is True]
    if not pasan:
        return None, "ninguna combinacion cumple D1-D4"

    # Paso 1: mayor struct_deterioration_H20 en calibracion
    def key_h20(r):
        return r["cal"]["by_horizon"][20]["pct_conf"]["struct_deterioration"] or 0.0
    max_h20 = max(key_h20(r) for r in pasan)
    top = [r for r in pasan if abs(key_h20(r) - max_h20) < 1e-9]

    # Desempate 1: mayor X_ATR (dentro de top)
    # Desempate 2: mayor Y_VOL
    # Desempate 3: N ascendente
    # Desempate 4: M ascendente
    top.sort(key=lambda r: (-r["X_ATR"], -r["Y_VOL"], r["N"], r["M"]))
    elegido = top[0]
    return elegido, f"{len(pasan)} combinaciones pasan D1-D4; elegido por ranking §6.2"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 5b.3 - Calibracion SOW ==")
    df, tickers, date_cut = load_dataset()
    print(f"Tickers en dataset: {len(tickers)}")
    feats, n_ok = precompute_features(df, tickers)
    print(f"Features construidas: {n_ok}")

    rows = run_grid(feats, date_cut)
    rows = add_D_flags(rows)

    # Serializar grid
    flat_rows = [flatten_row(r) for r in rows]
    grid_df = pd.DataFrame(flat_rows)
    grid_df.to_csv(OUT_GRID, index=False)
    print(f"Grid escrito: {OUT_GRID} ({len(grid_df)} filas)")

    # Seleccion
    elegido, motivo = select_candidate(rows)
    print(f"Seleccion: {motivo}")
    if elegido is not None:
        print(f"  -> N={elegido['N']} M={elegido['M']} X_ATR={elegido['X_ATR']} Y_VOL={elegido['Y_VOL']}")

    # Resumen
    summary = {
        "schema": "wyckoff_5b3_summary_v1",
        "date_cut": str(date_cut.date()),
        "n_tickers_universe": len(tickers),
        "n_tickers_with_features": n_ok,
        "n_combos": len(rows),
        "grid": {
            "N": list(GRID_N), "M": list(GRID_M),
            "X_ATR": list(GRID_X), "Y_VOL": list(GRID_Y),
        },
        "constants": {
            "PRECEDENT_WINDOW": PRECEDENT_WINDOW,
            "ATR_WINDOW": ATR_WINDOW,
            "HORIZONS": list(HORIZONS),
        },
        "criteria": {
            "D1_MIN_SOW": D1_MIN_SOW,
            "D2_MIN_CONFIRMED": D2_MIN_CONFIRMED,
            "D3_MIN_PCT_H20": D3_MIN_PCT_H20,
            "D4_MAX_DELTA_PP": D4_MAX_DELTA_PP,
        },
        "selection": {
            "motivo": motivo,
            "elegido": None if elegido is None else {
                "N": elegido["N"], "M": elegido["M"],
                "X_ATR": elegido["X_ATR"], "Y_VOL": elegido["Y_VOL"],
            },
            "cal": None if elegido is None else elegido["cal"],
            "hold": None if elegido is None else elegido["hold"],
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Resumen escrito: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()