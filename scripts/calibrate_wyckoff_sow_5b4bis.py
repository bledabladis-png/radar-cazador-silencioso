#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibracion 5b.4-bis - Grid 240 con 3 bloques y delta_hetero.

Protocolo: docs/auditoria/wyckoff/30_protocolo_fase_5b4bis_v3.md (v3)
Contrato:  docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md
Dictamen:  29_dictamen_5b4_v2.md

Read-only. NO modifica datos, config ni contrato.
Entrada:   data/stock_prices.parquet
Salida:
    outputs/audit/wyckoff_5b4bis_grid.csv
    outputs/audit/wyckoff_5b4bis_bootstrap.csv
    outputs/audit/wyckoff_5b4bis_hetero.csv
    outputs/audit/wyckoff_5b4bis_summary.json

Uso:
    py scripts/calibrate_wyckoff_sow_5b4bis.py
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
OUT_GRID = OUT_DIR / "wyckoff_5b4bis_grid.csv"
OUT_BOOT = OUT_DIR / "wyckoff_5b4bis_bootstrap.csv"
OUT_SUMMARY = OUT_DIR / "wyckoff_5b4bis_summary.json"
OUT_HETERO = OUT_DIR / "wyckoff_5b4bis_hetero.csv"

# --- Universo ---
MIN_OBS = 200

# --- Grid y horizontes ---
HORIZONS = (10, 20, 40)
GRID_N = (20, 30, 40, 60)
GRID_M = (5, 10, 15, 20, 30)
GRID_X = (0.25, 0.50, 0.75, 1.00)
GRID_Y = (1.10, 1.20, 1.50)

# --- Constantes de control (fijas) ---
PRECEDENT_WINDOW = 60
ATR_WINDOW = 20
STRUCT_DETERIORO = -0.10
T_NORM_STRONG = 0.30
PREC_STRUCT_STRONG = 0.30

# --- Bloques temporales (por fecha del landmark L) ---
# v3: 3 bloques predefinidos por calendario (protocolo 5b.4-bis seccion 3).
BLOCKS = (
    ("P1", pd.Timestamp("2021-10-01"), pd.Timestamp("2023-06-30")),
    ("P2", pd.Timestamp("2023-07-01"), pd.Timestamp("2025-03-31")),
    ("P3", pd.Timestamp("2025-04-01"), pd.Timestamp("2026-10-01")),
)

# --- Criterios duros (protocolo v3 seccion 5.1) ---
D1_MIN_CONFIRMED = 20
D1_MIN_BASELINE = 20
D2_LOWER_CI_STRICT = True   # lower_CI > 0 (filtro de desarrollo, no confirmatorio)
D3_FRACTION_REQUIRED = 2.0 / 3.0  # al menos ceil(2/3) de bloques evaluables con lift > 0

# --- Bootstrap ---
BOOT_B = 2000
BOOT_SEED = 20261002

# --- Metricas del outcome ---
KEYS = ("struct_deterioration", "price_weakness", "below_support", "lower_low")


def load_dataset():
    if not DATA_PARQUET.exists():
        raise SystemExit(f"NO existe {DATA_PARQUET}")
    df = pd.read_parquet(DATA_PARQUET)
    close = df.xs("Close", axis=1, level=0)
    cov = close.notna().sum()
    tickers = sorted(cov[cov >= MIN_OBS].index.tolist())
    return df, tickers


def precompute_features(df, tickers):
    """Precalcula struct, t_norm, close/low/high/volume, candidate por ticker."""
    feats = {}
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
    return feats


def block_of(date):
    """Devuelve la etiqueta del bloque que contiene date, o None."""
    for label, a, b in BLOCKS:
        if a <= date <= b:
            return label
    return None

def find_candidate_starts(candidate: pd.Series):
    """Posiciones donde candidate=True AND candidate_{t-1}=False."""
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


def first_sow_in_window(sow_arr, start_idx, end_idx):
    """Primer indice en [start_idx, end_idx] con SOW=1. None si no hay."""
    if sow_arr is None:
        return None
    a = int(start_idx)
    b = min(int(end_idx), len(sow_arr) - 1)
    if b < a or a < 0:
        return None
    window = sow_arr[a:b + 1]
    hits = np.where(window > 0)[0]
    if len(hits) == 0:
        return None
    return a + int(hits[0])


def build_episodes_landmark(feats, sow_cache, starts_cache, N, M,
                             min_gap_sessions=None):
    """Construye episodios con landmark L = t0 + M.

    - confirmed = SOW en [t0+1, L]
    - baseline  = no SOW en [t0+1, L]
    - L se calcula como indice; se descarta si L >= len(dates).
    - Censura: elegible para H solo si L+H < len(dates).
      El filtro de horizonte se aplica en la fase de agregacion
      (depende de h).

    Devuelve lista de dicts:
        {ticker, t0_idx, t0_date, L_idx, L_date, sow_idx, is_confirmed,
         block_L}
    """
    eps = []
    for tk, feat in feats.items():
        sow_arr = sow_cache.get(tk)
        starts = starts_cache.get(tk, np.array([], dtype=int))
        dates = feat["dates"]
        n = len(dates)
        last_kept_t0_idx = None
        for t0_idx in starts:
            t0_idx = int(t0_idx)
            if (min_gap_sessions is not None
                    and last_kept_t0_idx is not None
                    and t0_idx - last_kept_t0_idx < min_gap_sessions):
                continue
            L_idx = t0_idx + M
            if L_idx >= n:
                continue
            # Ventana [t0+1, L]. Si M == 0, ventana vacia -> siempre baseline.
            if M >= 1:
                sow_idx = first_sow_in_window(sow_arr, t0_idx + 1, L_idx)
            else:
                sow_idx = None
            is_conf = sow_idx is not None
            L_date = dates[L_idx]
            block = block_of(L_date)
            eps.append({
                "ticker": tk,
                "t0_idx": t0_idx,
                "t0_date": dates[t0_idx],
                "L_idx": L_idx,
                "L_date": L_date,
                "sow_idx": int(sow_idx) if sow_idx is not None else None,
                "is_confirmed": is_conf,
                "block_L": block,
            })
            last_kept_t0_idx = t0_idx
    return eps


def episode_metrics(feat, L_idx, N, h):
    """Metricas del outcome para un horizonte h, ancladas en L_idx.

    Devuelve dict {key: int|None} solo para el horizonte h.
    None si L_idx + h fuera de la serie o valores necesarios no validos.
    """
    n = len(feat["struct"])
    L_h = L_idx + h
    if L_h >= n:
        return None
    struct_L = feat["struct"].iloc[L_idx]
    if pd.isna(struct_L):
        return None
    close_L = feat["close"].iloc[L_idx]
    if pd.isna(close_L):
        return None

    # support[L] ex-ante: rolling min Low en [L-N, L-1]
    lo = max(0, L_idx - N)
    if L_idx - lo < 1:
        return None
    support_L = feat["low"].iloc[lo:L_idx].min()
    base_low_L = feat["close"].iloc[lo:L_idx].min()

    struct_h = feat["struct"].iloc[L_h]
    close_h = feat["close"].iloc[L_h]

    out = {
        "struct_deterioration": None,
        "price_weakness": None,
        "below_support": None,
        "lower_low": None,
    }
    if not pd.isna(struct_h):
        out["struct_deterioration"] = int(float(struct_h) < float(struct_L))
    if not pd.isna(close_h):
        out["price_weakness"] = int(float(close_h) < float(close_L) * 0.98)
        if not pd.isna(support_L):
            out["below_support"] = int(float(close_h) < float(support_L))
    seg = feat["close"].iloc[L_idx + 1: L_h + 1]
    if not seg.empty and not pd.isna(base_low_L):
        out["lower_low"] = int(float(seg.min()) < float(base_low_L))
    return out

def aggregate_episodes(episodes, feats, N, h):
    """Agrega episodios por bloque + total, para un horizonte h.

    Devuelve:
        {
          'by_block': {
             'A': {'n_conf': int, 'n_base': int,
                   'vals_conf': {key: [v]}, 'vals_base': {key: [v]},
                   'rows': [(ticker, is_confirmed, {key: v})]},
             ...
             'ALL': idem
          }
        }

    Cada episodio se cuenta en su bloque (por L_date).
    La censura ya se aplica al pedir episode_metrics(L_idx, h): si
    L+h fuera de serie, devuelve None y el episodio no se cuenta.
    """
    out = {"by_block": {}}
    for label in [b[0] for b in BLOCKS] + ["ALL"]:
        out["by_block"][label] = {
            "n_conf": 0,
            "n_base": 0,
            "vals_conf": {k: [] for k in KEYS},
            "vals_base": {k: [] for k in KEYS},
            "rows": [],
        }

    for e in episodes:
        tk = e["ticker"]
        feat = feats[tk]
        m = episode_metrics(feat, e["L_idx"], N, h)
        if m is None:
            continue
        group = "conf" if e["is_confirmed"] else "base"
        blocks_to_inc = []
        if e["block_L"] is not None:
            blocks_to_inc.append(e["block_L"])
        blocks_to_inc.append("ALL")
        for b in blocks_to_inc:
            bucket = out["by_block"][b]
            if group == "conf":
                bucket["n_conf"] += 1
            else:
                bucket["n_base"] += 1
            for k in KEYS:
                v = m[k]
                if v is not None:
                    if group == "conf":
                        bucket["vals_conf"][k].append(v)
                    else:
                        bucket["vals_base"][k].append(v)
            bucket["rows"].append((tk, e["is_confirmed"], m))
    return out


def pct_or_none(vals):
    return float(np.mean(vals)) if vals else None


def lift_or_none(vals_conf, vals_base):
    pc = pct_or_none(vals_conf)
    pb = pct_or_none(vals_base)
    if pc is None or pb is None:
        return None, pc, pb
    return pc - pb, pc, pb


def summarize_block(bucket):
    """Convierte un bucket en resumen con lifts."""
    res = {
        "n_conf": bucket["n_conf"],
        "n_base": bucket["n_base"],
    }
    for k in KEYS:
        lift, pc, pb = lift_or_none(bucket["vals_conf"][k], bucket["vals_base"][k])
        res[f"pct_conf_{k}"] = pc
        res[f"pct_base_{k}"] = pb
        res[f"lift_{k}"] = lift
    return res


def run_grid(feats, starts_cache, date_cut=None, min_gap_sessions=None):
    """Itera grid (N, M, X, Y). Devuelve lista de rows."""
    del date_cut  # no se usa: split diferido a 5b.X
    total_combos = len(GRID_N) * len(GRID_X) * len(GRID_Y) * len(GRID_M)
    rows = []
    n_done = 0
    t0 = time.time()

    for N in GRID_N:
        for X in GRID_X:
            for Y in GRID_Y:
                sow_cache = compute_sow_cache(feats, N, X, Y)
                for M in GRID_M:
                    episodes = build_episodes_landmark(
                        feats, sow_cache, starts_cache, N, M,
                        min_gap_sessions=min_gap_sessions,
                    )
                    # Precalcular metricas por horizonte (una sola vez)
                    # Auditor 2026-10-05 seccion 4: fuente unica de verdad.
                    # n_confirmed/n_baseline se calculan sobre ANALYSIS_ROWS
                    # (H completo), no sobre episodes crudos. Elimina la
                    # ambiguedad 635/629 documentada en el dictamen.
                    analysis_H20 = build_analysis_rows(episodes, feats, N, h=20)
                    row = {
                        "N": N, "M": M, "X_ATR": X, "Y_VOL": Y,
                        "n_episodes_raw": analysis_H20["n_episodes_raw"],
                        "n_dropped_h_censored": analysis_H20["n_dropped_h_censored"],
                        "n_dropped_metrics_none": analysis_H20["n_dropped_metrics_none"],
                        "n_analysis_rows": analysis_H20["n_analysis_rows"],
                        "n_confirmed": analysis_H20["n_confirmed_H_complete"],
                        "n_baseline": analysis_H20["n_baseline_H_complete"],
                        "by_horizon": {},
                    }
                    for h in HORIZONS:
                        agg = aggregate_episodes(episodes, feats, N, h)
                        row["by_horizon"][h] = {
                            label: summarize_block(agg["by_block"][label])
                            for label in agg["by_block"]
                        }
                        # Guardamos rows completos para bootstrap de H20
                        if h == 20:
                            row["_rows_H20"] = agg["by_block"]["ALL"]["rows"]
                            row["_rows_by_block_H20"] = {
                                label: agg["by_block"][label]["rows"]
                                for label in agg["by_block"]
                            }
                    rows.append(row)
                    n_done += 1
                    if n_done % 20 == 0:
                        el = time.time() - t0
                        print(f"  [{n_done}/{total_combos}] {el:.1f}s", flush=True)
    el = time.time() - t0
    print(f"Grid completo: {n_done} combinaciones en {el:.1f}s")
    return rows

def bootstrap_lift_H20(rows, B=BOOT_B, seed=BOOT_SEED):
    """Bootstrap por ticker-cluster del lift struct_deterioration_H20.

    rows: lista de (ticker, is_confirmed, {key: v}) para H20, bloque ALL.
    Remuestrea tickers completos, conservando todos sus episodios.
    Devuelve dict con:
        lift_point, lower_ci, upper_ci, n_tickers, n_episodes,
        n_confirmed, n_baseline, n_boot_validos
    """
    if not rows:
        return {
            "lift_point": None, "lower_ci": None, "upper_ci": None,
            "n_tickers": 0, "n_episodes": 0, "n_confirmed": 0,
            "n_baseline": 0, "n_boot_validos": 0,
        }
    # Agrupar por ticker
    by_tk = {}
    for tk, is_conf, m in rows:
        v = m.get("struct_deterioration")
        if v is None:
            continue
        by_tk.setdefault(tk, []).append((is_conf, v))
    tickers = list(by_tk.keys())
    if not tickers:
        # Auditor 2026-10-05 §16: rows no vacio pero todos con
        # struct_deterioration=None -> by_tk vacio. Cerrar camino.
        return {
            "lift_point": None, "lower_ci": None, "upper_ci": None,
            "p_one_sided": None,
            "n_tickers": 0, "n_episodes": len(rows), "n_confirmed": 0,
            "n_baseline": 0, "n_boot_validos": 0,
        }

    def _lift_from_subset(tks):
        n_c = n_b = s_c = s_b = 0
        for tk in tks:
            for is_conf, v in by_tk[tk]:
                if is_conf:
                    n_c += 1
                    s_c += v
                else:
                    n_b += 1
                    s_b += v
        if n_c == 0 or n_b == 0:
            return None
        return (s_c / n_c) - (s_b / n_b)

    lift_point = _lift_from_subset(tickers)
    n_conf = sum(1 for tk in tickers for (c, _) in by_tk[tk] if c)
    n_base = sum(1 for tk in tickers for (c, _) in by_tk[tk] if not c)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        sample = rng.choice(tickers, size=len(tickers), replace=True)
        v = _lift_from_subset(list(sample))
        if v is not None:
            boots.append(v)
    if len(boots) < 50:
        # Auditor 2026-10-05 §16: defensa contra rows validas pero todas
        # con struct_deterioration=None -> boots vacio. Sin esto,
        # np.percentile([], ...) es un camino de error.
        lower = upper = None
        p_one_sided = None
    else:
        lower = float(np.percentile(boots, 2.5))
        upper = float(np.percentile(boots, 97.5))
        # p-valor bootstrap one-sided (auditor 2026-10-05 seccion 9).
        # H0: lift <= 0, Ha: lift > 0.
        # Recentrar la distribucion bootstrap al lift_point y contar
        # cuantos valores recentrados >= lift_point.
        boots_arr = np.asarray(boots, dtype=float)
        boots_shifted = boots_arr - float(lift_point)
        n_ge = int((boots_shifted >= float(lift_point)).sum())
        p_one_sided = (1.0 + n_ge) / (1.0 + len(boots_arr))

    return {
        "lift_point": lift_point,
        "lower_ci": lower,
        "upper_ci": upper,
        "p_one_sided": p_one_sided,
        "n_tickers": len(tickers),
        "n_episodes": len(rows),
        "n_confirmed": n_conf,
        "n_baseline": n_base,
        "n_boot_validos": len(boots),
    }


def block_is_evaluable(block_summary):
    """Un bloque es evaluable para H20 si tiene muestra suficiente."""
    return (
        block_summary["n_conf"] >= D1_MIN_CONFIRMED
        and block_summary["n_base"] >= D1_MIN_BASELINE
    )


def evaluate_row(row, boot_res):
    """Aplica D1-D3 sobre una fila del grid (protocolo v3 seccion 5.1).

    D1: n_confirmed_H20_complete >= 20 AND n_baseline_H20_complete >= 20
        (en ALL, H20). La metrica se computa sobre ANALYSIS_ROWS
        (episodios con H completo). Coherente con 5b.X §2.1.
    D2: lower_CI(lift_H20_ALL) > 0 (filtro de desarrollo, no confirmatorio).
    D3: de los bloques EVALUABLES, al menos ceil(2/3) con lift > 0.
        Bloque evaluable = (n_conf>=20 AND n_base>=20). No evaluable = INSUFFICIENT_SAMPLE.
        Si hay 3 evaluables -> >= 2 con lift > 0.
        Si hay 2 evaluables -> >= 2 con lift > 0 (ceil(4/3)=2). Nota: 2/2.
        Si hay 1 evaluable  -> >= 1 con lift > 0.
    """
    import math
    h20 = row["by_horizon"][20]
    all_b = h20["ALL"]
    d1 = (all_b["n_conf"] >= D1_MIN_CONFIRMED
          and all_b["n_base"] >= D1_MIN_BASELINE)
    d2 = (boot_res["lower_ci"] is not None
          and boot_res["lower_ci"] > 0.0)
    bloques_eval = []
    bloques_lift_pos = 0
    for label, _, _ in BLOCKS:
        b = h20[label]
        if block_is_evaluable(b):
            bloques_eval.append(label)
            if b["lift_struct_deterioration"] is not None and b["lift_struct_deterioration"] > 0:
                bloques_lift_pos += 1
    n_eval = len(bloques_eval)
    if n_eval == 0:
        d3 = False
    else:
        req = int(math.ceil(n_eval * D3_FRACTION_REQUIRED))
        if req < 1:
            req = 1
        d3 = (bloques_lift_pos >= req)
    return bool(d1), bool(d2), bool(d3), bloques_eval, bloques_lift_pos


def stability_metrics(row):
    """mediana_lift y MAD_lift sobre los bloques evaluables (H20, struct_deterioration)."""
    lifts = []
    for label, _, _ in BLOCKS:
        b = row["by_horizon"][20][label]
        if block_is_evaluable(b):
            v = b["lift_struct_deterioration"]
            if v is not None:
                lifts.append(v)
    if not lifts:
        return None, None, 0
    arr = np.array(lifts)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return med, mad, len(lifts)


def select_candidate(rows, boot_map):
    """Aplica criterios D1-D3 y ranking (seccion 5.4 protocolo v3).

    Ranking v3:
      Paso 1: mayor n_bloques_con_lift_positivo (mas estabilidad).
      Paso 2: mayor mediana_lift.
      Paso 3: menor delta_hetero (mas homogeneidad).
      Paso 4: tiebreak lexicografico (N, M, X_ATR, Y_VOL) ascendente.
    M neutral: no se premia ni castiga por tamano.

    Devuelve (elegido, motivo).
    """
    pasan = []
    for i, r in enumerate(rows):
        d1, d2, d3, bl_eval, bl_pos = evaluate_row(r, boot_map[i])
        r["D1"] = d1
        r["D2"] = d2
        r["D3"] = d3
        r["_bloques_evaluables"] = bl_eval
        r["_bloques_lift_pos"] = bl_pos
        if d1 and d2 and d3:
            med, mad, n_bl = stability_metrics(r)
            r["_mediana_lift"] = med
            r["_mad_lift"] = mad
            r["_n_bloques_eval"] = n_bl
            # delta_hetero = max - min sobre bloques evaluables
            lifts_eval = []
            for label in bl_eval:
                b = r["by_horizon"][20][label]
                v = b["lift_struct_deterioration"]
                if v is not None:
                    lifts_eval.append(v)
            if len(lifts_eval) >= 2:
                r["_delta_hetero"] = float(max(lifts_eval) - min(lifts_eval))
            else:
                r["_delta_hetero"] = None
            pasan.append((i, r))
    if not pasan:
        return None, "ninguna combinacion pasa D1-D3"
    # Ranking v3:
    # 1. mayor n_bloques_lift_pos
    # 2. mayor mediana_lift
    # 3. menor delta_hetero (None -> +inf para no premiar)
    # 4. lexicografico
    def key(r):
        d_het = r["_delta_hetero"] if r["_delta_hetero"] is not None else 1e9
        return (
            -r["_bloques_lift_pos"],
            -(r["_mediana_lift"] if r["_mediana_lift"] is not None else -1e9),
            d_het,
            r["N"], r["M"], r["X_ATR"], r["Y_VOL"],
        )
    pasan.sort(key=lambda x: key(x[1]))
    elegido = pasan[0][1]
    return elegido, f"{len(pasan)} combinaciones pasan D1-D3; elegido por ranking v3"

def _short_key(k):
    return {
        "struct_deterioration": "struct",
        "price_weakness": "price",
        "below_support": "below",
        "lower_low": "lower",
    }[k]


def build_analysis_rows(episodes, feats, N, h=20, t0_min=None):
    """Fuente unica de verdad de la poblacion analizada.

    Auditor 2026-10-05 seccion 4: los pares 635/629 y 1753/1731 tienen
    que salir de una sola funcion para eliminar la ambiguedad entre
    "episodios con L" y "episodios con H completo".

    Pipeline:
        episodes_raw
            -> drop t0 < t0_min (si t0_min)
            -> drop H censurado (L_idx + h >= len(struct))
            -> drop metrics=None (struct_L o struct_H NaN)
            -> ANALYSIS_ROWS

    Args:
        episodes: lista de dicts producidos por build_episodes_landmark.
        feats: dict de features por ticker.
        N: window del SOW (usado por episode_metrics para support_L).
        h: horizonte. Default 20.
        t0_min: si se pasa, descarta episodios con t0_date < t0_min.

    Returns:
        {
          "n_episodes_raw": int,
          "n_dropped_oos": int,
          "n_dropped_h_censored": int,
          "n_dropped_metrics_none": int,
          "n_analysis_rows": int,
          "n_confirmed_H_complete": int,
          "n_baseline_H_complete": int,
          "rows": list[(ticker, is_confirmed, metrics)],
        }
    """
    n_raw = len(episodes)
    n_dropped_oos = 0
    n_dropped_h = 0
    n_dropped_metrics = 0
    rows = []
    for e in episodes:
        if t0_min is not None and e["t0_date"] < t0_min:
            n_dropped_oos += 1
            continue
        feat = feats[e["ticker"]]
        if e["L_idx"] + h >= len(feat["struct"]):
            n_dropped_h += 1
            continue
        m = episode_metrics(feat, e["L_idx"], N, h)
        if m is None:
            n_dropped_metrics += 1
            continue
        rows.append((e["ticker"], e["is_confirmed"], m))
    n_conf = sum(1 for _, c, _ in rows if c)
    n_base = len(rows) - n_conf
    return {
        "n_episodes_raw": n_raw,
        "n_dropped_oos": n_dropped_oos,
        "n_dropped_h_censored": n_dropped_h,
        "n_dropped_metrics_none": n_dropped_metrics,
        "n_analysis_rows": len(rows),
        "n_confirmed_H_complete": n_conf,
        "n_baseline_H_complete": n_base,
        "rows": rows,
    }


def flatten_row(r):
    """Convierte un row del grid en dict plano para CSV."""
    flat = {
        "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
        "n_episodes_raw": r["n_episodes_raw"],
        "n_dropped_h_censored": r["n_dropped_h_censored"],
        "n_dropped_metrics_none": r["n_dropped_metrics_none"],
        "n_analysis_rows": r["n_analysis_rows"],
        "n_confirmed": r["n_confirmed"],
        "n_baseline": r["n_baseline"],
        "D1": r.get("D1"),
        "D2": r.get("D2"),
        "D3": r.get("D3"),
        "n_bloques_evaluables": len(r.get("_bloques_evaluables", [])),
        "n_bloques_lift_pos": r.get("_bloques_lift_pos"),
        "mediana_lift": r.get("_mediana_lift"),
        "mad_lift": r.get("_mad_lift"),
        "delta_hetero": r.get("_delta_hetero"),
    }
    for h in HORIZONS:
        for label in [b[0] for b in BLOCKS] + ["ALL"]:
            b = r["by_horizon"][h][label]
            prefix = f"H{h}_{label}"
            flat[f"{prefix}_n_conf"] = b["n_conf"]
            flat[f"{prefix}_n_base"] = b["n_base"]
            for k in KEYS:
                sk = _short_key(k)
                flat[f"{prefix}_pct_conf_{sk}"] = b[f"pct_conf_{k}"]
                flat[f"{prefix}_pct_base_{sk}"] = b[f"pct_base_{k}"]
                flat[f"{prefix}_lift_{sk}"] = b[f"lift_{k}"]
    return flat


def flatten_bootstrap(boot, row):
    return {
        "N": row["N"], "M": row["M"], "X_ATR": row["X_ATR"], "Y_VOL": row["Y_VOL"],
        "lift_point": boot["lift_point"],
        "lower_ci": boot["lower_ci"],
        "upper_ci": boot["upper_ci"],
        "p_one_sided": boot.get("p_one_sided"),
        "n_tickers": boot["n_tickers"],
        "n_episodes": boot["n_episodes"],
        "n_confirmed": boot["n_confirmed"],
        "n_baseline": boot["n_baseline"],
        "n_boot_validos": boot["n_boot_validos"],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gap-sessions", type=int, default=None,
                    help="Gap minimo entre t0 consecutivos del mismo ticker.")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("== 5b.4-bis - 3 bloques + delta_hetero (v3) ==")
    if args.min_gap_sessions is not None:
        print(f"  min_gap_sessions = {args.min_gap_sessions}")
    df, tickers = load_dataset()
    print(f"Tickers en dataset: {len(tickers)}")
    feats = precompute_features(df, tickers)
    print(f"Features construidas: {len(feats)}")

    starts_cache = {tk: find_candidate_starts(feat["candidate"])
                    for tk, feat in feats.items()}
    total_starts = sum(len(v) for v in starts_cache.values())
    print(f"Candidate starts totales: {total_starts}")

    rows = run_grid(feats, starts_cache,
                    min_gap_sessions=args.min_gap_sessions)

    # Bootstrap por fila (sobre H20, bloque ALL)
    print(f"Bootstrap: B={BOOT_B}, seed={BOOT_SEED}")
    boot_map = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        boot_map[i] = bootstrap_lift_H20(r["_rows_H20"], B=BOOT_B, seed=BOOT_SEED)
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  boot [{i+1}/{len(rows)}] {el:.1f}s", flush=True)
    print(f"Bootstrap completado en {time.time()-t0:.1f}s")

    # Seleccion (asigna D1-D3 y aplica ranking)
    elegido, motivo = select_candidate(rows, boot_map)
    print(f"Seleccion: {motivo}")
    if elegido is not None:
        print(f"  -> N={elegido['N']} M={elegido['M']} "
              f"X_ATR={elegido['X_ATR']} Y_VOL={elegido['Y_VOL']}")

    # Serializar grid CSV
    flat_rows = [flatten_row(r) for r in rows]
    grid_df = pd.DataFrame(flat_rows)
    grid_df.to_csv(OUT_GRID, index=False)
    print(f"Grid CSV: {OUT_GRID} ({len(grid_df)} filas)")

    # Serializar bootstrap CSV
    boot_rows = [flatten_bootstrap(boot_map[i], rows[i]) for i in range(len(rows))]
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(OUT_BOOT, index=False)
    print(f"Bootstrap CSV: {OUT_BOOT} ({len(boot_df)} filas)")

    # Serializar summary JSON
    summary = {
        "schema": "wyckoff_5b4bis_summary_v1",
        "protocolo": "30_protocolo_fase_5b4bis_v3.md (v3)",
        "dictamen": "29_dictamen_5b4_v2.md",
        "n_tickers_universe": len(tickers),
        "n_tickers_with_features": len(feats),
        "n_combos": len(rows),
        "grid": {
            "N": list(GRID_N), "M": list(GRID_M),
            "X_ATR": list(GRID_X), "Y_VOL": list(GRID_Y),
        },
        "blocks": [
            {"label": b[0], "start": str(b[1].date()), "end": str(b[2].date())}
            for b in BLOCKS
        ],
        "constants": {
            "PRECEDENT_WINDOW": PRECEDENT_WINDOW,
            "ATR_WINDOW": ATR_WINDOW,
            "HORIZONS": list(HORIZONS),
        },
        "criteria": {
            "D1_MIN_CONFIRMED": D1_MIN_CONFIRMED,
            "D1_MIN_BASELINE": D1_MIN_BASELINE,
            "D2": "lower_CI(lift_H20) > 0",
            "D3": "ceil(2/3) de bloques evaluables con lift_H20 > 0",
            "BOOT_B": BOOT_B,
            "BOOT_SEED": BOOT_SEED,
        },
        "n_pasan_D1D3": sum(1 for r in rows if r.get("D1") and r.get("D2") and r.get("D3")),
        "n_D1": sum(1 for r in rows if r.get("D1")),
        "n_D2": sum(1 for r in rows if r.get("D2")),
        "n_D3": sum(1 for r in rows if r.get("D3")),
        "selection": {
            "motivo": motivo,
            "elegido": None if elegido is None else {
                "N": elegido["N"], "M": elegido["M"],
                "X_ATR": elegido["X_ATR"], "Y_VOL": elegido["Y_VOL"],
                "mediana_lift": elegido.get("_mediana_lift"),
                "mad_lift": elegido.get("_mad_lift"),
                "delta_hetero": elegido.get("_delta_hetero"),
                "n_bloques_evaluables": len(elegido.get("_bloques_evaluables", [])),
                "n_bloques_lift_pos": elegido.get("_bloques_lift_pos"),
                "by_horizon_H20": elegido["by_horizon"][20],
            },
        },
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"Summary JSON: {OUT_SUMMARY}")

    # CSV de heterogeneidad: una fila por combinacion con delta_hetero,
    # firma de signos por bloque, n_bloques_eval, n_bloques_pos.
    hetero_rows = []
    for r in rows:
        bloques_lifts = []
        signos = []
        for label, _, _ in BLOCKS:
            b = r["by_horizon"][20][label]
            v = b["lift_struct_deterioration"]
            ev = block_is_evaluable(b)
            bloques_lifts.append({"label": label, "evaluable": ev, "lift": v})
            if ev and v is not None:
                signos.append("+" if v > 0 else ("-" if v < 0 else "0"))
        hetero_rows.append({
            "N": r["N"], "M": r["M"], "X_ATR": r["X_ATR"], "Y_VOL": r["Y_VOL"],
            "delta_hetero": r.get("_delta_hetero"),
            "n_bloques_eval": len(r.get("_bloques_evaluables", [])),
            "n_bloques_pos": r.get("_bloques_lift_pos"),
            "firma_signos": "".join(signos),
            "lifts_por_bloque": ";".join(
                f"{b['label']}={'NA' if b['lift'] is None else round(b['lift'],4)}"
                + ("" if b["evaluable"] else "_INSUF")
                for b in bloques_lifts
            ),
            "D1": r.get("D1"), "D2": r.get("D2"), "D3": r.get("D3"),
        })
    hetero_df = pd.DataFrame(hetero_rows)
    hetero_df.to_csv(OUT_HETERO, index=False)
    print(f"Hetero CSV: {OUT_HETERO} ({len(hetero_df)} filas)")


if __name__ == "__main__":
    main()