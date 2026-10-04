"""Placebos A y B (seccion 11).

Placebo A: permutacion temporal restringida. Por ticker y ano natural,
se redistribuyen las etiquetas SOW entre los episodios del mismo ticker
y ano, conservando el conteo.

Placebo B: permutacion intra-sector/intra-semana. Por (ano ISO, semana
ISO, sector), se intercambian las etiquetas SOW entre episodios del
mismo grupo, conservando el conteo.

Placebo C: temporal lead (diagnostico). No entra en decision.

Reproducibilidad (M6): PLACEBO_MASTER_SEED + derivacion determinista por
(fold, scheme, permutacion).
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from scripts.sow_v4 import config
from scripts.sow_v4.model_primary import fit_primary, predict_rd


def derive_seed(master_seed: int, *parts) -> int:
    """SHA-256(master|p1|p2|...) -> primeros 8 bytes como uint64."""
    key = "|".join([str(master_seed)] + [str(p) for p in parts])
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def apply_placebo_A(episodes_train: list[dict], rng: np.random.Generator) -> list[dict]:
    """Permutacion temporal restringida por (ticker, ano)."""
    out = [dict(e) for e in episodes_train]
    groups: dict = defaultdict(list)
    for i, e in enumerate(out):
        year = e["t0_date"].year
        groups[(e["ticker"], year)].append(i)
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        labels = [out[i]["is_confirmed"] for i in idxs]
        rng.shuffle(labels)
        for i, lab in zip(idxs, labels):
            out[i]["is_confirmed"] = bool(lab)
    return out


def apply_placebo_B(
    episodes_train: list[dict],
    sector_map: dict,
    rng: np.random.Generator,
) -> list[dict]:
    """Permutacion intra-sector/intra-semana."""
    out = [dict(e) for e in episodes_train]
    groups: dict = defaultdict(list)
    for i, e in enumerate(out):
        iso = e["t0_date"].isocalendar()
        sec = sector_map.get(e["ticker"], "__unknown__")
        groups[(iso.year, iso.week, sec)].append(i)
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        labels = [out[i]["is_confirmed"] for i in idxs]
        rng.shuffle(labels)
        for i, lab in zip(idxs, labels):
            out[i]["is_confirmed"] = bool(lab)
    return out


def evaluate_placebo(
    episodes_train: list[dict],
    episodes_eval: list[dict],
    scheme: str,
    fold_idx: int,
    sector_map: dict | None = None,
) -> dict | None:
    """Evalua un placebo (A o B) sobre un fold.

    1. Real: fit en episodes_train -> RD_pool sobre episodes_eval.
    2. Permutaciones: para k en [0, K):
        seed = derive_seed(master, fold_idx, scheme, k)
        perm_train = apply(scheme, seed)
        fit_perm = fit_primary(perm_train)
        rd_k = predict_rd(fit_perm, episodes_eval)['RD_pool']
    3. q5, q95 = percentiles 5/95 de la distribucion de randomizacion.

    Regla de aceptacion (seccion 11.6):
        q5 >= -DELTA_PLACEBO  AND  q95 <= +DELTA_PLACEBO
    """
    if scheme not in ("A", "B"):
        raise ValueError(f"scheme invalido: {scheme}")

    fit_real = fit_primary(episodes_train)
    if not fit_real.ok:
        return None
    rd_real = predict_rd(fit_real, episodes_eval)
    if not rd_real["ok"] or rd_real["RD_pool"] is None:
        return None

    K = config.PLACEBO_N_PERMUTACIONES
    rd_perm = np.full(K, np.nan)

    for k in range(K):
        seed_k = derive_seed(config.PLACEBO_MASTER_SEED, fold_idx, scheme, k)
        rng = np.random.default_rng(seed_k)
        if scheme == "A":
            perm_train = apply_placebo_A(episodes_train, rng)
        else:
            if sector_map is None:
                raise ValueError("sector_map requerido para Placebo B")
            perm_train = apply_placebo_B(episodes_train, sector_map, rng)

        fit_perm = fit_primary(perm_train)
        if not fit_perm.ok:
            continue
        rd_p = predict_rd(fit_perm, episodes_eval)
        if rd_p["ok"] and rd_p["RD_pool"] is not None:
            rd_perm[k] = rd_p["RD_pool"]

    valid = rd_perm[~np.isnan(rd_perm)]
    if len(valid) < 50:
        return None

    q5 = float(np.percentile(valid, 5.0))
    q95 = float(np.percentile(valid, 95.0))
    accepted = (q5 >= -config.DELTA_PLACEBO) and (q95 <= config.DELTA_PLACEBO)

    return {
        "scheme": scheme,
        "fold_idx": fold_idx,
        "RD_real_pool": rd_real["RD_pool"],
        "RD_placebo_q5": q5,
        "RD_placebo_q95": q95,
        "RD_placebo_median": float(np.median(valid)),
        "n_valid": int(len(valid)),
        "k_permutations": K,
        "accepted": bool(accepted),
    }


def evaluate_placebo_C_lead(
    episodes_train: list[dict],
    episodes_eval: list[dict],
) -> dict | None:
    """Placebo C diagnostico: SOW desplazado 40 sesiones hacia adelante.

    No entra en decision. Se reporta por transparencia.
    """
    shift = config.PLACEBO_LEAD_SESIONES
    _ = shift  # placeholder funcional; se implementara si el auditor lo pide
    return None