# -*- coding: utf-8 -*-
"""Simulacion de potencia v8.

Protocolo v8, secciones 15, 16, 17.

Implementacion vectorizada:
  - Poblacion: conteos n_Y1_h por estrato.
  - MC: k_h ~ Hypergeometric(N_h, n_Y1_h, n_h).
  - Bootstrap: R_h_boot ~ Binomial(m_h, k_h/n_h).
  - Censo: contribucion fija.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import SEED_GLOBAL
from scripts.gold_standard.sampling_b_v8 import CELDAS

R_MC_SCREENING = 200
R_MC_CONFIRM = 500
B_SCREENING = 500
B_CONFIRM = 2000
INVALID_RATE_MAX = 0.01

PI_Y_GRID = (0.005, 0.008, 0.010, 0.012)
SE_GRID = (0.60, 0.70, 0.80)

TARGETS = {"se": 0.07, "sp": 0.07, "ppv": 0.08, "npv": 0.08}

N_SYN_STRESS = 100_000
K_CELDA_DEFAULT = 200
K_CELDA_MIN = 5


def _seed_from_master(master: int, tag: str) -> int:
    h = sha256(f"{master}_{tag}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**32)


def escenarios_factibles(q_D: float) -> list[dict]:
    out = []
    for pi in PI_Y_GRID:
        for se in SE_GRID:
            if pi * se > q_D + 1e-12:
                continue
            sp = 1.0 - (q_D - pi * se) / (1.0 - pi) if pi < 1.0 else 1.0
            if 0.0 <= sp <= 1.0:
                out.append({"pi_Y": pi, "Se": se, "Sp": sp})
    return out


def _hamilton(cuotas: np.ndarray, total: int) -> np.ndarray:
    cuotas = np.asarray(cuotas, dtype=float)
    base = np.floor(cuotas).astype(int)
    rem = total - int(base.sum())
    if rem == 0:
        return base
    restos = cuotas - base
    if rem > 0:
        for i in np.argsort(-restos)[:rem]:
            base[i] += 1
    else:
        i = 0
        order = np.argsort(restos)
        while rem < 0 and i < len(base):
            if base[order[i]] > 0:
                base[order[i]] -= 1
                rem += 1
            i += 1
    return base


def _poblacion_por_estrato(
    estratos: pd.DataFrame,
    pi_Y: float, Se: float, Sp: float, q_D_syn: float,
) -> np.ndarray:
    """n_Y1_h por estrato.

    Convencion diagnostica:
        Se = P(D=1 | Y=1)
        Sp = P(D=0 | Y=0)
        PPV = pi_Y * Se / q_D
        P(Y=1 | D=0) = pi_Y * (1-Se) / (1-q_D)
    """
    strata = estratos.reset_index(drop=True).copy()
    N_h = strata["N_h"].to_numpy(dtype=int)
    is_D1 = strata["celda"].isin(["C1", "C2"]).to_numpy()

    if q_D_syn <= 0 or q_D_syn >= 1:
        return np.zeros_like(N_h)
    p_Y1_D1 = pi_Y * Se / q_D_syn
    p_Y1_D0 = pi_Y * (1 - Se) / (1 - q_D_syn)

    ideal = np.where(is_D1, N_h * p_Y1_D1, N_h * p_Y1_D0)
    base = np.floor(ideal).astype(int)

    for is_d1_val, p_target in [(True, p_Y1_D1), (False, p_Y1_D0)]:
        mask = is_D1 == is_d1_val
        N_g = int(N_h[mask].sum())
        target = int(round(N_g * p_target))
        base_m = base[mask]
        ideal_m = ideal[mask]
        diff = target - int(base_m.sum())
        if diff > 0:
            restos = ideal_m - base_m
            for i in np.argsort(-restos)[:diff]:
                base_m[i] += 1
        elif diff < 0:
            restos = ideal_m - base_m
            i = 0
            order = np.argsort(restos)
            while diff < 0 and i < len(base_m):
                if base_m[order[i]] > 0:
                    base_m[order[i]] -= 1
                    diff += 1
                i += 1
        base[mask] = base_m

    return np.minimum(base, N_h)


def evaluar_escenario(
    pi_Y: float, Se: float, Sp: float,
    estratos_con_n: pd.DataFrame,
    R_MC: int = R_MC_SCREENING,
    B: int = B_SCREENING,
    master_seed: int = SEED_GLOBAL,
) -> dict:
    """Vectorizado. Vease protocolo §15-§17."""
    strata = estratos_con_n.reset_index(drop=True)
    N_h = strata["N_h"].to_numpy(dtype=int)
    n_h = strata["n_h"].to_numpy(dtype=int)
    is_D1 = strata["celda"].isin(["C1", "C2"]).to_numpy()
    w_h = N_h.astype(float) / np.maximum(n_h, 1)
    is_censo = (n_h == N_h)
    m_h = np.maximum(n_h - 1, 1).astype(float)

    f_h = n_h.astype(float) / np.maximum(N_h, 1).astype(float)
    lam_h = np.sqrt(np.maximum(m_h * (1 - f_h) / m_h, 0.0))
    lam_h = np.where(is_censo, 0.0, lam_h)
    lam_h = np.where(np.isfinite(lam_h), lam_h, 0.0)

    q_D_syn = pi_Y * Se + (1 - pi_Y) * (1 - Sp)
    n_Y1_h = _poblacion_por_estrato(strata, pi_Y, Se, Sp, q_D_syn)
    n_Y0_h = N_h - n_Y1_h

    rng_mc = np.random.default_rng(_seed_from_master(master_seed, "mc"))
    rng_boot = np.random.default_rng(_seed_from_master(master_seed, "boot"))

    w_h_total = w_h * n_h
    den_ppv = float((w_h_total * is_D1).sum())
    den_npv = float((w_h_total * (~is_D1)).sum())

    e_lists = {"se": [], "sp": [], "ppv": [], "npv": []}
    invalid_max = 0.0

    for _ in range(R_MC):
        k_h = rng_mc.hypergeometric(
            n_Y1_h, n_Y0_h, n_h
        ).astype(float)

        p_boot = np.where(n_h > 0, k_h / np.maximum(n_h, 1), 0.0)
        p_boot = np.clip(p_boot, 0.0, 1.0)

        R_h_boot = np.zeros((B, len(n_h)))
        non_censo = ~is_censo
        if non_censo.any():
            m_h_int = m_h[non_censo].astype(np.int64)
            p_non_censo = p_boot[non_censo].astype(np.float64)
            R_h_boot[:, non_censo] = rng_boot.binomial(
                m_h_int[None, :], p_non_censo[None, :],
                size=(B, int(non_censo.sum())),
            )
        if is_censo.any():
            R_h_boot[:, is_censo] = np.broadcast_to(
                k_h[None, is_censo], (B, is_censo.sum())
            )

        with np.errstate(invalid="ignore"):
            factor = (
                k_h[None, :] * (1 - lam_h[None, :])
                + lam_h[None, :] * (n_h[None, :] / m_h[None, :]) * R_h_boot
            )
        W_Y1 = w_h[None, :] * factor
        W_Y0 = w_h_total[None, :] - W_Y1

        num_se = (W_Y1 * is_D1[None, :]).sum(axis=1)
        den_se = W_Y1.sum(axis=1)
        Se_b = np.where(den_se > 0, num_se / den_se, np.nan)

        num_sp = (W_Y0 * (~is_D1)[None, :]).sum(axis=1)
        den_sp = W_Y0.sum(axis=1)
        Sp_b = np.where(den_sp > 0, num_sp / den_sp, np.nan)

        PPV_b = np.where(den_ppv > 0, num_se / den_ppv, np.nan)
        NPV_b = np.where(den_npv > 0, num_sp / den_npv, np.nan)

        for name, vals in [("se", Se_b), ("sp", Sp_b),
                            ("ppv", PPV_b), ("npv", NPV_b)]:
            valid = vals[~np.isnan(vals)]
            invalid_max = max(invalid_max, (B - len(valid)) / B)
            if len(valid) > 1:
                lo, hi = np.percentile(valid, [2.5, 97.5])
                e_lists[name].append(float((hi - lo) / 2.0))

    q95 = {k: (float(np.percentile(v, 95)) if v else float("nan"))
           for k, v in e_lists.items()}

    pass_estim = {k: (np.isfinite(q95[k]) and q95[k] <= TARGETS[k])
                  for k in TARGETS}
    pass_invalid = invalid_max <= INVALID_RATE_MAX
    return {
        "pi_Y": pi_Y, "Se": Se, "Sp": Sp,
        "q95_MC": q95,
        "pass_estimador": pass_estim,
        "pass_invalid": pass_invalid,
        "invalid_rate_max": invalid_max,
        "R_MC": R_MC, "B": B,
        "PASS": bool(all(pass_estim.values()) and pass_invalid),
    }