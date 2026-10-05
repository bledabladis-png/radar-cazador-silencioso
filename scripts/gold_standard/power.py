# -*- coding: utf-8 -*-
"""Simulacion del diseno para dimensionar n_B (Muestra B).

Metodo (protocolo v7, seccion 3.3):
  - Cuadricula de 27 escenarios: pi x Se x Sp.
  - Para cada escenario y cada n_B:
      1. Construir poblacion sintetica compatible con (pi, Se, Sp).
      2. Asignar n_h proporcional con minimo Rao-Wu-Yue.
      3. Muestrear n_h por estrato con probabilidades conocidas.
      4. Calcular Se, Sp, PPV, NPV ponderadas.
      5. Repetir B veces; IC empirico = percentiles 2.5 y 97.5.
  - Error e(metrica) = (p97.5 - p2.5) / 2.
  - n_B minimo valido = primer n_B que satisface TODAS las
    restricciones en TODOS los escenarios.
  - Si el minimo supera CAPACIDAD_MAX_NB: suspender.

NO usa formulas cerradas de proporciones. Simula el diseno real.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import (
    CAPACIDAD_MAX_NB,
    E_NPV,
    E_PPV,
    E_SE,
    E_SP,
    N_MIN_RAO_WU,
    SEED_GLOBAL,
)


# 27 escenarios del protocolo
def escenarios_default() -> list[dict]:
    out = []
    for pi in (0.10, 0.20, 0.30):
        for se in (0.60, 0.70, 0.80):
            for sp in (0.60, 0.70, 0.80):
                out.append({"pi": pi, "se": se, "sp": sp})
    return out


def _asignar_n_h(estratos: pd.DataFrame, n_B: int) -> np.ndarray:
    """Reparto proporcional con minimo Rao-Wu-Yue y sin superar N_h."""
    N = estratos["N_h"].to_numpy().astype(float)
    total = N.sum()
    if total <= 0 or n_B <= 0:
        return np.zeros_like(N, dtype=int)
    n = np.floor(N / total * n_B).astype(int)
    n = np.maximum(n, N_MIN_RAO_WU)
    n = np.minimum(n, N.astype(int))
    return n


def _generar_poblacion(
    estratos: pd.DataFrame,
    pi: float,
    se: float,
    sp: float,
    rng: np.random.Generator,
) -> dict:
    """Poblacion sintetica por estrato.

    Cada individuo tiene (ref, det) con:
      P(ref=1) = pi
      P(det=1 | ref=1) = se
      P(det=1 | ref=0) = 1 - sp
    """
    N_h = estratos["N_h"].to_numpy().astype(int)
    refs = []
    dets = []
    for N in N_h:
        ref = rng.random(N) < pi
        det = np.where(ref, rng.random(N) < se, rng.random(N) < (1 - sp))
        refs.append(ref.astype(np.int8))
        dets.append(det.astype(np.int8))
    return {"ref": refs, "det": dets, "N_h": N_h}


def _una_replica(
    poblacion: dict,
    n_h: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Una replica del diseno estratificado.

    Por estrato h: sample n_h indices sin reemplazo.
    Pesos w_i = N_h / n_h.
    """
    refs_sel = []
    dets_sel = []
    pesos_sel = []
    for h, (ref_h, det_h) in enumerate(
        zip(poblacion["ref"], poblacion["det"])
    ):
        N = len(ref_h)
        n = int(n_h[h])
        if n <= 0:
            continue
        if n >= N:
            idx = np.arange(N)
        else:
            idx = rng.choice(N, size=n, replace=False)
        refs_sel.append(ref_h[idx])
        dets_sel.append(det_h[idx])
        peso = (N / n) if n > 0 else 0.0
        pesos_sel.append(np.full(n, peso, dtype=float))
    if not pesos_sel:
        return {
            "se": np.nan, "sp": np.nan, "ppv": np.nan, "npv": np.nan,
        }
    ref = np.concatenate(refs_sel)
    det = np.concatenate(dets_sel)
    w = np.concatenate(pesos_sel)

    w_ref1 = w[ref == 1].sum()
    w_ref0 = w[ref == 0].sum()
    w_det1 = w[det == 1].sum()
    w_det0 = w[det == 0].sum()
    tp = w[(ref == 1) & (det == 1)].sum()
    tn = w[(ref == 0) & (det == 0)].sum()

    se_hat = tp / w_ref1 if w_ref1 > 0 else np.nan
    sp_hat = tn / w_ref0 if w_ref0 > 0 else np.nan
    ppv = tp / w_det1 if w_det1 > 0 else np.nan
    npv = tn / w_det0 if w_det0 > 0 else np.nan
    return {"se": se_hat, "sp": sp_hat, "ppv": ppv, "npv": npv}


def simular_escenario(
    pi: float,
    se: float,
    sp: float,
    estratos: pd.DataFrame,
    n_B: int,
    B: int = 500,
    seed: int = SEED_GLOBAL,
) -> dict:
    """Simula un escenario. Devuelve medias, IC empiricos y errores."""
    rng_pop = np.random.default_rng(seed)
    rng_sample = np.random.default_rng(seed + 1)
    pob = _generar_poblacion(estratos, pi, se, sp, rng_pop)
    n_h = _asignar_n_h(estratos, n_B)

    runs = []
    for b in range(B):
        rng_b = np.random.default_rng(seed + 100 + b)
        runs.append(_una_replica(pob, n_h, rng_b))

    df = pd.DataFrame(runs)
    out = {"pi": pi, "se_true": se, "sp_true": sp, "n_B": n_B,
           "n_h_sum": int(n_h.sum()), "B": B}
    for m in ("se", "sp", "ppv", "npv"):
        vals = df[m].dropna().to_numpy()
        if len(vals) == 0:
            out[f"{m}_mean"] = np.nan
            out[f"{m}_e"] = np.nan
            continue
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[f"{m}_mean"] = float(np.mean(vals))
        out[f"{m}_lo"] = float(lo)
        out[f"{m}_hi"] = float(hi)
        out[f"{m}_e"] = float((hi - lo) / 2.0)
    return out


def _cumple_restricciones(res: dict) -> bool:
    return (
        res.get("se_e", np.inf) <= E_SE
        and res.get("sp_e", np.inf) <= E_SP
        and res.get("ppv_e", np.inf) <= E_PPV
        and res.get("npv_e", np.inf) <= E_NPV
    )


def dimensionar_n_B(
    estratos: pd.DataFrame,
    escenarios: list[dict] | None = None,
    n_min: int = 100,
    n_max: int = CAPACIDAD_MAX_NB,
    paso: int = 50,
    B: int = 500,
    seed: int = SEED_GLOBAL,
) -> dict:
    """Busca el n_B minimo que cumple todas las restricciones.

    Devuelve dict con n_B_min, si supera capacidad, y detalles
    por escenario para el n_B elegido.
    """
    if escenarios is None:
        escenarios = escenarios_default()

    detalle_por_nB = []
    for n_B in range(n_min, n_max + 1, paso):
        resultados = [
            simular_escenario(
                e["pi"], e["se"], e["sp"], estratos, n_B, B=B, seed=seed
            )
            for e in escenarios
        ]
        todas_ok = all(_cumple_restricciones(r) for r in resultados)
        detalle_por_nB.append({
            "n_B": n_B,
            "todas_ok": todas_ok,
            "resultados": resultados,
        })
        if todas_ok:
            return {
                "n_B_min": n_B,
                "excede_capacidad": False,
                "capacidad_max": CAPACIDAD_MAX_NB,
                "detalle": detalle_por_nB,
            }
    return {
        "n_B_min": None,
        "excede_capacidad": True,
        "capacidad_max": CAPACIDAD_MAX_NB,
        "detalle": detalle_por_nB,
        "mensaje": (
            "Ningun n_B <= CAPACIDAD_MAX_NB cumple todas las "
            "restricciones. SUSPENDER y revisar diseno."
        ),
    }