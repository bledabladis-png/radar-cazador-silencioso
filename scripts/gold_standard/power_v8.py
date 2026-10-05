# -*- coding: utf-8 -*-
"""Simulacion de potencia v8.

Protocolo v8, secciones 15, 16, 17.

Componentes:
  - Escenarios factibles (§15.3) condicionados a q_D observado.
  - Generacion de poblacion sintetica con regla §17.2:
    redondeo global + largest remainder, NO round() por celda.
  - Monte Carlo externo R_MC (§16).
  - Bootstrap RWY interno B (§12).
  - Criterio PASS: q95_MC(e_theta) <= target por escenario y estimador.
  - invalid_rate <= 0.01.
  - Si 800 FAIL: buscar menor n_total <= 800 (§17.5).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.gold_standard.bootstrap_rwy import (
    ppv_metric,
    npv_metric,
    se_metric,
    sp_metric,
)
from scripts.gold_standard.constants import SEED_GLOBAL
from scripts.gold_standard.sampling_b_v8 import CELDAS

# Parametros de simulacion
R_MC_SCREENING = 200
R_MC_CONFIRM = 500
B_SCREENING = 500
B_CONFIRM = 2000
INVALID_RATE_MAX = 0.01

# Cuadricula gate (§15.3)
PI_Y_GRID = (0.005, 0.008, 0.010, 0.012)
SE_GRID = (0.60, 0.70, 0.80)

# Targets (§14)
TARGETS = {
    "se": 0.07, "sp": 0.07, "ppv": 0.08, "npv": 0.08,
}

METRICAS = {
    "se": se_metric,
    "sp": sp_metric,
    "ppv": ppv_metric,
    "npv": npv_metric,
}

N_SYN_STRESS = 100_000
K_CELDA_DEFAULT = 200
K_CELDA_MIN = 5


def _hamilton(cuotas: np.ndarray, total: int) -> np.ndarray:
    """Largest remainder / Hamilton."""
    cuotas = np.asarray(cuotas, dtype=float)
    base = np.floor(cuotas).astype(int)
    rem = total - int(base.sum())
    if rem == 0:
        return base
    restos = cuotas - base
    if rem > 0:
        order = np.argsort(-restos)
        for i in order[:rem]:
            base[i] += 1
    else:
        order = np.argsort(restos)
        i = 0
        while rem < 0 and i < len(base):
            if base[order[i]] > 0:
                base[order[i]] -= 1
                rem += 1
            i += 1
    return base

def escenarios_factibles(q_D: float) -> list[dict]:
    """§15.3: cuadricula factible de gate principal."""
    out = []
    for pi in PI_Y_GRID:
        for se in SE_GRID:
            if pi * se > q_D + 1e-12:
                continue
            # Sp implicita
            if pi < 1.0:
                sp = 1.0 - (q_D - pi * se) / (1.0 - pi)
            else:
                sp = 1.0
            if not (0.0 <= sp <= 1.0):
                continue
            out.append({
                "pi_Y": pi, "Se": se, "Sp": sp,
            })
    return out


def generar_poblacion_sintetica(
    N_h_celda: dict,
    pi_Y: float, Se: float, Sp: float,
    N_syn_total: int | None = None,
) -> dict:
    """§17.2: redondeo global + largest remainder.

    N_h_celda: dict {celda: N_Cj}.
       Si N_syn_total es None, usar los N_Cj tal cual (gate principal).
       Si N_syn_total no es None, escalar proporcionalmente (stress test).

    Devuelve dict {celda: (n_Y1, n_Y0)}.
    """
    # 1. Determinar N_Cj finales
    if N_syn_total is not None:
        total_orig = sum(N_h_celda.values())
        if total_orig == 0:
            raise ValueError("N_h_celda vacio")
        cuotas = np.array([N_h_celda.get(c, 0) for c in CELDAS], dtype=float)
        cuotas = cuotas * N_syn_total / total_orig
        N_Cj = _hamilton(cuotas, N_syn_total)
    else:
        N_Cj = np.array([N_h_celda.get(c, 0) for c in CELDAS], dtype=int)

    # 2. totales globales: TP sobre (C1, C2); FP sobre (C3, C4)
    N_C1, N_C2, N_C3, N_C4 = N_Cj
    N_D1 = N_C1 + N_C2
    N_D0 = N_C3 + N_C4

    total_TP = int(round(N_D1 * Se))
    total_FP = int(round(N_D0 * (1.0 - Sp)))

    # 3. distribuir TP entre C1 y C2 via Hamilton
    if N_D1 > 0:
        cuotas_tp = np.array([N_C1, N_C2], dtype=float) * (total_TP / N_D1)
        tp = _hamilton(cuotas_tp, total_TP)
        tp = np.minimum(tp, [N_C1, N_C2])
    else:
        tp = np.array([0, 0])

    if N_D0 > 0:
        cuotas_fp = np.array([N_C3, N_C4], dtype=float) * (total_FP / N_D0)
        fp = _hamilton(cuotas_fp, total_FP)
        fp = np.minimum(fp, [N_C3, N_C4])
    else:
        fp = np.array([0, 0])

    return {
        "C1": {"N": int(N_C1), "n_Y1": int(tp[0]), "n_Y0": int(N_C1) - int(tp[0])},
        "C2": {"N": int(N_C2), "n_Y1": int(tp[1]), "n_Y0": int(N_C2) - int(tp[1])},
        "C3": {"N": int(N_C3), "n_Y1": int(fp[0]), "n_Y0": int(N_C3) - int(fp[0])},
        "C4": {"N": int(N_C4), "n_Y1": int(fp[1]), "n_Y0": int(N_C4) - int(fp[1])},
    }

def _muestra_srswor(
    poblacion: dict,
    estratos_con_n: pd.DataFrame,
    rng: np.random.Generator,
) -> dict:
    """SRSWOR por estrato. Devuelve y_ref/y_det/w/estrato_id por unidad.

    Para simplificar, la unidad sintetica tiene atributos (celda, sector,
    periodo). La poblacion ya tiene la composicion correcta; el muestreo
    opera por estrato (celda, sector, periodo) segun los n_h fijados.
    """
    parts = []
    for _, row in estratos_con_n.iterrows():
        celda = row["celda"]
        n_h = int(row["n_h"])
        N_h = int(row["N_h"])
        if n_h <= 0:
            continue
        pop = poblacion.get(celda)
        if pop is None:
            continue
        # Distribuir n_h entre Y=1 e Y=0 segun proporcion poblacional
        N_h_pop = pop["N"]
        if N_h_pop == 0:
            continue
        prop_Y1 = pop["n_Y1"] / N_h_pop
        n_Y1 = int(round(n_h * prop_Y1))
        n_Y1 = min(n_Y1, pop["n_Y1"], n_h)
        n_Y0 = n_h - n_Y1
        if n_Y0 > pop["n_Y0"]:
            n_Y0 = pop["n_Y0"]
            n_Y1 = n_h - n_Y0
            n_Y1 = max(n_Y1, 0)
        # D de la celda
        d_val = 1 if celda in ("C1", "C2") else 0
        # y_ref: 1 para n_Y1 unidades, 0 para n_Y0
        y_arr = np.concatenate([np.ones(n_Y1, dtype=int),
                                 np.zeros(n_Y0, dtype=int)])
        d_arr = np.full(len(y_arr), d_val, dtype=int)
        w_arr = np.full(len(y_arr), N_h / max(n_h, 1), dtype=float)
        parts.append({"y": y_arr, "d": d_arr, "w": w_arr, "h": n_h})
    if not parts:
        raise ValueError("Muestra sintetica vacia")
    y = np.concatenate([p["y"] for p in parts])
    d = np.concatenate([p["d"] for p in parts])
    w = np.concatenate([p["w"] for p in parts])
    # estrato_id: indexado 0..len(estratos_con_n)-1
    estrato_id = np.concatenate([
        np.full(len(p["y"]), i, dtype=int)
        for i, p in enumerate(parts)
    ])
    return {"y": y, "d": d, "w": w, "estrato_id": estrato_id}


def _ci_para_estimador(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    w: np.ndarray,
    estrato_id: np.ndarray,
    N_h_map: dict,
    metric_fn,
    B: int,
    seed: int,
) -> dict:
    """Bootstrap RWY de un estimador. Devuelve e_theta, invalid_rate."""
    from scripts.gold_standard.bootstrap_rwy import rao_wu_yue_ci

    # Filtrar unidades cuyo estrato tiene n_h >= 2 o es censo
    res = rao_wu_yue_ci(
        y_ref, y_det, w, estrato_id, N_h_map,
        metric_fn=metric_fn, B=B, seed=seed,
    )
    invalid = B - res.n_replicas
    e_theta = (res.hi - res.lo) / 2.0 if not np.isnan(res.hi) else np.nan
    return {
        "point": res.point,
        "lo": res.lo,
        "hi": res.hi,
        "e_theta": e_theta,
        "n_validas": res.n_replicas,
        "n_invalidas": invalid,
        "invalid_rate": invalid / B if B > 0 else 0.0,
    }

def evaluar_escenario(
    pi_Y: float, Se: float, Sp: float,
    estratos_con_n: pd.DataFrame,
    R_MC: int = R_MC_SCREENING,
    B: int = B_SCREENING,
    master_seed: int = SEED_GLOBAL,
) -> dict:
    """§17: simula un escenario.

    1. Construir poblacion sintetica (§17.2) con los N_Cj = suma N_h por celda.
    2. R_MC muestras SRSWOR independientes (stream MC).
    3. Bootstrap B por muestra (stream bootstrap).
    4. Distribucion q95_MC de e_theta por estimador.
    5. invalid_rate maximo.
    """
    from scripts.gold_standard.seeds import rng_for_replica

    # N_Cj por celda
    N_h_celda = {}
    for celda in CELDAS:
        sub = estratos_con_n[estratos_con_n["celda"] == celda]
        N_h_celda[celda] = int(sub["N_h"].sum())

    # Poblacion sintetica con conteos exactos
    poblacion = generar_poblacion_sintetica(
        N_h_celda, pi_Y, Se, Sp, N_syn_total=None,
    )

    # Streams separados (§16.4)
    rng_mc = rng_for_replica(master_seed, 0)   # stream MC
    rng_boot_seed_base = 10_000

    # N_h_map para bootstrap
    N_h_map = {}
    for i, row in enumerate(estratos_con_n.itertuples()):
        N_h_map[i] = int(row.N_h)

    resultados_por_estimador = {k: [] for k in METRICAS}
    invalid_max = 0.0
    n_validas_min = B

    for r in range(R_MC):
        # Muestra SRSWOR
        muestra = _muestra_srswor(poblacion, estratos_con_n, rng_mc)
        seed_boot = rng_boot_seed_base + r

        for k, fn in METRICAS.items():
            res = _ci_para_estimador(
                muestra["y"], muestra["d"], muestra["w"],
                muestra["estrato_id"], N_h_map,
                metric_fn=fn, B=B, seed=seed_boot,
            )
            if not np.isnan(res["e_theta"]):
                resultados_por_estimador[k].append(res["e_theta"])
            invalid_max = max(invalid_max, res["invalid_rate"])
            n_validas_min = min(n_validas_min, res["n_validas"])

    # q95 MC
    q95 = {}
    for k, vals in resultados_por_estimador.items():
        if len(vals) == 0:
            q95[k] = np.nan
        else:
            q95[k] = float(np.percentile(vals, 95))

    pass_estim = {
        k: (not np.isnan(q95[k])) and (q95[k] <= TARGETS[k])
        for k in METRICAS
    }
    pass_invalid = invalid_max <= INVALID_RATE_MAX
    return {
        "pi_Y": pi_Y, "Se": Se, "Sp": Sp,
        "q95_MC": q95,
        "pass_estimador": pass_estim,
        "pass_invalid": pass_invalid,
        "invalid_rate_max": invalid_max,
        "n_validas_min": n_validas_min,
        "R_MC": R_MC, "B": B,
        "PASS": bool(all(pass_estim.values()) and pass_invalid),
    }


def dimensionar_v8(
    estratos_base: pd.DataFrame,
    q_D: float,
    R_MC: int = R_MC_SCREENING,
    B: int = B_SCREENING,
    master_seed: int = SEED_GLOBAL,
) -> dict:
    """§17.5: evaluar n_total=800 y, si falla, buscar menor n_total <= 800.

    estratos_base: DataFrame con celda, sector, periodo, N_h.
                   Los N_h son los del frame real.
    """
    # Paso 0: factibilidad de minimos
    from scripts.gold_standard.sampling_b_v8 import (
        verificar_factibilidad_minimos,
        asignar_n_h,
    )
    fact = verificar_factibilidad_minimos(estratos_base, K=K_CELDA_DEFAULT)
    if not fact["ok"]:
        return {
            "estado": "SUSPENDIDO",
            "motivo": "minimos exceden K",
            "detalle": fact,
        }

    escenarios = escenarios_factibles(q_D)
    if len(escenarios) == 0:
        return {"estado": "PROTOCOLO_INVALIDO",
                "motivo": "|S_factible|=0"}

    # Evaluar n_total=800
    estratos_n = asignar_n_h(estratos_base, K=K_CELDA_DEFAULT)
    resultados_800 = [
        evaluar_escenario(e["pi_Y"], e["Se"], e["Sp"],
                          estratos_n, R_MC=R_MC, B=B,
                          master_seed=master_seed)
        for e in escenarios
    ]
    pass_800 = all(r["PASS"] for r in resultados_800)
    if pass_800:
        return {
            "estado": "PASS_800",
            "n_total": 4 * K_CELDA_DEFAULT,
            "escenarios": resultados_800,
        }

    # Buscar menor n_total = 4K, K entero, 5 <= K <= 200
    # El protocolo no exige iterar exhaustivamente en cada K;
    # busca el menor K que pasa.
    for K in range(K_CELDA_MIN, K_CELDA_DEFAULT + 1, 5):
        estratos_n = asignar_n_h(estratos_base, K=K)
        res = [
            evaluar_escenario(e["pi_Y"], e["Se"], e["Sp"],
                              estratos_n, R_MC=R_MC, B=B,
                              master_seed=master_seed)
            for e in escenarios
        ]
        if all(r["PASS"] for r in res):
            return {
                "estado": "PASS_MENOR_N",
                "n_total": 4 * K,
                "K": K,
                "escenarios": res,
            }

    return {
        "estado": "SUSPENDIDO",
        "motivo": "n_total requerido > 800",
    }