"""Reglas de decision (seccion 13 del protocolo v4.1).

Cuatro estados:
- VALIDADO UNIVERSAL
- VALIDADO CONDICIONAL A REGIMEN DE ESTRES
- NO VALIDADO
- MUESTRA INSUFICIENTE

Reglas de bloqueo B1-B13. El gate de potencia se evalua antes (seccion 12.9),
por eso aqui se recibe ya como booleano y motivo.
"""
from __future__ import annotations

import math

from scripts.sow_v4 import config


def _ceil_2_3(n: int) -> int:
    return int(math.ceil(config.DECISION_PASS_FRACTION * n))


def _pass_in_regime(fold_bootstraps: dict, regime: str) -> int | None:
    """Un fold 'pasa' en regimen r si LowerCI_95%(RD_r) > DELTA_MIN.

    Usa B20 como primario. Devuelve 1/0 o None si no evaluable.
    """
    bs = fold_bootstraps.get("B20")
    if bs is None:
        return None
    key = "ci95_stress" if regime == "stress" else "ci95_normal"
    ci = bs.get(key)
    if ci is None or ci[0] is None:
        return None
    return 1 if ci[0] > config.DELTA_MIN else 0


def _fold_pass_B(fold_bootstraps: dict, regime: str, B_name: str) -> int | None:
    bs = fold_bootstraps.get(B_name)
    if bs is None:
        return None
    key = "ci95_stress" if regime == "stress" else "ci95_normal"
    ci = bs.get(key)
    if ci is None or ci[0] is None:
        return None
    return 1 if ci[0] > config.DELTA_MIN else 0


def _upper_ci_90_normal(fold_bootstraps: dict) -> float | None:
    bs = fold_bootstraps.get("B20")
    if bs is None:
        return None
    ci = bs.get("ci90_normal")
    if ci is None or ci[1] is None:
        return None
    return float(ci[1])


def decide(
    fold_results: list[dict],
    placebo_results: list[dict],
    gate_universal: bool,
    gate_conditional: bool,
    gate_muestra_insuficiente: bool,
) -> dict:
    """fold_results: lista de dicts con ok/rd_point/bootstraps/fold_idx
    placebo_results: lista de dicts con scheme/fold_idx/accepted
    gate_*: resultado del power analysis (seccion 12.9)
    """
    if gate_muestra_insuficiente:
        return {"estado": "MUESTRA INSUFICIENTE", "motivo": "gate_potencia"}

    eval_folds = [f for f in fold_results if f.get("ok")]
    n_eval = len(eval_folds)
    if n_eval < config.DECISION_MIN_EVAL:
        return {"estado": "MUESTRA INSUFICIENTE", "motivo": f"N_eval={n_eval}"}

    # Robustez B20/B40 por fold (B1)
    for f in eval_folds:
        for r in ("stress", "normal"):
            p20 = _fold_pass_B(f["bootstraps"], r, "B20")
            p40 = _fold_pass_B(f["bootstraps"], r, "B40")
            if p20 is not None and p40 is not None and p20 != p40:
                return {
                    "estado": "NO VALIDADO",
                    "motivo": f"B1_inestabilidad_B20_vs_B40_fold={f['fold_idx']}_reg={r}",
                }

    # Placebos (B2-c, dictamen 2026-10-04):
    # p_rand = (1 + count(RD_placebo >= RD_real)) / (K+1)
    # Pass si p_rand <= 0.05. Cada placebo individual debe cumplir:
    #   N_pass_p >= ceil(2/3 * N_eval)
    # No se agregan P1 y P2.
    placebo_required = _ceil_2_3(n_eval)
    by_scheme: dict = {}
    for pl in placebo_results:
        if pl is None:
            continue
        by_scheme.setdefault(pl["scheme"], []).append(pl)

    b2_details = {}
    for scheme in ("P1", "P2"):
        rows = by_scheme.get(scheme, [])
        n_pass = sum(1 for r in rows if r.get("b2c_pass"))
        b2_details[scheme] = {
            "n_pass": n_pass,
            "n_required": placebo_required,
            "n_evaluados": len(rows),
        }
        if n_pass < placebo_required:
            return {
                "estado": "NO VALIDADO",
                "motivo": f"B2c_placebo_{scheme}_pass={n_pass}<{placebo_required}",
                "b2_detalles": b2_details,
            }

    # Contar evaluables por regimen
    stress_passes = []
    normal_passes = []
    normal_upper_ok = []
    n_eval_stress = 0
    n_eval_normal = 0
    for f in eval_folds:
        ps = _pass_in_regime(f["bootstraps"], "stress")
        pn = _pass_in_regime(f["bootstraps"], "normal")
        if ps is not None:
            n_eval_stress += 1
            stress_passes.append(ps)
        if pn is not None:
            n_eval_normal += 1
            normal_passes.append(pn)
        up = _upper_ci_90_normal(f["bootstraps"])
        if up is not None:
            normal_upper_ok.append(up < config.DELTA_MIN)

    # Universal
    if (
        gate_universal
        and n_eval_stress >= config.DECISION_MIN_EVAL_STRESS
        and n_eval_normal >= config.DECISION_MIN_EVAL_NORMAL
    ):
        req_s = _ceil_2_3(n_eval_stress)
        req_n = _ceil_2_3(n_eval_normal)
        if sum(stress_passes) >= req_s and sum(normal_passes) >= req_n:
            return {"estado": "VALIDADO UNIVERSAL", "motivo": "ok"}

    # Condicional
    if (
        gate_conditional
        and n_eval_stress >= config.DECISION_MIN_EVAL_STRESS
        and n_eval_normal >= config.DECISION_MIN_EVAL_NORMAL
    ):
        req_s = _ceil_2_3(n_eval_stress)
        req_n = _ceil_2_3(n_eval_normal)
        if sum(stress_passes) >= req_s and sum(normal_upper_ok) >= req_n:
            return {"estado": "VALIDADO CONDICIONAL A REGIMEN DE ESTRES", "motivo": "ok"}

    return {"estado": "NO VALIDADO", "motivo": "criterios_no_alcanzados"}