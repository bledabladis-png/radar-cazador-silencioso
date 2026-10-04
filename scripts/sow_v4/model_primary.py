"""Modelo primario (seccion 5 del protocolo v4.1).

logit(P(Y=1)) = b0 + b1*SOW + b2*R_stress + b3*(SOW*R_stress)
              + b4*D1_prev + b5*D2_prev + b6*D3_prev

Muestra de ajuste: exclusivamente R_episode in {stress, normal} (garantizado
por episodes.build_episodes_for_combo).

g-computation: predicciones contrafactuales Y_hat_1 (SOW=1) y Y_hat_0 (SOW=0),
marginalizacion sobre X_0 observado. RD_r por regimen, RD_pool sobre todos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import statsmodels.api as sm

from scripts.sow_v4 import config

EPS_SEPARACION = 1e-6


@dataclass
class FitResult:
    ok: bool
    reason: str
    model: object = None
    coef: np.ndarray | None = None
    converged: bool = False
    finite_coefs: bool = False
    separation_frac: float = 1.0
    diagnostics: dict = field(default_factory=dict)


def _design_matrix(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(episodes)
    x = np.empty((n, 7), dtype=float)
    y = np.empty(n, dtype=float)
    for i, e in enumerate(episodes):
        sow = 1.0 if e["is_confirmed"] else 0.0
        r_stress = 1.0 if e["R_episode"] == "stress" else 0.0
        x[i, 0] = 1.0
        x[i, 1] = sow
        x[i, 2] = r_stress
        x[i, 3] = sow * r_stress
        x[i, 4] = e["D1_prev"]
        x[i, 5] = e["D2_prev"]
        x[i, 6] = e["D3_prev"]
        y[i] = float(e["Y"])
    return x, y


def _design_matrix_pooled(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de diseno del modelo pooled (seleccion INNER).

    logit(P(Y=1)) = b0 + b1*SOW + b2*D1 + b3*D2 + b4*D3
    Sin interaccion de regimen.
    """
    n = len(episodes)
    x = np.empty((n, 5), dtype=float)
    y = np.empty(n, dtype=float)
    for i, e in enumerate(episodes):
        sow = 1.0 if e["is_confirmed"] else 0.0
        x[i, 0] = 1.0
        x[i, 1] = sow
        x[i, 2] = e["D1_prev"]
        x[i, 3] = e["D2_prev"]
        x[i, 4] = e["D3_prev"]
        y[i] = float(e["Y"])
    return x, y


def fit_primary_pooled(episodes_train: list[dict]) -> FitResult:
    """Modelo pooled para seleccion INNER (seccion 9.2 v4.1 revisada).

    No incluye SOW*R_stress. El regimen NO determina la seleccion.
    """
    if len(episodes_train) < 10:
        return FitResult(ok=False, reason="n_train<10")
    x, y = _design_matrix_pooled(episodes_train)
    if y.min() == y.max():
        return FitResult(ok=False, reason="y_constante")
    try:
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
    except Exception as exc:
        return FitResult(ok=False, reason=f"fit_exception:{type(exc).__name__}")

    conv = bool(getattr(model, "mle_retvals", {}).get("converged", False))
    coef = np.asarray(model.params, dtype=float)
    finite = bool(np.all(np.isfinite(coef)))
    p_hat = np.asarray(model.predict(x), dtype=float)
    sep = np.mean((p_hat < EPS_SEPARACION) | (p_hat > 1.0 - EPS_SEPARACION))

    ok = conv and finite and (sep <= config.PRIMARY_MODEL_MAX_SEPARATION_FRAC)
    reason = "ok" if ok else (
        "no_converge" if not conv
        else "coef_no_finito" if not finite
        else "separacion"
    )
    return FitResult(
        ok=ok,
        reason=reason,
        model=model,
        coef=coef,
        converged=conv,
        finite_coefs=finite,
        separation_frac=float(sep),
        diagnostics={
            "n_train": len(episodes_train),
            "n_conf_train": sum(1 for e in episodes_train if e["is_confirmed"]),
            "n_base_train": sum(1 for e in episodes_train if not e["is_confirmed"]),
            "model": "pooled",
        },
    )


def predict_rd_pool(fit: FitResult, episodes_eval: list[dict]) -> dict:
    """RD pooled para seleccion INNER (seccion 9.3 v4.1 revisada).

    Mean sobre todos los episodios del VALID de:
      P_hat(Y=1 | SOW=1, X_i) - P_hat(Y=1 | SOW=0, X_i)
    """
    if not fit.ok:
        return {
            "ok": False,
            "reason": fit.reason,
            "RD_pool": None,
            "n_total": 0,
        }
    x, _ = _design_matrix_pooled(episodes_eval)
    x1 = x.copy(); x1[:, 1] = 1.0
    x0 = x.copy(); x0[:, 1] = 0.0
    p1 = np.asarray(fit.model.predict(x1), dtype=float)
    p0 = np.asarray(fit.model.predict(x0), dtype=float)
    return {
        "ok": True,
        "reason": "ok",
        "RD_pool": float(np.mean(p1 - p0)),
        "n_total": len(episodes_eval),
    }


def fit_primary(episodes_train: list[dict]) -> FitResult:
    if len(episodes_train) < 10:
        return FitResult(ok=False, reason="n_train<10")
    x, y = _design_matrix(episodes_train)
    if y.min() == y.max():
        return FitResult(ok=False, reason="y_constante")
    try:
        model = sm.Logit(y, x).fit(disp=False, maxiter=200)
    except Exception as exc:
        return FitResult(ok=False, reason=f"fit_exception:{type(exc).__name__}")

    conv = bool(getattr(model, "mle_retvals", {}).get("converged", False))
    coef = np.asarray(model.params, dtype=float)
    finite = bool(np.all(np.isfinite(coef)))

    p_hat = np.asarray(model.predict(x), dtype=float)
    sep = np.mean((p_hat < EPS_SEPARACION) | (p_hat > 1.0 - EPS_SEPARACION))

    ok = conv and finite and (sep <= config.PRIMARY_MODEL_MAX_SEPARATION_FRAC)
    reason = "ok" if ok else (
        "no_converge" if not conv
        else "coef_no_finito" if not finite
        else "separacion"
    )

    return FitResult(
        ok=ok,
        reason=reason,
        model=model,
        coef=coef,
        converged=conv,
        finite_coefs=finite,
        separation_frac=float(sep),
        diagnostics={
            "n_train": len(episodes_train),
            "n_conf_train": sum(1 for e in episodes_train if e["is_confirmed"]),
            "n_base_train": sum(1 for e in episodes_train if not e["is_confirmed"]),
        },
    )


def predict_rd(
    fit: FitResult, episodes_eval: list[dict]
) -> dict:
    """Devuelve RD_stress, RD_normal, RD_pool con n efectivos y cobertura.

    No calcula IC (eso lo hace bootstrap).
    """
    if not fit.ok:
        return {
            "ok": False,
            "reason": fit.reason,
            "RD_stress": None,
            "RD_normal": None,
            "RD_pool": None,
            "n_stress": 0,
            "n_normal": 0,
        }

    x, _ = _design_matrix(episodes_eval)
    x1 = x.copy()
    x1[:, 1] = 1.0
    x0 = x.copy()
    x0[:, 1] = 0.0
    p1 = np.asarray(fit.model.predict(x1), dtype=float)
    p0 = np.asarray(fit.model.predict(x0), dtype=float)
    delta = p1 - p0

    mask_stress = np.array([e["R_episode"] == "stress" for e in episodes_eval])
    mask_normal = np.array([e["R_episode"] == "normal" for e in episodes_eval])
    mask_any = mask_stress | mask_normal

    def _mean(mask):
        if mask.sum() == 0:
            return None
        return float(np.mean(delta[mask]))

    return {
        "ok": True,
        "reason": "ok",
        "RD_stress": _mean(mask_stress),
        "RD_normal": _mean(mask_normal),
        "RD_pool": _mean(mask_any),
        "n_stress": int(mask_stress.sum()),
        "n_normal": int(mask_normal.sum()),
        "n_total": int(mask_any.sum()),
    }