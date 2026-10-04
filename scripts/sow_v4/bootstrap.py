"""Bootstrap moving block sobre el panel de episodios (seccion 7).

Unidad de remuestreo: bloques temporales de B sesiones, con todos los
episodios cuyo t0 cae dentro del bloque. Asignacion exclusiva por t0
(seccion 7.2).

Modelo CONGELADO: no se reajusta dentro del bootstrap (regla bloqueante
B4). Solo se aplica `fit.model.predict` sobre la submuestra bootstrap.

Longitudes: B20 primario, B40 sensibilidad confirmatoria, B50 diagnostico.
"""
from __future__ import annotations

import numpy as np

from scripts.sow_v4.model_primary import FitResult, predict_rd


def _session_to_index(session_dates) -> dict:
    return {d: i for i, d in enumerate(session_dates)}


def bootstrap_rd(
    fit: FitResult,
    episodes_eval: list[dict],
    session_dates: list,
    B: int,
    n_replicas: int,
    seed: int,
) -> dict | None:
    """Moving block bootstrap.

    Devuelve dict con arrays de RD_stress, RD_normal, RD_pool.
    None si T < B o no hay episodios.
    """
    if not fit.ok:
        return None
    if not episodes_eval:
        return None

    s2i = _session_to_index(session_dates)
    try:
        t0_idx = np.array([s2i[e["t0_date"]] for e in episodes_eval], dtype=int)
    except KeyError:
        return None

    T = len(session_dates)
    if T < B:
        return None

    # Precomputar bloques: por cada posible inicio, indices de episodios
    # cuyo t0 esta en [start, start+B)
    blocks: list[np.ndarray] = []
    for start in range(T - B + 1):
        end = start + B
        mask = (t0_idx >= start) & (t0_idx < end)
        blocks.append(np.where(mask)[0])

    n_blocks_per_sample = int(np.ceil(T / B))
    rng = np.random.default_rng(seed)

    rd_stress = np.full(n_replicas, np.nan)
    rd_normal = np.full(n_replicas, np.nan)
    rd_pool = np.full(n_replicas, np.nan)

    for b in range(n_replicas):
        chosen: list[np.ndarray] = []
        for _ in range(n_blocks_per_sample):
            k = int(rng.integers(0, len(blocks)))
            chosen.append(blocks[k])
        if not chosen:
            continue
        idx_sel = np.concatenate(chosen) if chosen else np.array([], dtype=int)
        if len(idx_sel) == 0:
            continue
        sub = [episodes_eval[i] for i in idx_sel]
        rd = predict_rd(fit, sub)
        if not rd["ok"]:
            continue
        if rd["RD_stress"] is not None:
            rd_stress[b] = rd["RD_stress"]
        if rd["RD_normal"] is not None:
            rd_normal[b] = rd["RD_normal"]
        if rd["RD_pool"] is not None:
            rd_pool[b] = rd["RD_pool"]

    return {
        "rd_stress": rd_stress,
        "rd_normal": rd_normal,
        "rd_pool": rd_pool,
        "B": B,
        "n_replicas": n_replicas,
    }


def percentile_ci(arr: np.ndarray, low: float, high: float) -> tuple[float | None, float | None]:
    valid = arr[~np.isnan(arr)]
    if len(valid) < 50:
        return (None, None)
    return (float(np.percentile(valid, low)), float(np.percentile(valid, high)))


def ci95(arr: np.ndarray) -> tuple[float | None, float | None]:
    return percentile_ci(arr, 2.5, 97.5)


def ci90(arr: np.ndarray) -> tuple[float | None, float | None]:
    return percentile_ci(arr, 5.0, 95.0)