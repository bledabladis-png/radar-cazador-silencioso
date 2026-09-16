"""Paquete MTE (DT2 Fase 5).

API publica re-exportada desde submodulos:
- state: persistencia
- scoring: componentes + agregados
- decision: maquina de estados + confianza
- engine: compute_mte
"""
from __future__ import annotations

from .state import (
    load_previous_scenario,
    save_scenario,
    MTE_STATE_FILE,
    CURRENT_TEMPORAL_CONTRACT_VERSION,
)

from .scoring import (
    tanh,
    _get_last,
    sector_rotation_score,
    safe_haven_score,
    credit_stress_score,
    inflation_pressure_score,
    compute_msi,
    compute_ipi,
    score_scenarios,
    SCENARIO_WEIGHTS,
)

from .decision import (
    NORMAL_TRANSITIONS,
    EXCEPTION_TRANSITIONS,
    validate_transition,
    consensus_score,
    distance_to_threshold,
    compute_confidence,
    classify_mte,
)

from .engine import compute_mte


__all__ = [
    "compute_mte",
    "compute_msi",
    "compute_ipi",
    "validate_transition",
    "sector_rotation_score",
    "safe_haven_score",
    "inflation_pressure_score",
    "credit_stress_score",
    "classify_mte",
    "score_scenarios",
    "consensus_score",
    "distance_to_threshold",
    "compute_confidence",
    "load_previous_scenario",
    "save_scenario",
    "MTE_STATE_FILE",
    "CURRENT_TEMPORAL_CONTRACT_VERSION",
    "NORMAL_TRANSITIONS",
    "EXCEPTION_TRANSITIONS",
    "tanh",
    "_get_last",
    "SCENARIO_WEIGHTS",
]
