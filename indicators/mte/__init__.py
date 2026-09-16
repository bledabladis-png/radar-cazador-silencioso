"""Paquete MTE (DT2 Fase 3).

Re-exporta la API publica desde submodulos:
- mte_legacy: compute_mte + scoring + decision + state-machine (fases 3-6 los extraeran)
- state: load_previous_scenario, save_scenario

Todos los simbolos que antes se importaban desde indicators.mte siguen
resolviendose desde aqui.
"""
from __future__ import annotations

# Re-export de state (submodulo real)
from .state import (
    load_previous_scenario,
    save_scenario,
    MTE_STATE_FILE,
    CURRENT_TEMPORAL_CONTRACT_VERSION,
)

# Re-export de mte_legacy (API publica completa)
from .mte_legacy import (
    compute_mte,
    compute_msi,
    compute_ipi,
    validate_transition,
    sector_rotation_score,
    safe_haven_score,
    inflation_pressure_score,
    credit_stress_score,
    classify_mte,
    score_scenarios,
    consensus_score,
    distance_to_threshold,
    compute_confidence,
)


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
]