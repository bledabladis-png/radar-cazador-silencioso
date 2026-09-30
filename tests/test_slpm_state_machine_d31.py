# -*- coding: utf-8 -*-
"""D31 (2026-09-30): tests para indicators/state_machine.py y
indicators/slpm_v12.py (dos modulos SLPM con cobertura baja).

- state_machine: 12% -> cobertura de classify_leadership_state,
  get_opportunity_quadrant, validate_state.
- slpm_v12: 38% -> _safe_mean, compute_leader_breadth_v2,
  compute_leader_integrity, compute_flow_divergence_v2,
  evaluate_slpm_v12 (orquestador).
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from indicators.state_machine import (
    classify_leadership_state,
    get_opportunity_quadrant,
    validate_state,
)
from indicators.slpm_v12 import (
    _safe_mean,
    compute_leader_breadth_v2,
    compute_leader_integrity,
    compute_flow_divergence_v2,
    evaluate_slpm_v12,
)


# ============================================================
# state_machine.classify_leadership_state
# ============================================================
def test_classify_cobertura_baja_unresolved():
    """coverage < 0.30 -> UNRESOLVED con data_quality LOW."""
    r = classify_leadership_state(0.9, 0.9, 0.9, coverage=0.2)
    assert r["state"] == "UNRESOLVED"
    assert r["data_quality"] == "LOW"


def test_classify_lost():
    r = classify_leadership_state(-0.5, 0.2, 0.5, coverage=0.7)
    assert r["state"] == "LOST"
    assert r["data_quality"] == "HIGH"


def test_classify_structural_decay():
    r = classify_leadership_state(-0.25, 0.2, 0.5, coverage=0.7)
    assert r["state"] == "STRUCTURAL_DECAY"


def test_classify_confirmed():
    r = classify_leadership_state(0.3, 0.5, 0.6, coverage=0.7)
    assert r["state"] == "CONFIRMED"
    assert r["data_quality"] == "HIGH"


def test_classify_emerging():
    r = classify_leadership_state(0.3, 0.5, 0.3, coverage=0.7)
    assert r["state"] == "EMERGING"


def test_classify_unresolved_sin_condiciones():
    r = classify_leadership_state(0.1, 0.4, 0.4, coverage=0.5)
    assert r["state"] == "UNRESOLVED"


def test_classify_data_quality_medium():
    """coverage entre 0.30 y 0.60 -> MEDIUM en CONFIRMED."""
    r = classify_leadership_state(0.3, 0.5, 0.6, coverage=0.5)
    assert r["state"] == "CONFIRMED"
    assert r["data_quality"] == "MEDIUM"


# ============================================================
# state_machine.get_opportunity_quadrant
# ============================================================
@pytest.mark.parametrize("state,expected", [
    ("CONFIRMED", "Structural Strength"),
    ("EMERGING", "Structural Strength"),
    ("TACTICAL_CORRECTION", "Tactical Correction"),
    ("STRUCTURAL_DECAY", "Structural Weakness"),
    ("LOST", "Structural Weakness"),
    ("UNRESOLVED", "Transition"),
])
def test_get_opportunity_quadrant_estados(state, expected):
    assert get_opportunity_quadrant(state) == expected


def test_get_opportunity_quadrant_desconocido():
    assert get_opportunity_quadrant("XYZ") == "Transition"


# ============================================================
# state_machine.validate_state
# ============================================================
def test_validate_confirmed_ok():
    assert validate_state("CONFIRMED", 0.3, 0.5, 0.6) == []


def test_validate_confirmed_errores_multiples():
    errs = validate_state("CONFIRMED", 0.1, 0.2, 0.3)
    assert len(errs) == 3


def test_validate_structural_decay_ok():
    assert validate_state("STRUCTURAL_DECAY", -0.25, 0.3, 0.5) == []


def test_validate_structural_decay_errores():
    errs = validate_state("STRUCTURAL_DECAY", 0.0, 0.5, 0.5)
    assert len(errs) == 2


def test_validate_estado_sin_checks():
    """EMERGING/UNRESOLVED no tienen validacion -> []."""
    assert validate_state("EMERGING", 0.0, 0.0, 0.0) == []


# ============================================================
# slpm_v12._safe_mean
# ============================================================
def test_safe_mean_vacio():
    assert _safe_mean([]) == 0.0


def test_safe_mean_con_nan():
    assert _safe_mean([1.0, float("nan"), 3.0]) == 2.0


def test_safe_mean_todo_nan():
    assert _safe_mean([float("nan"), float("nan")]) == 0.0


# ============================================================
# slpm_v12.compute_leader_breadth_v2
# ============================================================
def _leader_metric(rs=1.1, mom=0.05, flow=1.0, wyckoff="MARKUP"):
    return {"rs": rs, "rs_momentum": mom, "flow_proxy_z": flow,
            "wyckoff_phase": wyckoff}


def test_breadth_v2_sin_datos():
    r = compute_leader_breadth_v2([])
    assert r["composite"] == 0.0
    assert r["n_used"] == 0
    assert r["coverage"] == 0.0


def test_breadth_v2_todos_favorables():
    leaders = [_leader_metric() for _ in range(5)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=5)
    assert r["rs_breadth"] == 1.0
    assert r["momentum_breadth"] == 1.0
    assert r["flow_breadth"] == 1.0
    assert r["wyckoff_breadth"] == 1.0
    assert r["coverage"] == 1.0


def test_breadth_v2_todos_desfavorables():
    leaders = [_leader_metric(rs=0.9, mom=-0.05, flow=-1.0,
                              wyckoff="MARKDOWN") for _ in range(5)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=5)
    assert r["rs_breadth"] == 0.0
    assert r["momentum_breadth"] == 0.0
    assert r["flow_breadth"] == 0.0
    assert r["wyckoff_breadth"] == 0.0


def test_breadth_v2_coverage_baja_efectiva():
    """coverage < 0.5 -> effective_composite = composite * coverage."""
    leaders = [_leader_metric() for _ in range(2)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=10)
    assert r["coverage"] == 0.2
    assert r["coverage_warning"] is True
    assert r["effective_composite"] < r["composite"]


def test_breadth_v2_fallback_rs_mom_20():
    """Sin rs_momentum, usa rs_mom_20 (A3.3-06)."""
    leaders = [{"rs": 1.1, "rs_mom_20": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}]
    r = compute_leader_breadth_v2(leaders, expected_leaders=1)
    assert r["momentum_breadth"] == 1.0


# ============================================================
# slpm_v12.compute_leader_integrity
# ============================================================
def test_integrity_sin_datos():
    r = compute_leader_integrity([])
    assert r["lis"] == 0.0
    assert r["n_leaders"] == 0


def test_integrity_normal():
    leaders = [_leader_metric()]
    r = compute_leader_integrity(leaders)
    assert r["n_leaders"] == 1
    assert -1.0 <= r["lis"] <= 1.0


def test_integrity_flow_cero_no_nan():
    """Fix 2026-09-29: flow_proxy_z=0.0 es legitimo, no se convierte a NaN."""
    leaders = [{"rs": 1.0, "rs_momentum": 0.0, "flow_proxy_z": 0.0,
                "wyckoff_phase": "RANGE"}]
    r = compute_leader_integrity(leaders)
    assert np.isfinite(r["lis"])


def test_integrity_flow_ausente_nan():
    """Sin flow_proxy_z -> NaN tratado como NaN (no crashea)."""
    leaders = [{"rs": 1.0, "rs_momentum": 0.0, "wyckoff_phase": "RANGE"}]
    r = compute_leader_integrity(leaders)
    assert np.isfinite(r["lis"])


# ============================================================
# slpm_v12.compute_flow_divergence_v2
# ============================================================
def test_flow_div_v2_sin_leaders():
    r = compute_flow_divergence_v2([], 0.5)
    assert pd.isna(r["leader_flow_div"])
    assert pd.isna(r["structural_flow_div"])


def test_flow_div_v2_normal():
    leaders = [_leader_metric(flow=1.0), _leader_metric(flow=0.5)]
    r = compute_flow_divergence_v2(leaders, sector_flow_proxy_z=0.2)
    assert r["leader_flow_div"] == pytest.approx(0.75 - 0.2)


def test_flow_div_v2_con_sector_price_flow():
    leaders = [_leader_metric(flow=1.0)]
    r = compute_flow_divergence_v2(
        leaders, sector_flow_proxy_z=0.5, sector_price_flow=0.2)
    assert r["sector_flow_vs_price_div"] == pytest.approx(0.3)


# ============================================================
# slpm_v12.evaluate_slpm_v12
# ============================================================
def test_evaluate_slpm_v12_sin_ranking():
    r = evaluate_slpm_v12(None, {}, [], 0.0)
    assert r["state"] == "UNRESOLVED"
    assert r["sector"] == ""


def test_evaluate_slpm_v12_normal():
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    sector_results = {"ranking": ranking}
    leaders = [_leader_metric() for _ in range(5)]
    with patch("indicators.slpm_v12.confirm_transition") as mock_ct:
        mock_ct.return_value = {
            "confirmed_state": "CONFIRMED", "previous_state": "CONFIRMED",
            "transition": False, "consecutive_count": 3,
        }
        r = evaluate_slpm_v12(
            None, sector_results, leaders, top_sector_flow=0.5,
            structural_scores={"XLK": 0.3},
            sector_persistence={"XLK": 0.6})
    assert r["sector_etf"] == "XLK"
    assert r["state"] == "CONFIRMED"
    assert r["opportunity_quadrant"] == "Structural Strength"


def test_evaluate_slpm_v12_persistence_none():
    """persistence None -> 0.5 con WARN."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    sector_results = {"ranking": ranking}
    with patch("indicators.slpm_v12.confirm_transition") as mock_ct:
        mock_ct.return_value = {
            "confirmed_state": "UNRESOLVED", "previous_state": "UNRESOLVED",
            "transition": False, "consecutive_count": 1,
        }
        r = evaluate_slpm_v12(
            None, sector_results, [], top_sector_flow=0.0,
            structural_scores={"XLK": 0.0}, sector_persistence=None)
    assert r["persistence"] == 0.5


def test_evaluate_slpm_v12_sin_leaders_anade_nota():
    """leader_metrics vacio -> anade nota al reason."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    with patch("indicators.slpm_v12.confirm_transition") as mock_ct:
        mock_ct.return_value = {
            "confirmed_state": "UNRESOLVED", "previous_state": "UNRESOLVED",
            "transition": False, "consecutive_count": 1,
        }
        r = evaluate_slpm_v12(None, {"ranking": ranking}, [], 0.0)
    assert "XLK" in r["state_reason"] or "Technology" in r["state_reason"]
