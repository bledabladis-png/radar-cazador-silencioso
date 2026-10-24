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
# N1 ronda 2 (2026-10-14): guard no finitos + reason_code
# ============================================================
def test_classify_nan_breadth_unresolved_insufficient_data_breadth():
    """Breadth NaN -> UNRESOLVED con reason_code especifico."""
    r = classify_leadership_state(0.3, float('nan'), 0.6, coverage=0.7)
    assert r["state"] == "UNRESOLVED"
    assert r["reason_code"] == "INSUFFICIENT_DATA_BREADTH"
    assert r["data_quality"] == "LOW"
    assert "no disponible" in r["reason"].lower() or "no finito" in r["reason"].lower()


def test_classify_inf_breadth_unresolved_insufficient_data_breadth():
    """Breadth +Inf no se clasifica: guard defensivo."""
    r = classify_leadership_state(0.3, float('inf'), 0.6, coverage=0.7)
    assert r["state"] == "UNRESOLVED"
    assert r["reason_code"] == "INSUFFICIENT_DATA_BREADTH"
    r2 = classify_leadership_state(0.3, float('-inf'), 0.6, coverage=0.7)
    assert r2["state"] == "UNRESOLVED"
    assert r2["reason_code"] == "INSUFFICIENT_DATA_BREADTH"


def test_classify_none_breadth_unresolved_insufficient_data_breadth():
    r = classify_leadership_state(0.3, None, 0.6, coverage=0.7)
    assert r["state"] == "UNRESOLVED"
    assert r["reason_code"] == "INSUFFICIENT_DATA_BREADTH"


def test_classify_string_breadth_unresolved_insufficient_data_breadth():
    r = classify_leadership_state(0.3, "abc", 0.6, coverage=0.7)
    assert r["state"] == "UNRESOLVED"
    assert r["reason_code"] == "INSUFFICIENT_DATA_BREADTH"


def test_reason_code_coverage_baja_insufficient_coverage():
    r = classify_leadership_state(0.9, 0.9, 0.9, coverage=0.2)
    assert r["state"] == "UNRESOLVED"
    assert r["reason_code"] == "INSUFFICIENT_COVERAGE"


@pytest.mark.parametrize("state,args,expected_code", [
    ("CONFIRMED", (0.3, 0.5, 0.6, 0.7), "CONFIRMED"),
    ("EMERGING", (0.3, 0.5, 0.3, 0.7), "EMERGING"),
    ("STRUCTURAL_DECAY", (-0.25, 0.2, 0.5, 0.7), "STRUCTURAL_DECAY"),
    ("LOST", (-0.5, 0.2, 0.5, 0.7), "STRUCTURAL_LOST"),
    ("UNRESOLVED", (0.1, 0.4, 0.4, 0.5), "MIXED_OR_INSUFFICIENT_SIGNALS"),
])
def test_reason_code_presente_en_todos_los_estados(state, args, expected_code):
    """Cada estado devuelve su reason_code especifico."""
    r = classify_leadership_state(args[0], args[1], args[2], coverage=args[3])
    assert r["state"] == state
    assert r["reason_code"] == expected_code


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
def _leader_metric(rs=1.1, mom=0.05, flow=1.0, wyckoff="MARKUP",
                   rs_ratio_20d=0.02):
    # N1 (2026-10-13): rs_ratio_20d es el criterio del Leader Breadth
    # (fail-closed). El campo 'rs' se mantiene por compatibilidad con
    # LIS pero ya no alimenta el criterio binario.
    return {"rs": rs, "rs_ratio_20d": rs_ratio_20d,
            "rs_momentum": mom, "flow_proxy_z": flow,
            "wyckoff_phase": wyckoff}


def test_breadth_v2_sin_datos():
    # N1 ronda 2 (2026-10-14): lista vacia -> NO_LEADERS + NaN.
    import math
    r = compute_leader_breadth_v2([])
    assert math.isnan(r["composite"])
    assert r["rs_breadth_status"] == "NO_LEADERS"
    assert r["leader_breadth_status"] == "NO_LEADERS"
    assert r["n_used"] == 0
    assert r["coverage"] == 0.0
    assert r["coverage_warning"] is True


def test_breadth_v2_todos_favorables():
    leaders = [_leader_metric() for _ in range(5)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=5)
    assert r["rs_breadth"] == 1.0
    assert r["momentum_breadth"] == 1.0
    assert r["flow_breadth"] == 1.0
    assert r["wyckoff_breadth"] == 1.0
    assert r["coverage"] == 1.0


def test_breadth_v2_todos_desfavorables():
    leaders = [_leader_metric(rs=0.9, rs_ratio_20d=-0.02, mom=-0.05,
                              flow=-1.0, wyckoff="MARKDOWN")
               for _ in range(5)]
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
# N1 ronda 2 (2026-10-14): tests contractuales del nuevo contrato
# ============================================================
import math


def test_n1_none_input_unavailable():
    r = compute_leader_breadth_v2(None)
    assert math.isnan(r["rs_breadth"])
    assert r["rs_breadth_status"] == "INPUT_UNAVAILABLE"
    assert r["leader_breadth_status"] == "INPUT_UNAVAILABLE"
    assert r["n_used"] == 0
    assert r["coverage_warning"] is True


def test_n1_lista_vacia_no_leaders():
    r = compute_leader_breadth_v2([])
    assert math.isnan(r["rs_breadth"])
    assert r["rs_breadth_status"] == "NO_LEADERS"
    assert r["leader_breadth_status"] == "NO_LEADERS"


@pytest.mark.parametrize("bad_value", [
    float("inf"), float("-inf"), float("nan"),
    "abc", "1.2.3", [], {}, object(),
    10**400,
])
def test_n1_valores_no_finitos_o_no_numericos_son_ausentes(bad_value):
    """Cualquier valor RS no convertible o no finito cuenta como
    ausente (C1+C8, incluye OverflowError)."""
    leaders = [{"rs_ratio_20d": bad_value, "rs_momentum": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}
               for _ in range(3)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=3)
    assert math.isnan(r["rs_breadth"])
    assert r["rs_breadth_status"] == "INSUFFICIENT_RS_RATIO"
    assert r["rs_breadth_missing"] == 3


def test_n1_rs_no_finito_no_se_clasifica_como_positivo():
    """+Inf no es positivo, -Inf no es negativo."""
    r = compute_leader_breadth_v2(
        [{"rs_ratio_20d": float("inf"), "rs_momentum": 0.05,
          "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}],
        expected_leaders=1)
    assert math.isnan(r["rs_breadth"])
    r = compute_leader_breadth_v2(
        [{"rs_ratio_20d": float("-inf"), "rs_momentum": 0.05,
          "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}],
        expected_leaders=1)
    assert math.isnan(r["rs_breadth"])


def test_n1_composite_nan_cuando_rs_no_disponible():
    """Composite dependiente de rs queda N/D; componentes
    independientes siguen como diagnostico."""
    leaders = [
        _leader_metric(rs_ratio_20d=0.02),
        {"rs": 1.1, "rs_momentum": 0.05,
         "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"},
    ]
    r = compute_leader_breadth_v2(leaders, expected_leaders=2)
    assert math.isnan(r["composite"])
    assert math.isnan(r["effective_composite"])
    assert r["leader_breadth_status"] == "INSUFFICIENT_RS_RATIO"
    assert r["momentum_breadth"] == 1.0
    assert r["flow_breadth"] == 1.0
    assert r["wyckoff_breadth"] == 1.0


def test_c6_composite_nan_cuando_falta_otro_componente():
    """C6: si falta momentum/flow/wyckoff, composite=NaN +
    INSUFFICIENT_COMPONENT_DATA."""
    leaders = [
        _leader_metric(),
        {"rs_ratio_20d": 0.02, "rs_momentum": 0.05,
         "wyckoff_phase": "MARKUP"},  # sin flow_proxy_z
    ]
    r = compute_leader_breadth_v2(leaders, expected_leaders=2)
    assert math.isnan(r["composite"])
    assert r["leader_breadth_status"] == "INSUFFICIENT_COMPONENT_DATA"
    assert math.isnan(r["flow_breadth"])
    assert r["rs_breadth"] == 1.0
    assert r["momentum_breadth"] == 1.0
    assert r["wyckoff_breadth"] == 1.0


def test_c6_falta_ambos_campos_momentum_es_ausente():
    leaders = [{"rs_ratio_20d": 0.02, "flow_proxy_z": 1.0,
                "wyckoff_phase": "MARKUP"}]
    r = compute_leader_breadth_v2(leaders, expected_leaders=1)
    assert math.isnan(r["momentum_breadth"])
    assert math.isnan(r["composite"])
    assert r["leader_breadth_status"] == "INSUFFICIENT_COMPONENT_DATA"


def test_c6_wyckoff_phase_invalida_cuenta_como_ausente():
    leaders = [{"rs_ratio_20d": 0.02, "rs_momentum": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "FOO"}]
    r = compute_leader_breadth_v2(leaders, expected_leaders=1)
    assert math.isnan(r["wyckoff_breadth"])
    assert r["leader_breadth_status"] == "INSUFFICIENT_COMPONENT_DATA"


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


# ============================================================
# N1 (2026-10-13, auditor): contrato fail-closed de rs_ratio_20d.
# Si falta el campo, rs_breadth = NaN + status INSUFFICIENT.
# NUNCA se cae al criterio de precio nominal (rs > 1.0).
# ============================================================

def test_n1_sin_rs_ratio_20d_no_cae_a_criterio_nominal():
    """Sin rs_ratio_20d: rs_breadth = NaN, status INSUFFICIENT.
    Aunque 'rs' > 1.0 (criterio antiguo), no se usa."""
    leaders = [{"rs": 5.0, "rs_momentum": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}
               for _ in range(5)]
    # NO tienen rs_ratio_20d
    r = compute_leader_breadth_v2(leaders, expected_leaders=5)
    import math
    assert math.isnan(r["rs_breadth"]), "no debe caer al criterio rs>1.0"
    assert r["rs_breadth_status"] == "INSUFFICIENT_RS_RATIO"
    assert r["rs_breadth_missing"] == 5


def test_n1_con_rs_ratio_20d_parcial():
    """N1 ronda 2 (2026-10-14): RS parcial -> fail-closed estricto.
    No se calcula sobre los validos (dictamen 1.3)."""
    import math
    leaders = [
        _leader_metric(rs_ratio_20d=0.02),
        _leader_metric(rs_ratio_20d=0.03),
        {"rs": 1.1, "rs_momentum": 0.05,
         "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"},  # sin rs_ratio
    ]
    r = compute_leader_breadth_v2(leaders, expected_leaders=3)
    assert math.isnan(r["rs_breadth"]), "parcial no debe calcular"
    assert r["rs_breadth_status"] == "INSUFFICIENT_RS_RATIO"
    assert r["leader_breadth_status"] == "INSUFFICIENT_RS_RATIO"
    assert r["rs_breadth_missing"] == 1


def test_n1_rs_ratio_20d_negativo_no_es_positivo():
    """rs_ratio_20d < 0 -> no positivo."""
    leaders = [_leader_metric(rs_ratio_20d=-0.05) for _ in range(3)]
    r = compute_leader_breadth_v2(leaders, expected_leaders=3)
    assert r["rs_breadth"] == 0.0
    assert r["rs_breadth_status"] == "OK"
    assert r["rs_breadth_missing"] == 0


# ============================================================
# N1 ronda 2 (2026-10-14): cortocircuito en evaluate_slpm_v12 (opcion D)
# Los tests mockean confirm_transition para evitar escritura a disco
# (colision entre workers xdist sobre outputs/state/slpm_state.json).
# ============================================================
import pytest as _pytest


@_pytest.fixture
def _mock_confirm_transition(monkeypatch):
    def _fake(sector_etf, instant_state):
        return {
            "confirmed_state": instant_state,
            "previous_state": instant_state,
            "transition": False,
            "consecutive_count": 1,
        }
    import indicators.slpm_v12 as _slpm
    monkeypatch.setattr(_slpm, "confirm_transition", _fake)
    return _fake


def test_evaluate_slpm_shortcircuit_cuando_rs_no_disponible(_mock_confirm_transition):
    """Si rs_breadth_status != OK, no se llama al clasificador.
    Dictamen opcion D: cortocircuito principal."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    # Lider sin rs_ratio_20d -> INSUFFICIENT_RS_RATIO -> cortocircuito
    leaders = [{"rs": 1.1, "rs_momentum": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}]
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, leaders, 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    assert r["state"] == "UNRESOLVED"
    assert r["state_reason_code"] == "INSUFFICIENT_DATA_RS"
    assert r["data_quality"] == "LOW"
    assert "rs_ratio_20d" in r["state_reason"]


def test_evaluate_slpm_shortcircuit_input_unavailable(_mock_confirm_transition):
    """leader_metrics=None -> INPUT_UNAVAILABLE propagado."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, None, 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    assert r["state"] == "UNRESOLVED"
    assert r["state_reason_code"] == "INPUT_UNAVAILABLE"
    assert r["data_quality"] == "LOW"


def test_evaluate_slpm_shortcircuit_no_leaders(_mock_confirm_transition):
    """leader_metrics=[] -> NO_LEADERS propagado."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, [], 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    assert r["state"] == "UNRESOLVED"
    assert r["state_reason_code"] == "NO_LEADERS"
    assert r["data_quality"] == "LOW"


def test_evaluate_slpm_shortcircuit_componente_faltante(_mock_confirm_transition):
    """Falta flow_proxy_z -> INSUFFICIENT_COMPONENT_DATA propagado."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    leaders = [{"rs_ratio_20d": 0.02, "rs_momentum": 0.05,
                "wyckoff_phase": "MARKUP"}]  # sin flow_proxy_z
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, leaders, 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    assert r["state"] == "UNRESOLVED"
    assert r["state_reason_code"] == "INSUFFICIENT_COMPONENT_DATA"
    assert r["data_quality"] == "LOW"


# ============================================================
# N1 ronda 2 (2026-10-14): integracion end-to-end del contrato
# Recorre compute_leader_breadth_v2 -> evaluate_slpm_v12 ->
# render_slpm_v12 verificando coherencia de los 4 casos y ausencia
# de conversion silenciosa a cero.
# ============================================================
import pytest as _pytest_e2e


@_pytest_e2e.fixture
def _mock_confirm(monkeypatch):
    def _fake(sector_etf, instant_state):
        return {
            "confirmed_state": instant_state,
            "previous_state": instant_state,
            "transition": False,
            "consecutive_count": 1,
        }
    import indicators.slpm_v12 as _s
    monkeypatch.setattr(_s, "confirm_transition", _fake)
    return _fake


@_pytest_e2e.mark.parametrize("leaders,expected_code,expected_breadth_status", [
    (None, "INPUT_UNAVAILABLE", "INPUT_UNAVAILABLE"),
    ([], "NO_LEADERS", "NO_LEADERS"),
    ([{"rs": 1.1, "rs_momentum": 0.05,
       "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}],
     "INSUFFICIENT_DATA_RS", "INSUFFICIENT_RS_RATIO"),
    ([{"rs_ratio_20d": 0.02, "rs_momentum": 0.05,
       "wyckoff_phase": "MARKUP"}],
     "INSUFFICIENT_COMPONENT_DATA", "OK"),
])
def test_e2e_n1_coherencia_cuatro_casos(
    leaders, expected_code, expected_breadth_status, _mock_confirm
):
    """Los 4 casos del dictamen §1.5 llegan coherentes al render."""
    from src.report.slpm import render_slpm_v12
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, leaders, 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    assert r["state"] == "UNRESOLVED"
    assert r["state_reason_code"] == expected_code
    assert r["data_quality"] == "LOW"

    breadth = r["leader_breadth_v2"]
    assert breadth["rs_breadth_status"] == expected_breadth_status
    if expected_breadth_status != "OK":
        assert math.isnan(breadth["rs_breadth"])
    assert math.isnan(breadth["composite"])
    assert math.isnan(breadth["effective_composite"])

    lines = render_slpm_v12(r)
    joined = chr(10).join(lines)
    assert "nan%" not in joined
    assert "N/D" in joined


def test_e2e_n1_caso_valido_no_se_convierte_a_cero(_mock_confirm):
    """Todos los componentes validos: composite calculado, no N/D."""
    ranking = [("XLK", "Technology", 0.5, "MARKUP")]
    leaders = [{"rs_ratio_20d": 0.02, "rs_momentum": 0.05,
                "flow_proxy_z": 1.0, "wyckoff_phase": "MARKUP"}
               for _ in range(5)]
    r = evaluate_slpm_v12(
        None, {"ranking": ranking}, leaders, 0.0,
        tactical_scores={"XLK": 0.5},
        structural_scores={"XLK": 0.5},
        sector_persistence={"XLK": 0.6},
    )
    breadth = r["leader_breadth_v2"]
    assert breadth["leader_breadth_status"] == "OK"
    assert math.isfinite(breadth["composite"])
    assert breadth["composite"] > 0


def test_valid_phases_cubre_todas_las_fases_del_pipeline():
    """_VALID_WYCKOFF_PHASES debe cubrir todas las fases que el pipeline
    puede emitir (legacy + orquestador nuevo). Si falta una, breadth
    se anula silenciosamente (fail-closed por componente)."""
    from indicators.slpm_v12 import _VALID_WYCKOFF_PHASES
    expected = {
        "MARKUP", "ACCUMULATION", "RANGE", "DISTRIBUTION", "MARKDOWN",
        "DISTRIBUTION_FORMING", "ACCUMULATION_FORMING", "CONFLICT",
        "INSUFFICIENT_DATA",
    }
    assert _VALID_WYCKOFF_PHASES == expected


def test_formings_y_conflict_no_anulan_leader_breadth():
    """Fases nuevas (FORMING, CONFLICT, INSUFFICIENT_DATA) no deben
    anular leader_breadth. Se comportan como RANGE: validas, no
    favorables (0.0 en wyckoff_map, False en _wyckoff_favorable)."""
    leaders = [
        {"rs_ratio_20d": 0.01, "rs_momentum": 0.01, "flow_proxy_z": 0.5,
         "wyckoff_phase": "DISTRIBUTION_FORMING"},
        {"rs_ratio_20d": 0.01, "rs_momentum": 0.01, "flow_proxy_z": 0.5,
         "wyckoff_phase": "ACCUMULATION_FORMING"},
        {"rs_ratio_20d": 0.01, "rs_momentum": 0.01, "flow_proxy_z": 0.5,
         "wyckoff_phase": "CONFLICT"},
        {"rs_ratio_20d": 0.01, "rs_momentum": 0.01, "flow_proxy_z": 0.5,
         "wyckoff_phase": "INSUFFICIENT_DATA"},
    ]
    r = compute_leader_breadth_v2(leaders, expected_leaders=4)
    assert r["leader_breadth_status"] == "OK"
    assert r["wyckoff_breadth"] == 0.0
    assert r["wyckoff_breadth_missing"] == 0
    # composite debe ser numerico, no NaN
    assert r["composite"] == r["composite"]
