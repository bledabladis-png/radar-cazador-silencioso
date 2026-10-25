# -*- coding: utf-8 -*-
"""Tests respuesta al dictamen de auditoria externa 2026-10-25.

Cubre:
- H-02: SOW confirma con 3 cierres bajo soporte-tol; 2 no confirman.
- H-03: FORMING invalida si el precio rompe el rango por cualquier lado.
"""
import pandas as pd

from indicators.wyckoff_event import ClassifiedBreak, Range
from indicators.wyckoff_orchestrator import (
    _acc_derived_phase, _dist_derived_phase,
)
from indicators.wyckoff_distribution import detect_sow_in_range


# =====================================================================
# H-02 - SOW: 3 confirman, 2 no
# =====================================================================

def _build_range_dist(start="2024-01-02", support=100.0, resistance=110.0, tol=1.0):
    return Range(
        range_id="X_test",
        ticker="X",
        timeframe="1d",
        formation_path="BC_AR",
        range_type="DISTRIBUTION_CANDIDATE",
        candidate_start_date=pd.Timestamp(start) - pd.Timedelta(days=60),
        start_available_at=pd.Timestamp(start),
        range_active_at=pd.Timestamp(start),
        support=support,
        resistance=resistance,
        tolerance_frac=0.05,
        tol_abs=tol,
        state="RANGE_ACTIVE",
        trend_context={},
    )


def test_h02_sow_confirma_con_tres_cierres_bajo_soporte():
    """H-02: 3 closes < support - tol confirman SOW.

    Prueba que la iteracion recorre la ventana completa (no solo 2
    posiciones como sugeria el pseudocodigo del informe preliminar).
    """
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    # range_active_at = idx 0; window comienza en idx 1
    # closes: [101, 98.5, 98.0, 97.5, 102, 103, 104]
    # 3 closes < 99 (idx 2,3,4) -> CONFIRMED
    df = pd.DataFrame(
        {"Close": [102.0, 101.0, 98.5, 98.0, 97.5, 102.0, 103.0, 104.0]},
        index=dates,
    )
    rng = _build_range_dist(start="2024-01-02")
    sow = detect_sow_in_range(df, "X", rng, as_of=df.index[-1])
    assert sow.status == "CONFIRMED", f"esperado CONFIRMED, fue {sow.status}"
    assert sow.event_date == dates[2]  # candidate = primer close < 99


def test_h02_sow_no_confirma_con_dos_cierres_y_recuperacion():
    """H-02: 2 closes bajo soporte, luego close > support + tol -> REJECTED."""
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    # 2 closes < 99 (idx 2,3), luego idx 4 cierra en 102 (>101 = support+tol)
    df = pd.DataFrame(
        {"Close": [102.0, 101.0, 98.5, 98.0, 102.0, 103.0, 104.0, 105.0]},
        index=dates,
    )
    rng = _build_range_dist(start="2024-01-02")
    sow = detect_sow_in_range(df, "X", rng, as_of=df.index[-1])
    assert sow.status == "REJECTED", f"esperado REJECTED, fue {sow.status}"


def test_h02_sow_no_confirma_sin_ruptura():
    """H-02: sin close < support - tol, NO_OBSERVED."""
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    df = pd.DataFrame(
        {"Close": [102.0, 101.0, 100.5, 100.0, 100.5, 101.0, 101.5, 102.0]},
        index=dates,
    )
    rng = _build_range_dist(start="2024-01-02")
    sow = detect_sow_in_range(df, "X", rng, as_of=df.index[-1])
    assert sow.status == "NOT_OBSERVED"


# =====================================================================
# H-03 - FORMING invalida si el precio rompe el rango
# =====================================================================

def _cb_dist_forming(support=100.0, resistance=110.0, tol=1.0):
    return ClassifiedBreak(
        ticker="X", as_of=pd.Timestamp("2026-10-23"), range_id="r1",
        classification="DISTRIBUTION_FORMING",
        structural_state="RANGE_ACTIVE", break_direction=None,
        range_type="DISTRIBUTION_CANDIDATE", trend_direction="UP",
        sow_status="NOT_OBSERVED", coherence_error=None, evidence={},
        range_support=support, range_resistance=resistance, range_tol_abs=tol,
    )


def _cb_acc_forming(support=100.0, resistance=110.0, tol=1.0):
    return ClassifiedBreak(
        ticker="X", as_of=pd.Timestamp("2026-10-23"), range_id="r1",
        classification="ACCUMULATION_FORMING",
        structural_state="RANGE_ACTIVE", break_direction=None,
        range_type="ACCUMULATION_CANDIDATE", trend_direction="DOWN",
        sow_status="NOT_OBSERVED", coherence_error=None, evidence={},
        range_support=support, range_resistance=resistance, range_tol_abs=tol,
    )


def test_h03_dist_forming_dentro_rango():
    """H-03: FORMING con precio dentro del rango sigue FORMING."""
    cb = _cb_dist_forming()
    assert _dist_derived_phase(cb, 105.0) == "DISTRIBUTION_FORMING"
    # Bordes: exactamente support - tol y resistance + tol: aun dentro
    assert _dist_derived_phase(cb, 99.0) == "DISTRIBUTION_FORMING"
    assert _dist_derived_phase(cb, 111.0) == "DISTRIBUTION_FORMING"


def test_h03_dist_forming_invalida_si_rompe_resistencia():
    """H-03: FORMING + price > resistance + tol -> RANGE (dist fallo)."""
    cb = _cb_dist_forming()
    assert _dist_derived_phase(cb, 112.0) == "RANGE"


def test_h03_dist_forming_invalida_si_rompe_soporte():
    """H-03: FORMING + price < support - tol -> RANGE (intento fallido)."""
    cb = _cb_dist_forming()
    assert _dist_derived_phase(cb, 98.0) == "RANGE"


def test_h03_acc_forming_dentro_rango():
    """H-03: FORMING con precio dentro del rango sigue FORMING."""
    cb = _cb_acc_forming()
    assert _acc_derived_phase(cb, 105.0) == "ACCUMULATION_FORMING"


def test_h03_acc_forming_invalida_si_pierde_soporte():
    """H-03: FORMING + price < support - tol -> RANGE (acc fallo)."""
    cb = _cb_acc_forming()
    assert _acc_derived_phase(cb, 98.0) == "RANGE"


def test_h03_acc_forming_invalida_si_rompe_resistencia():
    """H-03: FORMING + price > resistance + tol -> RANGE (rango roto arriba)."""
    cb = _cb_acc_forming()
    assert _acc_derived_phase(cb, 112.0) == "RANGE"


def test_h03_forming_sin_niveles_no_crashea():
    """H-03: cb sin niveles no crashea (defensivo)."""
    cb = ClassifiedBreak(
        ticker="X", as_of=pd.Timestamp("2026-10-23"), range_id=None,
        classification="DISTRIBUTION_FORMING",
        structural_state="NOT_EVALUABLE", break_direction=None,
        range_type=None, trend_direction=None,
        sow_status=None, coherence_error=None, evidence={},
        range_support=None, range_resistance=None, range_tol_abs=None,
    )
    # Sin niveles, no puede verificar el rango -> mantiene FORMING
    assert _dist_derived_phase(cb, 105.0) == "DISTRIBUTION_FORMING"
