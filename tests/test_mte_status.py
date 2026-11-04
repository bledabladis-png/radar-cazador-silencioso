# -*- coding: utf-8 -*-
from indicators.mte.status import (
    VALID, DEGRADED, INVALID, COVERAGE_MIN, derive_mte_state,
)


def test_valid_todo_ok():
    assert derive_mte_state("OK", 0.95, 0.6) == VALID
    assert derive_mte_state("OK", COVERAGE_MIN, 0.1) == VALID


def test_degraded_por_futures_stale():
    assert derive_mte_state("STALE", 0.95, 0.6) == DEGRADED


def test_degraded_por_coverage_baja_o_none():
    assert derive_mte_state("OK", 0.85, 0.6) == DEGRADED
    assert derive_mte_state("OK", None, 0.6) == DEGRADED


def test_degraded_por_confidence_no_finito():
    assert derive_mte_state("OK", 0.95, float("nan")) == DEGRADED
    assert derive_mte_state("OK", 0.95, None) == DEGRADED


def test_invalid_futuros_unavailable_y_coverage_none():
    assert derive_mte_state("UNAVAILABLE", None, 0.6) == INVALID
    assert derive_mte_state("UNAVAILABLE", 0.95, 0.6) == DEGRADED
