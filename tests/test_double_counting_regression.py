
# -*- coding: utf-8 -*-
"""
Tests de regresión de double counting.

Estos tests garantizan que las correcciones estructurales v1 no se reviertan.
Se ejecutan automáticamente en CI (pytest).
"""

import sys
import os
import inspect
import pytest

# Asegurar que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.weights import STRUCTURAL_WEIGHTS
from indicators.state_machine import classify_leadership_state, validate_state


def test_lis_no_es_input_de_classify_leadership_state():
    params = inspect.signature(classify_leadership_state).parameters
    assert "lis" not in params, "LIS no debe ser input decisorio de classify_leadership_state"


def test_tactical_no_es_input_de_classify_leadership_state():
    params = inspect.signature(classify_leadership_state).parameters
    assert "tactical_score" not in params, "Tactical no debe ser input decisorio de classify_leadership_state"


def test_persistence_sigue_como_input_de_classify_leadership_state():
    params = inspect.signature(classify_leadership_state).parameters
    assert "persistence" in params, "Persistence debe seguir como input decisorio de classify_leadership_state"


def test_leader_breadth_no_esta_en_structural_weights():
    assert "leader_breadth" not in STRUCTURAL_WEIGHTS, (
        "Leader Breadth no debe estar en Structural Weights para evitar doble entrada"
    )


def test_validate_state_no_recibe_lis():
    params = inspect.signature(validate_state).parameters
    assert "lis" not in params, "validate_state no debe recibir LIS como argumento"
