
# -*- coding: utf-8 -*-
"""
Tests de regresión de double counting.

Estos tests garantizan que las correcciones estructurales v1 no se reviertan.
Se ejecutan automáticamente en CI (pytest).
"""

import sys
import os

# Asegurar que la raíz del proyecto esté en sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.weights import STRUCTURAL_WEIGHTS


def test_leader_breadth_no_esta_en_structural_weights():
    assert "leader_breadth" not in STRUCTURAL_WEIGHTS, (
        "Leader Breadth no debe estar en Structural Weights para evitar doble entrada"
    )
