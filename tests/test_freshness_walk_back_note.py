# -*- coding: utf-8 -*-
"""Tests de la nota forward-reference en Data Freshness (S30, 2026-11-06).

Provenance (renderizado antes de Data Freshness) ya explica WALK_BACK
con 2 notas. Este test cubre la referencia inversa: Data Freshness
apunta a Provenance cuando walk_back_applied=True, para que la tabla
compacta no se lea como "todo CURRENT" cuando el equity va con fecha
efectiva anterior.
"""
from __future__ import annotations

import inspect

import pandas as pd

from src.report.freshness import render_data_freshness


def _render(walk_back_applied=False):
    return render_data_freshness(
        None, None, None,
        reference_date=pd.Timestamp("2026-11-06"),
        walk_back_applied=walk_back_applied,
    )


def test_param_walk_back_existe_con_default_false():
    sig = inspect.signature(render_data_freshness)
    assert "walk_back_applied" in sig.parameters
    assert sig.parameters["walk_back_applied"].default is False


def test_sin_walk_back_no_aparece_nota():
    joined = "".join(_render(walk_back_applied=False))
    assert "ver Provenance" not in joined
    assert "fecha efectiva del bloque de equity" not in joined


def test_con_walk_back_aparece_nota_apuntando_a_provenance():
    joined = "".join(_render(walk_back_applied=True))
    assert "ver Provenance" in joined
    assert "WALK_BACK" in joined
    assert "fecha efectiva del bloque de equity" in joined
