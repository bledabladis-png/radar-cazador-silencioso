# -*- coding: utf-8 -*-
# Golden de score_scenarios ANTES del refactor a SCENARIO_RULES (B6-H7).
# Captura el comportamiento EXACTO actual. Si el refactor cambia un
# solo digito, este test falla. Incluye bug latente cls=inf -> -996
# (el +3 de cls>0.85 se suma tras el early-exit -999), que se
# preserva hasta que se arregle como item separado.

import pytest
from indicators.mte.scoring import score_scenarios


GOLDEN = {
    (0.0, 0.0, 0.0, 0.0): {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 3, "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2},
    (0.5, 0.5, 0.9, 0.5): {"CRISIS": 16, "RECESSION": 7, "STAGFLATION": 6, "SOFT LANDING": 5, "EXPANSION": 0, "MIXED": 2},
    (-0.5, -0.5, -0.5, -0.5): {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 3, "SOFT LANDING": 4, "EXPANSION": 8, "MIXED": 2},
    (0.2, 0.2, 0.2, 0.2): {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 9, "SOFT LANDING": 5, "EXPANSION": 0, "MIXED": 2},
    (0.3, 0.3, 0.6, 0.0): {"CRISIS": 6, "RECESSION": 7, "STAGFLATION": 2, "SOFT LANDING": 5, "EXPANSION": 0, "MIXED": 2},
    (0.0, 0.0, 0.8, 0.0): {"CRISIS": 9, "RECESSION": 3, "STAGFLATION": 0, "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2},
    (0.0, 0.0, 0.9, 0.0): {"CRISIS": 12, "RECESSION": 3, "STAGFLATION": 0, "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2},
    (0.0, 0.0, 0.4, 0.0): {"CRISIS": 0, "RECESSION": 3, "STAGFLATION": 0, "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2},
    (0.0, 0.0, -0.3, 0.0): {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 3, "SOFT LANDING": 4, "EXPANSION": 3, "MIXED": 2},
    (0.0, 0.0, 0.0, 0.6): {"CRISIS": 0, "RECESSION": 0, "STAGFLATION": 8, "SOFT LANDING": 0, "EXPANSION": 0, "MIXED": 2},
}


@pytest.mark.parametrize("inputs,expected", list(GOLDEN.items()), ids=[str(k) for k in GOLDEN])
def test_golden_score_scenarios(inputs, expected):
    srs, shs, cls, ips = inputs
    got = score_scenarios(srs, shs, cls, ips)
    assert got == expected, "inputs=" + str(inputs) + ": got " + str(got) + ", expected " + str(expected)


def test_cls_nan_bloquea_crisis():
    """cls=NaN -> crisis=-999. Preserva bug latente con inf (item separado)."""
    got = score_scenarios(0.0, 0.0, float("nan"), 0.0)
    assert got["CRISIS"] == -999


def test_cls_neg_inf_bloquea_crisis():
    got = score_scenarios(0.0, 0.0, float("-inf"), 0.0)
    assert got["CRISIS"] == -999


def test_estructura_scores_completa():
    got = score_scenarios(0.0, 0.0, 0.0, 0.0)
    assert set(got.keys()) == {"CRISIS", "RECESSION", "STAGFLATION", "SOFT LANDING", "EXPANSION", "MIXED"}
