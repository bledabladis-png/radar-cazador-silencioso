# -*- coding: utf-8 -*-
"""B6-MTE-INF-NO-FINITE-RESTANTES (B4, 2026-11-01).

Tras B6-MTE-CLS-INF (ba1f2eac), la regla cls>0.85 de CRISIS tiene
finite=["cls"]. Pero RECESSION, STAGFLATION, SOFT LANDING y EXPANSION
seguian con reglas sin finite: un inf en la variable disparaba la
condicion y sumaba bonus. Este test verifica que, en esos 4 escenarios,
una variable no finita produce el mismo score que NaN en esa variable
(ambas se saltan la regla).

CRISIS no se testea aqui: sus reglas tienen finite=["cls"] por diseno
del refactor H7. Un srs=inf o shs=inf en CRISIS dispara las reglas
srs>0.3 y shs>0.3 (finite=["cls"] solo protege cls). Es un hallazgo
fuera del alcance de este item, se registrara si aparece.

Contrato: para (scenario, var) en {4 escenarios} x {srs, shs, cls, ips},
score con var=inf == score con var=nan (ambos saltan la regla).
"""
import pytest

from indicators.mte.scoring import score_scenarios


SCENARIOS = ["RECESSION", "STAGFLATION", "SOFT LANDING", "EXPANSION"]
VARS = ["srs", "shs", "cls", "ips"]
BASE = {"srs": 0.0, "shs": 0.0, "cls": 0.0, "ips": 0.0}


def _score(srs, shs, cls, ips):
    return score_scenarios(srs, shs, cls, ips)


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("var", VARS)
@pytest.mark.parametrize("valor", [float("inf"), float("-inf")])
def test_inf_no_suma_bonus(scenario, var, valor):
    """Con var=+-inf, score[scenario] == score[scenario] con var=nan."""
    kwargs_inf = dict(BASE); kwargs_inf[var] = valor
    kwargs_nan = dict(BASE); kwargs_nan[var] = float("nan")
    got_inf = _score(**kwargs_inf)
    got_nan = _score(**kwargs_nan)
    assert got_inf[scenario] == got_nan[scenario], (
        f"{scenario} con {var}={valor}: score={got_inf[scenario]} != "
        f"con {var}=nan score={got_nan[scenario]}. Regla sin finite."
    )


def test_todas_reglas_4_escenarios_tienen_finite():
    """Contrato estructural: 0 reglas sin finite en los 4 escenarios."""
    from indicators.mte.scoring import SCENARIO_RULES
    for scenario in SCENARIOS:
        for rule in SCENARIO_RULES[scenario]["rules"]:
            assert "finite" in rule, (
                f"{scenario}: regla sin finite: {rule}. "
                f"Anadir finite con las vars de la cond."
            )
