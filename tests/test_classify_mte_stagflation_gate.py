# -*- coding: utf-8 -*-
"""Tests del guard semantico de STAGFLATION (C2, dictamen 2026-10-16).

Contrato: `IPS > 0.15 AND SRS > 0` (el publicado en H15).
Si no se cumple, STAGFLATION no puede ser el escenario final
aunque su puntuacion bruta sea la mayor.

Los tests de `score_scenarios` (test_mte_scoring_characterization.py)
se mantienen intactos: ese calculo no cambia.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.mte.decision import classify_mte


def _patch_prev(monkeypatch, prev="MIXED", pending=None, reset=False):
    monkeypatch.setattr(
        "indicators.mte.decision.load_previous_scenario",
        lambda: (prev, pending, reset),
    )


def _patch_validate(monkeypatch, allowed=True):
    monkeypatch.setattr(
        "indicators.mte.decision.validate_transition",
        lambda *a, **kw: allowed,
    )


def _patch_confidence(monkeypatch, value=0.7):
    monkeypatch.setattr(
        "indicators.mte.decision.compute_confidence",
        lambda *a, **kw: value,
    )


def _setup(monkeypatch):
    _patch_prev(monkeypatch, prev="MIXED", pending=None, reset=False)
    _patch_validate(monkeypatch, True)
    _patch_confidence(monkeypatch, 0.7)


# ============================================================
# Casos del auditor (tabla §5)
# ============================================================
def test_stagfl_blocked_con_ips_negativo_y_srs_positivo(monkeypatch):
    """IPS=-0.41, SRS positivo -> STAGFLATION NO elegible."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=-0.41)
    assert scenario != "STAGFLATION"


def test_stagfl_blocked_con_ips_0149(monkeypatch):
    """IPS=0.149 (bajo umbral), SRS positivo -> NO elegible."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=0.149)
    assert scenario != "STAGFLATION"


def test_stagfl_blocked_con_ips_exactamente_015(monkeypatch):
    """IPS=0.15 (limite estricto), SRS positivo -> NO elegible."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=0.15)
    assert scenario != "STAGFLATION"


def test_stagfl_blocked_con_srs_cero(monkeypatch):
    """IPS=0.16, SRS=0 (limite estricto) -> NO elegible."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0, shs=0, cls=0.1, ips=0.16)
    assert scenario != "STAGFLATION"


def test_stagfl_elegible_con_ips_016_y_srs_positivo(monkeypatch):
    """IPS=0.16, SRS>0 -> STAGFLATION elegible (puede ganar)."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=0.16)
    # Con prev=MIXED y pending=None, la histeresis hace que MIXED
    # siga siendo el escenario final pero pending_out sea STAGFLATION.
    # Aqui comprobamos que STAGFLATION es CANDIDATO (via pending).
    # Simplificamos: usamos prev=STAGFLATION directamente.
    monkeypatch.setattr(
        "indicators.mte.decision.load_previous_scenario",
        lambda: ("STAGFLATION", None, False),
    )
    scenario2, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=0.16)
    assert scenario2 == "STAGFLATION"


def test_stagfl_blocked_con_ips_invalido(monkeypatch):
    """IPS NaN -> NO elegible."""
    _setup(monkeypatch)
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=float("nan"))
    assert scenario != "STAGFLATION"


def test_stagfl_gana_por_puntuacion_pero_no_elegible(monkeypatch):
    """Caso real del reporte: STAGFLATION gana el argmax con puntuacion
    alta, pero IPS=-0.41 lo descalifica. Debe ganar otro escenario."""
    _setup(monkeypatch)
    # srs>0, ips<0 -> STAGFLATION = SRS(2) + CLS(3) = 5; MIXED = 2
    # sin el guard, STAGFLATION ganaria. Con el guard, no.
    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=-0.41)
    assert scenario != "STAGFLATION"
    # El escenario ganador debe ser uno de los legitimos
    assert scenario in ("MIXED", "RECESSION", "SOFT LANDING",
                        "EXPANSION", "CRISIS")


# ============================================================
# Coherencia de confidence
# ============================================================
def test_confidence_se_calcula_sobre_escenario_elegido(monkeypatch):
    """Si STAGFLATION es descartado, la confidence debe calcularse
    sobre el escenario efectivamente elegido, no sobre el ganador bruto."""
    capturado = {}
    _patch_prev(monkeypatch, prev="MIXED", pending=None, reset=False)
    _patch_validate(monkeypatch, True)

    def _fake_confidence(srs, shs, cls, ips, scenario):
        capturado["scenario"] = scenario
        return 0.7

    monkeypatch.setattr(
        "indicators.mte.decision.compute_confidence",
        _fake_confidence,
    )

    scenario, _, _, _ = classify_mte(srs=0.5, shs=0, cls=0.1, ips=-0.41)
    assert capturado.get("scenario") == scenario
    assert scenario != "STAGFLATION"