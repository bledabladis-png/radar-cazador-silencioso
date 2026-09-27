"""Tests de la logica de estado e histeresis de MTE (DT2 Fase 1).

Cubre:
- load_previous_scenario: fichero valido, inexistente, version distinta.
- validate_transition: transiciones normales, excepciones con cls, bloqueos.

Sin tocar outputs/state/. Todo con tmp_path + monkeypatch.
"""
from __future__ import annotations

import json


class TestLoadPreviousScenario:

    def test_fichero_valido(self, tmp_path, monkeypatch):
        import indicators.mte as mte_module
        state = tmp_path / "mte_state.json"
        state.write_text(json.dumps({
            "schema_version": 1,
            "temporal_contract_version": mte_module.CURRENT_TEMPORAL_CONTRACT_VERSION,
            "scenario": "EXPANSION",
            "pending": "SOFT LANDING",
        }), encoding="utf-8")
        monkeypatch.setattr("indicators.mte.state.MTE_STATE_FILE", str(state))

        scenario, pending, temporal_reset = mte_module.load_previous_scenario()
        assert scenario == "EXPANSION"
        assert pending == "SOFT LANDING"
        assert temporal_reset is False

    def test_fichero_inexistente(self, tmp_path, monkeypatch):
        import indicators.mte as mte_module
        state = tmp_path / "no_existe.json"
        monkeypatch.setattr("indicators.mte.state.MTE_STATE_FILE", str(state))

        scenario, pending, temporal_reset = mte_module.load_previous_scenario()
        assert scenario == "MIXED"
        assert pending is None
        assert temporal_reset is False

    def test_version_distinta_hace_reset(self, tmp_path, monkeypatch):
        import indicators.mte as mte_module
        state = tmp_path / "mte_state.json"
        state.write_text(json.dumps({
            "schema_version": 1,
            "temporal_contract_version": "VERSION_OBSOLETA",
            "scenario": "EXPANSION",
            "pending": "SOFT LANDING",
        }), encoding="utf-8")
        monkeypatch.setattr("indicators.mte.state.MTE_STATE_FILE", str(state))

        scenario, pending, temporal_reset = mte_module.load_previous_scenario()
        assert scenario == "MIXED"
        assert pending is None
        assert temporal_reset is True

    def test_json_corrupto(self, tmp_path, monkeypatch):
        import indicators.mte as mte_module
        state = tmp_path / "mte_state.json"
        state.write_text("no soy json", encoding="utf-8")
        monkeypatch.setattr("indicators.mte.state.MTE_STATE_FILE", str(state))

        scenario, pending, temporal_reset = mte_module.load_previous_scenario()
        assert scenario == "MIXED"
        assert pending is None
        assert temporal_reset is False


class TestValidateTransition:

    def test_transicion_normal(self):
        from indicators.mte import validate_transition
        # EXPANSION -> SOFT LANDING esta en NORMAL_TRANSITIONS
        assert validate_transition("EXPANSION", "SOFT LANDING", cls=0.0) is True
        assert validate_transition("EXPANSION", "MIXED", cls=0.0) is True

    def test_transicion_excepcion_con_cls_alto(self):
        from indicators.mte import validate_transition
        # EXPANSION -> RECESSION es excepcion, requiere cls > 0.85
        assert validate_transition("EXPANSION", "RECESSION", cls=0.90) is True

    def test_transicion_excepcion_sin_cls_suficiente(self):
        from indicators.mte import validate_transition
        # cls = 0.50 no supera el umbral 0.85
        assert validate_transition("EXPANSION", "RECESSION", cls=0.50) is False

    def test_transicion_no_listada(self):
        from indicators.mte import validate_transition
        # CRISIS -> EXPANSION no esta en NORMAL_TRANSITIONS (CRISIS solo
        # permite RECESSION y MIXED) ni en EXCEPTION_TRANSITIONS.
        assert validate_transition("CRISIS", "EXPANSION", cls=0.99) is False
        # CRISIS -> STAGFLATION tampoco esta en ninguna tabla.
        assert validate_transition("CRISIS", "STAGFLATION", cls=1.0) is False