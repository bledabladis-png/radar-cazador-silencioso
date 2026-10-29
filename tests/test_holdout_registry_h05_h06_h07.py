"""Tests H-05, H-06, H-07 dictamen doc 76: holdout_registry.

H-05: trazabilidad historica de SOW-V6 y estado por holdout.
H-06: event_type diferencia CONFIRMATORY_EXECUTION de
      EXPLORATORY_DESIGN_INFLUENCE.
H-07: independence_note con la restriccion de interseccion.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit import holdout_registry as hr


@pytest.fixture
def isolated_registry(monkeypatch, tmp_path):
    """Registry en tmp_path, aislado del real."""
    new_path = tmp_path / "holdout_registry.json"
    monkeypatch.setattr(hr, "REGISTRY_PATH", new_path)
    return new_path


def _entry(**overrides):
    base = {
        "candidata": "TEST-CAND",
        "fecha_consumo": "2026-10-29",
        "informe": "docs/auditoria/wyckoff/test",
        "event_type": "CONFIRMATORY_EXECUTION",
        "experiment_manifest": "data/mappings/experiment_manifests/test.json",
        "identity_status": "VERIFIED",
        "manifest_sha256_en_consumo": "abc123",
    }
    base.update(overrides)
    return base


# ============================================================
# H-05: estructura, SOW-V6 con identidad no verificable, estados derivados
# ============================================================

def test_registry_estructura_completa(isolated_registry):
    reg = hr.load_registry()
    assert reg["schema_version"] == 1
    assert "independence_note" in reg
    assert set(reg["holdouts"].keys()) == {"cohort", "r3000"}
    for h in ("cohort", "r3000"):
        assert "history" in reg["holdouts"][h]
        assert "state" in reg["holdouts"][h]
        assert reg["holdouts"][h]["state"] == "SEALED"


def test_sow_v6_identidad_no_verificable_registry_real():
    """El registry versionado en el repo tiene SOW-V6 como UNVERIFIABLE."""
    real = hr.REGISTRY_PATH
    if not real.exists():
        pytest.skip("registry real no existe en este entorno")
    reg = json.loads(real.read_text(encoding="utf-8"))
    for h in ("cohort", "r3000"):
        hist = reg["holdouts"][h]["history"]
        assert len(hist) >= 1
        sow = [e for e in hist if e["candidata"] == "SOW-V6"]
        assert len(sow) == 1
        assert sow[0]["event_type"] == "EXPLORATORY_DESIGN_INFLUENCE"
        assert sow[0]["experiment_manifest"] is None
        assert sow[0]["identity_status"] == "UNVERIFIABLE_PRE_MANIFEST"
        assert sow[0]["manifest_sha256_en_consumo"] is None


def test_derive_state_sealed():
    assert hr.derive_state([]) == "SEALED"


def test_derive_state_exploratory():
    assert hr.derive_state([_entry(event_type="EXPLORATORY_DESIGN_INFLUENCE",
                                    experiment_manifest=None,
                                    identity_status="NOT_APPLICABLE")]) == "EXPLORATORY_CONSUMED"


def test_derive_state_confirmatory():
    assert hr.derive_state([_entry(event_type="CONFIRMATORY_EXECUTION")]) == "CONFIRMATORY_CONSUMED"


def test_derive_state_retired():
    e = _entry(event_type="CONFIRMATORY_EXECUTION", state_override="RETIRED_OR_SUPERSEDED")
    assert hr.derive_state([e]) == "RETIRED_OR_SUPERSEDED"


# ============================================================
# H-06: event_type solo CONFIRMATORY_EXECUTION o EXPLORATORY_DESIGN_INFLUENCE
# ============================================================

def test_event_type_validos():
    for et in ("CONFIRMATORY_EXECUTION", "EXPLORATORY_DESIGN_INFLUENCE"):
        errs = hr.validate_entry(_entry(event_type=et,
                                        experiment_manifest=None
                                        if et == "EXPLORATORY_DESIGN_INFLUENCE"
                                        else "x.json",
                                        identity_status="NOT_APPLICABLE"
                                        if et == "EXPLORATORY_DESIGN_INFLUENCE"
                                        else "VERIFIED"))
        assert not errs, f"{et}: {errs}"


def test_event_type_invalido_rechazado():
    errs = hr.validate_entry(_entry(event_type="RANDOM_TYPE"))
    assert any("event_type invalido" in e for e in errs)


def test_exploratory_no_puede_tener_manifest():
    """EXPLORATORY_DESIGN_INFLUENCE con experiment_manifest -> error."""
    errs = hr.validate_entry(_entry(
        event_type="EXPLORATORY_DESIGN_INFLUENCE",
        experiment_manifest="data/mappings/experiment_manifests/x.json",
        identity_status="VERIFIED",
    ))
    assert any("EXPLORATORY_DESIGN_INFLUENCE no debe tener experiment_manifest" in e for e in errs)


def test_confirmatory_sin_manifest_requiere_unverifiable():
    """CONFIRMATORY_EXECUTION sin manifest exige UNVERIFIABLE_PRE_MANIFEST."""
    errs = hr.validate_entry(_entry(
        event_type="CONFIRMATORY_EXECUTION",
        experiment_manifest=None,
        identity_status="VERIFIED",
    ))
    assert any("CONFIRMATORY_EXECUTION sin experiment_manifest" in e for e in errs)


# ============================================================
# H-07: independence_note con restriccion de interseccion 266/268
# ============================================================

def test_independence_note_menciona_266(isolated_registry):
    reg = hr.load_registry()
    note = reg["independence_note"]
    assert "266" in note, "la nota debe mencionar los 266 tickers compartidos"
    assert "independientes" in note.lower() or "no pueden" in note.lower()


# ============================================================
# verify_registry: coherencia completa
# ============================================================

def test_verify_detecta_state_inconsistente(isolated_registry):
    """Si el state guardado no coincide con el derivado, verify debe fallar."""
    reg = hr._empty_registry()
    reg["holdouts"]["cohort"]["history"] = [_entry()]
    reg["holdouts"]["cohort"]["state"] = "SEALED"  # incorrecto
    hr.save_registry(reg)
    errs = hr.verify_registry(hr.load_registry())
    assert any("state=" in e for e in errs)


def test_verify_acepta_registry_valido(isolated_registry):
    reg = hr._empty_registry()
    hr.save_registry(reg)
    errs = hr.verify_registry(hr.load_registry())
    assert not errs
