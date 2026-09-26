"""Tests de la histeresis SLPM v1.2 (F2-19 + F2-21).

Microciclo auditado. Cubre:
  - Reset por cambio de sector (F2-19).
  - Histeresis por candidato explicito (F2-21): el candidato debe
    aparecer N=2 runs consecutivos para confirmar la transicion.
  - Compatibilidad con schema v1 (state file sin sector_etf).
  - Comportamiento de INITIAL en ausencia de state previo.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import indicators.state_transition as st


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """Redirige SLPM_STATE_FILE a un tmp_path aislado por test."""
    p = tmp_path / "slpm_state.json"
    monkeypatch.setattr(st, "SLPM_STATE_FILE", str(p))
    return p


def _write_state(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_state(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 1. Init sin estado previo
# ============================================================
def test_initial_sin_state(state_file):
    out = st.confirm_transition("XLK", "CONFIRMED")
    assert out["confirmed_state"] == "CONFIRMED"
    assert out["previous_state"] is None
    assert out["transition"] == "INITIAL_CONFIRMED"
    assert out["consecutive_count"] == 1

    saved = _read_state(state_file)
    assert saved["schema_version"] == st.SCHEMA_VERSION
    assert saved["sector_etf"] == "XLK"
    assert saved["state"] == "CONFIRMED"
    assert saved["consecutive_count"] == 1
    assert saved["candidate_state"] is None
    assert saved["candidate_count"] == 0


# ============================================================
# 2. Cambio de sector resetea (F2-19)
# ============================================================
def test_sector_distinto_resetea(state_file):
    _write_state(state_file, {
        "state": "CONFIRMED",
        "consecutive_count": 153,
        "last_updated": "2026-09-26T00:00:00",
    })
    out = st.confirm_transition("XLF", "UNRESOLVED")
    assert out["confirmed_state"] == "UNRESOLVED"
    assert out["previous_state"] is None
    assert out["transition"] == "INITIAL_UNRESOLVED"
    assert out["consecutive_count"] == 1

    saved = _read_state(state_file)
    assert saved["sector_etf"] == "XLF"
    assert saved["state"] == "UNRESOLVED"
    assert saved["consecutive_count"] == 1


# ============================================================
# 3. Mismo estado incrementa
# ============================================================
def test_mismo_estado_incrementa(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 5,
        "candidate_state": None,
        "candidate_count": 0,
    })
    out = st.confirm_transition("XLK", "CONFIRMED")
    assert out["confirmed_state"] == "CONFIRMED"
    assert out["transition"] == "STABLE_CONFIRMED"
    assert out["consecutive_count"] == 6


# ============================================================
# 4. Transicion sin histeresis -> inmediata
# ============================================================
def test_transicion_sin_histeresis_inmediata(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 5,
        "candidate_state": None,
        "candidate_count": 0,
    })
    # (CONFIRMED, STRUCTURAL_DECAY) no esta en HYSTERESIS
    out = st.confirm_transition("XLK", "STRUCTURAL_DECAY")
    assert out["confirmed_state"] == "STRUCTURAL_DECAY"
    assert out["previous_state"] == "CONFIRMED"
    assert out["transition"] == "CONFIRMED_TO_STRUCTURAL_DECAY"
    assert out["consecutive_count"] == 1


# ============================================================
# 5. Candidato aislado (1 run) NO cambia el estado (F2-21)
# ============================================================
def test_candidato_aislado_no_cambia(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 100,
        "candidate_state": None,
        "candidate_count": 0,
    })
    # (CONFIRMED, EMERGING) esta en HYSTERESIS
    out = st.confirm_transition("XLK", "EMERGING")
    assert out["confirmed_state"] == "CONFIRMED"
    assert out["transition"] == "HOLDING_CONFIRMED_CANDIDATE_EMERGING"
    assert out["consecutive_count"] == 100

    saved = _read_state(state_file)
    assert saved["candidate_state"] == "EMERGING"
    assert saved["candidate_count"] == 1


# ============================================================
# 6. Candidato repetido 2 veces SI cambia
# ============================================================
def test_candidato_repetido_dos_veces_cambia(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 100,
        "candidate_state": "EMERGING",
        "candidate_count": 1,
    })
    out = st.confirm_transition("XLK", "EMERGING")
    assert out["confirmed_state"] == "EMERGING"
    assert out["previous_state"] == "CONFIRMED"
    assert out["transition"] == "CONFIRMED_TO_EMERGING"
    assert out["consecutive_count"] == 1

    saved = _read_state(state_file)
    assert saved["candidate_state"] is None
    assert saved["candidate_count"] == 0


# ============================================================
# 7. Candidato alterno resetea el contador
# ============================================================
def test_candidato_alterno_resetea_contador(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 100,
        "candidate_state": "EMERGING",
        "candidate_count": 1,
    })
    # Cambiamos a otro candidato distinto
    out = st.confirm_transition("XLK", "TACTICAL_CORRECTION")
    assert out["confirmed_state"] == "CONFIRMED"
    assert out["transition"] == "HOLDING_CONFIRMED_CANDIDATE_TACTICAL_CORRECTION"

    saved = _read_state(state_file)
    assert saved["candidate_state"] == "TACTICAL_CORRECTION"
    assert saved["candidate_count"] == 1


# ============================================================
# 8. Volver al estado confirmado borra el candidato
# ============================================================
def test_vuelta_al_confirmado_borra_candidato(state_file):
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 100,
        "candidate_state": "EMERGING",
        "candidate_count": 1,
    })
    out = st.confirm_transition("XLK", "CONFIRMED")
    assert out["confirmed_state"] == "CONFIRMED"
    assert out["transition"] == "STABLE_CONFIRMED"
    assert out["consecutive_count"] == 101

    saved = _read_state(state_file)
    assert saved["candidate_state"] is None
    assert saved["candidate_count"] == 0


# ============================================================
# 9. State file con schema v1 (sin sector) se trata como reset
# ============================================================
def test_schema_v1_resetea(state_file):
    _write_state(state_file, {
        "state": "UNRESOLVED",
        "consecutive_count": 153,
        "last_updated": "2026-09-26T01:46:00",
    })
    out = st.confirm_transition("XLK", "CONFIRMED")
    assert out["transition"] == "INITIAL_CONFIRMED"
    assert out["previous_state"] is None

    saved = _read_state(state_file)
    assert saved["schema_version"] == st.SCHEMA_VERSION
    assert saved["sector_etf"] == "XLK"


# ============================================================
# 10. State file corrupto se trata como reset
# ============================================================
def test_state_corrupto_resetea(state_file):
    state_file.write_text("not valid json {{{", encoding="utf-8")
    out = st.confirm_transition("XLK", "CONFIRMED")
    assert out["transition"] == "INITIAL_CONFIRMED"


# ============================================================
# 11. F2-19 verificado: leader rota sin heredar count
# ============================================================
def test_leader_rota_no_hereda_consecutive_count(state_file):
    # Run 1: lider = XLK, se acumula count 100
    _write_state(state_file, {
        "schema_version": st.SCHEMA_VERSION,
        "sector_etf": "XLK",
        "state": "CONFIRMED",
        "consecutive_count": 100,
        "candidate_state": None,
        "candidate_count": 0,
    })
    # Run 2: lider rota a XLF con estado distinto
    out = st.confirm_transition("XLF", "UNRESOLVED")
    assert out["consecutive_count"] == 1
    assert out["previous_state"] is None
    assert out["transition"] == "INITIAL_UNRESOLVED"