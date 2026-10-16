# -*- coding: utf-8 -*-
"""Tests H8: bloque de provenance en la cabecera del reporte.

Dictamen auditor externo 2026-10-16: sin SHA/run_id/sesion objetivo,
el reporte no puede certificarse. Este bloque los publica.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report.provenance import (
    render_provenance,
    build_provenance,
    _compute_validation_gate,
)


# ============================================================
# render_provenance
# ============================================================
def test_render_none_no_renderiza():
    assert render_provenance(None) == []


def test_render_empty_no_renderiza():
    assert render_provenance({}) == []


def test_render_completo():
    prov = {
        "commit_sha": "abc12345",
        "run_id": "12345",
        "run_attempt": "1",
        "target_session": "2026-10-16",
        "reference_date": datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc),
        "effective_date_equity": "2026-10-15",
        "validation_gate": "10/10",
        "generated_at": "2026-10-16T19:00:00Z",
    }
    body = "".join(render_provenance(prov))
    assert "**Provenance:**" in body
    assert "`commit_sha`: abc12345" in body
    assert "`run_id`: 12345 (attempt 1)" in body
    assert "`target_session`: 2026-10-16" in body
    assert "`effective_date_equity`: 2026-10-15" in body
    assert "`validation_gate`: 10/10" in body
    assert "`generated_at`: 2026-10-16T19:00:00Z" in body


def test_render_sin_run_id_local():
    prov = {"commit_sha": "abc12345", "target_session": "2026-10-16"}
    body = "".join(render_provenance(prov))
    assert "N/D (ejecucion local)" in body


def test_render_campos_faltantes_van_a_nd():
    prov = {"commit_sha": "abc12345"}
    body = "".join(render_provenance(prov))
    assert "`target_session`: N/D" in body
    assert "`effective_date_equity`: N/D" in body


# ============================================================
# build_provenance
# ============================================================
def test_build_commit_trunca_a_8():
    env = {"GITHUB_SHA": "fedcba9876543210"}
    p = build_provenance(reference_date=None, env=env)
    assert p["commit_sha"] == "fedcba98"


def test_build_sin_env_deja_commit_none():
    p = build_provenance(reference_date=None, env={})
    assert p["commit_sha"] is None


def test_build_run_id_y_attempt():
    env = {"GITHUB_RUN_ID": "999", "GITHUB_RUN_ATTEMPT": "3"}
    p = build_provenance(reference_date=None, env=env)
    assert p["run_id"] == "999"
    assert p["run_attempt"] == "3"


def test_build_validation_gate_10_10():
    vg = {"passed": True, "checks": ["a"] * 10, "errors": []}
    p = build_provenance(reference_date=None, validation_gate_result=vg)
    assert p["validation_gate"] == "10/10"


def test_build_validation_gate_fallida():
    vg = {"passed": False, "checks": ["a"] * 8, "errors": ["x", "y"]}
    p = build_provenance(reference_date=None, validation_gate_result=vg)
    assert p["validation_gate"] == "6/8"


def test_build_validation_gate_ausente():
    p = build_provenance(reference_date=None, validation_gate_result=None)
    assert p["validation_gate"] is None


def test_build_reference_date_se_preserva():
    ref = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
    p = build_provenance(reference_date=ref, env={})
    assert p["reference_date"] == ref


# ============================================================
# _compute_validation_gate (unidad)
# ============================================================
def test_compute_gate_none():
    assert _compute_validation_gate(None) is None


def test_compute_gate_sin_checks():
    vg = {"passed": True, "checks": [], "errors": []}
    # checks_n=0 -> "0/0" (conservador, no 10/10)
    assert _compute_validation_gate(vg) == "0/0"


# ============================================================
# Integracion: generate_daily_report acepta provenance
# ============================================================
def test_generate_daily_report_acepta_kwarg():
    """Verifica que generate_daily_report acepta provenance sin romper."""
    import inspect
    from src.report_generator import generate_daily_report
    sig = inspect.signature(generate_daily_report)
    assert "provenance" in sig.parameters
    assert sig.parameters["provenance"].default is None


def test_render_provenance_se_llama_desde_generator(monkeypatch):
    """Verifica que el bloque pasa por generate_daily_report via render_provenance."""
    import src.report_generator as rg
    capturado = {}
    def _fake_render(prov):
        capturado["prov"] = prov
        return []
    monkeypatch.setattr(rg, "render_provenance", _fake_render)

    prov = {"commit_sha": "deadbeef", "target_session": "2026-10-16"}
    try:
        rg.generate_daily_report(
            None, "MIXED", 0.5, None, "NEUTRAL", 0.5,
            None, "LOW", 0.5, {"regime": "ROTATIONAL"},
            [], [], [], [],
            provenance=prov,
            output_path=str(Path(__file__).parent / "_tmp_prov_report.md"),
        )
    except Exception:
        # Si otros renders fallan con None, no es problema del test.
        # Solo verificamos que render_provenance fue llamado.
        pass
    assert capturado.get("prov") == prov, (
        "render_provenance no fue llamado con el dict esperado"
    )


# ============================================================
# H8 + auditor 2026-10-16: coherencia temporal
# ============================================================
def test_coherencia_ok_si_target_igual_effective():
    from src.report.provenance import _temporal_coherence
    assert _temporal_coherence("2026-10-16", "2026-10-16") == "OK"


def test_coherencia_lag_admitted_dentro_de_max_lag():
    from src.report.provenance import _temporal_coherence
    assert _temporal_coherence("2026-10-16", "2026-10-15", contract_max_lag_days=5) == "LAG_ADMITTED"


def test_coherencia_incoherent_fuera_de_max_lag():
    """EQUITY_EOD con max_lag=0: target > effective es incoherente."""
    from src.report.provenance import _temporal_coherence
    assert _temporal_coherence("2026-10-16", "2026-10-15", contract_max_lag_days=0) == "INCOHERENT"


def test_coherencia_anomaly_effective_posterior():
    """effective_date posterior a target_session es anomalia."""
    from src.report.provenance import _temporal_coherence
    assert _temporal_coherence("2026-10-15", "2026-10-16") == "ANOMALY"


def test_coherencia_nd_sin_datos():
    from src.report.provenance import _temporal_coherence
    assert _temporal_coherence(None, "2026-10-15") == "N/D"
    assert _temporal_coherence("2026-10-16", None) == "N/D"


def test_build_provenance_marca_gate_incoherent():
    """Si target > effective con max_lag=0 y gate=10/10, marcar contradiccion."""
    vg = {"passed": True, "checks": ["a"] * 10, "errors": []}
    p = build_provenance(
        reference_date=None,
        target_session="2026-10-16",
        effective_date_equity="2026-10-15",
        validation_gate_result=vg,
        contract_max_lag_days=0,
        env={},
    )
    assert "INCOHERENT" in p["validation_gate"]
    assert p["temporal_coherence"] == "INCOHERENT"


def test_build_provenance_ok_coherente():
    vg = {"passed": True, "checks": ["a"] * 10, "errors": []}
    p = build_provenance(
        reference_date=None,
        target_session="2026-10-16",
        effective_date_equity="2026-10-16",
        validation_gate_result=vg,
        env={},
    )
    assert p["validation_gate"] == "10/10"
    assert p["temporal_coherence"] == "OK"


def test_build_provenance_lag_admitted_no_contradice():
    """Con max_lag=5, un lag de 1 dia es admisible y no marca INCOHERENT."""
    vg = {"passed": True, "checks": ["a"] * 10, "errors": []}
    p = build_provenance(
        reference_date=None,
        target_session="2026-10-16",
        effective_date_equity="2026-10-15",
        validation_gate_result=vg,
        contract_max_lag_days=5,
        env={},
    )
    assert p["validation_gate"] == "10/10"
    assert p["temporal_coherence"] == "LAG_ADMITTED"


def test_render_incluye_temporal_coherence():
    prov = {
        "commit_sha": "abc12345",
        "target_session": "2026-10-16",
        "effective_date_equity": "2026-10-15",
        "validation_gate": "10/10 (INCOHERENT)",
        "temporal_coherence": "INCOHERENT",
    }
    body = "".join(render_provenance(prov))
    assert "`temporal_coherence`: INCOHERENT" in body
