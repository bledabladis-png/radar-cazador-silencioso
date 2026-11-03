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
    infer_walk_back,
    _compute_validation_gate,
    _is_gate_passed,
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


def test_render_sin_run_id_dice_no_identificado():
    """H2 (2026-10-17): dict sin run_id (render puro, no build) ->
    "N/D (no identificado)". Antes decia "N/D (ejecucion local)",
    pero eso conflacia dos casos: (a) build_provenance genero un
    local-<hash> valido, (b) el dict llego sin run_id porque no se
    paso por build. Ahora se distinguen."""
    prov = {"commit_sha": "abc12345", "target_session": "2026-10-16"}
    body = "".join(render_provenance(prov))
    assert "N/D (no identificado)" in body
    assert "N/D (ejecucion local)" not in body


def test_render_con_local_run_id():
    """H2: run_id local-<hash> se publica con sufijo (ejecucion local)."""
    prov = {"commit_sha": "abc12345", "target_session": "2026-10-16",
            "run_id": "local-deadbeef"}
    body = "".join(render_provenance(prov))
    assert "`run_id`: local-deadbeef (ejecucion local)" in body


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


def test_build_validation_gate_11_11():
    """2026-10-30: el gate crecio a 11 checks. El literal "10/10" era
    hardcoded y mentia. Ahora debe emitir "11/11"."""
    vg = {"passed": True, "checks": ["a"] * 11, "errors": []}
    p = build_provenance(reference_date=None, validation_gate_result=vg)
    assert p["validation_gate"] == "11/11"
    assert _compute_validation_gate(vg) == "11/11"


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


# ============================================================
# _is_gate_passed (2026-10-30)
# ============================================================
def test_is_gate_passed_10_10():
    assert _is_gate_passed("10/10")


def test_is_gate_passed_11_11():
    assert _is_gate_passed("11/11")


def test_is_gate_passed_9_10_false():
    assert not _is_gate_passed("9/10")


def test_is_gate_passed_none_false():
    assert not _is_gate_passed(None)


def test_is_gate_passed_basura_false():
    assert not _is_gate_passed("10")
    assert not _is_gate_passed("10/10/10")


# ============================================================
# WALK_BACK (2026-11-03, TEMPORAL-COHERENCE-WALK-BACK)
# ============================================================
def test_coherencia_walk_back_es_estado_propio():
    """effective < target con flag walk_back -> WALK_BACK, no INCOHERENT."""
    from src.report.provenance import _temporal_coherence
    r = _temporal_coherence(
        "2026-11-02", "2026-10-30",
        contract_max_lag_days=0, effective_is_walk_back=True,
    )
    assert r == "WALK_BACK"


def test_coherencia_walk_back_no_aplica_si_effective_posterior():
    """effective > target es ANOMALY aunque el flag sea True."""
    from src.report.provenance import _temporal_coherence
    r = _temporal_coherence(
        "2026-10-30", "2026-11-02", effective_is_walk_back=True,
    )
    assert r == "ANOMALY"


def test_coherencia_walk_back_no_aplica_si_effective_igual():
    """t == e con flag walk_back -> OK (no hubo lag que explicar)."""
    from src.report.provenance import _temporal_coherence
    r = _temporal_coherence(
        "2026-11-02", "2026-11-02", effective_is_walk_back=True,
    )
    assert r == "OK"


def test_coherencia_sin_flag_sigue_incoherent():
    """Regresion: sin flag, lag > 0 con max_lag=0 sigue INCOHERENT.
    El fix no elimina la deteccion de incoherencia real."""
    from src.report.provenance import _temporal_coherence
    r = _temporal_coherence(
        "2026-11-02", "2026-10-30",
        contract_max_lag_days=0, effective_is_walk_back=False,
    )
    assert r == "INCOHERENT"


def test_build_provenance_walk_back_no_marca_gate_incoherent():
    """Flag walk_back + gate passed -> validation_gate sin (INCOHERENT)."""
    vg = {"passed": True, "checks": ["a"] * 11, "errors": []}
    p = build_provenance(
        reference_date=None,
        target_session="2026-11-02",
        effective_date_equity="2026-10-30",
        validation_gate_result=vg,
        contract_max_lag_days=0,
        effective_is_walk_back=True,
        env={},
    )
    assert p["validation_gate"] == "11/11"
    assert p["temporal_coherence"] == "WALK_BACK"
    assert p["effective_is_walk_back"] is True
    assert "INCOHERENT" not in p["validation_gate"]


# ============================================================
# infer_walk_back (2026-11-03, TEMPORAL-COHERENCE-WALK-BACK)
# ============================================================
def test_infer_walk_back_meta_ok_con_lag():
    """status OK + lag_days=3 -> True (walk-back sano)."""
    meta = {
        "status": "OK", "lag_days": 3,
        "date": "2026-10-30", "requested_date": "2026-11-02",
    }
    assert infer_walk_back(meta) is True


def test_infer_walk_back_meta_ok_sin_lag():
    """status OK + lag_days=0 -> False (sin retroceso)."""
    meta = {
        "status": "OK", "lag_days": 0,
        "date": "2026-11-02", "requested_date": "2026-11-02",
    }
    assert infer_walk_back(meta) is False


def test_infer_walk_back_insufficient_coverage():
    """status INSUFFICIENT_COVERAGE -> False (no fue walk-back, fue fallo)."""
    meta = {"status": "INSUFFICIENT_COVERAGE", "lag_days": 5}
    assert infer_walk_back(meta) is False


def test_infer_walk_back_meta_none():
    """meta None -> False (sin datos)."""
    assert infer_walk_back(None) is False


def test_infer_walk_back_meta_vacio():
    """dict vacio -> False."""
    assert infer_walk_back({}) is False


def test_infer_walk_back_lag_string_numerico():
    """lag_days como string numerico (defensivo) -> True."""
    assert infer_walk_back({"status": "OK", "lag_days": "3"}) is True


def test_infer_walk_back_lag_invalido():
    """lag_days invalido -> False (no exception)."""
    assert infer_walk_back({"status": "OK", "lag_days": "x"}) is False
    assert infer_walk_back({"status": "OK", "lag_days": None}) is False


# ============================================================
# Integracion: run.py cablea infer_walk_back -> build_provenance
# ============================================================
def test_run_py_pasa_effective_is_walk_back():
    """Regresion: si run.py olvidara el kwarg, build_provenance
    volveria a marcar INCOHERENT con lag>0.
    """
    import inspect
    import run as _run
    src = inspect.getsource(_run)
    assert "infer_walk_back(df_stocks_effective_meta)" in src
    assert "effective_is_walk_back=_prov_walk_back" in src
