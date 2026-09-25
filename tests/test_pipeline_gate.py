# -*- coding: utf-8 -*-
"""Tests del gate de disponibilidad (F-IAE-CRON-02 / F-IAE-GATE-01)."""
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.pipeline_gate as gate
from scripts.pipeline_gate import (
    GATE_PANEL_USA,
    MANIFEST_THRESHOLD,
    PROBE_MIN_COVERAGE,
    _manifest_satisfies,
    evaluate,
    resolve_target_session,
)


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """PROJECT_ROOT del modulo apunta a tmp_path."""
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    return tmp_path


def _write_manifest(root, expected, coverage, status="VALID"):
    p = root / "data" / "stock_prices.parquet.manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "quality": {
            "expected_session": expected,
            "coverage_pct_last": coverage,
            "status": status,
        }
    }), encoding="utf-8")


def _make_download_df(target_session, coverage_frac, tickers=GATE_PANEL_USA):
    """Simula yf.download con MultiIndex (Close, ticker)."""
    n = len(tickers)
    n_valid = int(round(n * coverage_frac))
    vals = [1.0] * n_valid + [float("nan")] * (n - n_valid)
    return pd.DataFrame(
        [vals],
        index=pd.DatetimeIndex([pd.Timestamp(target_session)]),
        columns=pd.MultiIndex.from_product([["Close"], list(tickers)]),
    )


def test_panel_fijo_20_tickers():
    assert len(GATE_PANEL_USA) == 20
    assert "AAPL" in GATE_PANEL_USA
    assert "MSFT" in GATE_PANEL_USA
    assert "BRK-B" not in GATE_PANEL_USA
    assert "SPY" not in GATE_PANEL_USA


def test_thresholds():
    assert PROBE_MIN_COVERAGE == 0.90
    assert MANIFEST_THRESHOLD == 0.95


def test_target_session_23_17_utc_thursday():
    now = datetime(2026, 9, 24, 23, 17)
    assert resolve_target_session(now) == "2026-09-24"


def test_target_session_23_17_utc_friday():
    now = datetime(2026, 9, 25, 23, 17)
    assert resolve_target_session(now) == "2026-09-25"


def test_target_session_03_17_utc_friday():
    now = datetime(2026, 9, 25, 3, 17)
    assert resolve_target_session(now) == "2026-09-24"


def test_target_session_07_17_utc_friday():
    now = datetime(2026, 9, 25, 7, 17)
    assert resolve_target_session(now) == "2026-09-24"


def test_target_session_11_17_utc_friday():
    now = datetime(2026, 9, 25, 11, 17)
    assert resolve_target_session(now) == "2026-09-24"


def test_target_session_4_slots_same_target():
    slots = [
        datetime(2026, 9, 24, 23, 17),
        datetime(2026, 9, 25, 3, 17),
        datetime(2026, 9, 25, 7, 17),
        datetime(2026, 9, 25, 11, 17),
    ]
    targets = {resolve_target_session(s) for s in slots}
    assert targets == {"2026-09-24"}


def test_target_session_23_17_saturday():
    now = datetime(2026, 9, 26, 23, 17)
    assert resolve_target_session(now) == "2026-09-25"


def test_target_session_03_17_saturday():
    now = datetime(2026, 9, 26, 3, 17)
    assert resolve_target_session(now) == "2026-09-25"


def test_target_session_domingo_viernes():
    now = datetime(2026, 9, 27, 23, 17)
    assert resolve_target_session(now) == "2026-09-25"


def test_target_session_sabado_tras_festivo():
    """2026-09-07 es Labor Day. Sabado 05-Sep 03:17 -> viernes 04-Sep."""
    now = datetime(2026, 9, 5, 3, 17)
    assert resolve_target_session(now) == "2026-09-04"


def test_target_session_martes_tras_labor_day():
    """Martes 08-Sep 03:17 -> viernes 04-Sep (salta finde + festivo)."""
    now = datetime(2026, 9, 8, 3, 17)
    assert resolve_target_session(now) == "2026-09-04"


def test_manifest_missing_not_satisfied():
    assert _manifest_satisfies(None, "2026-09-24") is False


def test_manifest_matching_coverage_high_satisfied():
    m = {"quality": {"expected_session": "2026-09-24",
                     "coverage_pct_last": 0.98}}
    assert _manifest_satisfies(m, "2026-09-24") is True


def test_manifest_matching_at_threshold_satisfied():
    m = {"quality": {"expected_session": "2026-09-24",
                     "coverage_pct_last": 0.95}}
    assert _manifest_satisfies(m, "2026-09-24") is True


def test_manifest_matching_below_threshold_not_satisfied():
    m = {"quality": {"expected_session": "2026-09-24",
                     "coverage_pct_last": 0.94}}
    assert _manifest_satisfies(m, "2026-09-24") is False


def test_manifest_different_session_not_satisfied():
    m = {"quality": {"expected_session": "2026-09-23",
                     "coverage_pct_last": 1.0}}
    assert _manifest_satisfies(m, "2026-09-24") is False


def test_manifest_no_quality_block_not_satisfied():
    assert _manifest_satisfies({}, "2026-09-24") is False


def test_current_when_manifest_fresh(isolated_project):
    """Manifest ya cubre target_session -> CURRENT, sin invocar probe."""
    _write_manifest(isolated_project, "2026-09-24", 0.99)
    result = evaluate("2026-09-24")
    assert result["state"] == "CURRENT"
    assert result["should_run"] is False
    assert result["expected_session"] == "2026-09-24"


def test_ready_when_probe_ok(isolated_project):
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "READY"
    assert result["should_run"] is True
    assert result["probe_coverage"] == 1.0


def test_ready_when_probe_at_threshold(isolated_project):
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    df = _make_download_df("2026-09-24", coverage_frac=0.90)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "READY"
    assert result["should_run"] is True


def test_not_ready_when_probe_below_threshold(isolated_project):
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    df = _make_download_df("2026-09-24", coverage_frac=0.85)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "NOT_READY"
    assert result["should_run"] is False


def test_not_ready_when_probe_all_nan(isolated_project):
    """Fila existe pero Close=NaN (caso del fallo 25-Sep)."""
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    df = _make_download_df("2026-09-24", coverage_frac=0.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "NOT_READY"
    assert result["should_run"] is False


def test_not_ready_when_row_missing(isolated_project):
    """Yahoo devuelve fila con fecha equivocada."""
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    df = _make_download_df("2026-09-23", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "NOT_READY"
    assert result["should_run"] is False


def test_error_when_probe_raises(isolated_project):
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    with patch.object(gate.yf, "download",
                      side_effect=RuntimeError("network down")):
        result = evaluate("2026-09-24")
    assert result["state"] == "ERROR"
    assert result["should_run"] is False
    assert "network down" in result["reason"]


def test_not_ready_when_probe_empty(isolated_project):
    """DataFrame vacio -> NOT_READY (no error)."""
    _write_manifest(isolated_project, "2026-09-23", 1.0)
    with patch.object(gate.yf, "download", return_value=pd.DataFrame()):
        result = evaluate("2026-09-24")
    assert result["state"] == "NOT_READY"
    assert result["should_run"] is False


def test_no_manifest_triggers_probe(isolated_project):
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "READY"
    assert result["should_run"] is True


def test_corrupted_manifest_triggers_probe(isolated_project):
    p = isolated_project / "data" / "stock_prices.parquet.manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{invalid json", encoding="utf-8")
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result = evaluate("2026-09-24")
    assert result["state"] == "READY"


def test_state_should_run_consistency(isolated_project):
    # CURRENT -> should_run=False
    _write_manifest(isolated_project, "2026-09-24", 0.99)
    result_current = evaluate("2026-09-24")
    assert result_current["state"] == "CURRENT"
    assert result_current["should_run"] is False

    # Manifest para sesion distinta -> probe se invoca
    _write_manifest(isolated_project, "2026-09-23", 1.0)

    # READY -> should_run=True
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df):
        result_ready = evaluate("2026-09-24")
    assert result_ready["state"] == "READY"
    assert result_ready["should_run"] is True

    # NOT_READY -> should_run=False
    df_low = _make_download_df("2026-09-24", coverage_frac=0.5)
    with patch.object(gate.yf, "download", return_value=df_low):
        result_notready = evaluate("2026-09-24")
    assert result_notready["state"] == "NOT_READY"
    assert result_notready["should_run"] is False


# ---------------- resolve_slot_flags ----------------

from scripts.pipeline_gate import (
    CRON_SLOTS,
    LAST_SLOT_CRON,
    resolve_slot_flags,
)


def test_cron_slots_contiene_4():
    assert len(CRON_SLOTS) == 4


def test_last_slot_cron_es_ultimo_de_cron_slots():
    assert CRON_SLOTS[-1] == LAST_SLOT_CRON


def test_resolve_slot_vacio():
    assert resolve_slot_flags("") == (False, False)


def test_resolve_slot_manual():
    assert resolve_slot_flags("manual") == (False, False)


def test_resolve_slot_desconocido():
    assert resolve_slot_flags("0 0 * * *") == (False, False)


def test_resolve_slot_1_no_last():
    assert resolve_slot_flags("17 23 * * *") == (True, False)


def test_resolve_slot_2_no_last():
    assert resolve_slot_flags("17 3 * * *") == (True, False)


def test_resolve_slot_3_no_last():
    assert resolve_slot_flags("17 7 * * *") == (True, False)


def test_resolve_slot_4_is_last():
    assert resolve_slot_flags("17 11 * * *") == (True, True)


def test_resolve_slot_cada_cron_slots_conocido():
    for slot in CRON_SLOTS:
        known, _ = resolve_slot_flags(slot)
        assert known is True, f"{slot} no reconocido"


# ---------------- retry del probe ----------------

from scripts.pipeline_gate import (
    PROBE_MAX_RETRIES,
    PROBE_RETRY_SLEEPS,
    _probe_panel,
)


def test_retry_constants():
    assert PROBE_MAX_RETRIES == 3
    assert len(PROBE_RETRY_SLEEPS) == PROBE_MAX_RETRIES - 1


def test_probe_panel_ok_sin_retry(isolated_project):
    """Primer intento OK -> no llama time.sleep."""
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    with patch.object(gate.yf, "download", return_value=df) as mock_dl:
        with patch.object(gate.time, "sleep") as mock_sleep:
            cov, err = _probe_panel("2026-09-24")
    assert err is None
    assert cov == 1.0
    assert mock_dl.call_count == 1
    assert mock_sleep.call_count == 0


def test_probe_panel_not_ready_sin_retry(isolated_project):
    """Coverage=0 pero sin error -> sin retry (Yahoo aun sin Close)."""
    df = _make_download_df("2026-09-24", coverage_frac=0.0)
    with patch.object(gate.yf, "download", return_value=df) as mock_dl:
        with patch.object(gate.time, "sleep") as mock_sleep:
            cov, err = _probe_panel("2026-09-24")
    assert err is None
    assert cov == 0.0
    assert mock_dl.call_count == 1
    assert mock_sleep.call_count == 0


def test_probe_panel_error_retry_exitoso(isolated_project):
    """Primer intento excepcion, segundo OK -> 1 sleep, sin error final."""
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    side = [RuntimeError("timeout"), df]
    def fake_download(*a, **kw):
        r = side.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    with patch.object(gate.yf, "download", side_effect=fake_download) as mock_dl:
        with patch.object(gate.time, "sleep") as mock_sleep:
            cov, err = _probe_panel("2026-09-24")
    assert err is None
    assert cov == 1.0
    assert mock_dl.call_count == 2
    assert mock_sleep.call_count == 1


def test_probe_panel_error_agota_retries(isolated_project):
    """3 intentos fallidos -> ERROR tras 2 sleeps."""
    with patch.object(gate.yf, "download",
                      side_effect=RuntimeError("network down")) as mock_dl:
        with patch.object(gate.time, "sleep") as mock_sleep:
            cov, err = _probe_panel("2026-09-24")
    assert cov == 0.0
    assert err is not None
    assert "network down" in err
    assert mock_dl.call_count == PROBE_MAX_RETRIES
    assert mock_sleep.call_count == PROBE_MAX_RETRIES - 1


def test_probe_panel_error_tercero_ok(isolated_project):
    """Excepcion, excepcion, OK -> 2 sleeps, resultado OK."""
    df = _make_download_df("2026-09-24", coverage_frac=1.0)
    side = [RuntimeError("a"), RuntimeError("b"), df]
    def fake_download(*a, **kw):
        r = side.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
    with patch.object(gate.yf, "download", side_effect=fake_download) as mock_dl:
        with patch.object(gate.time, "sleep") as mock_sleep:
            cov, err = _probe_panel("2026-09-24")
    assert err is None
    assert cov == 1.0
    assert mock_dl.call_count == 3
    assert mock_sleep.call_count == 2


# ---------------- formato de CRON_SLOTS ----------------

def test_cron_slots_formato_posix_5_campos():
    for slot in CRON_SLOTS:
        parts = slot.split()
        assert len(parts) == 5, f"{slot} no tiene 5 campos"


def test_cron_slots_minuto_17():
    """Todos los slots arrancan en :17 (auditor 2026-09-25)."""
    for slot in CRON_SLOTS:
        minute = slot.split()[0]
        assert minute == "17", f"{slot} no arranca en :17"


def test_cron_slots_orden_cronologico_recuperacion():
    """Los slots siguen el orden cronologico de la ventana de recuperacion.

    Arranca a las 23:17 del dia D y continua 03:17, 07:17, 11:17 del dia
    D+1. El orden numerico de horas UTC no es [3, 7, 11, 23] porque el
    slot "23" precede al "3" en el ciclo real.
    """
    horas = [int(s.split()[1]) for s in CRON_SLOTS]
    assert horas == [23, 3, 7, 11], f"horas en orden inesperado: {horas}"


def test_cron_slots_unicos():
    assert len(set(CRON_SLOTS)) == len(CRON_SLOTS)


def test_cron_slots_horas_esperadas():
    """23, 3, 7, 11 UTC: cubren 5h17m a 17h17m post-cierre USA."""
    horas = [int(s.split()[1]) for s in CRON_SLOTS]
    assert horas == [23, 3, 7, 11]
