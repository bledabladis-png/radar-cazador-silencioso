# -*- coding: utf-8 -*-
"""Tests de integracion I3/I8 + escenarios del dictamen externo 2026-10-17.

Cobertura de los 8 escenarios exigidos en la seccion 8 del dictamen:

| # | Escenario                                              | Cubierto por                                 |
|---|--------------------------------------------------------|----------------------------------------------|
| 1 | Guard falla por cobertura global -> BLOCKED con motivo | test_attempt_blocked_por_guard_result        |
| 2 | Guard no puede escribir JSON -> attempt registra       | test_guard_json_corrupto_bloquea (H20)       |
| 3 | JSON del Guard malformado -> no validacion satisfact.  | test_guard_json_corrupto_bloquea (H20)       |
| 4 | Resultado de otra ejecucion -> rechazado               | test_guard_run_id_mismatch_bloquea (H20)     |
| 5 | VG falla + Guard falla -> precedencia con ambas causas | test_escenario_5_vg_y_guard_ambas_causas     |
| 6 | Receipt de otra ejecucion -> distinguir origen         | test_escenario_6_receipt_origen (documental) |
| 7 | STAGFLATION previo + IPS/SRS invalidos -> state coer.  | test_mte_guard_post_hist_persistence (H22)   |
| 8 | Falta FUTURE_SETTLEMENT -> UNAVAILABLE                 | test_mte_futures_status_unavailable (H23)    |

Este fichero cubre 5, 6, y las extensiones de I3/I8 que el auditor
pidio explicitamente (verificar consumidores del receipt, no solo
la funcion que lo escribe).
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import write_completion_attempt as wca
from scripts import pipeline_gate as gate


# --- Helpers -----------------------------------------------------------

def _setup_attempt_env(monkeypatch, tmp_path, run_id="12345", run_attempt="1"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(wca, "ROOT", tmp_path)
    monkeypatch.setattr(wca, "MANIFESTS", ())
    monkeypatch.setenv("GITHUB_RUN_ID", run_id)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", run_attempt)
    monkeypatch.setenv("GITHUB_SHA", "abc")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("TARGET_SESSION", "2026-10-16")
    monkeypatch.setattr(wca, "OUT_PATH", tmp_path / "attempt.jsonl")
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _guard_payload(ok=False, failures=None, run_id=12345, run_attempt=1,
                   status="failed"):
    return {
        "schema_version": 1,
        "status": status,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "target_session": "2026-10-16",
        "generated_at": "2026-10-17T06:00:00Z",
        "threshold": 0.95,
        "ok": ok,
        "failures": failures if failures is not None else ["x: y"],
    }


def _entry(tmp_path):
    return json.loads(
        (tmp_path / "attempt.jsonl").read_text("utf-8").strip().splitlines()[0]
    )


# --- Escenario 5: VG falla + Guard falla -> ambas causas disponibles ---

def test_escenario_5_vg_y_guard_ambas_causas(tmp_path, monkeypatch):
    """VG falla (INCOMPLETE por precedencia) y Guard tambien falla.

    Precedencia documentada: INCOMPLETE gana. Pero la causa del Guard
    NO debe desaparecer de reasons. Ambas son auditables.
    """
    _setup_attempt_env(monkeypatch, tmp_path)
    _write_json(
        tmp_path / "outputs" / "state" / "validation_gate_result.json",
        {"passed": False, "checks_n": 10, "errors_n": 2},
    )
    _write_json(
        tmp_path / "outputs" / "state" / "guard_coverage_result.json",
        _guard_payload(
            ok=False,
            failures=[
                "data/stock_prices.parquet.manifest.json: "
                "coverage_pct_last=0.927 < 0.95"
            ],
        ),
    )

    wca.main()
    entry = _entry(tmp_path)
    assert entry["outcome"] == "INCOMPLETE", (
        "Precedencia documentada: VG gana la etiqueta"
    )
    # Ambas causas disponibles (escenario 5 del dictamen):
    assert any("validation_gate" in r for r in entry["reasons"])
    assert any("0.927" in r for r in entry["reasons"]), (
        "El detalle del Guard debe aparecer en reasons aunque VG gane"
    )


# --- Escenario 6: receipt de otra ejecucion / sesion ya completada ---

def test_escenario_6_receipt_origen_documentado(tmp_path, monkeypatch):
    """El attempt actual marca SUCCESS por existencia de receipt.

    NO distingue si el receipt lo emitio esta ejecucion o una anterior.
    El contrato de completion_attempt.jsonl define SUCCESS como estado
    de la SESION. El auditor (H20) permite mantener la etiqueta pero
    pide que se deje explicito. Este test documenta el comportamiento
    actual y anota el gap.

    Cambio futuro recomendado (no aplicado): anadir `receipt_origin` o
    `receipt_run_id` al entry, comparando con GITHUB_RUN_ID.
    """
    _setup_attempt_env(monkeypatch, tmp_path, run_id="99999")
    _write_json(
        tmp_path / "outputs" / "state" / "completion_receipt.json",
        {"schema_version": 1, "status": "COMPLETED",
         "run_id": 11111, "target_session": "2026-10-16"},
    )

    wca.main()
    entry = _entry(tmp_path)
    # Comportamiento actual: SUCCESS sin comprobar origen
    assert entry["outcome"] == "SUCCESS"
    # Gap documentado: el entry no expone que el receipt es de otro run
    assert "receipt_origin" not in entry, (
        "Si receipt_origin existe, actualizar este test y el contrato "
        "del attempt."
    )


# --- I3 extendido: orden estricto por indice de step ------------------

def test_i3_orden_estricto_steps_yml():
    """I3: guard -> generate receipt -> upload receipt -> attempt.

    Verifica el orden por indice del step en el job run-system, no por
    find() sobre el texto (mas robusto frente a referencias en
    comentarios o notas).

    Tambien verifica que guard_coverage_result.json se genera antes
    que el receipt: sin eso, el receipt podria emitirse sin la
    validacion contractual de la enmienda v1.1.
    """
    yml = ROOT / ".github" / "workflows" / "daily_run.yml"
    d = yaml.safe_load(yml.read_text("utf-8"))
    steps = d["jobs"]["run-system"]["steps"]
    names = [s.get("name", "") for s in steps]

    def idx(predicate):
        for i, name in enumerate(names):
            if predicate(name):
                return i
        return None

    i_guard = idx(lambda n: n == "Guard coverage")
    i_generate = idx(lambda n: n == "Generate completion receipt")
    i_upload_receipt = idx(lambda n: n == "Upload completion receipt")
    i_attempt = idx(lambda n: n.startswith("Write completion attempt"))
    i_upload_attempt = idx(lambda n: n == "Upload completion attempt")

    for name, i in [
        ("Guard coverage", i_guard),
        ("Generate completion receipt", i_generate),
        ("Upload completion receipt", i_upload_receipt),
        ("Write completion attempt", i_attempt),
        ("Upload completion attempt", i_upload_attempt),
    ]:
        assert i is not None, f"step '{name}' no encontrado en run-system"

    # Orden requerido por I3: guard antes que generate receipt.
    assert i_guard < i_generate, (
        f"Guard ({i_guard}) debe ir antes que Generate receipt ({i_generate})"
    )
    # Generate antes que Upload.
    assert i_generate < i_upload_receipt
    # Attempt va despues del receipt (siempre, if:always()).
    assert i_upload_receipt < i_attempt
    assert i_attempt < i_upload_attempt

    # if: success() en generate/upload; if: always() en attempt.
    assert steps[i_generate].get("if") == "success()"
    assert steps[i_upload_receipt].get("if") == "success()"
    assert steps[i_attempt].get("if") == "always()"
    assert steps[i_upload_attempt].get("if") == "always()"


# --- I8 extendido: guard ok=True no produce CURRENT en el gate --------

def test_i8_guard_ok_no_produce_current(monkeypatch, tmp_path):
    """I8: un guard_coverage_result.json con ok=True no sustituye al
    receipt ni al manifest para el gate.

    find_completion_receipt solo busca el artifact del receipt. Nunca
    lee guard_coverage_result.json. El gate no debe considerar una
    sesion completada por tener guard OK.

    Verifica tambien que _manifest_satisfies sigue siendo fail-closed:
    sin parquet sibling, un manifest sintetico no produce CURRENT.
    """
    # Monkeypatch del ROOT del gate para apuntar a tmp
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    # Fabricar guard_coverage_result.json ok=True
    (tmp_path / "outputs" / "state").mkdir(parents=True, exist_ok=True)
    _write_json(
        tmp_path / "outputs" / "state" / "guard_coverage_result.json",
        _guard_payload(ok=True, failures=[], status="passed"),
    )

    # Sin receipt, sin manifest valido, sin parquet -> el gate debe
    # devolver CURRENT? NO. evaluate() no consulta el guard result.
    # Con token ausente, find_completion_receipt devuelve None.
    assert gate.find_completion_receipt("2026-10-16") is None, (
        "find_completion_receipt debe ignorar guard_coverage_result.json"
    )

    # _manifest_satisfies sigue fail-closed sin parquet sibling
    manifest = {
        "artifact": {"sha256": "0" * 64},
        "quality": {
            "expected_session": "2026-10-16",
            "coverage_pct_last": 0.99,
        },
    }
    assert gate._manifest_satisfies(manifest, "2026-10-16") is False