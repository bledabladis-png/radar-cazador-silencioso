# -*- coding: utf-8 -*-
"""Centinela H12 (auditoria 2026-11-07): el gate debe reintentar en
slots intermedios cuando el receipt declara walk-back.

Bug real (produccion 06-11-2026): el run 42156577761 (03:26 UTC)
hizo walk-back a 04-11 por cobertura incompleta de Yahoo small caps.
El receipt se genero para target_session=2026-11-05 sin declarar
walk-back. Los 4 runs posteriores del dia (10:51, 12:55, 15:01,
18:03) devolvieron CURRENT (should_run=false) por el receipt
existente. El reporte de las 18:03 seguia con datos del 04-11.

Causa raiz:
- receipt no incluia effective_session (writer no lo propagaba
  del manifest, aunque el manifest ya lo tenia).
- evaluate() devolvia CURRENT si existia receipt para
  target_session, sin mirar si era walk-back.

Fix:
- writer persiste quality['effective_session'] en el receipt.
- _validate_receipt_schema acepta effective_session opcional
  (compat con receipts pre-fix).
- evaluate(target, is_last_slot=False):
  - receipt con effective < target y not is_last_slot -> READY.
  - receipt con effective < target y is_last_slot -> CURRENT
    (acepta walk-back para no bloquear el reporte final).
  - receipt con effective == target -> CURRENT.
  - receipt sin effective_session (pre-fix) -> CURRENT.
"""
from __future__ import annotations

from unittest.mock import patch

from scripts import pipeline_gate as pg


def _receipt(effective_session="__absent__", run_id=999):
    r = {
        "schema_version": pg.RECEIPT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "target_session": "2026-11-05",
        "workflow": next(iter(pg.VALID_RECEIPT_WORKFLOWS)),
        "run_id": run_id,
        "run_attempt": 1,
        "completed_at": "2026-11-06T03:49:00Z",
        "commit_sha": "abc123",
        "manifest_sha256": "0" * 64,
        "manifest_coverage_pct": 0.998,
        "validation_gate": "{0}/{0}".format(11),
        "pipeline_conclusion": "success",
        "contract_validation_status": "OK",
    }
    if effective_session != "__absent__":
        r["effective_session"] = effective_session
    return r


class TestEvaluateReintentaWalkBack:
    """El core del fix: walk-back en slot intermedio -> READY."""

    def test_walk_back_intermedio_devuelve_ready(self):
        with patch.object(pg, "find_completion_receipt",
                          return_value=_receipt("2026-11-04")):
            r = pg.evaluate("2026-11-05", is_last_slot=False)
        assert r["state"] == "READY"
        assert r["should_run"] is True
        assert "walk-back receipt" in r["reason"]
        assert r["effective_session"] == "2026-11-04"
        assert r["receipt_run_id"] == 999

    def test_walk_back_ultimo_slot_devuelve_current(self):
        with patch.object(pg, "find_completion_receipt",
                          return_value=_receipt("2026-11-04")):
            r = pg.evaluate("2026-11-05", is_last_slot=True)
        assert r["state"] == "CURRENT"
        assert r["should_run"] is False
        assert r["effective_session"] == "2026-11-04"

    def test_effective_igual_a_target_devuelve_current(self):
        with patch.object(pg, "find_completion_receipt",
                          return_value=_receipt("2026-11-05")):
            r = pg.evaluate("2026-11-05", is_last_slot=False)
        assert r["state"] == "CURRENT"
        assert r["should_run"] is False

    def test_receipt_sin_effective_compat_current(self):
        """Receipts pre-fix: sin campo. Debe seguir siendo CURRENT."""
        with patch.object(pg, "find_completion_receipt",
                          return_value=_receipt("__absent__")):
            r = pg.evaluate("2026-11-05", is_last_slot=False)
        assert r["state"] == "CURRENT"
        assert r["should_run"] is False
        assert r["effective_session"] is None

    def test_effective_anterior_es_walk_back(self):
        """effective=2026-11-03 (2 dias atras): tambien READY."""
        with patch.object(pg, "find_completion_receipt",
                          return_value=_receipt("2026-11-03")):
            r = pg.evaluate("2026-11-05", is_last_slot=False)
        assert r["state"] == "READY"
        assert r["effective_session"] == "2026-11-03"

    def test_sin_receipt_probe_fluye(self):
        """Sin receipt: evaluate cae al probe habitual (no walk-back)."""
        with patch.object(pg, "find_completion_receipt", return_value=None), \
             patch.object(pg, "_probe_panel", return_value=(0.95, None)):
            r = pg.evaluate("2026-11-05", is_last_slot=False)
        assert r["state"] == "READY"
        assert r["should_run"] is True
        assert "probe coverage" in r["reason"]


class TestValidadorAceptaEffectiveSession:

    def test_effective_session_string_aceptado(self):
        r = _receipt("2026-11-04")
        assert pg._validate_receipt_schema(r, "2026-11-05") is True

    def test_effective_session_ausente_aceptado(self):
        r = _receipt("__absent__")
        assert pg._validate_receipt_schema(r, "2026-11-05") is True

    def test_effective_session_tipo_invalido_rechazado(self):
        r = _receipt(20261104)  # int en vez de str
        assert pg._validate_receipt_schema(r, "2026-11-05") is False


class TestWriterPropagaEffectiveSession:
    """El writer lee quality['effective_session'] y lo pone en el receipt.

    Asercion estatica (centinela de codigo fuente) en vez de mock
    runtime: el writer llama a helpers internos cuyos nombres no
    son contrato. Si en el futuro se refactoriza el writer y el
    campo deja de propagarse, el test falla por regex/AST.
    """

    def test_writer_source_contiene_effective_session(self):
        from pathlib import Path
        src = Path("scripts/write_completion_receipt.py").read_text(encoding="utf-8")
        assert '"effective_session": quality.get("effective_session")' in src, (
            "El writer debe propagar quality['effective_session'] al "
            "receipt. Si se refactoriza, actualizar este centinela."
        )
