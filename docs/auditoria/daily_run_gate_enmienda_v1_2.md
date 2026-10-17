# Addendum v1.2 a la enmienda contractual v1.1

**Contrato base:** `docs/auditoria/daily_run_gate_contrato_v1.md` (2026-10-01).
**Enmienda previa:** `docs/auditoria/daily_run_gate_enmienda_v1_1.md` (2026-10-16).
**Fecha:** 2026-10-17.
**Estado:** ABIERTO a firma externa. Motivado por evidencia real (run 39001863568).
**Schema del receipt:** se mantiene en v1. No hay cambio estructural.

---

## 1. Motivo

El run `39001863568` (verificacion forzada del pipeline, 04:15 UTC del
17-oct, head `8174e4f7`) fallo en el step `Guard coverage` con:

    [FAIL] data/stock_prices.parquet.manifest.json: coverage_pct_last=0.927052 < 0.95

El `completion_attempt.jsonl` emitido (con `if: always()`) registro:

    {"outcome": "INCOMPLETE",
     "failed_contracts": [],
     "reasons": ["sin receipt y sin causa identificada"]}

La causa real (fallo del guard de cobertura) no aparece por ningun lado
del attempt.

## 2. Causa raiz

`scripts/write_completion_attempt.py::_detect_outcome()` solo lee:

- `outputs/state/completion_receipt.json` (receipt emitido -> SUCCESS).
- `outputs/state/validation_gate_result.json` (VG fail -> INCOMPLETE).
- Los contratos criticos via `validate_artifact_contracts()` sobre los
  manifests (contrato falla -> BLOCKED).

No hay comunicacion guard -> attempt. Cuando el guard bloquea por cobertura
global, sha256, `last_date` o status INVALID de un manifest no critico, el
attempt no lo ve. La enmienda v1.1 exige "outcome y causas", pero su contrato
del attempt no incluia el resultado del guard.

## 3. Alcance

### 3.1. Cambia

- `scripts/guard_coverage.py`: escribe `outputs/state/guard_coverage_result.json`
  en ambos caminos (OK y FAIL). Formato:

      {"ok": true,  "threshold": 0.95, "failures": []}
      {"ok": false, "threshold": 0.95, "failures": ["<path>: <motivo>", ...]}

- `scripts/write_completion_attempt.py::_detect_outcome()`: lee
  `outputs/state/guard_coverage_result.json`. Si `ok is False`, emite
  `outcome: BLOCKED`, `failed_contracts: ["guard:<failure>", ...]`,
  `reasons: ["guard_coverage: N fallo(s)", "<failure>", ...]`.

### 3.2. No cambia

- Schema del receipt (v1). Invariantes I1-I8 preservadas.
- Prioridad de `_detect_outcome()`:
  1. Receipt existe -> SUCCESS.
  2. VG `passed=False` -> INCOMPLETE.
  3. Guard `ok=False` -> BLOCKED (nuevo).
  4. Contrato critico falla -> BLOCKED.
  5. Sin causa -> INCOMPLETE "sin receipt y sin causa identificada".
- `guard_coverage.py --threshold` (sin cambio).
- Comportamiento del guard sobre `validate_artifact_contracts()`.

### 3.3. Anade

- Fichero `outputs/state/guard_coverage_result.json` como artifact por
  ejecucion. Se sube con `completion_attempt.jsonl` o como artifact
  propio (decision de CI, no de contrato).
- Dos tests en `tests/test_guard_coverage.py`:
  `test_guard_escribe_result_ok`, `test_guard_escribe_result_fallo`.
- Dos tests en `tests/test_write_completion_attempt.py`:
  `test_attempt_blocked_por_guard_result`,
  `test_attempt_vg_prioritario_sobre_guard`.

## 4. Justificacion

El contrato del attempt (seccion 3 de la enmienda v1.1) exige "outcome y
causas". El run 39001863568 demuestra que hay un caso (guard de cobertura
falla por manifest no critico) en el que el attempt queda ciego. El fix es
contract-additive: solo anade una rama de lectura y una escritura, no
cambia comportamiento existente.

## 5. Rollback

- Quitar la escritura de `guard_coverage_result.json` en
  `guard_coverage.py` (linea `_write_result`).
- Quitar la rama de lectura del guard en `_detect_outcome()`.
- Eliminar los 4 tests nuevos.
- Schema del receipt no cambia: no hay migracion.

## 6. Firma

**ABIERTO a firma externa.** Evidencia: run `39001863568` (attempt ciego
ante el fallo real del guard). Fix minimo, contract-additive. Tests
cubren los dos caminos (guard falla -> BLOCKED, VG falla -> INCOMPLETE
prioritario sobre guard). Sin tocar schema del receipt.

**Fin del addendum.**