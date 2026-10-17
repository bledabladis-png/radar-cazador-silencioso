# Addendum v1.2 a la enmienda contractual v1.1

**Contrato base:** `docs/auditoria/daily_run_gate_contrato_v1.md` (2026-10-01).
**Enmienda previa:** `docs/auditoria/daily_run_gate_enmienda_v1_1.md` (2026-10-16).
**Fecha:** 2026-10-17.
**Estado:** APROBADO 2026-10-17 (dictamen externo del 2026-10-17; ver seccion 6). Motivado por evidencia real (run 39001863568).
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

### 3.4. Ampliacion 2026-10-17: guard C2 post-histeresis y futures_status

Auditoria del reporte del run 38994345784 (viernes 16-oct) detecto
dos incoherencias adicionales en el subsistema MTE:

**A. Guard C2 post-histeresis (indicators/mte/decision.py).**
`classify_mte` aplica `scores.pop("STAGFLATION")` cuando no se cumple
`IPS > 0.15 AND SRS > 0`. Pero despues, si `validate_transition(prev,
new) == False`, la histeresis hace `new_scenario = prev_scenario`. Si
`prev == "STAGFLATION"`, la salida final puede ser STAGFLATION aunque
el guard lo haya descartado del argmax. Caso real: run 38994345784 con
`ips=-0.387`, `srs=0.326`, `prev=STAGFLATION`. Fix: aplicar el mismo
guard POST-histeresis. Si `final_scenario == "STAGFLATION"` y no cumple
la condicion, forzar `MIXED` y `pending_out=None`. Dos tests nuevos:
`test_stagfl_no_persiste_por_histeresis_con_prev_stagflation`,
`test_stagfl_si_persiste_con_ips_y_srs_validos`.

**B. futures_status real (indicators/mte/engine.py).**
`engine.py:143` escribia `'futures_status': 'BLOCKED'` como literal.
Vestigio del hallazgo F3-05 (OilPriceAPI sin futuros) cerrado el
2026-09-30 al sustituir por Yahoo (R5). Desde entonces `FUTURE_SETTLEMENT`
esta activo en `temporal_contracts`. El state mentia. Fix: leer
`temporal_meta.by_contract.FUTURE_SETTLEMENT.status`. Un test nuevo:
`test_futures_status_refleja_temporal_meta`.

Ambos son contract-additive: no cambian schema del receipt ni
invariantes I1-I8.

## 4. Justificacion

El contrato del attempt (seccion 3 de la enmienda v1.1) exige "outcome y
causas". El run 39001863568 demuestra que hay un caso (guard de cobertura
falla por manifest no critico) en el que el attempt queda ciego. El fix es
contract-additive: solo anade una rama de lectura y una escritura, no
cambia comportamiento existente.

## 5. Rollback

El addendum agrupa tres cambios de naturaleza distinta. El rollback
se hace por grupos independientes (H21 del dictamen externo
2026-10-17). Revertir un grupo no revierte los otros.

### 5.1. Grupo A: comunicacion Guard -> attempt

- `scripts/guard_coverage.py`: eliminar `_read_target_session`,
  `_split_failure` y el cuerpo nuevo de `_write_result`. Restaurar
  la firma `(ok, threshold, failures)` y la escritura no-atomica.
- `scripts/write_completion_attempt.py`: eliminar
  `_read_guard_result()` y la lectura del guard en
  `_detect_outcome()`. Revertir `failed_contracts` a
  `["guard:<mensaje>"]` si se quiere volver al addendum v1.2
  pre-dictamen; mantener `GUARD_COVERAGE` /
  `GUARD_COVERAGE_UNAVAILABLE` si solo se retira la parte MTE.
- Tests a eliminar (Grupo A):
  - `tests/test_guard_coverage_result_h19.py` (8 tests).
  - Los 7 tests H20 y los 2 actualizados en
    `tests/test_write_completion_attempt.py`.
  - `tests/test_integration_i3_i8_scenarios.py::test_escenario_5_*`
    y `test_i8_guard_ok_no_produce_current`.

### 5.2. Grupo B: guard post-histeresis MTE

- `indicators/mte/decision.py`: revertir el bloque "C2 ampliacion
  2026-10-17" (el que fuerza MIXED si `final_scenario ==
  STAGFLATION` y no cumple `IPS > 0.15 AND SRS > 0`).
- Tests a eliminar (Grupo B):
  - `tests/test_mte_guard_post_hist_persistence.py` (5 tests).
  - En `tests/test_classify_mte_stagflation_gate.py`:
    `test_stagfl_no_persiste_por_histeresis_con_prev_stagflation`
    y `test_stagfl_si_persiste_con_ips_y_srs_validos`.

### 5.3. Grupo C: futures_status derivado del contrato

- `indicators/mte/engine.py`: restaurar
  `'futures_status': 'BLOCKED'` (literal pre-2026-10-17) o
  mantener la lectura del metadato sin la rama
  `or 'UNAVAILABLE'`, segun se prefiera.
- Tests a eliminar (Grupo C):
  - `tests/test_mte_futures_status_unavailable.py` (6 tests).
  - `tests/test_mte_engine.py::test_futures_status_refleja_temporal_meta`.

### 5.4. Invariantes

Schema del receipt no cambia en ningun grupo: no hay migracion de
artefactos. I1-I8 se mantienen en los tres casos.

## 6. Firma

**APROBADO 2026-10-17.** Fundamento: dictamen externo del
2026-10-17 sobre el addendum v1.2. El dictamen aprobo el fondo y
condiciono la firma a cinco precisiones (H19-H23). Las cinco
condiciones se han resuelto:

- H19 (identidad de ejecucion, escritura atomica, manejo de
  excepciones): commit `c8a265ff`, 8 tests en
  `tests/test_guard_coverage_result_h19.py`.
- H20 (validacion de identidad y `failed_contracts` estables):
  commit `00c70f4a`, 7 tests nuevos + 2 actualizados en
  `tests/test_write_completion_attempt.py`.
- H21 (rollback por grupos A/B/C): seccion 5 ampliada arriba.
- H22 (persistencia del state MTE, no solo la etiqueta): commit
  `5dbdce8e`, 5 tests en
  `tests/test_mte_guard_post_hist_persistence.py`.
- H23 (`futures_status` = UNAVAILABLE si falta metadato): commit
  `20c21c2b`, 6 tests en
  `tests/test_mte_futures_status_unavailable.py`.
- Escenario 5 del dictamen (VG + guard: ambas causas) + I3/I8
  extendidos: commit `745df9ed`, 4 tests en
  `tests/test_integration_i3_i8_scenarios.py`.

CI verde en el run `39009989796` (head `745df9ed`). Suite local
3641 passed + 5 skipped (`ESTADO_SISTEMA.md`, commit `35fcdef7`).

La certificacion de produccion queda pendiente del primer run
scheduled con `target_session` nuevo (lunes 2026-10-19 23:17 UTC),
que verificara de punta a punta la enmienda v1.1 + este addendum.

**Fin del addendum.**
