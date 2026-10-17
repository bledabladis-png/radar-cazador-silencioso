# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-17 04:04 UTC)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa al cierre: 3588 passed + 5 skipped (local).
                En CI: 3551 passed + 33 skipped + 9 deselected (los 28 que se skipean requieren parquets gitignored).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)
Gobernanza:     docs/gobernanza/REGISTRO.yaml (fuente canonica de items auditables).
                Recuento: 49 FIXED, 3 BLOCKED_EXTERNAL, 2 WONT_FIX, 1 OPEN, 1 SUPERSEDED, 1 FALSE_POSITIVE.
Contrato gate:  docs/auditoria/daily_run_gate_contrato_v1.md +
                docs/auditoria/daily_run_gate_enmienda_v1_1.md (2026-10-16)
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).
                Verificado en runs 39000723877, 39000887521, 39001130522.

---

## 2. SESION 17-OCT

### 2.0. Contexto

Tres tramos:
- 01:00-02:46: diagnostico y fix de los runs fallidos 23:17 UTC (16-oct,
  exit 2) y 01:19 UTC (17-oct, exit 1). Verificacion de la enmienda
  v1.1 en produccion.
- 02:46-03:00: cierre documental (HANDOFF, bitacora, poda 00_ARRANQUE,
  items gobernanza).
- 03:00-04:04: prevencion (fix code smell + 3 tests + workflow
  ci_tests.yml + cierre CI-PREVENTION-GAP).

### 2.1. Tramo 01:00-02:46 - fixes P0 y verificacion v1.1

- P0-A pyyaml: commit `529f4ed2` introdujo `import yaml` en 4 ficheros
  sin declarar pyyaml en requirements.txt. Localmente pasaba por
  `optuna` (dependencia transitiva no declarada). En CI: 3 collection
  errors, exit 2. Fix `4be5a90c`: `pyyaml==6.0.3`.
- P0-B shallow clone: tras fix pyyaml, "Run tests" fallo (exit 1) en
  `test_gobernanza.py::test_registro_real_pasa`. `actions/checkout@v4`
  hace fetch-depth: 1. El test verifica existencia de 37 SHAs del
  REGISTRO contra git. Fix `bd30aecb`: `fetch-depth: 0` en el job
  `run-system`.
- P1 US_EQUITY INVALID: no era causa, era estado heredado del ultimo
  run valido. Resuelto por el run exitoso.
- Verificacion en prod, run `38994345784` (dispatch 02:17 UTC, head
  `bd30aecb`): success. Receipt COMPLETED con
  `contract_validation_status: OK`, `manifest_coverage_pct: 0.9757`,
  `guard_coverage: OK`, `validation_gate: 10/10`,
  `pipeline_conclusion: success`. Attempt: `outcome: SUCCESS`.
- Verificacion del camino CURRENT: run `38998607809` (dispatch 03:23
  UTC). Gate devolvio CURRENT, `run-system` skipped, `issue-manager`
  success. Comportamiento esperado confirmado en produccion.
- 12 items `VERIFIED_IN_CI_PENDING` -> `VERIFIED_IN_PRODUCTION` en
  REGISTRO (`4e2822b5`).

### 2.2. Tramo 02:46-03:00 - cierre documental

- HANDOFF reescrito con el ciclo 17-oct (`33550d82`).
- Bitacora: entrada 2026-10-17 con el tramo 01:00-02:46 (`b63df200`).
- Poda `00_ARRANQUE`: 11528 -> 10192 B + enmienda v1.1 reflejada
  (`ccff8fca`).
- Item `CI-PUSH-NO-VERIF` abierto con `siguiente_accion` errata
  (`c91cb76d`), rectificado tras verificar que pyflakes y gobernanza
  ya eran gates del daily_run (`a792fa91`).

### 2.3. Tramo 03:00-04:04 - prevencion

- Fix code smell confirmado con probe: `_collect_failed_contracts`
  duplicaba `failed_contracts` si un manifest generaba N razones
  BLOCKED (2 contratos criticos simulados -> 2 entradas identicas).
  Movido el append fuera del bucle. `8e96531c`.
- `tests/test_imports_declarados.py`: AST + requirements.txt. + negativo
  con mock (`30bb02c4`). Coste ~0.5s. Habria cazado pyyaml.
- `tests/test_workflow_fetch_depth.py`: todo job con pytest exige
  `fetch-depth: 0`. 5 tests (1 positivo + 4 negativos) (`0c862502`).
  Coste ~0.1s. Habria cazado shallow clone.
- `.github/workflows/ci_tests.yml`: suite completa en cada push a main
  (paths-ignore: data/, outputs/), independiente del gate. `042033a4`.
- Item `CI-PREVENTION-GAP`: OPEN -> FIXED (`019adbdc`) ->
  VERIFIED_IN_PRODUCTION (`161964a3`) con runs 39000723877 y
  39000887521 como evidencia.
- `00_ARRANQUE` + bitacora: bloque CI tests anadido (`bb64fff8`).

---

## 3. FRENTES CERRADOS EN LA SESION

- pyyaml no declarado (P0-A).
- shallow clone incompatible con test de gobernanza (P0-B).
- US_EQUITY INVALID (P1): resuelto por el run valido.
- Verificacion en produccion de enmienda v1.1 (ambos caminos: READY y
  CURRENT).
- 12 items PENDING -> PRODUCTION en REGISTRO.
- Code smell `failed_contracts` duplicado.
- Prevencion: 3 tests + workflow ci_tests.yml + item CI-PREVENTION-GAP
  cerrado.

---

## 4. PENDIENTES VIVOS

### 4.1. Verificacion completa de enmienda v1.1 desde cero

**Hito:** primer run con `target_session` nuevo (lunes 2026-10-19
23:17 UTC). Los runs de hoy verifican los dos caminos del gate pero
parten de un receipt ya emitido. El lunes sera el primer ciclo
completo: gate READY (sin receipt), pipeline, guard, receipt nuevo
con `contract_validation_status`. Bloqueante para dar la enmienda v1.1
por cerrada de punta a punta.

### 4.2. Otros frentes

- **H5.3** (OPEN): cron 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)**: bloqueado por auditor externo.
- **Wyckoff 5b.X v3**: bloqueado por datos (`n_confirmed_H20_complete
  >= 550` post-2026-10-02, ~2027).
- **SOW v6**: pendiente firma externa.
- **Ubuntu 26**: `ubuntu-latest` migra el 19-oct-2026. Los workflows
  ya pinnen `ubuntu-24.04`.
- **Issue #5**: `[health-check] WARN - discrepancias no criticas`.
  Deuda de proceso.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests/ -q --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0 (o behind N si el bot ha pusheado; `git pull --ff-only`).
- working tree limpio.
- Tests: ver `ESTADO_SISTEMA.md`. Referencia al cierre: 3588 passed + 5 skipped.
- pyflakes silencio, compileall OK.
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.

Nota: cada push a main (que toque algo fuera de data/ y outputs/)
dispara `.github/workflows/ci_tests.yml`. La suite se valida en CI
sin esperar al proximo slot del daily_run.

---

## 6. NOTAS CRITICAS

- **H7 fue un bug real.** Test AST en
  `tests/test_audit_contracts_rolling_nan.py`. Si algun fix futuro
  necesita rolling sobre series con huecos, dropna previo.
- **H2 es falso positivo.** `rs_mom` (WLS) invariante a escala.
- **No usar run.py en horario de mercado USA** (13:30-20:00 UTC).
- **Receipt fail-closed (enmienda v1.1).** Si
  `validate_artifact_contracts()` devuelve BLOCKED o UNAVAILABLE,
  no se emite receipt. Siguiente slot hara READY.
- **Contrato EQUITY_EOD `critical=True`.** `market_data` con
  `US_EQUITY.status=INVALID` o `coverage_at_session < 0.90` bloquea
  la publicacion del receipt.
- **Shallow clone rompe tests contractuales de gobernanza.** Todo
  job que corra pytest necesita `fetch-depth: 0`. Ahora hay test
  estatico (`test_workflow_fetch_depth.py`) que lo verifica.
- **`import yaml` requiere `pyyaml` en `requirements.txt`.** No
  confiar en dependencias transitivas. Test estatico
  (`test_imports_declarados.py`) lo verifica.
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta
  ausencia de schedule. Los scheduled de GitHub pueden retrasarse
  horas (observado en 17-oct).
- **Gobernanza canonica.** Cualquier item de auditoria va a
  `docs/gobernanza/REGISTRO.yaml`. El HANDOFF, la bitacora y el
  indice son VISTAS; nunca fuente de estado.
- **Anclas temporales.** Cualquier fecha literal en asignacion modulo
  o default de funcion debe estar en `docs/gobernanza/ANCLAS.yaml`.
- **EOL del corpus:** LF.

---

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_1.md`.
Indice auditoria externa: `docs/auditoria/radar/audits/INDEX_H1_H18.md`.
Expedientes: `docs/auditoria/wyckoff/`, `docs/auditoria/iae/golden/`,
`docs/auditoria/s03_deep/`.

---

**Fin del handoff.**