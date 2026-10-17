# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-17 02:46 UTC)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa al cierre: 3570 passed + 5 skipped (con filtro CI).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)
Gobernanza:     docs/gobernanza/REGISTRO.yaml (fuente canonica de items auditables)
Contrato gate:  docs/auditoria/daily_run_gate_contrato_v1.md +
                docs/auditoria/daily_run_gate_enmienda_v1_1.md (2026-10-16)

---

## 2. SESION 17-OCT (fix CI + verificacion en produccion de la enmienda v1.1)

### 2.0. Contexto

Los runs scheduled 23:17 UTC (16-oct) y 01:19 UTC (17-oct) fallaron
en "Run tests" con exit 2 y exit 1 respectivamente. Diagnostico,
dos fixes P0, dispatch manual, verificacion en produccion.

### 2.1. Hallazgos

- **P0-A - pyyaml no declarado.** El commit `529f4ed2` (gobernanza
  fases 4-7) introdujo `import yaml` en `scripts/verifica_gobernanza.py`,
  `scripts/genera_indice_auditorias.py`, `tests/test_gobernanza.py`,
  `tests/test_anclas_temporales.py` sin anadir `pyyaml` a
  `requirements.txt`. En local pasaba porque `optuna` lo instalaba
  como dependencia transitiva (no declarada tampoco). En CI: 3
  collection errors, exit 2.
- **P0-B - shallow clone.** Tras fix pyyaml, "Run tests" volvio a
  fallar (exit 1) en `test_gobernanza.py::test_registro_real_pasa`.
  Causa: `actions/checkout@v4` hace `fetch-depth: 1` por defecto.
  El test verifica existencia de los 37 SHAs referenciados en
  `REGISTRO.yaml` contra git; solo existia HEAD.
- **P1 - US_EQUITY INVALID.** El `completion_attempt` del run fallido
  (`38990355276`) reportaba `US_EQUITY: status=INVALID` en
  `market_data.parquet.manifest.json`. No causa del fallo: era
  estado heredado del ultimo run valido (el pipeline no corrio).
### 2.2. Implementado

- `4be5a90c`: `pyyaml==6.0.3` en `requirements.txt`.
- `bd30aecb`: `fetch-depth: 0` en el checkout del job `run-system`
  (solo ese; `gate`, `issue-manager` y LSE scraper no ejecutan
  pytest ni dependen de historia).
- `4e2822b5`: 12 items `VERIFIED_IN_CI_PENDING` ->
  `VERIFIED_IN_PRODUCTION` en `REGISTRO.yaml`.
- `8ec3fbd7`: eliminar cifra literal de tests en `00_ARRANQUE.md`
  (drift con `ESTADO_SISTEMA.md`).

### 2.3. Verificacion en produccion

Run `38994345784` (dispatch 02:17 UTC, head `bd30aecb`): success
completo. `completion-receipt-2026-10-16` con
`contract_validation_status: OK`, `manifest_coverage_pct: 0.9757`,
`guard_coverage: OK`, `validation_gate: 10/10`,
`pipeline_conclusion: success`. `completion-attempt`:
`outcome: SUCCESS`, `failed_contracts: []`.

Con esto, la enmienda v1.1 queda verificada en produccion (guard +
receipt compartiendo `validate_artifact_contracts()`).

---

## 3. FRENTES CERRADOS EN LA SESION

- pyyaml no declarado (P0-A).
- shallow clone incompatible con test de gobernanza (P0-B).
- US_EQUITY INVALID (P1): resuelto por el run valido.
- Verificacion en produccion de enmienda v1.1.
- 12 items `PENDING` -> `PRODUCTION` en REGISTRO.

---

## 4. PENDIENTES VIVOS

### 4.1. Documental

- **`00_ARRANQUE.md` sobre tope.** 11528 B > 11264 B (tope
  autoimpuesto). Pendiente: poda + anadir enmienda v1.1,
  `validation.py`, `EQUITY_EOD critical=True` y contrato del gate.
- **Divergencia de comando pytest.** `00_ARRANQUE` sin `-n 12`;
  HANDOFF anterior con `-n 12`; `pytest-xdist` no esta en
  `requirements.txt`. Decision: alinear a `00_ARRANQUE` (sin
  `-n 12`, coincide con el CI). Esta version del HANDOFF lo aplica.
### 4.2. Gobernanza - proceso

- **Deuda detectada.** La sesion 16-oct pusheo al menos dos commits
  que rompian CI real (`529f4ed2` con `import yaml` sin dep, y el
  commit que introdujo `test_registro_real_pasa` incompatible con
  shallow clone). Mismo patron que la autocritica anterior (3 push
  con tests en rojo). Abrir item en `REGISTRO.yaml` como deuda de
  proceso.

### 4.3. Otros frentes vivos

- **H5.3** - cron trimestral 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)** - bloqueado por auditor externo.
- **Wyckoff 5b.X v3** - bloqueado por datos
  (`n_confirmed_H20_complete >= 550` post-2026-10-02, ~2027).
- **SOW v6** - pendiente firma externa.
- **Ubuntu 26** - `ubuntu-latest` migra el 19-oct-2026. Los workflows
  ya pinnen `ubuntu-24.04` (commit `7c913031`).
- **Issue #5** - `[health-check] WARN - discrepancias no criticas`.
  Deuda de proceso.

### 4.4. Deudas documentadas (sin accion pendiente)

Ver `00_ARRANQUE.md` seccion 4.4. Sin cambios.

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
- Tests: ver `ESTADO_SISTEMA.md`. Referencia al cierre: 3570 passed + 5 skipped.
- pyflakes silencio, compileall OK.
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.
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
- **Shallow clone rompe tests contractuales de gobernanza.** El
  job `run-system` necesita `fetch-depth: 0`. Si se anade otro
  job que corra pytest, replicarlo.
- **`import yaml` requiere `pyyaml` en `requirements.txt`.**
  No confiar en dependencias transitivas (optuna lo instalaba
  en local).
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta
  ausencia de schedule.
- **Gobernanza canonica.** Cualquier item de auditoria va a
  `docs/gobernanza/REGISTRO.yaml`. El HANDOFF, la bitacora y el
  indice son VISTAS; nunca fuente de estado.
- **Anclas temporales.** Cualquier fecha literal en asignacion modulo
  o default de funcion debe estar en `docs/gobernanza/ANCLAS.yaml`.
- **EOL del corpus:** LF. (`00_ARRANQUE.md` es LF; versiones previas
  del HANDOFF decian CRLF incorrectamente.)

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