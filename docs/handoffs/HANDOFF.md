# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-16 noche)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests:          ver `Consolidacion_Documentos/ESTADO_SISTEMA.md` (fuente autoritativa).
                Referencia orientativa: ~3493 passed + 5 skipped (con filtro CI).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)
Gobernanza:     docs/gobernanza/REGISTRO.yaml (fuente canonica de items auditables)
Contrato gate:  docs/auditoria/daily_run_gate_contrato_v1.md +
                docs/auditoria/daily_run_gate_enmienda_v1_1.md (2026-10-16)

---

## 2. SESION 16-OCT (auditoria del reporte + segundo dictamen + enmienda v1.1)

### 2.0. Contexto

Tercera sesion del dia. Se audita el reporte generado por el run
38834597307 (16-oct 01:20 UTC). 10 hallazgos clasificados.
Emitido informe al auditor externo. Recibido dictamen con 6
precisiones. Aplicadas las fases aprobadas.

### 2.1. Hallazgos criticos (auditoria del reporte)

- **N-C1**: `market_data.parquet` con `last_date=2026-10-14` y
  `US_EQUITY.coverage_at_session=0.0` (contrato EQUITY_EOD con
  `max_lag_days=0`). El pipeline no aborto y el receipt se emitio.
- **N-C2**: `guard_coverage.py` verifica `last > expected` pero no
  `last < expected`.
- **N-C3**: `guard_coverage.py` ignora `by_market` cuando la cobertura
  global pasa el threshold.
- **N-C4**: `write_completion_receipt.py` no verifica `last_date` vs
  `expected_session` del manifest.
- **N-C5**: el receipt emitido con el pipeline desfasado impidio la
  recuperacion por slots posteriores (idempotencia protegiendo una
  finalizacion invalida).
- **N-C6**: `breadth_metrics.py` usaba `df_stocks.index[-1]`
  (posicion fisica), violando R2.
- **N-C7**: sin control explicito de alineacion de sesion esperada
  en `health_check.py`.
- **C2**: STAGFLATION ganaba el `argmax` sin que `IPS > 0.15`.
- **C3**: asterisco de saturacion ausente en el reporte CI
  (`primary_flow_z_raw` no propagado al df in-memory de SSGA).
- **M1, M2, M3**: alerta H1 categorica con |Z| pequeno;
  FLOW_CONFIDENCE empate no documentado; Data Freshness vs Sector
  Breadth sin distincion de dimensiones.

### 2.2. Segundo dictamen (precisiones)

- Criticidad explicita por contrato, no umbral dinamico de volumen.
- Validacion compartida en `src/temporal_contracts/validation.py`.
- Registro de intentos como artifact por ejecucion (no commit al repo).
- Test R3 explicito con datos controlados.
- Guard semantico STAGFLATION con `IPS > 0.15 AND SRS > 0`.
- Enmienda contractual v1.1 publicada antes de tocar el guard.

### 2.3. Implementado

**Commits M1, M2, M3, C3, N-C7, N-C6** — cerrados y verificados.

**Enmienda v1.1** (`docs/auditoria/daily_run_gate_enmienda_v1_1.md`):
- Cambia: guard_coverage ya puede modificarse. Anadida validacion
  contractual compartida con el receipt.
- No cambia: schema del receipt (v1). Invariantes I1-I8 preservadas.
- Anade: campo `critical: bool` en registry, funcion
  `validate_artifact_contracts()`, registro de intentos como artifact.

**`src/temporal_contracts/validation.py`**:
- Estados: OK | WARN | BLOCKED | UNAVAILABLE.
- Reglas: contrato critico con INVALID o coverage < min -> BLOCKED;
  contrato critico con n=0 -> BLOCKED; by_market sin el contrato
  critico -> UNAVAILABLE; by_market ausente -> WARN (no hay desglose).
- `EQUITY_EOD` marcado `critical=True`. Los demas `critical=False`.

**`scripts/guard_coverage.py`**: anade validacion contractual. El
guard original (cobertura global + sha256 + last>expected) se preserva.

**`scripts/write_completion_receipt.py`**: anade la misma validacion
como precondicion. Si `BLOCKED` o `UNAVAILABLE`, no se emite receipt
(fail-closed).

### 2.4. Commits del dia

  4b7caba6  M1+M2+M3
  a4f19514  fix tests H1 a banda neutral + limpieza pyflakes
  e6b2fc0b  C3: primary_flow_z_raw en SSGA return
  e87f94f1  N-C7: check alineacion sesion esperada
  79a24106  N-C6: fecha efectiva canonica breadth + cap R4
  1859a143  enmienda v1.1
  db5cc2db  validation.py + criticidad en registry
  4ac3ec0c  integracion guard + receipt (con 4 tests rojos; corregido en df5b6974)
  df5b6974  fix tests post-integracion

---

## 3. FRENTES CERRADOS EN LA SESION

- Auditoria del reporte del run 38834597307 (10 hallazgos).
- M1, M2, M3 (redaccion + notas + alineacion sesion).
- C3 (provider SSGA).
- N-C6 (breadth R2).
- N-C7 (health check alineacion).
- Enmienda v1.1 + validation.py + integracion guard/receipt.

---

## 4. PENDIENTES VIVOS

### 4.1. Commits 4, 5, 6 del dictamen (proxima sesion)

**Commit 4 — Registro de intentos incompletos como artifact.**
- `scripts/write_completion_attempt.py` (nuevo).
- `outputs/state/completion_attempt.jsonl` (una linea JSON por intento).
- Subida como artifact `completion-attempt-<run_id>` en `daily_run.yml`
  con `if: always()`.
- Campos: `run_id`, `run_attempt`, `target_session`,
  `attempt_timestamp`, `outcome`, `failed_contracts`, `reasons`,
  `manifest_hashes`.
- Tests: no-mutacion, artefacto subido con always, campos obligatorios.

**Commit 5 — Test R3 explicito.**
- Verificar que la fecha efectiva de `sector_breadth.csv` y
  `sector_breadth_momentum.csv` se obtiene via `resolve_effective_date()`
  sobre el universo combinado, no por igualdad entre `max(date)`.
- Datos controlados: filas desordenadas, observacion final no elegible.
- Verifica R2 + R3 sin crear nueva abstraccion.

**Commit 6 — Guard semantico STAGFLATION.**
- `indicators/mte/decision.py::classify_mte`.
- Condicion: `IPS > 0.15 AND SRS > 0` (la del contrato H15).
- Si no se cumple, STAGFLATION no puede ser el escenario final.
- Recalcular confidence sobre el escenario efectivamente elegido.
- Tests obligatorios (tabla del auditor):
  IPS=-0.41 (No), 0.149 (No), 0.15 (No), 0.16+SRS=0 (No),
  0.16+SRS>0 (Si), IPS invalido (No).
- **No sustituir** los tests de `score_scenarios` existentes.

### 4.2. Verificacion del cron tras enmienda v1.1

El primer run del bot tras `df5b6974` verificara:
- `guard_coverage.py` con validacion contractual.
- `write_completion_receipt.py` con la misma validacion.

Con `market_data` en 14-oct y `expected_session` 15-oct, el guard
ahora deberia fallar por contrato EQUITY_EOD incumplido. Verificar
en el cron de la madrugada del 17-oct.

### 4.3. Aviso GitHub

`ubuntu-latest` migra a Ubuntu 26 el 19-oct-2026.

### 4.4. Issue #5

`[health-check] WARN - discrepancias no criticas`. Deuda de proceso.

### 4.5. Nota propia (autocritica de la sesion)

En esta sesion pusheé 3 veces con tests en rojo (el commit 4ac3ec0c
y dos iteraciones previas de otros bloques). Corregido. Motivo:
prisa. Regla no negociable: suite verde antes de commit.

---

## 5. COMO ARRANCAR

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests/ -q -n 12 --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = el que devuelva `git log --oneline -1`.
- ahead 0, behind 0 (o behind N si el bot ha pusheado).
- working tree limpio.
- ~3493 passed + 5 skipped.
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
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta
  ausencia de schedule.
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