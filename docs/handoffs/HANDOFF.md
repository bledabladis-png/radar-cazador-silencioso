# HANDOFF - Radar de Rotacion Sectorial

Documento de traspaso entre sesiones. Autocontenido. Pegar al inicio
del nuevo chat junto con `Consolidacion_Documentos/00_ARRANQUE.md`.

**Este fichero se sobrescribe cada sesion.** El historico queda en git.
No crear variantes con fecha.

---

## 1. ESTADO AL CIERRE (2026-10-17 06:06 UTC)

Rama activa:    main
HEAD:           consultar `git log --oneline -1` (no literal; el corpus no fija SHA para evitar deriva)
Sincronizacion: main == origin/main
Tests local:    ver `Consolidacion_Documentos/ESTADO_SISTEMA.md`. Al cierre: 3607 passed + 5 skipped.
Tests CI:       3574 passed + 33 skipped + 9 deselected (los 28 que se skipean requieren parquets gitignored).
Validation Gate: 10/10
Working tree:   limpio (salvo data/ y outputs/ que el bot actualiza)
Corpus:         v3 (Consolidacion_Documentos/00-08 + ESTADO_SISTEMA)
Gobernanza:     docs/gobernanza/REGISTRO.yaml. Al cierre: 50 FIXED, 3 BLOCKED_EXTERNAL, 2 WONT_FIX, 1 OPEN, 1 SUPERSEDED, 1 FALSE_POSITIVE.
Contrato gate:  docs/auditoria/daily_run_gate_contrato_v1.md +
                docs/auditoria/daily_run_gate_enmienda_v1_1.md +
                docs/auditoria/daily_run_gate_enmienda_v1_2.md (ABIERTO a firma)
CI de tests:    .github/workflows/ci_tests.yml (push a main, paths-ignore: data/, outputs/).
                Ultimo run verde: 39008384072 (head 237b080a, 2m21s).

---

## 2. SESION 17-OCT

### 2.0. Contexto

Cinco tramos:
- 01:00-02:46: P0 pyyaml + P0 shallow clone + verificacion v1.1 en prod.
- 02:46-03:00: cierre documental (HANDOFF, bitacora, 00_ARRANQUE).
- 03:00-04:04: prevencion (fix failed_contracts + 3 tests + ci_tests.yml).
- 04:04-05:35: auditoria del reporte del run 38994345784 (7 hallazgos
  H1-H5, A, B), 6 commits de fix, verificacion local con run.py real.
- 05:35-06:06: 5 commits con CI rojo por depender de gitignored,
  detectado y corregido. CI verde.

### 2.1. Tramos 01:00-04:04

Ver resumen previo:
- pyyaml (`4be5a90c`), fetch-depth (`bd30aecb`), verificacion v1.1
  (`38994345784`, receipt COMPLETED), 12 items PRODUCTION
  (`4e2822b5`), HANDOFF (`33550d82`), bitacora (`b63df200`), poda
  `00_ARRANQUE` (`ccff8fca`), items gobernanza (`c91cb76d`,
  `a792fa91`, `d8c42736`), addendum v1.2 inicial (`05a86817`),
  ci_tests.yml (`042033a4`, `161964a3`), `00_ARRANQUE` + bitacora
  (`bb64fff8`), HANDOFF 04:04 (`43fe1592`), retirar verify_v11.yml
  (`d34c15bb`).

### 2.2. Tramo 04:04-05:35: auditoria del reporte 38994345784

7 hallazgos, todos cerrados y verificados en reporte regenerado
(run.py local 2026-10-17 05:35 UTC):

- **H1 (data_quality)**: Yahoo Finance cobertura N/D desde 2026-09-12.
  Regresion del commit 1a4e7ea2 (cambio sector_breadth.csv por
  market_data.parquet sin reimplementar cobertura). Fix `dfb8f4d4`:
  rama parquet calcula cobertura sectorial (11 ETFs).
- **H2 (data_quality)**: 11 fuentes con coverage_logic=None. Fix
  `5dd68152`: 'sectorial'/'rows'+coverage_col/'from_row' por fuente.
- **H3 (ssga)**: FEZ dShares N/D por diff sobre 1 fila (last_df
  groupby.tail(1)). Fix `2fef491e`: shares_change en df completo.
- **H4 (flows_international)**: LYXI Z-score N/D sin motivo. Fix
  `a662d882`: leer flow_zscore_regime, anotar MAD0_NAN/INSUFFICIENT/
  MAD0_SAME + nota explicativa.
- **H5 (stock_leader)**: Spring/SOS celda vacia para 0. Fix
  `65ed84c6`: 0/1 explicitos.
- **A (MTE)**: STAGFLATION publicado con IPS=-0.387 (guard C2 solo
  en argmax, no post-histeresis). Fix `05a86817`: guard post-hist,
  forzar MIXED.
- **B (MTE)**: futures_status='BLOCKED' hardcoded (vestigio F3-05).
  Fix `05a86817`: leer temporal_meta.FUTURE_SETTLEMENT.status.

Registro: `REP-AUDIT-20261017` (agrupado, VERIFIED_IN_PRODUCTION).
Reversion de data/ y outputs/ tras verificar (politica 01_METODO §4).

### 2.3. Tramo 05:35-06:06: CI roto y reparado

5 commits consecutivos con CI rojo (39005974500, 39006131075,
39006407799, 39006745624, 39006311775 cancelled):
`test_data_quality_yahoo_coverage` y
`test_data_quality_coverage_12_sources` leian
`data/market_data.parquet` y `outputs/history/*.csv` (gitignored).
En local pasaban, en CI no. Patron "verificar en local, no en CI"
repetido.

Fix `237b080a`: reescribir ambos con fixtures sinteticos en tmp_path
(parquet de 11 sectores, CSVs de rows/sectorial/from_row) + test
estructural (declaracion de coverage_logic, sin archivos).
Verificado en CI run 39008384072 (3574 passed, 33 skipped).

---

## 3. FRENTES CERRADOS EN LA SESION

- pyyaml, shallow clone, US_EQUITY INVALID (P0/P1 del 16-oct).
- Enmienda v1.1 verificada en produccion (ambos caminos del gate).
- 12 items PENDING -> PRODUCTION.
- Code smell `failed_contracts` duplicado.
- Prevencion: 3 tests + ci_tests.yml + `CI-PREVENTION-GAP` cerrado.
- Auditoria del reporte 38994345784: 7 hallazgos (H1-H5, A, B).
- CI rota por gitignored: 5 commits detectados, fix + verificacion CI.

---

## 4. PENDIENTES VIVOS

### 4.1. Verificacion completa de enmienda v1.1 + v1.2 desde cero

**Hito:** primer run con `target_session` nuevo (lunes 2026-10-19
23:17 UTC). Los runs de hoy verifican caminos parciales. El lunes
sera el primer ciclo completo: gate READY sin receipt, pipeline,
guard, receipt nuevo con `contract_validation_status`.

### 4.2. Addendum v1.2 abierto a firma externa

`docs/auditoria/daily_run_gate_enmienda_v1_2.md`. Cubre:
- 3.1-3.3: attempt lee guard_coverage_result (fail-closed ciego
  ante fallo del guard).
- 3.4: guard C2 post-histeresis y futures_status real.

Firma pendiente antes de considerar el addendum como contrato
cerrado. La implementacion ya esta en produccion.

### 4.3. Otros frentes

- **H5.3** (OPEN): cron 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)**: bloqueado por auditor externo.
- **Wyckoff 5b.X v3**: bloqueado por datos (~2027).
- **SOW v6**: pendiente firma externa.
- **Ubuntu 26**: `ubuntu-latest` migra el 19-oct-2026. Los workflows
  ya pinnen `ubuntu-24.04`.
- **Issue #5**: `[health-check] WARN - discrepancias no criticas`.
- **Deuda de proceso**: "verificar en CI, no solo en local" se ha
  repetido 3 veces en 24h (16-oct, 17-oct madrugada, 17-oct
  madrugada-ci). Candidato a regla en `01_METODO`.

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
- Tests: ver `ESTADO_SISTEMA.md`. Referencia al cierre: 3607 passed + 5 skipped.
- pyflakes silencio, compileall OK.
- Gobernanza: `[OK] Gobernanza coherente`.
- Indice H1-H18: `[OK] indice actualizado`.
- Bitacora: `[OK] Bitacora dentro de la regla`.

Nota: cada push a main (fuera de data/ y outputs/) dispara
`.github/workflows/ci_tests.yml`. Los tests nuevos deben funcionar
sin archivos gitignored. Usar tmp_path + fixtures sinteticos.

---

## 6. NOTAS CRITICAS

- **H7 fue un bug real.** Test AST en
  `tests/test_audit_contracts_rolling_nan.py`.
- **H2 es falso positivo.** `rs_mom` (WLS) invariante a escala.
- **No usar run.py en horario de mercado USA** (13:30-20:00 UTC).
- **Receipt fail-closed (enmienda v1.1).** BLOCKED o UNAVAILABLE
  -> no se emite receipt.
- **Contrato EQUITY_EOD `critical=True`.** `market_data` con
  `US_EQUITY.status=INVALID` o `coverage_at_session < 0.90` bloquea
  la publicacion del receipt.
- **Shallow clone rompe tests contractuales de gobernanza.** Todo
  job con pytest necesita `fetch-depth: 0`.
- **`import yaml` requiere `pyyaml` en `requirements.txt`.**
- **Tests nuevos no deben leer archivos gitignored** (data/,
  outputs/). Usar `tmp_path` + fixtures sinteticos. Verificado
  2026-10-17: 5 commits con CI rojo por esto.
- **Schedule vs Dispatch.** Un dispatch reciente NUNCA oculta
  ausencia de schedule.
- **Gobernanza canonica.** Items a `docs/gobernanza/REGISTRO.yaml`.
- **Anclas temporales.** Fechas literales a `ANCLAS.yaml`.
- **EOL del corpus:** LF. `data/quality.py` y `.gitignore` tienen
  EOL mixto (deuda documentada, sin accion).

---

## 7. REFERENCIAS

Corpus normativo: `Consolidacion_Documentos/` (12 docs).
Gobernanza: `docs/gobernanza/` (README, REGISTRO.yaml, ANCLAS.yaml).
Contrato del gate: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_2.md`.
Indice auditoria externa: `docs/auditoria/radar/audits/INDEX_H1_H18.md`.
Expedientes: `docs/auditoria/wyckoff/`, `docs/auditoria/iae/golden/`,
`docs/auditoria/s03_deep/`.

---

**Fin del handoff.**