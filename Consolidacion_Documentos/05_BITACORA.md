# 05 - BITACORA

**Registro de sesiones recientes. NO normativo. NO se pega al arrancar.**
**Para sesiones anteriores, ver `git log`.**
**Se poda por antiguedad: mantener las ultimas 10 sesiones + las del dia en curso. Al anadir una nueva, la mas antigua fuera del dia se elimina o se resume en 04_HISTORICO.md. Ver seccion 5.**

---

## 1. PROPOSITO

Bitacora de las ultimas sesiones. Contiene:
- Que se hizo.
- Que quedo pendiente.
- Commits asociados.
- Proximo paso sugerido.

**No es 04_HISTORICO.md.** El historico es tematico y estable. La bitacora es cronologica y se poda.

---

## 2. FORMATO DE UNA ENTRADA

Cada sesion se registra asi:

    ### YYYY-MM-DD — Titulo corto

    **Objetivo.** Que se pretendia hacer.

    **Hecho.**
    - Bloque 1.
    - Bloque 2.

    **Commits.** <lista de SHAs> (o rango).

    **Pendiente.**
    - Item 1.
    - Item 2.

    **Proximo paso sugerido.** Que deberia hacerse a continuacion.

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 10 fuera del dia en curso, la mas antigua se elimina o se resume en el historico. Ver reglas completas en §5.

---

## 3. SESIONES

### 2026-11-08 — Sesion 34: 3 hermanos de normalizacion + MACRO-SCORE false positive

**Objetivo.** Cerrar deudas de codigo abiertas, no solo documentacion. Foco
en la clase "ticker crudo vs normalizado en output de writers", tras
detectar que IAE publicaba BRK-B y BRK.B como activos distintos.

**Hecho.**
- MACRO-SCORE-EMPTY-LATENTE (OPEN desde 2026-10-31) -> FALSE_POSITIVE.
  Guard de render_regimenes existe desde d5b94209. compute_macro_regime
  nunca devuelve None/NaN en conf. Gap real: test centinela. Anadido
  tests/test_macro_score_empty_guard.py (5 tests verdes).
- CATALOG-BRK-FALLBACK-NORMALIZE FIXED (d41c223e). security_identity.py
  cargaba cusip_ticker_exceptions y etf_holdings sin normalizar. Mismo
  CUSIP producia BRK-B por crosswalk y BRK.B por fallback. Fix:
  normalize_yahoo_ticker en ambas ramas. E2E run.py real.
- RS-INTERNAL-TICKER-NORMALIZE FIXED (9d712c28). rs_internal.py publicaba
  ticker crudo (BF.B) en lugar de ticker_norm (BF-B). Ademas no pasaba
  replace_date=True a persist -> fila huerfana coexistia con la nueva.
  Dos fixes + 2 tests. E2E run.py real.
- WYCKOFF-HOLDINGS-TICKER-NORMALIZE FIXED (4132dae3). _evaluate_holding
  publicaba ticker crudo aunque usaba ticker_norm para lookup. 1 linea +
  test nuevo. Verificado por probe + unit test; E2E sin regresion (la
  condicion no se dispara en el run actual).
- Deuda S3 limpiada: 19 ficheros .bak/.orig (18 duplicados identicos de
  sector_wyckoff_distribution v1 + 1 p1_h02_cobertura_efectiva).
- Las 3 docs-sync: cierres coherentes de HANDOFF + ARRANQUE + REGISTRO +
  ESTADO_SISTEMA. TEST-COUNT-DUP cerrado de facto.

**Commits.** `6026e699`, `d41c223e`, `9d712c28`, `4132dae3`, `15487996`,
`e0e0c7f4`, `c6ebc36b`.

**Pendiente.**
- 7 OPEN restantes: H5.3 (cron 13F 20-nov), REP-AUDIT-LAG (17-nov),
  TEST-FRESHNESS-UMBRAL-VS-CADENCIA (vence ya), TEMPORAL-EFFECTIVE-DATE /
  MTE-DEGRADED-STATE / FRESHNESS-POR-BLOQUE (decision producto),
  SEMANTICA-SIN-CONTRATO, DICT3-H04 (registro historico).
- 2 BLOCKED_EXTERNAL sin novedad: C2-920, WYCKOFF-5BX.
- Scheduled 23:17 UTC del 2026-11-09: primer run con los 3 fixes en main.
  Verificar nota 713 (pendiente por cadencia desde S33-bis), CSVs Wyckoff,
  crosswalk estable, sin regresiones.

**Proximo paso sugerido.** Verificar scheduled 2026-11-09 23:17 UTC
(artefacto daily-report del run). Despues: TEST-FRESHNESS-UMBRAL-VS-CADENCIA
(cierre o prorroga segun observador) y las 3 decisiones de producto.

### 2026-11-07 — Sesion 32: 5 fixes de frescura y aritmetica (auditoria 2026-11-07)

**Objetivo.** Cerrar bloque auditoria 2026-11-07 (12 hallazgos),
verificar frescura de datos contra fuentes reales, confirmar
autonomia 13F.

**Hecho.**
- H5 (PCR lag): el render publicaba PCR del 4/11 bajo fecha VIX
  5/11 sin declarar. Causa raiz: last_date existia en pcr_data,
  se perdia en market_data. Fix: propagar + lag N d + nota. 12
  tests.
- H2 (A/D sin fecha): A/D Net, NH/NL, A/D Line sin declarar
  fecha efectiva. Render lee ad['effective_date'] que ya existia.
  14 tests.
- H3 (derivados breadth sin DEGRADED): Sector Regime Matrix y
  Matriz de Evidencia consumian breadth sin avisar walk-back.
  Mecanismo input_status reutilizando indicators/mte/status.py.
  Persistido como columna + nota en render. 6 ficheros, 15 tests.
- H11 (NIPC aritmetico): Total != suma breakdowns (+1). Causa:
  int() truncando. Fix: int(round) + derivar Total = suma
  breakdowns + nota si difiere. N/D preservado en NaN/None.
  6 tests.
- H12-fix (gate walk-back): bug de frescura detectado por el
  operador. Run 42156577761 (06-11 03:26) hizo walk-back a 04-11,
  los 4 runs posteriores devolvieron CURRENT por receipt existente.
  Reporte de las 18:03 seguia con datos del 04-11. Fix: receipt
  incluye effective_session; evaluate(target, is_last_slot=False)
  devuelve READY si effective < target y no es ultimo slot.
  10 tests.
- Autonomia 13F verificada: workflow active, cron
  `17 6 20 2,5,8,11 *` introducido 25-sep-2026. Primer disparo
  real 20-nov-2026 06:17 UTC (Q3-2026). 0 runs scheduled en
  historial porque no ha habido oportunidad. Retry H5.2 + issue
  si falla.

**Commits.** 20 desde f4747ce5. Los 5 de codigo:
`994212c8`, `136309e4`, `bcbe468a`, `551367e4`, `660bc65a`.
8 syncs documentales + otros.

**Pendiente.**
- 3 OPEN en REGISTRO (H1, H4, H10): decision de producto.
- 5 semanticos sin item (H6, H7, H8, H9, H12-auditor): decision
  de producto o cubiertos.
- Deuda TEST-COUNT-DUP: literales duplicados en 4 sitios,
  desincronizado 5 veces en S31-S32.
- Deuda GATE-PANEL-SOLO-BLUE-CHIPS: gate mide frescura sobre 20
  blue chips. Mitigado por H12-fix.

**Proximo paso sugerido.** Decision de producto sobre H1/H4/H10.
Recomendacion para H4: WONT_FIX razonado (confidence ya declarada
'no calibrada', estado publicado por canal separado). Abrir items
formales para H6/H7/H8/H9/H12-auditor si el operador quiere
visibilidad en gobernanza.

---

### 2026-11-06 — Sesion 31: 9/10 huecos del cuestionario S30

**Objetivo.** Responder al saliente S30 y cerrar los huecos reales
detectados en el cuestionario (22 preguntas: 5 prioritarias + 17
sistematicas). Prioridad: verificacion pura > bugs > tests > docs.

**Hecho.**
- 10 huecos reales identificados. 9 cerrados; el #1 (`issues: write`
  en CI) solo verificable tras scheduled 07-11.
- Hallazgo grave no anticipado: el AST guard de `REPORT-NEGATIVE-ZERO`
  (S30) estaba roto en Python 3.14 (local + CI). PEP 701 envuelve el
  `format_spec` en `ast.JoinedStr`; el guard solo miraba
  `ast.Constant`. Pasaba el test en falso. Al reparar el guard se
  destaparon 5 sitios no migrados (validation_gate:103,105,
  confirmation:134, etf_flows:34, header:106).
- LSE a REGISTRO como WONT_FIX razonado: scraper es repo externo
  privado con cron propio, no rama paralela. Ampliar a 86 tickers
  requiere Refinitiv no disponible.
- Euronext/XETRA/BME cubiertos como worst_market en Sector Breadth
  (antes solo LSE). Codigo productivo limpio verificado.
- Q15 (MTE UNCONFIRMED): umbral real vive en `volatility_mte.py:94`
  (no en `scoring.py` como apunto el saliente). 12 tests parametrizados
  + 1 de sincronizacion texto<->logica (extrae el umbral del texto
  contractual con regex, sin hardcodear).
- 3 sincronizaciones de conteo de tests (4556 -> 4567 -> 4579). El
  patron estructural (conteo duplicado en 4 sitios) queda anotado
  como deuda TEST-COUNT-DUP en HANDOFF §3.6.
- Gobernanza: item nuevo `REPORT-NEGATIVE-ZERO-GUARD-PY314` (FIXED,
  d4e80b9b). Enmienda a `REPORT-NEGATIVE-ZERO` para no dejar
  evidencia falsa en el registro canonico. 303 -> 304 items.

**Commits.** `99142a6a`, `de9df670`, `d4e80b9b`, `4dc1c32f`,
`ff8d040e`, `69cab02a`, `ea4863c4`, `dfc5dc5d`, `af1ea520`,
`f258d95e`.

**Pendiente.**
- #3 Q12 CSV crudo (severidad alta): los 101 runs del observatorio
  S30 solo estan en prosa en 04_HISTORICO. Requiere sesion dedicada
  para reconstruir + persistir CSV reproducible + script regenerador.
- #1 issues:write CI: verificar scheduled 07-11 ~03:26 UTC.
- Deuda TEST-COUNT-DUP (conteo duplicado en 4 sitios, 3
  desincronizaciones esta sesion).
- Deuda FIXED-EVIDENCIA-FALSA (pattern posible en otros FIXED).

**Proximo paso sugerido.** Sesion dedicada a Q12 CSV crudo, o
verificar #1 tras scheduled 07-11. En paralelo: abrir items formales
para TEST-COUNT-DUP y FIXED-EVIDENCIA-FALSA en REGISTRO.

---

### 2026-11-06 — Sesion 30: 4 deudas codigo S29 + 3 bugs reporte + Q10 + Q12

**Objetivo.** Cerrar deuda codigo del cuestionario S29 (18 preguntas al saliente)
antes de Q10/Q12/Q15. Prioridad: bugs reales > deuda durable > documental.

**Hecho.**
- Q6 anti-drift (`3761fb0e`): `test_derivacion_por_clase` verifica los 8 campos de `_build_registry_entry`.
- Q7 guard (`24a6a74f`): `resolve_security_identity` raise si 3 kwargs None y cusip no vacio. Footgun real de HON 05-11.
- Q14 mojibake durable (`9a368461` + `a0d167fa` + `a041e3a0`): heuristica Unicode, auto-descubrimiento de roots, whitelist por fichero, 3 centinelas. Autocritica: `9a368461` sobrescribio test existente sin `.orig`, subsanado en `a0d167fa`.
- Q13 enforcement (`9443f7da`): `scripts/hooks/` con pre_commit_check.py, pre_push_check.py, install.py. Stubs sh en .git/hooks, validados en 2 pushes reales. 12 tests centinela.
- Bug E -0.00 (`a37e1a9a`): `f"+.2f"` sobre z=-0.001 -> `-0.00` en Alertas de Divergencia. Fix durable: `src/formatting.py:format_signed_float`. 5 sitios migrados. 21 tests (18 param + 2 integracion + 1 AST centinela).
- Bug B WARN stale (`7d2724b6`): Sector Breadth decia "sesion NYSE pendiente" cuando la causa era LSE 18.6%. Fix: `_compute_stale_context` deriva el mercado con menor coverage desde `_compute_by_market`. 10 tests nuevos. `test_pipeline_breadth_metrics` a 5 keys.
- Bug F granularidad (`7d0c6732`): Data Freshness (Yahoo CURRENT, sectores 11/11) vs Provenance (WALK_BACK, tickers 408/530). Fix: nota forward-reference cuando walk_back_applied. 3 tests.
- Cierre documental intermedio (`e9cba8e6`): ESTADO_SISTEMA + 04_HISTORICO S30 + 00_ARRANQUE 3.
- Q12 gobernanza (`bbd9e4b4`): CRON-SLOTS-OBS -> FIXED (21/22 dias con >=1 run); SCHEDULER-DELAYS -> WONT_FIX (p95 <=8h). Ventana 22 dias, 101 runs cruzados con run-system.
- Q10 check_yahoo_revision (`cf75d79a`): script con --with-issues. Abre/cierra issue label yahoo-revision. `daily_run.yml`: issues: write + env GH_TOKEN. 12 tests.
- Diagnostico LSE (no commit): scraper cubre 20/86 LSE tickers. 70 FTSE 250/small caps sin cierre Yahoo. WONT_FIX razonado.

**Commits.** `24a6a74f`, `3761fb0e`, `9a368461`+`a0d167fa`+`a041e3a0`, `9443f7da`, `a37e1a9a`, `7d2724b6`, `7d0c6732`, `e9cba8e6`, `bbd9e4b4`, `cf75d79a`, cierre formal S30.

**Pendiente.**
- Q15 (confidence<0.5 MTE + FRESHNESS_* + SATURATED/MAD0_SAME): sesion aparte con decision producto. Baja urgencia.
- WYCKOFF-5BX: BLOCKED_EXTERNAL (~2027). C2-920: BLOCKED_EXTERNAL.

**Proximo paso sugerido.** Sesion de Q15 con decision producto, o reabrir frentes pendientes segun REGISTRO.

### 2026-11-05 — Sesion 29: 3 bugs reales (D1, HON, BKNG) + 5 items del saliente

**Objetivo.** Cerrar los 5 items del saliente (FORCE-DISPATCH,
POWERSHELL-TRAMPS, CHECK-YAHOO-REVISION, SEMANTICA-SIN-CONTRATO,
TEST-FRESHNESS) + investigar los 2 que quedaban OPEN sin causa raiz
(COMMIT-46A3B370, CHECK-API-KEYS). Durante la investigacion se
descubrieron 3 bugs reales.

**Hecho (20+ commits pusheados).**

Bugs reales (todos con E2E y test que cierra la clase):

- **D1 TEMPORAL-CONTRACTS-NO-WIRED** (`1644adb3`). CONTRACTS_REGISTRY
  era dict paralelo de 9 entradas a _IMPLEMENTED_CONTRACTS (13).
  resolve_all_contracts iteraba list_contracts() -> 9. Fix: derivar
  de _IMPLEMENTED_CONTRACTS + atributos de clase. Test anti-drift.
  E2E run 42020621979: 13 contratos resueltos.
- **HON-REVERSE-SPLIT-CUSIP** (`cc576403`). CUSIP pre/post reverse
  split 2026-06-29. 507 filings Q2 con CUSIP viejo caian a
  UNRESOLVED_IDENTITY -> EXIT artificial. Fix: 2 entries en
  cusip_equivalence.csv. n_exit HON 902 -> 784. Residual 784: 690
  venta real, 42 sin filing, 11 CUSIP raro. 6 tests nuevos.
- **BKNG-SPLIT-DELTA** (`0dbddfcc`). Split 24:1 entre Q1 y Q2. delta
  artificial +688M acciones. Fix: nuevo split_detection.py con
  deteccion por doble ratio (precio y shares), exclusion de bonos y
  ETFs apalancados. sum_delta +688M -> -27.7M. 8 splits detectados.
  14 tests.
- **VERIFICAR-MTE-STATE-PROD** (`b73294f7`, item creado FIXED).
  Run 42020621979 confirma schema v2, mte_state=DEGRADED,
  futures_status=STALE.

Items del saliente cerrados (5):

- FORCE-DISPATCH-EMERGENCIA (`29688163`): runbook §4.5 reescrito
  con causa real (force salta el if: del job, no bypassea
  pipeline_gate.py), backup PRE_SHA, procedimiento 8 pasos, 2
  opciones de reversion.
- POWERSHELL-TRAMPS-NO-DOCUMENTADAS (`63af55dc`): §6quater en 01d
  con 5 trampas verificadas (if inline, exit, array vacio, comilla
  simple, git checkout LASTEXITCODE=0).
- CHECK-YAHOO-REVISION-FRECUENCIA (`e1f8090e`): bug de ubicacion.
  check_yahoo_revision estaba en health_check.yml (current==head,
  nunca dispara). Movido a daily_run.yml post-run.py pre-commit.
  Verificado: [OK] sesion nueva (2026-11-03 -> 2026-11-04).
- SEMANTICA-SIN-CONTRATO-INVENTARIO (`bce81bf4`, OPEN con inventario
  parcial). 6 candidatos auditados: 3 con contrato, 3 sin
  (FRESHNESS_*, confidence<0.5, SATURATED/MAD0_SAME). Decision
  contrato vs WONT_FIX pendiente.
- TEST-FRESHNESS-UMBRAL-VS-CADENCIA (`76bd01e1`, hipotesis REFUTADA).
  Los 3 tests failed NO eran umbral. Tras refresh real del bot
  (3172eaf5): 4469 passed, 0 failed. Observatorio
  freshness_gap_observer.py recoge serie hasta 11-10.

Incidente CI corregido (`340bfadc`): los 2 scheduled del 05-11
fallaron en Run tests porque los scripts nuevos no estaban en
CI_ENTRYPOINTS ni cubiertos por _local_names(). Fix + extension.
Autocritica: la regla 01_METODO §3 existe por esto.

Items del saliente aclarados:

- COMMIT-46A3B370-AUDIT (`fbc60969`, WONT_FIX): 10 hits
  documentales, 0 en codigo. Commit edito archivo declarado obsoleto.
- CHECK-API-KEYS-HUERFANAS (`fbc60969`, FALSE_POSITIVE): las 6 API
  keys tienen consumidor real.

Bugs de coherencia documental:

- HEAD literal stale por auto-referencia en 00_ARRANQUE y HANDOFF.
- 22 commits -> 31 (27 contenido + 4 cierre).
- "Se auto-resuelven con el proximo refresh" falso. Corregido a
  hipotesis refutada.
- 4 mojibakes en 05_BITACORA (triple UTF-8 -> latin-1 -> UTF-8).
- COMMIT-46A3B370-AUDIT seguia OPEN en 00_ARRANQUE tras cerrarlo.
- 9f59365a (HON) quedo huerfano tras rebase. SHA real: cc576403.

**Verificacion.**

- pyflakes LIMPIO, compileall OK.
- Suite completa: 4469 passed, 0 failed, 2 skipped.
- gobernanza [OK], poda [OK].
- E2E run.py local: 17.1 min, gate 11/11, 13 contratos.
- Run scheduled 42020621979 (head 5055a2f3): success. 3 cambios
  nuevos verificados (Observe freshness gap, Check yahoo revision,
  13 contratos). Reporte auditado: 0 strings de modulos retirados,
  0 mojibakes, 0 nan%/-0.00, N/D 8 (normal).

**Pendiente.**

- Cierre documental de sesion 29 (HANDOFF + 05_BITACORA +
  04_HISTORICO). 00_ARRANQUE y REGISTRO ya actualizados.
- SEMANTICA-SIN-CONTRATO-INVENTARIO: decision 3 conceptos.
- 06-nov: DAILY-RUN-FAIL-40992758129 (si no vuelve, WONT_FIX).
- 10-nov: CRON-SLOTS-OBS, SCHEDULER-DELAYS, TEST-FRESHNESS
  (>=5 filas observatorio).
- 17-nov: RECEIPT-AUDIT-LAG. 20-nov: H5.3.

**Commits.** `1644adb3`, `fbc60969`, `4a8987da`, `3aa295fb`,
`29688163`, `af33648d`, `63af55dc`, `f47db432`, `e1f8090e`,
`44ef1ca8`, `bce81bf4`, `76bd01e1`, `f4684b33`, `340bfadc`,
`5055a2f3`, `c707a1ee`, `cc576403`, `0dbddfcc`, `87e09b73`,
`f1202d9e`, `90a78259`.

**Proximo paso sugerido.** 2026-11-06: revisar DAILY-RUN-FAIL-40992758129
(si no reaparece, WONT_FIX). 2026-11-10: los 3 observacionales +
TEST-FRESHNESS con >=5 filas. Decision SEMANTICA cuando haya margen.

### 2026-11-04 — Sesion 28: 7 items FIXED con codigo real

**Objetivo.** Cerrar con codigo real, no solo documentacion, los 4 items
de auditoria 11-03 pendientes (B1/B2/B3/B5) y 2 items relacionados
(INVESTIGAR-TESTS, WYCKOFF-5BX-INSTRUMENTACION). Desbloquear
generate_estado_sistema.py, que llevaba 2 sesiones sin regenerar.

**Hecho (22 commits).**

- VERIFICAR-*-PROD FIXED (`bc7e0db8`). Evidencia run `41764334709`
  (head `45ba9bc2`). WALK_BACK en reporte L13-15. IAE: 11 tickers del
  auditor en tabla correcta. WYCKOFF-PHASE: Matriz de Regimen publica
  fases con senal (`DIST_FORMING (20%, N=5)`).
- B2 DECISION-MTE-DEGRADED-STATE FIXED (`b73294f7` + `cca7a372`).
  `indicators/mte/status.py` con VALID/DEGRADED/INVALID. `mte_state`
  en JSON (schema v1->v2) y en render. 5 tests centinela.
- INVESTIGAR-TESTS FIXED (`729317d7` + `3b6bfc90`). 7 tests protegidos,
  1 falso positivo (`test_verify_actual_pasa`), 1 bug real
  (`test_intersection_main_devuelve_0` escribia en `data/mappings/`).
- B3 DECISION-FRESHNESS-POR-BLOQUE FIXED (`18f77740` + `fdd83b34`).
  `validation_gate` hardcodeaba `age > 14` (FINRA) y `age > 5` (CBOE).
  Fix: importa `FRESHNESS_CBOE`/`FRESHNESS_FINRA` de `config.settings`
  y usa `[1]` (frontera RECENT/STALE). FINRA 14 -> 28. 5 tests.
- B1 DECISION-EFFECTIVE-DATE-POR-MODULO FIXED (`568decd5`, `3efd8037`,
  `3fc0746e`, `7ecd595c`, `324941a4`). 4 TemporalContracts nuevos
  (SEC_13F, FINRA_DARKPOOL, SSGA_PRIMARY_FLOW, FRED_MACRO).
  `_IMPLEMENTED_CONTRACTS` 9 -> 13. Justificacion de las 2 fuentes
  no cubiertas: HoldingsPIT (log PIT append-only, no mide lag);
  BME/Euronext (cubiertos por EQUITY_EOD + INDEX_EOD_EUROPA).
- WYCKOFF-5BX-INSTRUMENTACION FIXED (`54d4a23f` + `f0f21518` +
  `d3a0c09e`). `indicators/wyckoff_h20.py` + `scripts/wyckoff_h20_
  counter.py`. 11 tests centinela. Wire en `daily_run.yml` job
  `run-system` idx 20. Primer valor: 27/68 confirmados (ACC=2, DIST=25).
- generate_estado_sistema.py desbloqueado (`ab75e39c` + `778c3089`).
  `run()` con timeout, `section_tests()` con `-n auto` + `timeout=600`.
  Regenerado ESTADO_SISTEMA.md: 137s (antes >10 min colgado). 5 fixes
  colaterales: pyflakes limpio (3 imports sin usar), `test_generate_
  estado_failclosed` (`_run` acepta timeout), `test_ci_entrypoints_
  import` (`wyckoff_h20_counter` en `CI_ENTRYPOINTS`).

**Pendiente al cierre.**

- Cierre documental de sesion 28 (HANDOFF + 00_ARRANQUE + 05_BITACORA +
  04_HISTORICO). HANDOFF y 00_ARRANQUE ya actualizados.
- 5 items OPEN de documentacion/investigacion: COMMIT-46A3B370-AUDIT,
  SEMANTICA-SIN-CONTRATO-INVENTARIO, CHECK-YAHOO-REVISION-FRECUENCIA,
  FORCE-DISPATCH-EMERGENCIA, POWERSHELL-TRAMPS-NO-DOCUMENTADAS.
- Verificar `mte_state` en prod (proximo scheduled post-fix).

**Proximo paso sugerido.** Elegir entre: (1) CHECK-YAHOO-REVISION-
FRECUENCIA (bug real potencial), (2) integrar `timestamps.py` en
pipeline_contractual (modulo huerfano), (3) FASE-A-REGIMEN-ECONOMIA-
REAL (3-5 sesiones).

---
### 2026-11-03 — Sesion 27 parte 5: consolidacion documental

**Objetivo.** Tras el fix WYCKOFF-PHASE-SEMANTICA, dejar la
documentacion al dia: contrato nuevo de la Matriz de Regimen
schema v3, arquitectura actualizada, 2 lecciones nuevas, tests
adaptados a v3, ESTADO_SISTEMA regenerado.

**Hecho.**

1. `1335cb72` test(schema): adaptar 4 tests al schema v3.
   T1 y T3 esperaban `.schema == "2"`; T3 renombrado v1_a_v3;
   T2 usa v3 (no v2) para `sin backup`; T6 usa v3 (mismo que
   spec). `test_matriz_nota_fase_dominante` adaptado a la nota
   nueva (RANGE no es fase). ESTADO_SISTEMA regenerado:
   4396 -> 4403 passed (+7: 3 IAE + 4 sector_regime).

2. `7e4fe196` docs(arquitectura):
   - `docs/auditoria/sector_regime_matrix_v3.md` nuevo (4918 B,
     LF). Contrato semantico del schema v3: 3 columnas nuevas
     (`wyckoff_signal_phase`, `_share`, `_n`), pool de 6 fases con
     senal, desempate determinista, invariantes I1-I5, ejemplo con
     datos reales del 30-10, rollback.
   - `02_ARQUITECTURA.md` L216-224: bloque nuevo distinguiendo
     `wyckoff_phase` (moda bruta con RANGE, auditoria) vs
     `wyckoff_signal_phase` (moda de fases con senal, informativa).
   - `01_METODO.md` seccion 3: leccion "Escala de `pct_*`: 0-100,
     no 0-1". Bug real del render: `* 100` daba 2000% donde el
     valor era 20%. Detectado en probe antes de commitear.

3. `ded662ed` docs(metodo): leccion "`Out-String` convierte
   `\n` a `\r\n` al capturar". Dos diagnosticos erroneos en la
   misma sesion por usar `git cat-file -p | Out-String` para
   verificar EOL de blob. El blob estaba bien (LF, 4918 B); el
   instrumento de medida estaba roto. Regla: leer binario con
   `[IO.File]::ReadAllBytes` o usar `git ls-files --eol` / `git
   cat-file -s`.

**Verificacion.** pytest 25 passed (6 writer + 1 nota + 9
sector_matrix + 9 mas). ESTADO_SISTEMA 4403 passed, exit 0.
Blob en HEAD del contrato v3: 4918 B, LF (verificado con
`git cat-file -s` y lectura binaria). verifica_gobernanza [OK]
tras cada commit.

**Pendiente / deuda activa.**
- `00_ARRANQUE` 20305 B (margen 175 B). Proximo contenido
  obliga recorte o division.
- `01_METODO` 20369 B (margen 111 B). Idem.
- 3 items de verificacion prod OPEN (WALK-BACK, IAE-CONSENSO,
  WYCKOFF-PHASE). Proximo run scheduled real los cierra con la
  misma descarga.

**Commits.** `1335cb72`, `7e4fe196`, `ded662ed`.

**Proximo paso sugerido.** Esperar run scheduled real (slot
23:17 UTC del 03-11 + delays). Verificar los 3 items con una
descarga del artifact daily-report. Tras verificacion, recorte
de `00_ARRANQUE` y `01_METODO` antes de anadir contenido nuevo.

### 2026-11-03 — Sesion 27 parte 4: fase dominante excluye RANGE

**Objetivo.** El usuario pregunta por que la columna 'Fase dominante'
de la Matriz de Regimen Sectorial es RANGE en los 11 sectores, si
supuestamente el modulo Wyckoff no clasifica ETFs. Auditoria completa.

**Hallazgo (no era bug de calculo).** No hay contradiccion: lo que
se retiro en sesion 3 fue clasificar el ETF como entidad. Lo que hay
es clasificacion de holdings. Pero la columna era decorativa:
RANGE ganaba 85-95% en 8 de 11 sectores. Porque RANGE = ausencia de
senal, no fase Wyckoff. La moda bruta siempre va a ser RANGE mientras
la mayoria de holdings no completan cascada. La columna decia la
verdad pero ocultaba la senal real (la 2a fase).

**Evidencia (datos reales 30-10).** XLK RANGE=44% (2o DIST_FORMING=
20%), XLF RANGE=85% (2o MARKDOWN=9%), XLP RANGE=90%, XLU RANGE=95%,
XLRE RANGE=95%. RANGE gana por 24-90 puntos. No es ruido.

**Fix (`a89eb1f5`).** 3 columnas aditivas (schema v2->v3).
`wyckoff_phase` original intacta (historico no se rompe).
- `wyckoff_signal_phase`: moda de las 6 fases con senal (ACC,
  ACC_FORMING, MARKUP, DIST, DIST_FORMING, MARKDOWN). Fallback
  'RANGE' si no hay senal. Desempate determinista: MARKDOWN >
  DIST > DIST_FORMING > ACC_FORMING > ACC > MARKUP.
- `wyckoff_signal_share`: % en la fase ganadora (escala 0-100).
- `wyckoff_signal_n`: count absoluto.
Render reformateado: 'DIST_FORMING (20%, N=5)' o 'RANGE (sin senal)'.
Nota reescrita: declara la semantica y remite a wyckoff_phase del CSV
para auditoria historica.

**Resultado 2026-10-30.** XLK->DIST_FORMING(20%), XLF->MARKDOWN(9%),
XLV->MARKUP(12%), XLE->DIST(10%), XLY->DIST_FORMING(10%),
XLP->DIST(5%), XLI->MARKDOWN(11%), XLB->DIST(10%),
XLU->MARKDOWN(5%), XLRE->DIST(5%), XLC->MARKDOWN(11%).

**Verificacion.** pytest test_sector_regime_matrix.py: 9 passed (4
nuevos). Bloque sector_context/regime/phase: 41 passed. Probe con CSV
real 30-10: los 11 sectores con share correcto (sin doble *100).
Fallback RANGE verificado con XLK forzado a 100%. pyflakes +
compileall silencio. verifica_gobernanza [OK]. REGISTRO 274 -> 276
items (WYCKOFF-PHASE-SEMANTICA FIXED + VERIFICAR-WYCKOFF-PHASE-PROD
OPEN).

**Commits.** `a89eb1f5`, `78431c49`.

**Pendiente.** `VERIFICAR-WYCKOFF-PHASE-PROD` OPEN: verificar en el
proximo run scheduled real que la Matriz publica fases con senal.
Resto sin cambio (VERIFICAR-WALK-BACK-PROD, VERIFICAR-IAE-CONSENSO-PROD,
06-nov, 10-nov x2, 17-nov, 20-nov, INVESTIGAR-TESTS,
2 BLOCKED_EXTERNAL, 3 deudas auditoria).

**Proximo paso sugerido.** El proximo run scheduled real (post-23:17
UTC del 03-11) verifica 3 items en el mismo artifact:
VERIFICAR-WALK-BACK-PROD, VERIFICAR-IAE-CONSENSO-PROD y
VERIFICAR-WYCKOFF-PHASE-PROD. Los 3 se cierran con la misma descarga.

### 2026-11-03 — Sesion 27 parte 3: auditoria externa 11-03 (IAE consenso)

**Objetivo.** Responder al informe de auditoria externa 2026-11-03.
El auditor trae 4 hallazgos (9.3, 10.1, 10.2, 10.3). Prioridad:
9.3 es bug real, los otros 3 son deuda de producto/diseno.

**Bug real 9.3 (FIXED).** El reporte publicaba 11 tickers
clasificados contra sus propios recuentos en 'Cambio neto por
ticker (13F)':
- En 'Acumulando' con mas reduciendo que aumentando: AAPL (2544
  vs 2885), GOOGL (2291 vs 2529), GOOG (1970 vs 2251), HD (1673
  vs 1748), PG (1641 vs 1684).
- En 'Distribuyendo' con mas aumentando que reduciendo: CB (826
  vs 790), NOW (946 vs 769), VLO (707 vs 705), WDC (590 vs 589),
  EOG (661 vs 575), LHX (688 vs 562).
Causa raiz: `top_accumulating` filtraba `sum_delta > 0` y
`top_distributing` `sum_delta < 0` (variacion neta de ACCIONES),
pero el encabezado dice 'mas gestores aumentan/reducen' (consenso
de GESTORES, n_increase vs n_decrease). El ranking SI era por
consenso; solo el filtro no. El docstring del modulo lo declaraba
('ranking por consenso, no por suma bruta'). Codigo contradecia su
propio contrato desde `c0f2a204` (01-11). Primer run que publico la
seccion fue 41575090209 (03-11); los scheduled del 01-11 y 02-11
fueron gate=false (fin de semana).

**Fix 9.3.** `5bbc7681`: filtro por consenso (n_inc>n_dec en
acc; n_inc<n_dec en dis). `sum_delta` sigue para desempate del
orden. 3 tests centinela:
- `test_top_accumulating_filtra_por_consenso_no_por_sum_delta`.
- `test_top_distributing_filtra_por_consenso_no_por_sum_delta`.
- `test_consistencia_encabezado_y_columnas` (toda fila cumple el
  encabezado de su tabla).
Probe con los 11 casos reales del run: los 5 pasan a Distribuyendo,
los 6 pasan a Acumulando. Controles MSFT/NVDA/HON/CAT se mantienen.
Gap de tests detectado: los existentes verificaban ORDEN pero no
FILTRO. Caso `n_inc<n_dec con sum_delta>0` no estaba cubierto.

**3 deudas OPEN (decision de producto, no codigo).**
- `TEMPORAL-EFFECTIVE-DATE-POR-MODULO` (10.1): el reporte sigue
  con una sola `effective_date_equity` mientras modulos declaran
  fechas heterogeneas en Data Freshness. El fix WALK_BACK resuelve
  el warning espurio, no la declaracion por modulo. Decision:
  coherencia global conservadora vs fecha por bloque. 1-2 sesiones.
- `MTE-DEGRADED-STATE` (10.2): MTE publica `STALE` sin degradar
  confianza. Auditor propone VALID/DEGRADED/INVALID. Diseno de
  contrato. 1 sesion diseno + 1-2 implementacion. Sin urgencia:
  STALE visible una vez.
- `FRESHNESS-POR-BLOQUE` (10.3): `CURRENT` es umbral por fuente,
  no actualidad. Auditor pide declarar por bloque cuando una
  conclusion depende de datos de sesion anterior. Observabilidad,
  no bug. El sistema ya tiene la informacion; falta presentacion.

**Verificacion.** pytest test_iae_per_ticker.py 12 passed (3
nuevos). Bloque IAE 142 passed. Probe con los 11 casos reales del
run 41575090209: swap correcto. pyflakes + compileall silencio.
verifica_gobernanza [OK]. REGISTRO 269 -> 274 items (9.3 FIXED,
10.1/10.2/10.3 OPEN, VERIFICAR-IAE-CONSENSO-PROD OPEN).

**Commits.** `5bbc7681`, `f7b823dc`, `9766714f`.

**Pendiente.** `VERIFICAR-IAE-CONSENSO-PROD` OPEN: verificar en
el proximo run scheduled real que los 11 tickers salen en la tabla
correcta. `VERIFICAR-WALK-BACK-PROD` OPEN (sin cambio). Resto sin
cambio (06-nov, 10-nov x2, 17-nov, 20-nov, INVESTIGAR-TESTS,
2 BLOCKED_EXTERNAL).

**Proximo paso sugerido.** El proximo run scheduled real (post-23:17
UTC del 03-11) verifica AMBOS items: `VERIFICAR-WALK-BACK-PROD`
(si hubo walk-back) y `VERIFICAR-IAE-CONSENSO-PROD` (siempre que la
seccion IAE publique). Los 11 tickers del auditor deben estar en la
tabla correcta. Si falla, reabrir IAE-CONSENSO-VS-SUM-DELTA.

### 2026-11-03 — Sesion 27: revision de los 10 workflows

**Objetivo.** Tras cerrar sesion 26, revisar los 10 workflows del repo
para detectar drift entre runbook, YAML y produccion. Sin cambio de
producto, auditoria pura.

**Contexto previo.** Al arrancar, pull con 2 commits del bot
(`db8ed703` update_macro_manual + `532bb4c2` Daily hist/state). El
run scheduled real 41575090209 (03-11 03:55 UTC, 21 min, gate=true,
HEAD `db8ed703`) confirma los fixes sesion 26 en produccion: 6/6
wyckoff CSVs en artifact, 2 notas *Filtro:* en reporte_diario.md
(5/12 y 25/55), 0 fallbacks.

**Hecho (2 commits de codigo + gobernanza).**

- **H2 + H4** (`e351d486`). daily_run.yml: glob
  `sec_nport_positions_*.csv` (el path sin sufijo nunca existio;
  productor escribe `_2026q2`/`_2026q3`). `continue-on-error: true`
  en `qqq_sec_nport_cross_validation.py` (control descriptivo, no
  debe bloquear publicacion si un CSV auxiliar tiene schema raro).
- **H1** (`d84ca390`). 07_RUNBOOK.md seccion 2 reestructurada:
  - update_sec_nport movido de 2.3 (trimestral dia 20) a 2.2
    (semanal miercoles). Llevaba 18 dias desfasado.
  - update_macro_manual anadido a 2.1 (diario).
  - update_qqq_sec_flow anadido en 2.3.1 (semestral 15 ene/jul).
  - Matiz 1-oct reescrito con correccion fechada.
  - Nuevo `tests/test_runbook_vs_workflows.py` (2 tests): compara
    crons documentados con YAML real y exige todos los update_*
    documentados. Cualquier drift futuro rompe el test.
- **Gobernanza** (`041fecc7`). VERIFICAR-SCHEDULED-387938E6 -> FIXED
  con evidencia del run 41575090209. Nuevo REVISION-WORKFLOWS-
  2026-11-03 -> FIXED con H1/H2/H4 + H3 FALSE_POSITIVE + H5 no
  accionable. REGISTRO: 266 -> 267 items.

**Falsos positivos descartados (no ocultos).**
- H3 (6 API keys parecian huerfanas): falso positivo. Grep inicial
  no cubria `data/providers/`, cada key tiene >=1 uso.
- H5 (formato L266-287 inconsistente): YAML valido, cosmetico.
  No accionable (mantener consistencia implicaria refactor masivo
  de formato sin valor).

**Commits.** `e351d486`, `d84ca390`, `041fecc7`.

**Pendiente.**
- INVESTIGAR-TESTS-MAIN-SIN-MONKEYPATCH-PATH: OPEN (8 tests, ROI
  bajo, sesion dedicada).
- DAILY-RUN-FAIL-40992758129: revisar 06-nov.
- Observacionales: 10-nov x2, 17-nov, 20-nov.
- 2 BLOCKED_EXTERNAL (C2-920, WYCKOFF-5BX).

**Proximo paso sugerido.** Continuar con 00_ARRANQUE y HANDOFF para
cerrar S27 documentalmente. Despues, sesion tranquila sin urgencias
hasta 06-nov (DAILY-RUN-FAIL) o 10-nov (observacionales).

### 2026-11-03 — Sesion 27 parte 2: TEMPORAL-COHERENCE-WALK-BACK

**Objetivo.** Tras el cierre del bloque 1 (revision de workflows),
revisar el reporte de produccion con datos. El usuario detecto que
verificar "tests verdes" no equivale a verificar el contenido que ve
el usuario. Al abrir el `reporte_diario.md` del run real
41575090209 aparece `temporal_coherence: INCOHERENT` y
`validation_gate: 11/11 (INCOHERENT)`.

**Verificacion previa (Wyckoff con datos).** Primera auditoria
cosmetica (2 notas *Filtro:*). Segunda con datos reales: 110 filas
cruzadas por (group, ticker), 6/6 sub-tablas, 0 discrepancias entre
.md y CSV filtrado, nota *Filtro:* exactamente en las 2 sub-tablas
donde n_filt < n_total. Los 21 errores del primer intento eran del
script del auditor (duplicados AMD/GOOGL entre QQQ/DIA/SPY).

**Bug real detectado.** Dos criterios incompatibles sobre el mismo
hecho:
- `pipeline_gate`/`guard_coverage`/`write_completion_receipt`
  aceptan walk-back si `coverage_pct_effective >= 0.90` (contrato
  vigente desde 2026-10-31, commit a692fafb).
- `provenance._temporal_coherence` comparaba lag vs
  `contract_max_lag_days=0` sin conocer el walk-back.
Resultado: el reporte publico `INCOHERENT` en un run que el gate
acepto. El propio codigo (H8, 2026-10-16) lo marcaba como
contradiccion sin resolverla.

**Hecho (4 commits + gobernanza).**

1. `3c488a22` feat(provenance): estado WALK_BACK. Nuevo param
   `effective_is_walk_back=False` en `_temporal_coherence` y
   `build_provenance`. `infer_walk_back(effective_meta)` como fuente
   de verdad: True si status==OK y lag_days>0. 5 tests.
2. `e25e051d` feat(pipeline): run.py cablea. Calcula
   `_prov_walk_back = infer_walk_back(df_stocks_effective_meta)`
   y lo pasa a `build_provenance`. 8 tests (7 infer_walk_back +
   1 integracion run.py).
3. `1abddd73` feat(report): nota explicativa en
   `render_provenance` cuando temporal_coherence==WALK_BACK. 4 tests.
4. `7f1b6e4f` docs(contrato): `temporal_coherence_walk_back_v1.md`
   (contrato semantico, 7 KB) + `daily_run_gate_enmienda_v1_3.md`
   (enmienda formal, 4.4 KB) + item REGISTRO
   `TEMPORAL-COHERENCE-WALK-BACK` FIXED commit 1abddd73.

**Decision arquitectonica (no se relaja tolerancia).**
- `contract_max_lag_days` sigue siendo 0. El fix no es un numero
  mas alto; es un estado nuevo con nombre propio.
- Gate y reporte comparten `coverage_pct_effective` como fuente unica
  (enmienda v1.3).
- `ANOMALY` gana siempre sobre `WALK_BACK` (integridad).
- `INCOHERENT` sigue siendo detectable: sin walk-back explicito y
  lag>0, se preserva. El test `test_coherencia_sin_flag_sigue_incoherent`
  lo cubre.

**Verificacion.**
- pytest tests/test_report_provenance.py: 49 passed (17 nuevos reales
  en 4 commits + 32 existentes).
- Probe con manifest real 03-11:
  `infer_walk_back({status:OK, lag_days:3})` -> True;
  `build_provenance(walk_back=True)` -> WALK_BACK, gate sin marca;
  `build_provenance(walk_back=False)` -> INCOHERENT (deteccion
  preservada).
- Bloque report/provenance: 92 passed.
- ESTADO_SISTEMA regenerado: 4377 -> 4396 passed (+19; 17 explicados,
  +2 sin explicacion cerrada, anotado como observacion abierta).

**Falso positivo reconocido.** El commit message de `7f1b6e4f` dice
"18 tests nuevos". Realmente son 17 (5 coherencia + 7 infer_walk_back
+ 4 nota render + 1 integracion run.py). Error de contabilidad en el
mensaje, ya pusheado. No se enmienda (ruido).

**Pendiente real.** `VERIFICAR-WALK-BACK-PROD` OPEN: ningun run
scheduled post-fix ha publicado aun. El proximo run scheduled con
walk-back (Close parcial) es la verificacion real. Si publica
`WALK_BACK` y gate 11/11 sin marca, se cierra VERIFIED_IN_PRODUCTION.
Si publica `INCOHERENT`, se reabre TEMPORAL-COHERENCE-WALK-BACK.

**Commits.** `3c488a22`, `e25e051d`, `1abddd73`, `7f1b6e4f`,
`45731e94`.

**Pendiente.** `VERIFICAR-WALK-BACK-PROD` OPEN (proximo run scheduled
con walk-back). Resto sin cambio (06-nov, 10-nov x2, 17-nov, 20-nov,
INVESTIGAR-TESTS, 2 BLOCKED_EXTERNAL).

**Proximo paso sugerido.** Cierre documental (HANDOFF + 00_ARRANQUE)
y esperar run scheduled. Al arrancar la proxima sesion, si el ultimo
run scheduled tiene walk-back, verificar `temporal_coherence` en el
artifact antes de cualquier otro trabajo.

### 2026-11-02 — Sesion 26: GUARD-COVERAGE fix + WYCKOFF filtro weight

**Objetivo.** Dos frentes independientes: (1) bug en test que escribia
en outputs/state de produccion; (2) peticion de producto: filtrar el
.md de Wyckoff por weight + entregar los 6 CSVs como artifact.

**Hecho.**
- GUARD-COVERAGE-TEST-CONTAMINA-STATE FIXED (`1f935b1d`, `cd6b5bee`).
  test_guard_coverage_validation.py llamaba gc.main() sin monkeypatch
  de RESULT_PATH. `ROOT = Path(__file__).resolve().parents[1]` es
  absoluto en import-time; `chdir(tmp_path)` no lo reubica. El test
  sobrescribia outputs/state/guard_coverage_result.json con run_id=0,
  target_session=2026-10-15 stale. Centinela
  test_guard_no_escribe_state_real. Leccion en 01_METODO seccion 3
  (`713f8d3d`).
- WYCKOFF-MD-FILTRO-WEIGHT FIXED (`8b57bd73`, `0ac42079`).
  MIN_WEIGHT_PCT=0.5 en src/report/wyckoff_holdings.py + nota runtime
  n_filt/n_total + fallback visible. 6 wyckoff CSVs al artifact
  daily-report en daily_run.yml (retention-days: 7). 5 tests nuevos.
  Impacto 2026-10-30: usa_index_acc 12->5, usa_index_dist 55->25.
  Sector USA y EU sin cambio (weights >= 0.5% ya).
- HANDOFF corregido (`442bbb3d`): 4358 -> suite filtrada; DICT3-TOP20-
  COBERTURA FIXED `0ab436d9`.
- VERIFICAR-SCHEDULED-387938E6 corregido (`0e765d10`): los 3 slots
  pre-cierre del 02-11 son gate=false (TARGET_SESSION=2026-10-30,
  1 min). Sigue OPEN, verificable 03-11.

**Commits.** `1f935b1d`, `cd6b5bee`, `442bbb3d`, `713f8d3d`, `d372dd9e`,
`8b57bd73`, `0ac42079`, `30b5e9de`, `0e765d10`.

**Pendiente.**
- VERIFICAR-SCHEDULED-387938E6: 03-11 verificar primer run real
  (gate=true, duracion >= 10 min, headSha >= 387938e6).
- INVESTIGAR-TESTS-MAIN-SIN-MONKEYPATCH-PATH OPEN (8 sospechosos).
- Resto OPEN sin cambio (06/10/17/20-nov).

**Proximo paso sugerido.** 2026-11-03: `gh run list --workflow=daily_run.yml
--limit 5` + `gh run view <id>`. Si run real success con headSha >=
387938e6, verificar artifact daily-report (6 wyckoff CSVs +
reporte_diario.md con 2 notas *Filtro:*). Si verde, cerrar
VERIFICAR-SCHEDULED-387938E6.

### 2026-11-01 — Sesion 25: FASE-A (analisis economico 30-oct)

**Objetivo.** Ejecutar los 5 puntos tecnicos del analista del reporte
30-oct. Prioritaria (a) del analista (bloque macro oficial) no
abierta: decision de producto.

**Hecho.**
- FASE-A-COBERTURA-DUAL (50ab0e1f): cobertura dual derivada de datos
  (n_descargados, coverage_rs_sobre_descarga, peso_total_etf,
  coverage_peso). Sin constantes de porcentaje. Render: 2 columnas +
  nota derivada. Test de header pasa a validar por NOMBRE de columna,
  no por count("|").
- FASE-A-FLUJOS-CAPAS-DERIVADAS (180f0ad6): FLOW_SYNTHESIS_LABELS dict
  como fuente unica. Regla y nota de no-sumabilidad derivadas de
  flow_synthesis.keys(). Test centinela falla si productor anade
  capa sin label.
- FASE-A-INDEPENDENCIA-ESTADISTICA (180f0ad6): config/shared_inputs.py
  con SHARED_INPUTS (6 variables verificadas por Gate 0). Helper
  _shared_vars_note en helpers. 13 tests fuerzan coherencia registro-
  codigo.
- FASE-A-IAE-PER-TICKER (c0f2a204): agregacion 13F por ticker. Gate 0:
  243 tickers del radar con datos, densidad alta (min 353 gestores,
  mediana 1919). Filtros de anomalia por ratio (pct_new, pct_exit,
  pct_increase) capturan IPO SPCX (12-jun-2026) y spin-offs HON/SOLS
  (oct-2025), DD/Qnity (nov-2025) sin lista de tickers. Ranking por
  consenso de gestores. Render en src/report/iae.py: Top 20
  acumulando/distribuyendo + nota de exclusiones.
- FASE-A-AMPLIAR-UNIVERSO-PESO (bcd31652, opcion 90%): 25 tickers en
  universe_expansions.yaml (8 bloques *_90). WBD declarado en XLC y
  filtrado. Universo: 505 -> 530. Test centinela: 4 tests (AST filter,
  foto universo, WBD en YAML no en universo, cobertura >=90% por ETF).
  Tests genericos de test_universe_expansions.py adaptados para excluir
  DELISTED declarados.
- Otros trabajos del dia: B6 completa (4 FIXED + 2 WONT_FIX), B3
  (1 FIXED), B2/B7/B1/B8 WONT_FIX. Fase 0/1 (2 fixes). 01_METODO
  dividido. README sin cifras caducas. Transversales cerrados.

**Commits (41).** Desde b69c17fe hasta bdfb74bb. Detalle en git log.

**Pendiente.**
- 4 observacionales con fecha (CRON-SLOTS-OBS y SCHEDULER-DELAYS
  10-nov; RECEIPT-AUDIT-LAG 17-nov; H5.3 20-nov).
- 2 BLOCKED_EXTERNAL: C2-920, WYCKOFF-5BX (~2027).
- 2 OPEN DICT3: DICT3-TOP20-COBERTURA, DICT3-H04-INTERVALS.
- FASE-A-REGIMEN-ECONOMIA-REAL: no abierta. Decision de producto.

**Proximo paso sugerido.** Observacionales auto-resuelven. Prioritaria
(a) del analista (bloque macro oficial separado) pendiente de decision
de producto.

### 2026-11-01 — Sesion 24: Fase 2 completa (B1-B8) + Fase 0/1 + transversales

**Objetivo.** Completar Fase 2 (auditoria por bloques B1-B8) + Fase 0
(inventario hardcodes) + Fase 1 (clasificacion) + cerrar transversales.
Ejecutar la hoja de ruta del plan de consistencia temporal.

**Hecho.**
- B4-propia-1 + B4-propia-2 (WONT_FIX, 0 bugs): engines,
  sector_regime_matrix, sector_regime, macro_regime, evidence_matrix,
  liquidity_v2, tactical_engine, volatility_regime, structural_engine,
  financial_conditions. 5 BAJA razonadas (except con WARN).
- B6 completa: RATIO-NAMES (3d21264a), MTE-CLS-INF (ba1f2eac),
  ILOC-BARRIDO (b661a0d6), MTE-INF-NO-FINITE (ac0a607e) FIXED.
  FLOWS-CURRENCY y FLOWS-REGISTRY WONT_FIX razonados (B5).
- B3: WYCKOFF-RENDERS-HUERFANOS FIXED (3a516268). 2 renders sin
  productor eliminados + 2 tests + 1 docstring adaptado.
- B2: 28 modulos auditados, WONT_FIX (0 bugs).
- B7: 36 modulos IAE auditados, WONT_FIX (0 bugs).
- B1: 13 modulos temporales, WONT_FIX (0 bugs).
- B8: 19 scripts audit + CI, WONT_FIX (0 bugs).
- Fase 0: 302 hits en 15 tipos. Fase 1: 2 fixes reales
  (336b48ab: ssga SECTOR_TICKERS derivado de config; 00_ARRANQUE
  8->11 pares). Resto NO_BUG razonado (FASE1-CENSO FIXED).
- Docs: 01_METODO dividido (5675d720, secciones 6/6bis/6ter/7 ->
  01d_POWERSHELL_AUDITORIA.md, METODO <20KB). README bloque Fuentes
  sin OilPriceAPI ni cifras caducas (6cf37a56).
- Transversales: FASE2-BARRIDO-CONSUMIDORES-ROTOS FIXED (39df77b0,
  test_ci_entrypoints_import 27 entry points); FASE0-BARRIDO-SCRIPTS
  WONT_FIX razonado; DT-NOW-HELPER-HUERFANO FALSE_POSITIVE;
  IWM-HOLDINGS-EXPANSION WONT_FIX razonado.

**Commits (32).** Rango: b69c17fe..9c4fe3dd (+ cierre 24 en este
commit). Cada item FIXED con su par (codigo+gobernanza); WONT_FIX y
FALSE_POSITIVE con 1 commit.

**Pendiente.**
- 4 observacionales con fecha (10-nov x2, 17-nov, 20-nov).
- 2 BLOCKED_EXTERNAL: C2-920, WYCKOFF-5BX.
- 2 OPEN DICT3: TOP20-COBERTURA, H04-INTERVALS.

**Proximo paso sugerido.** Observacionales auto-resuelven en fecha.
DICT3-TOP20-COBERTURA espera respuesta del auditor.

### 2026-10-31 — Sesion 22: diagnostico de estado + cierre PCR-CONTRACT

**Objetivo.** Verificar el estado real del sistema tras el commit
387938e6 (docs arranque-21). Alinear el corpus documental con la
fuente autoritativa (ESTADO_SISTEMA.md). Cerrar el frente
PCR-CONTRACT (OPEN desde sesion 20).

**Hecho.**
- Diagnostico de estado: 4 documentos leidos (00, 01, 01c, HANDOFF),
  cuestionario de 23 preguntas al saliente. Hallazgos:
  - ARRANQUE §5 con literales 3816/3829 obsoletos.
  - METODO §4 con 'Validation Gate 10/10' obsoleto (real 11/11
    dinamico desde 090a3439).
  - METODO §11 con limite 11 KB ya superado (ARRANQUE real 19.5 KB).
  - README con bloque Resumen desfasado (2267 tests, 10/10, 313
    tickers).
  - 02_ARQUITECTURA con bloque de 10 checks; real 11 (faltaba
    Schema versioning Fase 1).
- Alineacion documental (3 commits docs + 1 lecciones):
  - 18ca4405: Validation Gate N/N dinamico en 4 docs (ARRANQUE,
    METODO, 02_ARQUITECTURA, README).
  - af019902: limite ARRANQUE 11->20 KB + trampa cancelled != failure
    en METODO §6.
  - bfba1763: leccion 'holdings se normaliza en origen' en 01c.
- Gobernanza: 6 deudas registradas (ca9cd95e), REGISTRO 189 -> 195:
  HOLDINGS-NORMALIZACION-EN-ORIGEN, HANDOFF-SIN-RESPUESTA-ENTRANTE-21,
  DAILY-RUN-FAIL-40992758129, VERIFICAR-SCHEDULED-387938e6-2026-11-02,
  METODO-EXCEDE-LIMITE-PROPIO, README-DESFASE-FUENTES-DATOS.
- PCR-CONTRACT cerrado:
  - 623b5dd3: eliminar 'percentile' duplicado en return de
    compute_pcr_signals (dead code, F601 no lo detecta).
  - 78a0e106: nuevo test_pcr_contract.py (3 tests) + par PCR al
    manifiesto DICT_CONTRACT_PAIRS (9 pares). Par eliminado de
    whitelist (2 -> 1). La whitelist declaraba producer
    src/pipeline/market_data.py::_compute_pcr (passthrough), el
    par real es indicators/options.py::compute_pcr_signals.
- CI: 6 pushes. 3 X en gh run list eran cancelled por concurrency
  (cancel-in-progress), no failure. Confirmado por ancestro contra
  78a0e106 (success). Trampa #5 (af019902) validada en vivo el
  mismo dia que se documento.

**Commits.** 18ca4405, af019902, bfba1763, ca9cd95e, 623b5dd3,
78a0e106 (6 commits).

**Pendiente.**
- SECTOR-RESULTS-CONTRACT: whitelist 1 -> 0. Producer declarado
  compute_sectors_base no coincide con el descrito en HANDOFF
  (compute_sector_rank_score). Merece analisis propio, sesion 23.
- HOLDINGS-NORMALIZACION-EN-ORIGEN: E2E obligatorio por METODO §4.
- Verificar scheduled con HEAD actual: primer run real lunes
  2026-11-02.
- HANDOFF a HEAD 5a976288.

**Cierre real.** 0e3b0dd3 (HOLDINGS-NORMALIZACION-EN-ORIGEN
FIXED tras rollback + B'). CI verde 41157786156. La sesion cerro
2 frentes grandes (DICT-CONTRACT-FRAMEWORK al 100%, HOLDINGS
causa raiz) y dejo 6 deudas de handoff OPEN.

**Reintento HOLDINGS (contexto).** El primer intento (B explicito)
rompio 16 tests con KeyError ticker_norm porque los tests sinteticos
no pasaban por load_stocks_data. Rollback. Segundo intento (B')
con ensure_ticker_norm + get_ticker_norm como red de seguridad:
tests existentes intactos, causa raiz cerrada via assert +
centinela AST que prohibe normalize_yahoo_ticker en los 5
consumidores. E2E verificado contra baseline 20261031_003442.

**Proximo paso sugerido.** Sesion 23: verificar daily_run real del
lunes 2026-11-02 (headSha 771f5ecc) y cerrar condiciones 4-5 de
DICT3-TOP20-COBERTURA (SHA en REGISTRO, mantener frentes
paralelos).

**Extension de sesion (misma fecha, 2026-10-31):**

1. Scheduled 41159854920 fallo en guard_coverage (coverage_pct_last=
   0.760396 a las 02:24 UTC). run.py paso (18 min, Gate 11/11).
   Diagnostico: coverage_pct_last media la ULTIMA fila fisica del
   parquet, que a esa hora era del 30-oct con Close parcial. Fix
   a692fafb: walk-back via resolve_effective_date (min 0.90);
   manifest publica effective_session + coverage_pct_effective;
   guard_coverage/receipt/pipeline_gate prefieren effective con
   fallback. coverage_pct_last intacto (tests ancla no rotos).
   CI verde 41162803916. Item: GUARD-COVERAGE-LAST-DATE-PARCIAL.

2. Invariante 2 de P1-H03 ejecutado (dictamen externo 2026-10-31).
   Script scripts/audit/p1_h03_invariante2.py. 272 tickers x 1313
   fechas, 290.694 filas. Resultado: 0 violaciones, 0 solo-A, 1
   solo-B (BRK.B, 1086 filas RANGE via BRK-B). 62.5 min, 12
   workers. Matiz documentado en doc 72 seccion 8: los 269 comunes
   no son duales, no hay comparacion pre/post literal por falta de
   snapshot pre-fix por ticker. Mas fuerte que SKIP original, mas
   debil que contrato literal.

3. Evidencia f71eb49c verificada contra padre: XLF n_total 22->23
   en 1086 fechas, count_range +1 en 1086, resto 0. Doc 70-v2d
   seccion 6.

4. Numeracion seccion 5 de doc 70-v2d corregida (5.2 duplicado ->
   5.3, 5.3 -> 5.4). Commit 771f5ecc.

5. Cuestionario al saliente resuelto: C.9 (25/25 receipts = 25
   writers), C.10 (autonomia funciona, #7 WARN no bloqueante),
   D.14 (01b vigente, 13 patrones), D.15 (poda segura). C.12
   PSY/PS derivado a item nuevo PSY-PS-PROBE-HISTORICO. B.6
   derivado a DICT3-TOP20-COBERTURA (canal activo).

**Commits de la extension:** a692fafb (guard fix), 771f5ecc
(invariante 2 + doc 70-v2d + doc 72), + este docs(cierre-22d).

**Frentes cerrados hoy (recuento):** DICT-CONTRACT-FRAMEWORK
100%, HOLDINGS-NORMALIZACION-EN-ORIGEN, GUARD-COVERAGE-LAST-DATE-
PARCIAL, DAILY-RUN-FAIL-41159854920, HANDOFF-SIN-RESPUESTA-
ENTRANTE-21. DICT3-TOP20-COBERTURA queda OPEN con 3/5 condiciones
hechas. P1-H03 invariante 2 atendido con matiz.

**Cierre tardio (misma fecha, tras probar WYCKOFF-EVENTS-ARTIFACT):**

Se intento B completo para el frente WYCKOFF-EVENTS-ARTIFACT
(doc 80): wirear PSY/PS + exponer 8 detectores de evidencia pasiva
al CSV (16 columnas nuevas, schema v1->v2). Pre-registro firmado en
167a4587. Wireado aplicado en local, benchmark 62->70ms/ticker.
**Detectado durante la verificacion: 3 de 4 tickers (AAPL, JPM,
NVDA) no completan la cascada BC->AR->Range->SOW; los 8 detectores
no se ejecutan para ellos. XOM si. Esto invalida el diseno del
pre-registro.**

Ademas los detectores solo evaluan la ultima barra (doc 62
explicito). En produccion PSY ocurre ~1 vez cada 50 sesiones por
ticker; NOT_OBSERVED seria el valor esperado el 98% de los dias.

Rollback del wireado (`git checkout -- indicators/wyckoff_*.py`,
sin diff en main). Reclasificado WYCKOFF-EVENTS-ARTIFACT a
WONT_FIX. Doc 80 seccion 10 documenta el hallazgo completo.
PSY-PS-PROBE-HISTORICO pasa a FIXED con el probe:
- ETFs sectoriales (11): bear PSY 0/495 PS 2/495; bull PSY 10/2035
  PS 0/2035.
- Holdings (20 tickers): bear PSY 0/374 PS 2/374; bull PSY 35/3843
  PS 8/3960.
Conclusion: los detectores funcionan. El problema es de diseno (solo
ultima barra + cascada condicional), no de calibracion.

HOLDINGS-SNAPSHOTS-XLF-VACIO (observacion de sesion 22):
WONT_FIX razonado. No es bug, es diseno PIT append-only: solo se
escribe (source_file, etf) cuando cambia el hash. 2026-10-26 solo
cambiaron 4 ETFs europeos; los 15 USA (XLF incluido) no cambiaron
respecto al 23-oct.

ARRANQUE seccion 4.2 rectificado: "PSY/PS no disparan" era enganoso.
Ahora describe el estado real (dead code funcional, sin wirear,
diseno de ultima barra impide aportacion visible).

**Frentes cerrados hoy (recuento final):** DICT-CONTRACT-FRAMEWORK
100%, HOLDINGS-NORMALIZACION-EN-ORIGEN, GUARD-COVERAGE-LAST-DATE-
PARCIAL, DAILY-RUN-FAIL-41159854920, HANDOFF-SIN-RESPUESTA-
ENTRANTE-21, PSY-PS-PROBE-HISTORICO (FIXED), WYCKOFF-EVENTS-
ARTIFACT (WONT_FIX razonado), HOLDINGS-SNAPSHOTS-XLF-VACIO
(WONT_FIX razonado). DICT3-TOP20-COBERTURA queda OPEN con 3/5
condiciones hechas. P1-H03 invariante 2 atendido con matiz.

**SOW-BEAR-CONFIRMATION cerrado WONT_FIX (sesion 22, tarde):**

Contrato v1.10.1 reescrito (corrige 3 errores de v1.10: usa
detect_sow_in_range, agrega por holdings con N=3, fija SPY). Modulo
indicators/sow_bear.py + 12 tests T1-T10 implementados. Probe
empirico antes del wiring:

- 9 fechas bear historicas (2022-05-17 a 2022-12-28, muestreo
  cada 20 sesiones): 0/9 con >=1 sector disparando.
- Hoy (2026-10-30, SPY no bear): 0/11 sectores con >=3 SOW
  CONFIRMED en M=5.

Causa: la cascada BC->AR->Range->SOW solo completa en minoria de
holdings (XLF 6/34, XLRE 2/20, XLU 3/21). SOW es evento de ruptura
puntual (~1/ano/holding). Pedir 3 coincidentes en M=5 sesiones es
estadisticamente casi imposible. La metrica esta mal elegida para
el sustrato.

Rollback: revert f043b87e (modulo + tests + doc 57 v1.10.1
eliminados). Contrato v1.10 original conservado como referencia del
error de diseno. Idea valida, metrica incorrecta. Si en el futuro
se redefine para medir estado (fases DISTRIBUTION/MARKDOWN) en vez
de evento puntual, pre-registro nuevo.

**Fase 0 hardcodes cerrada (misma fecha, sesion 22):**

Barrido mecanico completo del codigo productivo (src, indicators,
regimes) en busca de hardcodes que pierdan consistencia en el
tiempo. 662 hits brutos, 221 en productivo, 15 bugs reales fix
en 6 commits (fc07b634, f3a7270e, a7a0c7b8, c5d6fa25, 3aa51d51,
898cee63). Detalle por item en REGISTRO (FASE0-*).

Bugs cerrados por grupo:

1. lista_sectores (2): MTE sector lists divergentes con
   config.tickers (XLC); etf_flows lista de 11 sectores
   hardcodeada. Impacto medido: srs 0.4057 -> 0.4202, MSI
   53.28 -> 53.68, escenario RECESSION sin cambio. Golden
   actualizado. +2 tests centinela.

2. io (2): open('r') sin encoding en 2 sitios sobre
   liquidity_state.json (UTF-8 leido con cp1252 en Windows,
   UnicodeDecodeError silenciado).

3. fecha_iso_string (1): temporal_filter FULL_PERIOD
   hardcodeado a '2026-03-31' (2 trimestres desfasado). Fix:
   None + raise explicito. Ancla IAE_FULL_PERIOD eliminada.

4. except_exception (8): 6 except Exception -> except
   (ValueError, TypeError) en bloques pd.Timestamp/to_datetime
   (patron 1 de 01b_PATRONES). 2 dead code eliminados.

5. ticker_hardcoded (2): etf_flows lista sectores hardcodeada;
   EU_INDEX_ETFS duplicada en 3 sitios. Fuente unica en
   config.tickers. +1 test centinela AST.

6. threshold_sin_const (0 fixes): 53 candidatos revisados uno
   a uno. 0 duplicacion real. Los umbrales 0.05/0.1/0.2/0.3
   que se repiten tienen significados distintos. WONT_FIX
   razonado.

7. datetime_now (0 fixes): 8 llamadas reales en productivo, 7
   justificadas (ejecucion/metadato). 1 helper huerfano
   (_ref_to_date en data_quality) anotado como OPEN.

Deuda abierta: OBSERVABILIDAD-SILENCIOSA (7 fallbacks sin
WARN), DT-NOW-HELPER-HUERFANO, FASE0-BARRIDO-SCRIPTS (441
hits en scripts/ calibracion, prioridad baja).


**Fase 2 - Auditoria por bloques tematicos (misma fecha, sesion 22):**

Plan original: 8 bloques (B1-B8) para auditar el sistema modulo a
modulo con los 13 patrones de 01b_PATRONES + 5 nuevos (dead code
funcional, cascada abortada, literal fragil, fuente de verdad
duplicada, fallback silencioso sin WARN). Cerrados hoy B5 y B1.

B5 (outputs/persistencia) - 4 sub-bloques, 3 bugs + 2 deudas:

- B5.1 (afada0ea) output_registry + output_writer. Bug ALTA: todo
  el bloque de migracion schema v1->v2 (backup .vN.csv,
  _migrated_this_call, no sobrescribir, no borrar .schema hasta
  exito) no tenia tests. Grep de schema_version|_migrated|.v1.csv
  en tests/: cero. Misma clase 1fb791ab (test verde con mock,
  integracion no ejercitada). T5 destapo bug real: si
  shutil.copy2 fallaba, el codigo silenciaba y continuaba como
  append normal -> CSV mezclaba v1 con v2. Fix fail-closed (raise
  OSError con contexto) + WARN si falla escribir .schema. T1-T6
  nuevos.

- B5.2 (6146f5c4) src/utils.py (manifest, coverage, helpers
  temporales). 5 funciones auditadas, 0 bugs funcionales, 4
  fallbacks silenciosos con WARN anadido: _try_cleanup (tmp
  huerfano), _compute_by_market (get_market UNKNOWN),
  write_artifact_with_manifest (resolve_effective_date cae a
  fallback), _nan_usa_on_non_market_days (get_market silenciado).
  Fixes de ayer (coverage_pct_effective, effective_session)
  siguen coherentes. append_dedup, _last_row_coverage_ok,
  truncate_to_expected_session, get_effective_meta,
  writer_observation_date: auditados sin bug.

- B5.3 (d1097ab0) validation_gate.py. Bug ALTA: check 11 (Schema
  versioning) envuelto en except Exception que reportaba
  _add_check(..., True, 'no evaluable'). Un import roto de
  REGISTRY, un REGISTRY vacio por bug o cualquier excepcion en el
  glob dejaba pasar el gate sin auditar schemas. Misma clase que
  GUARD-COVERAGE-LAST-DATE-PARCIAL. Fix: except especifico +
  _add_check(..., False, ...). Test centinela: forzar REGISTRY
  inaccesible via sys.modules. NO tocados: wyckoff_holdings_data=
  None -> OK 'no disponible' en checks 1/6/10 es contrato
  documentado (holiday mode).

- B5.4 (d1097ab0) finalize.py. Bug MEDIA: save_regime_history
  hacia macro_score.iloc[-1] sin guard. run.py:344 llama
  directamente sin try/except. Si macro_score es None o Series
  vacia (all_signals vacio por fallo masivo), IndexError tumba el
  run sin contexto. Bug latente. Fix: guard -> skip NO_DATA.
  Test. Deuda MACRO-SCORE-EMPTY-LATENTE OPEN: el writer ya no
  crashea, el render del reporte no verificado para ese caso.
  Diferible a B6.

B1 (temporal) - 2 bugs + 2 deudas:

- B1-FX-PER-PAIR (88964dbf) FX_DAILY_CUT declaraba
  per_pair_max_lag {EURUSD:0, USDJPY:0, USDCNY:1} pero resolve()
  colapsaba a max() via compute_status. Bug real con datos
  2026-10-30: EURUSD lag 1 (max 0) e USDJPY lag 1 (max 0)
  violaban contrato pero el agregado decia STALE. Fix: resolver
  per-par, agregado = peor (BLOCKED > INSUFFICIENT > STALE > OK),
  per_member con detalle. Verificado: EURUSD/USDJPY INSUFFICIENT,
  USDCNY STALE, agregado INSUFFICIENT. 4 tests nuevos. E2E: py
  run.py exit=0 (14.6 min), FX_DAILY_CUT=INSUFFICIENT (antes
  STALE).

- B1-RUN-IMPORT-REGRESION (88964dbf) regresion introducida por
  3aa51d51 (fix ticker_hardcoded): al mover USA_INDEX_ETFS/
  EU_INDEX_ETFS de wyckoff_holdings_report a config.tickers,
  run.py:21 seguia importando del modulo antiguo. CI verde porque
  NINGUN test importa run.py. E2E (primer intento del fix FX)
  lo destapo: ImportError al arrancar el pipeline. Misma clase
  1fb791ab aplicada a entry points. Fix: import de config.tickers.
  Test centinela test_run_module_import.py (import run + AST check
  de origen de constantes). Cierra la clase para run.py.

- B1-PER-TICKER-LAG-DECORATIVO (WONT_FIX) 5 contratos declaran
  per_ticker_lag pero solo usan max_lag_days (colapsado a max).
  Solo INDEX_EOD_EUROPA tiene discrepancia real (^FTSE 1 vs max
  5). Mismo bug que FX pero sin impacto medido: contrato no
  critico, agregado ya STALE/OK. ROI bajo. Generalizar
  compute_status si el auditor lo pide.

- FASE2-BARRIDO-CONSUMIDORES-ROTOS (OPEN) B1-RUN-IMPORT-REGRESION
  expuso que CI no verifica entry points (run.py, scripts/). Un
  refactor de firma o movimiento de constante puede romper un
  consumidor no verificado. Barrido diferido a B6/B8.

**Regresiones propias del dia (autocritica):**

Dos regresiones introducidas por fixes propios, ambas destapadas
por E2E, no por CI:

1. HOLDINGS B explicito (0004caec, revertido): normalizar en
   origen rompio 16 tests porque los fixtures sinteticos no
   pasaban por load_stocks_data. Rollback a B' (ensure_ticker_norm
   + get_ticker_norm como red). Leccion: pre-registro subestimo
   los tests afectados (10 ficheros vs 20+ tests).

2. run.py import (3aa51d51, fix en 88964dbf): mover constantes
   sin actualizar consumidor. CI verde, E2E rojo. Leccion: CI no
   verifica entry points. Test centinela nuevo cierra la clase.

Patron comun: "fix aplicado, consumidor no verificado". Es la
clase 1fb791ab en su forma mas pura. El framework dict_contract
ya la cubre para dicts internos; falta para entry points
(FASE2-BARRIDO-CONSUMIDORES-ROTOS OPEN).

**Frentes cerrados hoy (acumulado final):**

Fase 0 hardcodes completa (15 bugs fix en 6 commits), Fase 2 B5
completo (3 bugs + 2 deudas), Fase 2 B1 completo (2 bugs + 2
deudas). Total sesion 22: 28 commits en origin/main, 20 bugs
reales fix (15 de Fase 0 + 3 de B5 + 2 de B1), 7 items WONT_FIX
razonados, 5 deudas OPEN con siguiente_accion. REGISTRO 189 ->
219 items.

Pendiente proxima sesion: B2 (pipelines descarga), B3 (Wyckoff),
B4 (motores/regimen), B6 (reporte), B7 (IAE), B8 (scripts audit).
Orden sugerido: B6 (por FASE2-BARRIDO-CONSUMIDORES-ROTOS pendiente
+ MACRO-SCORE-EMPTY-LATENTE), despues B4 (por fix XLC del MTE, no
auditados los otros motores), despues B3/B2/B7/B8.

### 2026-10-31 — Sesion 21: coherencia analisis vs descarga + BRK.B residual

**Objetivo.** Auditar el reporte generado por el run local
20261030_224230 en busca de errores, inconsistencias, bugs silenciosos.
Encontrar la causa del gap `n_valid_return20=33 < n_valid_rs=34` en XLF
(expuesto por O2).

**Hecho.**
- Diagnostico: dos bugs estructurales, ambos en la clase
  "el sistema descarga X pero analiza Y":
  1. Cap desincronizado (O2): compute_all_stock_metrics usaba
     head(TOP_N_CANDIDATES=15) mientras get_stock_list descargaba 34
     XLF por universe_expansions. Fix 1cb15d52: universo = holdings
     en df_stocks.Close, normalizar, guard 100.
  2. BRK.B residual: holdings usa BRK.B, parquet BRK-B. WYCKOFF-
     TICKER-NORMALIZATION (2026-10-27) resuelto en 2 modulos pero no
     en rs_internal.py y sector_concentration.py. Fix e04b53ff.

**Verificacion empirica.**
- Run local 20261030_230703 (pre O2): n_valid_rs XLF=14 (33 descargados).
- Run local 20261031_003442 (post O2 + BRK): n_valid_rs XLF=34,
  ret20=34, gap=0 en los 11 sectores.
- XLF cobertura 18% -> 45%. XLV 14->26. XLU 15->22. XLY 15->22.
  XLI 42->50. XLK 33->39.

**Commits.** 1cb15d52 (O2), e04b53ff (BRK) + este docs.

**Pendiente.**
- Verificacion en scheduled real (proximo run-system con HEAD
  e04b53ff o posterior).
- PCR-CONTRACT, SECTOR-RESULTS-CONTRACT (whitelist justificada,
  sesion 22+).
- 4 observacionales con fecha (H5.3 20-nov, RECEIPT-AUDIT-LAG
  17-nov, CRON-SLOTS-OBS / SCHEDULER-DELAYS 10-nov).
- Decision IWM-HOLDINGS-EXPANSION.

**Proximo paso sugerido.** Verificar scheduled con e04b53ff.
Confirmar los 11 fixes del reporte (Fase dominante, N valid,
Futures Status, Severidad, [INFO] catalogo, 0.00, gate 11/11,
pending inactivo por nh_nl>0) + los 2 fixes de analisis (O2,
BRK). Si todo verde, cerrar el ciclo del reporte 41037457375.

### 2026-10-30 — Sesion 20: contrato dict framework (plan de cierre clase 1fb791ab)

**Objetivo.** Cerrar la clase 1fb791ab en el codigo, no en el proceso
humano. Cualquier modulo A que produce un dict (return o state JSON)
y modulo B que lo consume via A_output.get(X): si A no publica X,
B obtiene None silenciosamente. Aplicar el patron a todos los pares
productor-consumidor del sistema que el AST/schema puedan cubrir.

**Hecho.** 6 commits:
- d100ba01: helper AST generico (tests/_dict_contract.py) +
  refactor MTE. 16 tests refactorizados (comportamiento identico).
- f364ad2e: liquidity_state (top-level bidireccional + nested
  regression + whitelist justificada). 5 tests.
- 053db86c: iae_section (downstream + consistencia 4 ramas). 3
  tests.
- 457742f7: darkpool (AST puro + top-level exacto). 3 tests.
- 4056c033: fix BUG REAL mte_scenario (render leia
  confirmation_data.get('mte_scenario') pero ningun productor lo
  escribia, rama RECESSION CAPITULATION muerta) + schemas
  flows_secondary y confirmation. 6 tests.
- e8a2a4ba: manifiesto + heuristico + cross_module. 3 tests.

**Descubrimientos.**
- Inventario de 32 render_* con dict-like: 10 DataFrames
  disfrazados (clasificador df_attrs los distingue), 2 dicts reales
  cubiertos, 2 en whitelist (PCR-CONTRACT, SECTOR-RESULTS-CONTRACT),
  2 consumidores del mismo productor (render_alerts lee ratios de
  confirmation_data).
- Bug real de la clase: render_confirmation leia mte_scenario sin
  productor. Rama RECESSION CAPITULATION nunca ejecutada. Detectado
  por inventario, no por run.
- Heuristico con clasificador df_attrs (df.empty es Attribute, no
  Call) evita 10 falsos positivos por DataFrame.

**Commits.** d100ba01, f364ad2e, 053db86c, 457742f7, 4056c033,
e8a2a4ba + este docs. Suite 4133 passed, 2 skipped.

**Pendiente.**
- PCR-CONTRACT (OPEN): productor indicators/options.py::
  compute_pcr_signals, consumer render_sentimiento_opciones (21
  lecturas). Cubrir sesion 20+ si ROI.
- SECTOR-RESULTS-CONTRACT (OPEN): productor indicators/
  sector_rank_score.py::compute_sector_rank_score, consumer
  render_ranking_sectorial_unificado. Cubrir sesion 20+ si ROI.
- Verificacion scheduled. WBD fuera de get_stock_list; 15
  expansions descargados; fixes semanticos del reporte (Fase
  dominante, N valid, Futures Status, Severidad, INFO catalogo,
  0.00 sin -0.00).

**Proximo paso sugerido.** Verificar scheduled. Cerrar items
observacionales con fecha (H5.3 20-nov, RECEIPT-AUDIT-LAG 17-nov,
CRON-SLOTS-OBS 10-nov, SCHEDULER-DELAYS 10-nov). Decidir ROI de
PCR-CONTRACT y SECTOR-RESULTS-CONTRACT.

### 2026-10-30 — Sesion 19b: contrato bidireccional MTE (clase 1fb791ab)

**Objetivo.** Cerrar la clase 1fb791ab, no solo la instancia. Bug
original: state tiene X, return no, render no lo ve. 1fb791ab
anadio el render de futures_status pero no expuso la clave en el
return; run local 20261030_201420 lo detecto (7/8, faltaba [4]).
No bastaba con arreglar futures_status: pending repetia el patron.

**Hecho.**
- 5637d45e: futures_status en return de _compute_mte (integracion).
  Test unitario con dict mockeado pasaba; integracion real no
  ejercitada. Bug de proceso.
- 5b96192f: pending en return + render_mte anade "Pending
  transition" + AST guard downstream (render lee X => X en return).
- db1e0a0f: 4 tests de render de pending (pinta, no pinta si None,
  no rompe si ausente, no altera scenario).
- fc6f63c4: AST guard bidireccional. Upstream (state X no whitelist
  => X en return) + whitelist justificada + regresion de los 2
  campos. Cierra la clase en las 4 direcciones.

**Verificacion empirica.**
- Re-run local 19: 7/8. Fallo [4] Futures Status.
- Re-run local 19b (post 5637d45e): 8/8, "Futures Status: STALE"
  visible en L153.
- Re-run local 19c no se ejecuto: la transicion SOFT LANDING ->
  RECESSION ya esta confirmada en state (pending=null). El punto
  "Pending transition visible" no se puede ejercitar en un run
  real hoy. Cubierto por test unitario test_mte_pending_pinta_linea.
- Suite 4108 passed, 2 skipped.

**Commits.** 5637d45e, 5b96192f, db1e0a0f, fc6f63c4 + este docs.

**Pendiente.**
- Verificacion en scheduled real (proximo run-system con HEAD >=
  fc6f63c4). Los 8 fixes previos siguen verificados en local;
  pending solo en tests (state transitorio).
- H5.3 (20-nov), C2, SOW-BEAR, WYCKOFF-5BX: sin cambios.

**Proximo paso sugerido.** Verificar scheduled. Si verde, cerrar
items observacionales con fecha. Decidir pre-registro IWM.

### 2026-10-30 — Sesion 19: cierre deuda del reporte 41037457375

**Objetivo.** Leer el reporte del run 41037457375 (primer scheduled
verde sobre 89b6e0a) y cerrar cada hallazgo, no documentarlo.

**Hecho.** 9 fixes + 1 WONT_FIX razonado, cada uno con test:
- VALIDATION-GATE-LITERAL-10-10 (090a3439): receipt "10/10" con 11
  checks reales. N/N dinamico en provenance, receipt y pipeline_gate.
  32 tests en 3 ficheros.
- MTE-FUTURES-STATUS-OCULTO (1fb791ab): mte_state.json STALE sin
  publicar. Linea en render_mte.
- LIQUIDITY-SEVERIDAD-OCULTA (1fb791ab): severidad_cruda y diff_bp
  ocultos. Linea en render_regimenes.
- CROSS-PROVIDER-VALIDATION-MUERTO (fac7d5f1): script que solo
  consultaba Yahoo con 2 tickers hardcoded. Eliminado con step y
  upload. Las 5 API keys restantes (POLYGON, ALPHA_VANTAGE,
  TWELVE_DATA, FINNHUB, FMP) tienen consumidores reales (DataRouter +
  BackupProvider), no son deuda.
- CATALOG-COVERAGE-WARN-PERPETUO (55a63833): WARN 83.97% con
  coverage_quality=COMPLETE. INFO si COMPLETE, WARN solo si regresion.
  DELISTED fuera del denominador. 12 -> 26 tests en test_iae_section.py.
- FASE-WYCKOFF-SEMANTICA (2c2a1c3e): header "Fase Wyckoff" -> "Fase
  dominante" (es moda de holdings, no fase del ETF) + nota.
- SECTOR-CONCENTRATION-LOW-COVERAGE (2c2a1c3e): columna "N valid" en
  Concentracion (Cob 18-25% sobre 4-5 tickers era invisible).
- NH-NL-N-VALID-OBSERVABILITY (2c2a1c3e): columna "N valid NH/NL" en
  Sector Breadth (NH=0/NL=0 sin n_valid visible).
- NEGATIVE-ZERO-CROSS-ASSET (71371872): -0.00 en XLE 60d FX. _fmt
  local -> _fmt_signed (FU-003a).
- LIDERAZGO-DOBLE-TAUTOLOGICO: WONT_FIX razonado (top-5 por retorno
  siempre bate; nota de sesgo ya en reporte).

**Commits.** 090a3439, 1fb791ab, fac7d5f1, 55a63833, 2c2a1c3e,
71371872 + este docs. Suite 4097 passed al cierre.

**Pendiente.**
- Verificacion empirica en proximo scheduled (23:17 UTC 30-10 o
  03:17 UTC 31-10): Fase dominante, N valid, Futures Status STALE,
  Severidad TENSION_ELEVADA, [INFO] catalogo, 0.00 XLE 60d FX.
- H5.3 (20-nov), C2, SOW-BEAR, WYCKOFF-5BX: sin cambios.

**Proximo paso sugerido.** Verificar scheduled. Si verde, cerrar
items observacionales con fecha. Decidir pre-registro IWM. Documentar
reglas force=true y push en 07_RUNBOOK.

### 2026-10-30 — Sesion 18: ticker health (WBD) + 15 tickers expansions

**Objetivo.** Cerrar el frente WBD (adquirida por Paramount Skydance)
y los 15 tickers fuera de top-20 sin datos. Pasar de diagnostico
manual a sistema autonomo.

**Hecho.**
- Paquete minimo previo: 3 commits de saneamiento docs/tests
  (b9d7aee7 contrato workflow, db1713fe HANDOFF/BACKLOG/METODO,
  e3400a88 rediseno test_h03). 4024 passed.
- 15 tickers fuera de top-20 anadidos a universe_expansions.yaml
  (c879df71): XLF 9, XLV 4, XLU 1, XLY 1. Causa raiz:
  TOP_N_SECTOR_COMPONENTS=20 + head(20) en get_stock_list.
- WBD diagnostico cerrado: adquirida por Paramount Skydance
  (cierre 2026-10-06), delisted, Yahoo solo sirve ultima barra
  pre-cierre (2026-10-05). Cross-listings tambien terminan alli.
  No hay fix tecnico dentro de las reglas (Yahoo, no imputar).
- Sistema autonomo ticker_health F1-F4:
  - F1 fc6232d8: yaml + modulo (estados ACTIVE/DELISTED/STALE/
    TRUNCATED_IPO). 16 tests.
  - F2 38bc1c42: get_stock_list filtra DELISTED (506 -> 505).
  - F3 90e269ff: p1_h02 reporta n_excluidos, WBD fuera de
    sin_datos. 3 tests AST.
  - F4 78ce37a2: check_ticker_health en health_check WARN si ticker
    ACTIVE stale no clasificado. Dispara issue automatico. 7 tests.
- Lecciones de proceso: 4 trampas de PowerShell pisadas (Find
  -Filter excluye .txt, ReadAllBytes resuelve CWD .NET,
  @()+ConvertFrom-Json no desenvuelve, Select-String -SimpleMatch
  desactiva regex). Ancla temporal HEALTH_CHECK_CONFIRMED_INCOMPLETE
  rota por insertar bloque antes de tiempo (amend 78ce37a2).

**Commits.** b9d7aee7, db1713fe, e3400a88, 89b6e0a2 (paquete
minimo), c879df71 (15 expansions), fc6232d8, 38bc1c42, 90e269ff,
78ce37a2 (ticker_health F1-F4).

**Pendiente.**
- Verificacion empirica del scheduled: artifact p1-h02 con XLF
  85.11%, XLV 85.28%, XLU 85.27%, XLY 85.80%. WBD en n_excluidos.
- ticker_health [OK] con WBD fuera de get_stock_list.
- 35 holdout cerrados por SOW-HOLDOUT WONT_FIX.
- HONA/SPCX se resuelven con sesiones.

**Proximo paso sugerido.** Verificar scheduled. Cerrar los items
observacionales con fecha (H5.3, RECEIPT-AUDIT-LAG,
CRON-SLOTS-OBS, SCHEDULER-DELAYS). Considerar pre-registro IWM
(texto ya consensuado con auditor). Documentar reglas force=true
y push en 07_RUNBOOK.

### 2026-10-28 — Sesion 15: saneamiento pipeline + 4 tickets deuda

**Objetivo.** Arrancar con estado limpio y atacar tickets de deuda.

**Hecho.**
- 3 bugs P0 corregidos:
  - MAX_NEW vs universe_expansions (64866361). Desbloquea pipeline.
  - .schema gitignored -> migracion en CI (b246154c).
  - daily_run.yml sin build_catalog_csvs (9e456745).
- 1 bug P0 de gobernanza: REGISTRO corrompido por append sin
  separador (61de4d7a). Validador reforzado con 2 checks nuevos.
- 4 tickets de deuda cerrados (2 commits por ticket: test +
  REGISTRO):
  - PARSE-SSGA-FEZ-TEST-MISSING (290f7977 + a04d259d): 6 tests.
  - EU-INDEX-NO-CAP-TEST-MISSING (62e04dba + c98feba1): 5 tests.
  - TEST-SCRIPTS-SYS-PATH-DYNAMIC (90080ff8 + 01e1cdf0): 4 tests.
  - COMPLETION-ATTEMPT-NO-STEP-AWARENESS (206702f9 + 76983040):
    5 tests.
- Tests: 3845 -> 3880 (+35). Gobernanza: 164 -> 165 items.
- 13 commits pusheados. HEAD al cierre: 76983040.

**Commits.** b2f3e30f..76983040 (13 commits: 5 de saneamiento,
8 de cierre de los 4 tickets de deuda).

**Pendiente.**
- Verificar proximo run scheduled (~04-06Z del 29/10): append de
  sector_wyckoff_distribution.csv (no regresion a 12 lineas),
  sector_regime_matrix.csv, receipt escrito.
- EU-HOLDINGS-DOWNLOAD-CAP: verificar en proximo run que el
  parquet tiene los 112 tickers EU.
- Deudas externas: SOW-BEAR-CONFIRMATION, SOW-HOLDOUT-IMMUTABLE,
  DICT3-*, TOP20-*, IWM-HOLDINGS-EXPANSION, C2-920, WYCKOFF-5BX.

**Proximo paso sugerido.**
1. Verificar el primer run scheduled con el .schema versionado:
   confirmar append de sector_wyckoff_distribution. Si no, ticket.
2. Verificar el step Regenerar catalog CSVs en el scheduled.
3. Si ambos OK: EU-HOLDINGS-DOWNLOAD-CAP verificable -> cerrar.

### 2026-10-28 (continuacion) - Sesion 15 parte 2: bug hunt + SOW-HOLDOUT

**Objetivo.** Continuar el saneamiento iniciado en la primera mitad
del dia. Derivado a: bug hunt de codigo + respuesta a 2 dictamenes
externos del mismo dia.

**Hecho.**

Bug hunt de codigo (5 frentes, 14 bugs reales, guards AST):
- P7 look-ahead (4 commits): iloc[-N:] con huecos
  (slice_last_n_sessions), endDate NY Fed determinista,
  resolve_observation_date vs resolve_reference_date (7 funciones
  migradas), bug de propagacion del raise en 3 funciones.
- Except silenciosos (1): fail-close manifest, str(be),
  run.py pass->print.
- Cache freshness (1): validacion de tickers en cache-hit, CSV
  legacy eliminado, CACHE_MIN_COVERAGE.
- Rate limiting (1): backoff desde attempt+1, sleep entre retries
  individuales (try/finally).

SOW-HOLDOUT-IMMUTABLE (6 commits):
- Infraestructura H-01/H-03: holdout_manifest.py (SHA-256 +
  hash canonico), holdout_intersection.py (interseccion 266/268).
- Fix CI: skipif en tests que hashean parquets gitignored.
- Doc 74: 5 hechos del analisis + decisiones H-04 (opcion B con
  historial) y H-07 (declarar poblacion real).
- H-02: capture_experiment_env.py (commit, prereg ancestro de
  HEAD, requirements SHA, validators, modules, params, seed,
  holdout manifest).
- Contrato v1.1 redactado: doc 58 de 259 -> 482 lineas, 14
  secciones, H-01..H-08.
- Backlog bloqueado externo consolidado en 1 documento (12 items,
  indice unico).

**Commits (parte 2).** 1cd608bc, 86ff433c, cee53c02, a4a28d9e,
a6a8f017, b8b0de96, 4150a68e, 780930e6, 45d48c5d, f7749f69,
93bc595d, d90f2600, a5a26bdb, 43bf21e0.

**Pendiente.**
- Enviar v1.1 SOW-HOLDOUT + doc 74 al auditor externo (canal).
- SOW-BEAR v1.10.1: no redactar hasta decision del usuario.
- Tests T6-T10 de SOW-HOLDOUT: no implementar hasta firma v1.1.
- Verificar proximo run scheduled (~04-06Z): .schema append,
  Regenerar catalog CSVs, EU-HOLDINGS-DOWNLOAD-CAP.

**Proximo paso sugerido.**
1. Enviar v1.1 + doc 74 al auditor (usuario).
2. Verificar proximo run scheduled (operativo).
3. Backlog externo en BACKLOG_BLOQUEADO_EXTERNO.md como indice.

## 4. REFERENCIA A SESIONES ANTERIORES

Para sesiones anteriores al 2026-09-24, ver `git log --oneline`. Resumen tematico en `04_HISTORICO.md`.

---

## 5. REGLAS DE PODA

- La bitacora mantiene las ultimas **10 sesiones**. Ademas, siempre
  conserva las del dia en curso, aunque superen 10 (una jornada larga
  puede generar 6-7 entradas, y perder la de la manana mientras se
  trabaja la de la noche no aporta).
- Al anadir una nueva, si hay >10 fuera del dia en curso, la mas antigua
  se elimina. Nota: la regla era 'ultimas 5' hasta 2026-09-30. Nunca se
  aplico (12 entradas acumuladas). Se cambia al umbral real observado
  en lugar de fingir que se poda.
- **Antes de eliminar**, verificar que sus decisiones clave estan en `04_HISTORICO.md`. Si no, resumir primero.

---

**Fin de la bitacora.**

---

### 2026-11-01 — B6 (auditoria reporte): 20 FIXED, 5 OPEN heredados

**Objetivo.** Auditar los 21 modulos de `src/report/*` + `report_generator.py`
con los 18 patrones de `01b_PATRONES.md` + 5 nuevos. Cerrar hallazgos de
B6 con test + evidencia + item REGISTRO.

**Hecho.**
- 21 modulos auditados. 20 items B6 FIXED (13 hallazgos H1-H13 + 7
  paridad guards). 5 OPEN heredados a B4 (3) y B5 (2).
- Hallazgos H1-H13: ver HANDOFF seccion 2. Resumen: input vacio macro,
  side-channel liquidity_state x2 (header + freshness), mojibake/header
  coding decl x3, IHR bands derivadas, MTE SCENARIO_RULES data-driven,
  guards NaN en confirmation/alerts/iae, paridad de guards con alerts
  en synthesis/breadth/market_context/sectorial, fixes mecanicos.
- Refactor data-driven en `indicators/mte/scoring.py`: `SCENARIO_RULES`
  dict con pre_invalid/finite/rules. Golden test pre-refactor (10
  combinaciones). Nota del scoring derivada via `_fmt_scenario_rules_note`.
- IHR bands derivadas: `_fmt_ihr_bands()` en helpers + `IHR_LABELS` en
  `options_metrics.py`. Antes 3 bandas hardcoded que ya divergian de
  `classify_ihr` (5 categorias).
- 2 renders huerfanos detectados (WYCKOFF-RENDERS-HUERFANOS, B3):
  `render_representatividad_lider` y `render_divergencia_sector_lideres`
  con productores retirados en `4bc45fd3` (sesion 3).
- Revertidos S1-S5 sobre `sector_context.py`: parcheaban NaN que solo
  existian en tests artificiales, no en CSV de produccion.

**Commits.** `22bef269`, `ddcdd0fb`, `74e2e446`, `466198ff`, `5e5d7286`,
`f4757230`, `1c30c571`, `98548787`, `09a74ac4`, `a2e4ac94`, `d072e6b8`,
`72e9dee2`, `127848db`, `66c75262`, `b62e645e`, `6981f19a`, `b69c17fe`,
`56f6c867`, `f3b34202`, `a4994a91`, `48b78253`, `ebc9ac5d`, `5d3bdcf2`,
`bcbddbbf`.

**Pendiente.**
- B4 (motores/regimen): 3 items heredados de B6 + auditoria completa.
- B3 (Wyckoff): 1 item heredado (renders huerfanos) + auditoria completa.
- B5 (outputs/persistencia): 2 items heredados de B6 (currency column,
  registry coverage).
- B2, B7, B8: sin auditar.

**Proximo paso sugerido.** B4: cerrar los 3 items heredados de B6
(ILOC-BARRIDO, MTE-CLS-INF-LATENTE, RATIO-NAMES-DUPLICADA) + auditar
`regimes/*`, `sector_regime_matrix`, `sector_rank_score`, `engines`,
`evidence_matrix`.
