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

    ### YYYY-MM-DD Ã¢â‚¬â€ Titulo corto

    **Objetivo.** Que se pretendia hacer.

    **Hecho.**
    - Bloque 1.
    - Bloque 2.

    **Commits.** <lista de SHAs> (o rango).

    **Pendiente.**
    - Item 1.
    - Item 2.

    **Proximo paso sugerido.** Que deberia hacerse a continuacion.

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 10 fuera del dia en curso, la mas antigua se elimina o se resume en el historico. Ver reglas completas en Ã‚Â§5.

---

## 3. SESIONES

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

### 2026-10-27 (sesion 14) - 9 tickets cerrados + 5 bugs reales + 3 docs contrato

**Objetivo.** Aprovechar nucleos del procesador para tests (derivado
a: cerrar deuda acumulada y endurecer CI).

**Hecho.**
- Paralelizacion: pytest-xdist==3.8.0, -n auto --dist loadfile.
  Suite 246s -> 120s. Comando de arranque actualizado con dos variantes
  (smoke check + suite autoritativa).
- 9 tickets cerrados: PCT-FLOAT-ROUNDING-DIVERGENCE (FIXED con doble
  run byte-identico), ARRANQUE-C6-GHOST (renombrado, premisa falsa),
  HANDOFF-MD-OBSOLETO, GOVERNANZA-CI-NO-STEP (step Verify governance
  en CI), GOVERNANZA-HANDOFF-OBSOLETO (verificador lee HANDOFF raiz),
  TOP20-EU-HOLDINGS-INCOMPLETOS, CURRENT-PHASE-BASIS-CSV (WONT_FIX
  medido), ACCUMULATION-ZERO (nota en reporte), VALIDATION-GATE-NO-
  STANDALONE (WONT_FIX), YAHOO-PARTIAL-HISTORY-PREEXISTING (WONT_FIX).
- 5 bugs reales destapados y corregidos:
  1. FEZ 27 tickers sin sufijo (Yahoo resolvia a ADRs erroneos:
     DG=Dollar General, EL=Estee Lauder, BN=Brookfield, AD=Kroger).
     27 mapeos verificados + validacion Yahoo. NDA FI excluido por D11.
  2. sys.path roto en update_european_holdings.py. Regresion de
     c4f99795 (25-oct). Test test_scripts_sys_path.py ampliado a 5
     scripts. Diagnostico inicial "26 dias fallando" era FALSO:
     corregido en REGISTRO.
  3. PIT snapshot no era append-only real. sort_by global reescribia
     el historico entero en orden alfabetico (diff 3000+3000 lineas
     cosmeticas). Fix: sort_by=None. Test de contrato anadido.
  4. Truncamiento [:20] en 3 parsers EU. Causa raiz de
     TOP20-EU-HOLDINGS-INCOMPLETOS. FEZ 20->47, DAXEX 20->32,
     ISF.L 20->88, LYXI 20->35. Cobertura 95/85/84/99%.
  5. head(20) en get_stock_list para indices EU. 112 tickers fuera
     del parquet aunque el CSV estuviera completo. Fix:
     EU_INDEX_ETFS_NO_CAP. get_stock_list 359 -> 471.
- 3 documentos redactados: 57_contrato_v1_10.md (SOW-BEAR-CONFIRMATION),
  58_contrato_holdout_inmutable.md (SOW-HOLDOUT-IMMUTABLE, corrige
  cifra 7/8 -> 35/40), 59_pre_registro_reaccumulation.md
  (DICT3-REACCUMULATION, R1-R5 + C1-C3).
- 2 tickets nuevos: EU-HOLDINGS-DOWNLOAD-CAP (IN_PROGRESS,
  verificacion proximo run), TOP20-USA-INDICES (medicion real:
  SPY n=167, QQQ n=41, DIA n=20).

**Commits.** 20+ commits: dc809efd, e5d1cc6b, ffd3c401, a390494a,
af811cfc, 353bcea8, a9a90cfa, 6fd537f8, f29a44f6, a1dbb692,
dbdbf5c5, 678a9a8d, ee7aac87, f939fb7e, 194f77dd, f31dbd8b,
17619aa9, 46a3b370, 3a682884, 3a383592, 14feb408, 0bff883f,
2683ed68, 57bdaa52, 1a719507, b31b878a, ea7f45df, 49fa3a82.

**Lecciones aprendidas.**
- `git add` selectivo omitio scripts en varios commits. Local pasaba
  (working tree con fix), CI fallaba (version antigua). Antes de cada
  commit de gobernanza: `git status -sb` completo.
- `AppendAllLines` no acepta encoding. Leer + concatenar + WriteAllLines.
- Here-string multilinea en consola interactiva rompe el parser. Usar
  arrays de lineas.
- Diagnostico "26 dias fallando" en REGISTRO era falso; corregido tras
  verificar cronologia de runs con gh. Leccion: verificar premisas del
  ticket antes de actuar.
- Runs locales SIEMPRE en consola visible, nunca Start-Process Hidden
  (instruccion del usuario).

**Pendiente.**
- Verificar EU-HOLDINGS-DOWNLOAD-CAP end-to-end en proximo run autonomo.
- Enviar 3 docs al auditor externo (57, 58, 59).
- IWM-HOLDINGS-EXPANSION: decision de producto pendiente (top-20 cubre
  6.8%, llegar a 50% requiere ~700 tickers).

**Proximo paso sugerido.** Verificar autonomia de EU en proximo run
(2026-10-28 11:17 UTC). Confirmar cobertura FEZ/DAXEX/ISF.L/LYXI
~95/85/84/99% con `py scripts/audit/p1_h02_cobertura_efectiva.py`.
Si OK: EU-HOLDINGS-DOWNLOAD-CAP -> FIXED. Si no: investigar
cache-check de download_stock_prices.
### 2026-10-27 (sesion 13) - Normalizacion tickers + regen historico + run.py

**Objetivo.** Cerrar `WYCKOFF-CLASSIFY-PHASE-HANG`, arreglar el bug real
detectado durante el probe (WYCKOFF-TICKER-NORMALIZATION), regenerar
el historico con el fix y actualizar `data_quality.csv`.

**Hecho.**
- WYCKOFF-CLASSIFY-PHASE-HANG: probe 360/360 OK, 21.6s. Falso positivo.
  `classify_phase` no cuelga. REGISTRO OPEN -> FALSE_POSITIVE.
- WYCKOFF-TICKER-NORMALIZATION descubierto: holdings usa `BRK.B`,
  parquet usa `BRK-B`. 3 sitios afectados (`_evaluate_holding`,
  `_get_tickers`, `_procesar_ticker`). XLF perdia BRK.B (12.32%
  de peso). Fix `4e94bf2e` + 1 test regresion.
- Snapshot diario: 527 -> 528 filas. XLF `ev_n` 22 -> 23,
  `ev_peso` 63.74 -> 76.05. Resto de grupos sin cambio.
- p1_h02 calcula `evaluable` real via `compute_wyckoff_snapshot`
  (era proxy con_datos). `68e64e6b`.
- Regeneracion historica (1086 fechas, 56.5 min, 248 tickers):
  solo XLF cambia (`n_total` 22 -> 23 en las 1086 fechas). 0
  regresiones. Invariantes P1-H03 post-swap 5/6 OK. `f71eb49c`.
- `p1_h03_invariantes.py::check_5` reformulado: compara activo
  contra `v2_full.csv` post-swap, no contra `.bak` pre-swap.
- `py run.py` manual (~23 min): `data_quality.csv` a 2026-10-26
  (624 filas). Gate 11/11. 6 CSVs holdings regenerados.
  `sector_wyckoff_distribution.csv` revertido a HEAD (ruido de
  float ULP sin cambio semantico). `e86d6f04`.

**Commits.** Rango `4dba82d2..e86d6f04` (3 commits):
- 4dba82d2 REGISTRO WYCKOFF-TICKER-NORMALIZATION -> FIXED + doc 72 §7.
- f71eb49c Regeneracion historica con BRK.B en XLF.
- e86d6f04 data(outputs): run 2026-10-27.
Mas en el mismo dia (sesion 12 continuada):
- 842c9aaf WYCKOFF-CLASSIFY-PHASE-HANG -> FALSE_POSITIVE.
- 4e94bf2e fix normalizacion tickers.
- 68e64e6b p1_h02 calcula evaluable real.
- 89e24705 REGISTRO WYCKOFF-TICKER-NORMALIZATION registrado.

**Pendiente.**
- DICT3-TOP20-COBERTURA: solicitar cierre a A.2c (doc 70-v2d). Auditor.
- DICT3-H04-INTERVALS: OPEN, rechazado por auditor H-04.
- DICT3-H04-INFERENCE-V2: OPEN, 4 decisiones de diseno.
- TOP20-FALTANTES-DATOS: 52 tickers (7 holdout SOW, 1 MRNA
  ausente, 44 descarga nueva).
- TOP20-EU-HOLDINGS-INCOMPLETOS, TOP20-USA-INDICES, TOP20-EU-INDICES.
- SOW-HOLDOUT-IMMUTABLE, GOVERNANZA-CI-NO-STEP.
- Ruido de float ULP entre regenerate_wyckoff_history.py y
  sector_metrics.py (candidato a ticket `PCT-FLOAT-ROUNDING-
  DIVERGENCE`).

**Proximo paso sugerido.** Prioridad tecnica: ticket de redondeo
de `pct_*` (evita ruido en cada `py run.py` post-regeneracion
historica). Prioridad datos: TOP20-FALTANTES-DATOS (descarga de los
44 + decision sobre 7 holdout SOW). Prioridad gobernanza: GOVERNANZA-
CI-NO-STEP (5 min, cierra el hueco de verificacion en CI).

### 2026-10-26 (sesion 12) - Dictamen 70-v2 (6 acciones) + dictamen H-04

**Objetivo.** Ejecutar las 6 acciones del dictamen externo 2026-10-26
sobre doc 70-v2, y responder al dictamen H-04 recibido al final de la
sesion.

**Hecho.**

Dictamen 70-v2 (4 acciones ejecutables + 1 separacion + gobernanza):
- P1-H01: selector con estados REACHED/UNREACHABLE/INVALID_UNIVERSE.
  `max_missing_weight_ratio=0.0` default. 15->18 tests. Consumidores
  adaptados (`_alcanzado`/`_ok` -> `_estado`), ninguno leia el tercer
  elemento (verificado por AST).
- P1-H02: cobertura efectiva por grupo (18 grupos). Script
  `p1_h02_cobertura_efectiva.py` + doc 71. Universo real de tickers
  sin datos: 52, no 8 (handoff subestimaba x6.5). Peor sector USA
  XLF 76.19% sobre grupo; peores EU FEZ 66.26%, ISF.L 60.80%.
  TOP20-FALTANTES-DATOS expandido.
- P1-H03: invariantes del backfill. Script `p1_h03_invariantes.py` +
  doc 72. 5/6 OK, 1 SKIP (bloqueado por hang de classify_phase).
  Correccion: `v2_full_target_only.csv` es artefacto intermedio
  defectuoso; base valida `v2_full.csv.bak`.
- P1-H04: nota PIT visible en reporte. `render_wyckoff_sectorial` +
  2 tests. Solo reporte (no sidecar JSON).
- P1-H06: 6 tickets gobernanza (PERSIST-REPLACE-DATE, PCR-SKIP-NO-DATA,
  TOP20-USA-INDICES, TOP20-EU-INDICES, SOW-HOLDOUT-IMMUTABLE,
  GOVERNANZA-CI-NO-STEP).
- P1-H05: separacion A.2c vs H-04 hecha. P1-H05 no se cierra: el
  dictamen H-04 rechaza la inferencia, pero la separacion esta
  registrada.

Dictamen H-04 (recibido al final de la sesion):
- Doc 73: respuesta formal. 5 hallazgos aceptados (P0-H04-01..03,
  P1-H04-04..05). Criterio §9 declarado no concluyente, no
  reinterpretado.
- DICT3-H04-INTERVALS: OPEN sin cambio de estado. siguiente_accion
  reescrito con los 5 hallazgos y las 4 decisiones.
- Nuevo DICT3-H04-INFERENCE-V2: 4 decisiones de diseno, sin codigo
  hasta aprobacion del auditor.
- Verificado: sistema NO publica IC95 en reporte (0 hits en
  src/report/). structure_ratio solo descriptiva. Sin cambios de
  codigo necesarios.

Ciclo documental 70-v2: v2b (aceptacion), v2c (solicitud
reconsideracion), v2d (cierre 5/6 acciones).

Bug detectado: classify_phase cuelga con al menos un ticker. Traza:
_trend_component en wyckoff_v1.py:99 alinea Series con indices
incompatibles. Bug preexistente, no introducido. Registrado como
WYCKOFF-CLASSIFY-PHASE-HANG (OPEN).

**Commits.** Rango `ebdeac02..74b09c7e` (13 commits):
- b6b75e55 doc 70-v2b
- 46a5bfe3 REGISTRO 6 items
- 6a591e67 correcciones post-saliente
- b3793d7c P1-H01 selector
- c080761d P1-H01 en REGISTRO
- 2d84ae82 P1-H02
- adfbb69b P1-H03
- f967985d P1-H04
- edd4cbf0 P1-H04 en REGISTRO
- 52a7ce8d doc 70-v2c
- 5b855ce5 doc 73 + REGISTRO H-04
- fc0b695b doc 70-v2d
- 74b09c7e WYCKOFF-CLASSIFY-PHASE-HANG

**Pendiente.**
- DICT3-TOP20-COBERTURA: solicitar cierre como FIXED a A.2c
  (doc 70-v2d). Auditor externo.
- DICT3-H04-INTERVALS: OPEN, decision sobre nuevo protocolo.
- DICT3-H04-INFERENCE-V2: OPEN, 4 decisiones de diseno.
- TOP20-FALTANTES-DATOS: 52 tickers (7 holdout SOW, 1 MRNA ausente,
  44 descarga nueva).
- TOP20-EU-HOLDINGS-INCOMPLETOS, TOP20-USA-INDICES, TOP20-EU-INDICES.
- SOW-HOLDOUT-IMMUTABLE, GOVERNANZA-CI-NO-STEP.
- WYCKOFF-CLASSIFY-PHASE-HANG: fix candidato `.to_numpy()` en
  _trend_component.

**Proximo paso sugerido.** Confirmar WYCKOFF-CLASSIFY-PHASE-HANG
como ticket prioritario (desbloquea invariante 2 de P1-H03 y columna
evaluable real de P1-H02). Alternativa: arrancar TOP20-FALTANTES-DATOS
(descarga de los 44 + decision sobre los 7 holdout SOW).

### 2026-10-26 (sesion 11) - A.2c TOP20 + dictamen condicionado + 2 bugs colaterales

**Objetivo.** Cerrar `DICT3-TOP20-COBERTURA` fases A.2a-c: cambio de
selector de holdings por sector USA de `nlargest(20)` a acumular por
peso hasta 85%, con regeneracion completa del historico. Primera
respuesta al dictamen externo 2026-10-26 sobre doc 70 v1.

**Cambios en produccion:**

1. Expansion declarativa (`404dc7c5`): `data/universe_expansions.yaml`
   con 28 tickers XLI. `get_stock_list()` fuente 5. Tests: 8.
2. Selector `_select_target_holdings` (`557048f8`): sin `idxmax()`,
   denominador = peso total grupo, `MIN_TICKERS_PER_GROUP=20` evita
   regresion de grupos >85%. Tests: 15.
3. `selection_mode` en `compute_wyckoff_snapshot` (`0c14a8a3`).
   `run.py` pasa `target_weight`. Tests: 4.
4. `regenerate_wyckoff_history` parametrizado (`5e4fe2fc`).
5. Backfill completo: 1086 fechas × 11 sectores, 50 min. 0 regresiones
   por fila (date, sector) en 11946 filas. Swap a activo + backup
   `pre_top20_target.csv` (hash 7D988C841C06E567..., versionado).
6. A.2c completo (`e7f27cd0`): 21 ficheros regenerados.

**Bugs colaterales corregidos:**

- `persist()` es append-only por merge_keys. Los 6 CSVs de holdings
  son subconjuntos filtrados por fase: un ticker que sale deja su
  fila vieja huerfana. Detectado con GE/PWR/VRT en XLI (14 filas CSV
  vs 11 snapshot). Fix: `replace_date=True` en `persist` + flag en
  `wyckoff_holdings`. Tests: 2. Commit `f4528b49`.
- `compute_pcr_signals` retornaba `None` sin `skip()` cuando CBOE
  no responde. El gate detectaba "sin receipt pcr_history" y fallaba
  el run entero. Fix: `skip('pcr_history', 'NO_DATA')` en los dos
  return None tempranos. Tests: 4. Commit `9338dcd4`.

**Efecto en cobertura:**

- XLI: 20→44 (57.2% → 85.8%).
- XLF: 19→22 (72.8% → 85+%).
- XLK: 20→22 (79.6% → 85+%).
- XLE, XLC, XLB, XLP, XLRE: sin cambio (ya cumplian >85%, el
  min_tickers=20 evita regresion).
- XLU, XLV, XLY: seleccionan tickers extra pero NO estan en parquet
  (CNP, CI/MRNA/BSX/HCA/COR/CAH, NKE). Cobertura efectiva sin cambio.
  Ticket `TOP20-FALTANTES-DATOS` abierto.

**Re-ejecucion H-03/H-04 sobre nueva base:**

Doc 69 §13 (anexo, no modifica §2-§10):
- H-03: 30/30 (identico, opera sobre parquet).
- H-04: `rho_bar` 0.524→0.471, ALL annual IC [0.268,0.754]→
  [0.294,0.765], ancho multiway/naive 2.4×. Criterio §9 sigue sin
  discriminar. `DICT3-H04-INTERVALS` OPEN.

**Dictamen externo 2026-10-26 sobre doc 70-v2:**

APROBACION CONDICIONADA de A.2c. NO aprobacion del cierre global
de `DICT3-TOP20-COBERTURA` (sigue OPEN). 6 acciones pendientes:
P1-H01 (selector con estados REACHED/UNREACHABLE/INVALID_UNIVERSE),
P1-H02 (tabla cobertura efectiva ponderada), P1-H03 (6 invariantes
backfill), P1-H04 (sesgo PIT visible en artefactos consumidos),
P1-H06 (gobernanza separada + tickets índices USA/EU/IWM),
P1-H05 (H-04 separado, no mezclar con A.2c).

**Correccion documental detectada por auditor:** el resumen decia
"resto no cambia porque ya cumplia". Falso: XLU 83.1%, XLV 78.9%,
XLY 84.5% NO cumplian nominalmente. No cambian materialmente porque
sus tickers extra no estan en parquet. Doc 70-v2b debe corregirlo.

**Incidente durante la sesion:** `git checkout -- outputs/` revertis
el CSV regenerado (52 min de backfill perdidos). Re-ejecutado. Leccion:
backup `.bak` antes de revertir.

**Commits (11):** 404dc7c5, 557048f8, 0c14a8a3, 5e4fe2fc, efd53cfe,
f4528b49, 9338dcd4, e7f27cd0, a14b463b, 8a0a8b1d. Suite: 3809
passed, 2 skipped. CI verde en cada push.

**Estado al cierre:** A.2c en produccion con target_weight sectores
USA. DICT3-TOP20-COBERTURA sigue OPEN (deuda: 6 acciones del
dictamen). DICT3-H04-INTERVALS OPEN. 4 frentes nuevos abiertos
como tickets: P1-H01 a P1-H06, TOP20-FALTANTES-DATOS,
TOP20-EU-HOLDINGS-INCOMPLETOS, indices USA/EU/IWM.

### 2026-10-25 (actualizacion de traspaso)

**Objetivo.** Actualizar HANDOFF + 00_ARRANQUE + 04_HISTORICO +
05_BITACORA para replicar el trabajo en otra IA con contexto completo.

**Hecho.**
- HANDOFF reescrito sesion 10. Estado real: HEAD f5cb8834, 3755/2
  tests, gate 11/11, 25 writers, PIT activo, structure_ratio continua,
  sector_regime_matrix v2.
- 00_ARRANQUE §3: gate 10/10 -> 11/11, 24 writers -> 25, bloque
  dictamen anadido.
- 04_HISTORICO: seccion sesion 10 con detalle por hallazgo.
- 05_BITACORA: esta entrada.

**Commits.** pendiente.

**Pendiente.**
- DICT3-H04-INTERVALS, DICT3-H05-FILLNA-FUTURE, DICT3-TOP20-COBERTURA,
  DICT3-REACCUMULATION.
- SOW-BEAR-CONFIRMATION (contrato v1.10).
- Deuda documental: 62 (Tests 3776 -> 3755), 00_ARRANQUE (writers
  20 -> 25).

**Proximo paso sugerido.** DICT3-H05-FILLNA-FUTURE (acotado) o cerrar
deuda documental residual (62 + 00_ARRANQUE).

### 2026-10-25 (sesion 10) - Respuesta al dictamen externo (doc 68)

**Objetivo.** Cerrar los huecos documentales del dictamen externo
2026-10-25 tras los cierres de codigo (H-03, H-06, PIT, structure_ratio
continuo, Xetra).

**Hecho.**
- `docs/auditoria/wyckoff/68_respuesta_dictamen_2026-10-25.md`: doc
  formal de respuesta. Detalle por hallazgo, pruebas aportadas, pruebas
  pendientes, cambios estructurales.
- REGISTRO:
  - `DICT3-WYCKOFF-AUDIT-2026-10-25` FIXED (frente global).
  - `DICT3-H04-INTERVALS` OPEN (ICs, tamanos muestrales, solapamiento).
  - `DICT3-H05-FILLNA-FUTURE` OPEN (bar_flags sin romper v1.8).
  - `DICT3-TOP20-COBERTURA` OPEN (universo dinamico 80-90%).
  - `DICT3-REACCUMULATION` OPEN (re-acumulacion continuacion).
  - `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE` -> SUPERSEDED por Bloque 2.
  - Nota en REP-AUDIT-20261022-WYCKOFF-P7-B4-B5-B3 (B4 = H-05 dictamen).
- HANDOFF: §3 CERRADOS con DICT3 FIXED + SUPERSEDED. §4.2 anade los
  4 DICT3 OPEN. Arreglada numeracion §4 (habia dos 4.1 duplicados).
- Bug menor: doc 67 no existe como fichero (el analisis se hizo en
  chat). Doc 68 lo sustituye.

**Commits.** pendiente.

**Pendiente.**
- `DICT3-H04-INTERVALS`: congelar protocolo antes de inferencia.
- `DICT3-H05-FILLNA-FUTURE`: bar_flags en build_ticker_df.
- `DICT3-TOP20-COBERTURA`: universo dinamico.
- `DICT3-REACCUMULATION`: frente nuevo con pre-registro.
- `SOW-BEAR-CONFIRMATION`: contrato v1.10 pendiente auditor.
- `WYCKOFF-5BX`: ~2027.
- Deuda documental residual: 62 (Tests 3776 vs 3705), 00_ARRANQUE
  writers 20 vs 24.

**Lecciones.** "10 anos" del informe preliminar era error propio:
el parquet market_data tiene 2021-10-04 pero el historico Wyckoff
regenerado empieza en 2022-08-11 (1086 sesiones, ~4.2 anos). El
doc 67 que se menciono en mensajes previos nunca se commiteo; doc 68
lo sustituye con las cifras correctas.

### 2026-10-25 (sesion 9) - Fase 3: regeneracion historica

**Objetivo.** Regenerar `sector_wyckoff_distribution` con el universo
correcto (top-20, >=200 barras) sobre todo el historico.

**Hecho.**
- `scripts/regenerate_wyckoff_history.py`: 12 workers, checkpoint,
  validacion obligatoria contra `.csv` activo.
- 11946 filas (1086 fechas), 2022-08-11 -> 2026-10-23.
- 49.8 min, 234.197 invocaciones classify_phase_at.
- Validacion: 11 filas x 20 columnas identicas en 2026-10-23.
- Swap: `.csv` activo = regenerado. `.v1.csv` (342) queda como backup.

**Hallazgo critico.**
`structure_positive = (pct_acc + pct_markup) >= 50` da 0/11946 en el
historico completo. `pct_range` domina (media por sector 47.4%-91.6%).
El umbral es inalcanzable. Fase 5 requiere rediseño de la metrica.

**Commits.** Codigo/datos en este commit. Doc en commit siguiente.

**Pendiente.**
- Fase 5: rediseño de structure_score con pre-registro. El umbral `>=50`
  no se toca hasta tener evidencia empirica sobre la que decidir.
- Fase 6: descarga unificada (bloqueada por falta de 4 ficheros).

**Lecciones.** Una metrica binaria con umbral fijo puede estar muerta
durante meses sin que nadie lo note, si el test no mide su discriminacion
temporal. Con 1086 fechas x 11 sectores, 0/11946 es evidencia solida de
que `>=50` no discrimina.
### 2026-10-25 (sesion 8) - Fase 4b: rename compute_sector_scores

**Objetivo.** Resolver confusion de nombres entre dos 'scores' del
reporte: `compute_sector_scores` (6 componentes sobre el ETF) y
`compute_structural_score` (3 componentes via STRUCTURAL_WEIGHTS).

**Hecho.**
- Rename `compute_sector_scores` -> `compute_sector_rank_score`.
- 31 reemplazos en 7 ficheros.
- Sin cambio de contrato ni calculo.
- Suite 3717/2. Run real verificado.

**Commits.** Codigo en este commit.

**Pendiente.**
- Fase 3: regeneracion historica (proyecto separado).
- Fase 5: structure_score con pre-registro (>=50 da 0/11).
- Fase 6: descarga unificada (bloqueada por 4 ficheros).

**Lecciones.** Un nombre generico ('sector_scores') oculta la
ambiguedad. Renombrar explicita lo que el codigo hace.
### 2026-10-25 (sesion 7) - Fase 2: schema_v en frescura

**Objetivo.** Cerrar asimetria Fase 1: `.schema` existia pero el
reporte no lo publicaba.

**Hecho.**
- `_read_schema_version()` lee `<path>.schema` (entero).
- `compute_data_quality` anade `schema_v` por fila post-loop.
- `render_calidad_datos` publica columna Schema entre Cobertura y Notas.
- Nota al pie con semantica (vN de `.schema`, v1 implicito, doc 64).

**Commits.** Codigo en este commit.

**Pendiente.**
- Fase 3: regeneracion historica (proyecto separado).
- Fase 4b: renombrar `compute_sector_scores` (trend, no estructura).
- Fase 5: `structure_score` con pre-registro.
- Fase 6: descarga unificada.

**Lecciones.** Simetria: un contrato que se valida internamente (gate
check 11) debe exponerse al operador (reporte). Si no, el gate sabe
algo que el usuario no ve.
### 2026-10-25 (sesion 6) - Fase 4: unificacion estructura

**Objetivo.** Cerrar la divergencia agregado vs CSVs detectada en
auditoria: 4 caminos paralelos de estructura sectorial.

**Hecho.**
- `compute_wyckoff_snapshot`: una pasada de `classify_phase`, top-20
  por grupo + 200 barras. 14 columnas (12 contrato + n_bars + evaluable).
- `sector_wyckoff_distribution` v3.0: consume snapshot. Universo pasa
  de "todos + 60" a "top-20 + 200".
- 6 CSVs: proyecciones del mismo snapshot.
- `run.py`: calcula una vez, pasa a los 2 consumidores.
- `phase_since` ISO date (antes timestamp con hora).
- Test coherencia: agregado == CSVs por fase, 11 sectores.
- Hotfix urgente: fixture de tests borraba `.schema` reales y
  sobrescribia `.v1.csv`. Recuperado desde git. `persist` endurecido.

**Commits.** `9960cdac` (codigo). Hotfix y datos en commits siguientes.

**Pendiente.**
- Fase 4b: renombrar `compute_sector_scores` (mide tendencia, no estructura).
- Fase 5: `structure_score` continuo. `>=50` sigue dando 0/11.
- Fase 6: descarga unificada (Router fed/treasury).

**Lecciones.** Una fixture de test que limpia artefactos de produccion
es un bug estructural, no un detalle. El `.v1.csv` (historico 342 filas)
se recupero desde git solo porque estaba en un commit. Si no, se pierde
permanentemente. Los tests de integracion con path real requieren
mtime cutoff o path_override.
### 2026-10-25 (sesion 5) - Fase 1: schema versioning outputs

**Objetivo.** Cerrar el incidente de schema mezclado detectado en
auditoria post-sesion-4 (auditoria completa solicitada por el usuario:
descarga, fuentes, Wyckoff, arquitectura).

**Diagnostico.** Commit `415487ff` (sesion 3) anadio FORMING/CONFLICT
a `sector_wyckoff_distribution` sin versionar. CSV con 330 filas v1 +
11 filas v2 sin senal. `persist()` ciego, gate y data_quality no
detectaban.

**Hecho.**
- `OutputSpec.schema_version` (default 1, sector_wyckoff_distribution=2).
- `persist()`: COPIAR backup v{old} si schema difiere, escribir nuevo
  sin appendear, escribir `.schema` lateral.
- Gate check 11: schema CSV vs spec, reporta MIGRATED.
- `.gitignore`: `outputs/**/*.schema` + `outputs/history/*.v[0-9]*.csv`.
- `finalize.save_regime_history`: path_override (retrocompatible).
- `conftest.py`: fixture autouse limpia `.schema` tras tests.
- `test_output_schema.py`: 6 tests.
- Run real 15 min, gate 11/11, CSV regenerado (11 filas v2).

**Commits.** `c9a848b5` (codigo), `a88e85da` (CSV).

**Pendiente (Fase 2-6).**
- Fase 2: data_quality con schema_v en frescura.
- Fase 3: regeneracion historica (proyecto separado).
- Fase 4: unificacion de 4 caminos de estructura sectorial.
- Fase 5: structure_score continuo.
- Fase 6: descarga unificada (Router).

**Lecciones.** Un cambio de schema sin version explicita se convierte
en historico envenenado silencioso. El append ciego es la causa raiz
del bug. `git log -p` + analisis de columnas por fecha fue como se
detecto. Requirio ~3h de auditoria para localizar el commit `415487ff`.

### 2026-10-25 (sesion 4 continuacion 2) - Cierre STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE

**Objetivo.** Cerrar la ultima deuda de codigo de sesion 3.

**Hecho.**
- Retirado `leader_breadth=0.5` de la firma de
  `regimes/structural_engine.py::compute_structural_score`.
- Confirmado en auditoria que el cuerpo NO usaba el parametro:
  `STRUCTURAL_WEIGHTS` = {rs_structural 0.50, flow_structure 0.30,
  persistence 0.20}, suma 1.0. Cosmetico puro, no sesga el calculo.
- AST: 10 llamadores, todos sin el kwarg.
- 28 focalizados + 3705/2 suite verde, pyflakes/compileall limpios.

**Commits.** `7679d3fb`.

**Pendiente.**
- `REPORT-ENCODING-MOJIBAKE-ADC` (nuevo): mojibake preexistente [FALSE_POSITIVE 2026-10-25: reporte en UTF-8 correcto, mojibake era consola PowerShell]
  en bloque Clasificacion de Evidencia del reporte.
- `CURRENT-PHASE-BASIS-CSV`, `ACCUMULATION-ZERO`,
  `VALIDATION-GATE-NO-STANDALONE`, `IWM-HOLDINGS-EXPANSION`,
  `SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE`,
  `PODA-BITACORA-DIA-EN-CURSO` (documentales u observacionales).
- SOW-BEAR-CONFIRMATION (contrato v1.10).

**Lecciones.** Un parametro con default neutro que nadie pasa puede
parecer sesgo silencioso pero ser cosmetico. La verificacion del
cuerpo (no solo de la firma) lo confirma en 2 minutos. En este caso
el frente era prioridad 2 en HANDOFF; debio ser prioridad 3 desde
el principio.

### 2026-10-25 (sesion 4 continuacion) - Cierre C4.5-SLPM-STRINGS-RESIDUALES

**Objetivo.** Cerrar C4.5-SLPM-STRINGS-RESIDUALES (deuda de
sesion 3).

**Hecho.**
- 9 ficheros productivos limpiados: `dependency_tracker.py`
  (v1.2->v1.3, DIRECT sin SLPM/stock_leader/Leader Breadth,
  liquidity_v2, +WLS), `sectorial.py:54`, `rankings.py:90,145`,
  `engines.py:2`, `rs_internal.py:6`, `momentum.py:73`,
  `report_generator.py:169` (bloque huerfano),
  `helpers.py` (Acciones lideres), `run.py` (limpieza CSVs de
  lideres).
- Commit codigo `c1e4af2e`.
- Run real 14.4 min, gate 10/10, 24 writers OK, cero menciones
  SLPM en reporte, cero "Acciones lideres".
- Nueva deuda `REPORT-ENCODING-MOJIBAKE-ADC` (OPEN): mojibake [FALSE_POSITIVE 2026-10-25: reporte en UTF-8 correcto, mojibake era consola PowerShell]
  preexistente en el bloque clasificacion de evidencia del
  Anti-Double-Counting ("ClasificaciÃ³n", "CrÃ©dito"). Detectado
  en A5 del run real. No introducido por C4.5.

**Commits.** `c1e4af2e` (codigo).

**Pendiente.**
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE` (cosmetico confirmado).
- `REPORT-ENCODING-MOJIBAKE-ADC` (nuevo, verificar fix en escritor [FALSE_POSITIVE 2026-10-25: reporte en UTF-8 correcto, mojibake era consola PowerShell]
  del reporte).
- `helpers.py` ya sin `analisis_lideres.csv`, pero la seccion
  "Acciones lideres" desaparecio del reporte: verificar que no
  rompe el contrato de la tabla Data Freshness.
- SOW-BEAR-CONFIRMATION (contrato v1.10).

**Lecciones.** El grep de C4.5 inicial (sesion 3) habia listado 3
notas; el grep expandido post-Commit A dio 9 ficheros. La leccion
de HANDOFF §6 sobre "auditar el reporte en 2 dimensiones" aplica
tambien al codigo: grep por fichero, no por linea conocida.

### 2026-10-25 (sesion 4) - Cierre VERIFY-LEADER-SELECTION-ROTO

**Objetivo.** Cerrar el primer item tras sesion 3.

**Hecho.**
- `git rm validation/verify_leader_selection.py` (importaba
  `config.index_tickers`, borrado en C4.4; leia `analisis_lideres*.csv`
  no generados desde C4.3).
- Retirados del `daily_run.yml`: step "Verify leader selection"
  (202-203), artefactos `analisis_lideres*.csv` (226-227) y
  `slpm_state.json` (239).
- `git rm outputs/state/slpm_state.json` (fosil sin productor).
- `data_quality.py:233` comentario corregido.
- `settings.py:87-90` bloque `# SLPM COVERAGE` huerfano eliminado.
- `generate_docs.py:214` fallback muerto eliminado.
- `run.py:62` comentario sin `verify_leader_selection`.

**Commits.** `37bca52d` (codigo).

**Pendiente.**
- `C4.5-SLPM-STRINGS-RESIDUALES` (siguiente, con run real + auditoria
  2 dimensiones).
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE` (cosmetico confirmado:
  `leader_breadth` no se usa en el cuerpo de `compute_structural_score`).
- `helpers.py:136` y `run.py:64` (residuales `analisis_lideres`,
  inocuos, van con C4.5).
- SOW-BEAR-CONFIRMATION (contrato v1.10).

**Lecciones.** El test de gobernanza exige commit alcanzable en
`refs/heads/` o `refs/tags/` y evidencia >= 20. No se puede marcar
`FIXED` en el mismo commit que crea el cambio. Secuencia correcta:
commit de codigo -> commit documental con hash.

### 2026-10-24 (sesion 3 continuacion) - C4.4 + bug rotacion + verificacion real

**Objetivo.** Verificar con pipeline real el estado del sistema
holdings-based. Al ejecutar el primer run completo se detectaron 2
fallos que invalidaban el "cierre" de la primera parte de la sesion.

**Hallazgo 1: C4.4 nunca se ejecuto.** El reporte tenia 2 secciones
legacy de indices con mojibake (`## Ãndices`). El sistema legacy de
`index_phase`/`index_leaders` coexistia con el holdings-based.

**Hallazgo 2: rotacion sectorial silenciada.** El `except` de
`sectors_base.py:56` capturaba un `ValueError: not enough values to
unpack (expected 4, got 3)` en `sector_rank_history.py:54`. La tupla
del ranking paso a 3 en C5.1; el consumidor seguia esperando 4. El
CSV `sector_rank_deltas.csv` no se actualizaba desde el 10:44.

**Hecho.**
- C4.4: retirados `index_phase.py`, `index_leaders.py`,
  `indices_intl.py`, `config/index_tickers.py`,
  `render_indices_internacionales`, 12 ficheros de test. Adaptados
  `run.py`, `report_generator.py`, `synthesis.py`,
  `test_output_contract_ast.py`,
  `test_render_modules_low_coverage_d29.py`, `generate_docs.py`.
  REGISTRO: 4 items FIXED sin test (test retirado).
- Bug rotacion: unpacking 4 -> 3 en `sector_rank_history.py`. Tests
  adaptados. `traceback.print_exc()` anadido en el except de
  `sectors_base.py`.
- 3 runs completos del pipeline verificados (exit 0, gate 10/10,
  24 writers, cero mojibake, cero secciones legacy, rotacion
  calculada, 6 CSVs holdings con fecha).
- Deudas nuevas en REGISTRO: `C4.5-SLPM-STRINGS-RESIDUALES`,
  `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE`,
  `VERIFY-LEADER-SELECTION-ROTO`.

**Commits.** `8819ba70` (fix rotacion), `b8f0a4b9` (artefactos del
run), `f7fdbeba` (C4.4).

**Lecciones.**
- `pyflakes .` sobre todo el repo, no por fichero.
- Here-strings `@"..."@` de PowerShell rompen los triple-quote de
  Python con backticks. Usar `@'...'@`.
- Auditar reporte en 2 dimensiones: headers `^## ` + menciones
  textuales + mojibake + N/D. La auditoria por headers dejo pasar 11
  menciones a SLPM.
- Cuando un except captura y sigue, verificar si el problema esta
  silenciado. Bug de rotacion duro desde C5.1 por un except con
  print + skip(NO_DATA).

**Pendiente.**
- C4.5: limpiar strings SLPM residuales en reporte.
- `structural_engine.py::compute_structural_score` zombie
  `leader_breadth=0.5`.
- `validation/verify_leader_selection.py` roto desde C4.3/C4.4.
- SOW-BEAR-CONFIRMATION (contrato v1.10).
- Items observacionales con fecha (H5.3, etc.).

**Proximo paso sugerido.** C4.5 con verificacion por pipeline real.

### 2026-10-24 (sesion 3) - Plan holdings: demolicion + construccion

**Objetivo.** Ejecutar el plan 63: retirar el sistema de lideres por
sector basado en fase ETF (SLPM, stock_leader, index_leaders,
classify_wyckoff_phase) y construir el sistema Wyckoff holdings-based
con 6 CSVs descargables.

**Hecho.**
- **Plan 63** en `docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md`.
- **C1** (`83187f8c`): modulo `indicators/wyckoff_holdings_report.py`.
  6 tablas (usa_sector_acc/dist, usa_index_acc/dist, eu_index_acc/dist).
- **C2.1** (`a3a63766`): 6 OutputSpec nuevos. Wrapper
  `src/pipeline/wyckoff_holdings.py`. Cableado en `run.py`.
- **C2.2** (`9b5b34cd`): renderers en `src/report/wyckoff_holdings.py`.
  2 secciones nuevas en el reporte.
- **C3.1** (`0c5b7fb3`): retirada SLPM. Borrados `slpm_v12.py`,
  `state_machine.py`, `state_transition.py`, `src/pipeline/slpm.py`,
  `src/report/slpm.py`. `SLPM_WEIGHTS`, `SLPM_EXPECTED_LEADERS`,
  `_forzar_lideres_slpm` eliminados.
- **C3.2** (`676ab956`): Validation Gate adaptado. Firma
  `slpm_v12_data` -> `wyckoff_holdings_data`. 3 checks sustituidos
  (SLPM v1.2 -> Wyckoff holdings CSVs; Opportunity Map -> Cobertura
  holdings; Anti-Double-Counting -> Sanity holdings). Gate 10/10.
- **C3.3** (`cf7659fb`): `generate_docs.py` limpio.
- **C4.1** (`753b7f22`): `indicators/stock_metrics.py` extraido de
  `stock_leader.py`.
- **C4.2** (`2ede0215`): `src/pipeline/stocks_data.py` extraido de
  `leaders.py`.
- **C4.3** (`4bc45fd3`): borrados `stock_leader.py`, `leaders.py`,
  `leader_representativeness.py`, `sector_leader_divergence.py`,
  `sector_annotations.py`. Registry 26 -> 24 writers.
- **C5.1** (`8d9b90a0`): fase ETF retirada del ranking. Tupla de 4 a 3.
  `classify_wyckoff_phase` fuera de `sector_regime.py`.
  `sector_regime_matrix.py` deriva `structure_positive` de holdings.
- **C5.2** (`c47cb420`): `classify_wyckoff_phase` y familia borradas de
  `wyckoff_v1.py`. 24+4 tests de contrato v1 borrados.
- **Fix** (`96276d05`): import huerfano en
  `test_struct_score_sector_indices.py` que pyflakes de CI detecto (en
  local solo se corrio pyflakes por ficheros tocados).
- **C6.1** (`a100d282`, `813ad19f`, `0a0597b3`, `a3471303`): HANDOFF,
  00_ARRANQUE, 02_ARQUITECTURA, 62 actualizados.
- **C6.2** (parcial): 5 items nuevos en REGISTRO, rama
  `feature/wyckoff-cascade` borrada, ESTADO_SISTEMA regenerado.

**Decision arquitectonica.** El ETF sectorial suaviza la senal; la
deteccion de fase se hace directamente sobre los holdings. 6 CSVs
descargables por sector/indice (USA + EU). Cobertura top-20 por weight
(57.2%-99.0% del peso del grupo).

**Commits clave.** `977ad888`, `bdc80626`, `f1edc145`, `16f2dd8a`,
`83187f8c`, `a3a63766`, `9b5b34cd`, `0c5b7fb3`, `676ab956`,
`cf7659fb`, `753b7f22`, `2ede0215`, `4bc45fd3`, `8d9b90a0`,
`c47cb420`, `96276d05`, `a100d282`, `813ad19f`, `0a0597b3`,
`a3471303`, `ebae6fe9`.

**Pre-registros fallidos (heredados).** Sesion 3 no anade nuevos. Los
60 y 61 siguen vigentes.

**Limitaciones documentadas.**
- `current_phase_basis` no va al reporte (solo al CSV).
- `ACCUMULATION=0` en produccion (transitorio por diseno).
- IWM fuera (top-20 = 6.2% de 1977 holdings).
- Cobertura USA 231/504 holdings en parquet.
- Validation Gate no reproducible standalone.
- TWST en CONFLICT (ambiguedad real).
- `sector_regime_matrix.py`: `structure_positive` pasa de fase ETF a
  holdings (C5.1); validacion pendiente.

**Pendiente.**
- C6.2 residual: `04_HISTORICO.md` (esta sesion).
- PSY/PS con pre-registro.
- Renderizar evidencia pasiva (mSOW/mSOS/LPSY/LPS/ST).
- Contrato v1.10 (SOW-BEAR-CONFIRMATION).
- Auditoria externa del sistema holdings-based.

**Proximo paso sugerido.** Pegar `HANDOFF.md` + `00_ARRANQUE.md`.
C6.2 (04_HISTORICO) primero.

**Lecciones operativas.**
- `pyflakes .` sobre todo el repo, no por fichero.
- Antes de cambiar firmas de tuplas/dicts, enumerar TODOS los mocks
  via AST, no regex de texto.


### 2026-10-24 (sesion 2) - Cierre SOW universal + reconstruccion completa sistema Wyckoff

**Objetivo.** Cerrar el frente SOW universal tras 7 experimentos
(Bloque A) y reconstruir el sistema Wyckoff por stock con acumulacion
y distribucion cableadas al pipeline (Bloque B).

**Hecho (Bloque A).**
- `SOW-V6` -> WONT_FIX. Senal real vs placebo en R3000 (3423 tickers,
  lift +0.123 vs -0.077, 21.5 sd), pero no universal (4/9 folds
  positivos, IC95 entre anos [-0.058, +0.303] cruza 0).
- `SOW-BEAR-CONFIRMATION` abierto (IN_PROGRESS). Requiere contrato
  v1.10 firmado por auditor externo.
- Evidencia en `docs/auditoria/wyckoff/51-56`.

**Hecho (Bloque B).**
- **Decision arquitectonica**: dejar de usar ETFs sectoriales e
  indices como fuente de fases Wyckoff. Cascadas por stock con
  deteccion de acumulacion y distribucion, orquestadas.
- **15 detectores en codigo**:
  - Distribucion: BC, AR, SOW + auxiliares ST_dist, PSY, mSOW, LPSY,
    false_breakout_up (UT/UTAD).
  - Acumulacion: SC, AR, SOS + auxiliares ST_acc, PS, mSOS, LPS,
    false_breakout_down (Spring).
- **Orquestador** (`wyckoff_orchestrator.py`) con regla simetrica de
  vigencia (rango vigente mientras precio no lo rompio por ningun
  lado). Estados: DIST/MARKDOWN/ACC/MARKUP + FORMING + CONFLICT +
  INSUFFICIENT_DATA. Separa `phase` (actual) y `event_phase`
  (historico).
- **Cableado al pipeline**: `sector_wyckoff_distribution.py` migrado
  de `classify_wyckoff_phase` a `classify_phase`.
- **Reporte**: tabla sectorial con AccF/Acc/Markup/Range/DistF/Dist/
  Markdown/Conflict + N + Cob.
- **Anotacion de titulos sectoriales** con composicion interna de
  holdings: `## Sector: XLK (MARKUP)` -> `*Composicion interna (22
  componentes): 3 Markup (14%), 8 Range (36%), 9 DistF (41%)...*`.
- **Estado en produccion**: 219 RANGE, 53 DIST_FORMING, 17 DIST,
  17 MARKDOWN, 13 MARKUP, 7 ACC_FORMING, 1 CONFLICT (327 stocks).

**Pre-registros fallidos (no ocultos).**
- `60_current_phase_rule_v1.md`: regla de vigencia unidireccional.
  Fallo C1 (18.6% < 20%) en holdout S&P 500. No se ajusto.
  Reemplazada por regla simetrica (definicion, no umbral).
- `61_utad_spring_pending_v1.md`: UTAD_PENDING/SPRING_PENDING. Fallo
  A3 (26 vs 3, ratio 8.67x > 5x por asimetria natural del mercado).
  No se ajusto.

**Limitaciones documentadas.**
- PSY/PS no disparan con umbrales actuales (0/50 en smoke real).
  Deuda de calibracion, no se recalibro post-hoc.
- `sector_regime.py` sigue usando `classify_wyckoff_phase` legacy
  para el titulo del sector. Anotacion de composicion mitiga.
- mSOW/mSOS/LPSY/LPS/ST estan en `evidence` pero no se renderizan.

**Commits clave.** `156738cd`, `6a1eff10`, `c45ac6ab`, `3c0b13f3`,
`37398f54`, `1d502cac`, `a8bd235a`, `17ecbe1d`, `4edba814`,
`415487ff`, `bd81abe4`, `bfe491bf`, `40ac2d07`, `64432648`.

**Pendiente.**
- Migrar `sector_regime.py` al nuevo orquestador (decision de
  producto).
- Renderizar evidencia pasiva en el reporte.
- Calibrar PSY/PS con pre-registro.
- Auditoria externa del sistema completo.
- Contrato v1.10 (SOW-BEAR-CONFIRMATION).

**Proximo paso sugerido.** Retomar en nuevo chat con
`docs/handoffs/HANDOFF.md` + `docs/auditoria/wyckoff/62_sistema_completo_2026-10-24.md`
pegados al inicio.

### 2026-10-23 - Verificacion scheduled target 2026-10-22 + auditoria autonomia

**Objetivo.** Verificar en CI real los 4 fixes criticos de la sesion
2026-10-22 (target 2026-10-22 en scheduled). Auditar autonomia
completa del sistema. Documentar.

**Hecho.**
- **Scheduled target 2026-10-22 verificado**: run `39884468711`
  (23-10 01:31 UTC, slot 22-10 23:17 UTC, delay 2h14m, `run-system:
  success`).
  - Liquidez: `NEUTRAL (Signal Consistency: 90%)`, no N/A.
  - `sector_correlation_summary` (P7): receipt OK, 2026-10-22.
  - `pcr_history` (side effect fix): receipt OK, 2026-10-22.
  - `cboe_vix3m` en origin/main: 4298 filas, 0 no-bursatiles.
  - `data_quality.date = 2026-10-22` y `receipt_audit.target =
    2026-10-22`. Contrato temporal integro. 20 writers, 18 OK + 2
    no-OK esperados (sector_rank_deltas NO_DATE_CHECK,
    sector_flow_characteristics LAGGED).
- **Auditoria autonomia (REP-AUDIT-20261022-AUTONOMY-AUDIT)**:
  FALSE_POSITIVE. Verificado con datos reales:
  - FRED caido: `download_series` devuelve `None`, no reescribe CSV.
  - `pipeline_gate` usa probe live (no parquet local). Parquet stale
    + Yahoo fresco = READY correcto.
  - 5/5 slots daily_run success 22-10; 2/2 update_macro_manual success.
  - Cadencia por fuente dentro de ventana (diaria≤3d, semanal≤10d,
    mensual≤60d).
- **5 huecos auditoria dia anterior investigados**:
  - Health Check 19-oct: FALSE_POSITIVE (regla antigua FAIL con 2/5).
  - ETF/European Holdings 1-oct: FALSE_POSITIVE (fallaron 6h y 4h
    antes del commit `1bead8ad` con `sys.path.insert`).
  - Index Holdings 1-oct: FIXED `e54f95b6` (refactor rows.append
    defensivo, `All arrays must be of the same length`).
  - cron_slots WARN 3/5: FALSE_POSITIVE (delays GitHub 2-4h, regla
    actual clasifica WARN correctamente).
  - manifest close_nan_last: FALSE_POSITIVE (WBD delisted + 19 BME).
  - iae_section WARN: FIXED `e54f95b6` (heading flexible 13F).

**Commits.** `e54f95b6`, `cdb93364`, `5bf4b4b3`, `2b5b5341`,
`41855de6`.

**Pendiente.** Ningun bug abierto. Ver HANDOFF.md.

**Proximo paso sugerido.** Retomar en nuevo chat con
`docs/handoffs/HANDOFF.md` pegado al inicio.

### 2026-10-22 - Auditoria externa #4 + 11 bugs reales + gobernanza

**Objetivo.** Cerrar auditoria externa #4 (P1-P8). Derivado: 11 bugs
reales detectados en revision cruzada (reporte, liquidez, Wyckoff,
parquets). Todos con test + verificacion en produccion local.

**Hecho.**

*Auditoria externa #4 (P1-P8).*
- P1 (RS nominal): FALSE_POSITIVE. rs_z usa rs_mom, invariante a escala.
- P2 (cobertura XLK): FALSE_POSITIVE. Denominador = universo completo
  del ETF, no top-20.
- P3 (XLF -5.0): BUG REAL. Umbral aplicaba al bruto, render redondeado.
  FIXED, 9 tests.
- P4 (IHR): FIXED. "IHR (Index/Equity PCR Ratio, proxy)", formula,
  nota de no-institucional. 4 tests.
- P5 (NIPC): FALSE_POSITIVE, H14.
- P6 (temporal_coherence): FIXED. Nota de alcance.
- P7 (rolling window gaps): FIXED en `8c40f7e6`. Detalle abajo.
- P8 (Voyager N-PORT): FIXED. Nota de residuales.

*FLS WALCL/CP.* WONT_FIX tras diagnostico estadistico. WALCL
`-z(pct_change_252)` |z|>5 12.3%; alternativas peores. CP `z(nivel)`
|z|>5 4.0%; `pct_change_252` mejora |z|>5 pero empeora `z_ult`
(+1.45 -> +2.19). Item FLS-COMPONENT-DESIGN FIXED.

*Bug 1 - router liquidez (`d405d539`).* `router.get_fed_data`
priorizaba FRED API (lag 5d) sobre CSVs `data/macro_manual/` (bot
diario). Resultado: `Liquidez Real: N/A` en reporte, contradiciendo a
FLS. Fix: CSV-first, API-fallback. Verificado: `NEUTRAL 90%`.

*Bug 2 - v1 contamina state v2 (`83f79a93`).* `regimes/liquidity.py`
(v1) escribia a `liquidity_state.json` con esquema incompatible. Los
tests de v1 sobreescribian el state de produccion. Fix: v1 ->
`liquidity_v1_state.json` (gitignored).

*Bug 3 - docstring v2 stale (`83f79a93`).* v2 declaraba "NO PRODUCCION"
siendo productivo desde 14-10. Reescrito.

*Bug 4 - ref rota (`83f79a93`).* `finalize.py:110` citaba item
inexistente `REP-AUDIT-20261017-capa4-commitB`. Fix -> `CAPA4`.

*Bug 5 - verifier acepta SHA huerfanos (`236c2a64`).* `git cat-file -e`
acepta objetos huerfanos. `for-each-ref --contains` para validar
reachability. Test con commit huerfano real.

*Bug 6 - v2 liquidez solo API (`5375f58b`).* `effr.csv` no existia;
v2 leia solo FRED API (lag 2d). Fix: `update_macro_manual.py` anade
EFFR; `_fetch_liquidity_v2_data` CSV-first con interseccion;
`effr.csv` con 2233 filas. State pasa a coincidir con CSV.

*Bug 7 - `-0.0` en `_score_continuo` (`5375f58b`).* `score(0.0) = -0.0`
visible en state. Fix: `+ 0.0`.

*Bug 8 - `compute_pcr_signals` side effects (`9a2bddcc`).* Escribia
`pcr_history.csv` sin pasar por `persist`/registry, sin receipt.
Diagnosticos del dia modificaban el working tree. Fix: flag
`persist_history=False` por defecto; `market_data.py` pasa `True`.

*Bug 9 - sector_correlation P7 (`8c40f7e6`).* `iloc[-window:]` sobre
serie con huecos: ventana cubre mas dias naturales que N sesiones. Fix:
nueva `get_trading_days` en `src/market_calendar.py` + reindex antes
del slice.

*Bug 10 - cboe_vix3m no-bursatiles (`7fcbe05f`).* Unico parquet de
produccion trackeado. 2 filas en `2010-12-31`, `2021-12-31`
(observacion New Year en viernes). Fix: `_post_filter_non_trading_days`
en `update_cboe.py` + parquet limpiado (4299 -> 4297, manifest
actualizado). Verificado en `origin/main`: 0 no-bursatiles.

*Bug 11 - Wyckoff v1.8 B5/B3 (`1cc8387a`).* B5: spring/SOS con
`min_periods=1` disparaban SOS espurios (XLF 7 en primeras 20
sesiones). Fix: `min_periods=EVENT_WINDOW` + patron H7
(`dropna().rolling().reindex()`). B3: `max_age_m` default 10 vs
contrato v1.9=30. WONT_FIX razonados: B2 (P7 en
`_trend_component`/`_atr_normalized`/`wyckoff_stability`: medias
moviles requieren dropna, no reindex); B4 (`fillna(Close)`/`fillna(0)`
es contrato v1.8, protegido por `test_regresion_i2_nan_internos_no_degenera`;
medicion 0/2512 filas H=L=C en `market_data.parquet`).

*Gobernanza.* FLS-COMPONENT-DESIGN OPEN->FIXED; LIQ-V2-DATA-SOURCE
OPEN->SUPERSEDED (duplicado del FIXED); GOVERNANCE-SHA-REACHABILITY
nuevo FIXED; CBOE-VIX3M-NON-TRADING-DAYS nuevo FIXED;
WYCKOFF-P7-B4-B5-B3 nuevo FIXED. Plan ROLLING-WINDOW corregido (error
factual: `sector_correlation_summary` si existe).

*Docs.* Trampas PowerShell + catalogo EOL/BOM (commit `6dcf637c`);
lecciones 1-9 en `01_METODO §6ter` (`e5bb6f43`, `d9e2e73c`,
`e35b314c`).

**Commits.** bc66ba6f, 3a3b06a9, 5ad74e56, 07854913, 5a1e0c37,
2686b154, a4f13d1c, 6dcf637c, 735ee713, e5bb6f43, 8b97089b,
d9e2e73c, d405d539, 9df5e7b8, e35b314c, 5375f58b, 2ec8abcd,
9a2bddcc, eae74527, ce9f62a0, 8c40f7e6, b6fba669, 7fcbe05f,
e26b7c7b, 1cc8387a, 1bb56473.

**Pendiente.**
- Verificar run scheduled target 2026-10-22 (slot 23:17 UTC + delays).
- DISTRIBUTION apagada: sesion dedicada de re-diseno SOW.
- Wyckoff 5b.X / SOW v6: bloqueados externos.
- HANDOFF automatico: deuda estructural.

**Proximo paso sugerido.** Verificar scheduled 2026-10-22 con
`verify_run_20261022.ps1` (criterios: liquidez NEUTRAL no N/A,
`cboe_vix3m` sin no-bursatiles, receipt `pcr_history`).

### 2026-10-21 - Auditoria externa #3 + REVIEW-EXTERNAL cerrado

**Objetivo.** Atender los 8 hallazgos H1-H8 del reporte 2026-10-20
(run 39528211520) y cerrar REVIEW-EXTERNAL.

**Hecho.**
- H1-H8: 7 FIXED con nota o renombrado, 1 FALSE_POSITIVE (H3
  sesgo de seleccion). H1 = Struct Wyckoff vs struct_score. H2 =
  edades runtime vs CSV. H4 = SLPM 0/5 trazabilidad. H5 = XLK
  MARKUP -5.52% (struct_score mide tendencia de fondo). H6 =
  nota P2.2 XLE siempre visible. H7 = orden por Score, no por
  Struct Wyckoff. H8 = FRED NEUTRAL -> ABUNDANTE, FALSE_POSITIVE.
- Notas derivadas de constantes (`STRUCTURAL_WEIGHTS`,
  `STRUCT_STRONG`, `FLOW_PROXY_WEIGHTS`, `EXPECTED_SECTOR_COUNT`,
  `TOP_N_CANDIDATES`, `TOP_N_LEADERS`). Protegido con
  `tests/test_report_notas_derivadas.py` (4 tests monkeypatch).
- REVIEW-EXTERNAL FIXED. FLS-SOFR confirmado en prod (run
  39528211520, gate 10/10, FLS 63/100).

**Commits.** bd32577a, ec0e7b8b, 100b55d6, 714fba5a, 9ebfacdd,
fd9e202d, c073f2a5, 10cf8ea5.

**Pendiente.** Auditoria externa #4 (nueva, 22 oct).

**Proximo paso sugerido.** Recibir auditoria #4 y atenderla.

### 2026-10-20 (tarde) - Dos auditorias externas + FLS

**Objetivo.** Atender dos auditorias externas del reporte 2026-10-19.
La primera: 14 items (4 criticas, 10 sospechosas). La segunda: 7 items
(Flow Proxy, trazabilidad temporal, sintesis credito).

**Hecho.**
- 14 commits. 11 items FIXED con test. 7 falsos positivos documentados
  en REVIEW-EXTERNAL.
- B1/B3/B4 (0aec4ce5): sintesis credito sin atribuir HIGH_STRESS al
  credito. Nota Flow Proxy vs Flujo Primario. Struct -> Struct Score.
- B2 (41b1c8bf): veto por variables materiales en evidence_matrix.
  XLC con price<0 y breadth<0 ya no sale FAVORABLE.
- A2 (47071a51): health_check no da WARN en trimestrales con dispatch OK
  dentro de max_days.
- A1 (9f6afa26): post_receipt_audit status LAGGED cuando receipt.date
  != target_session. Primer LAGGED real sera en run 2026-10-20.
- P1 (eaed52a7): nota Flow Proxy tabla (composite ETF) vs matriz
  (mediana componentes). Son metricas distintas y pueden discrepar.
- P2 (68e4e40c): nota temporal en provenance, recuento votos P/N/0 en
  matriz, nota regla Wyckoff.
- C8 (7475108f): nota FLS saturacion por nivel. Item FLS-COMPONENT-DESIGN
  OPEN.
- C10 (33a12a10): ratio |Z| extremo destacado en Cross-Asset.
- C6 (878ad4fc): nota materialidad Europa (media simple, no AUM).
- FLS (38f17674): SOFR como spread SOFR-IORB, no nivel. Diagnostico:
  |z|>5 17.7% de sesiones con nivel, 5.7% con spread. FLS 79 -> 63.
  stressed 3/5 -> 2/5.
- Falsos positivos: C1 (Wyckoff +1), C2 (Flow Proxy 30% WLS),
  C3 (SLPM NO_LEADERS), C4 (datos desfasados), C5 (NIPC 13F),
  C7 (spread 10Y-3M), C9 (IPI).

**Commits.** 0aec4ce5, 41b1c8bf, 47071a51, 7f329d07, af0be35d,
9f6afa26, 7475108f, 33a12a10, 878ad4fc, eaed52a7, 2ddc7488,
68e4e40c, ec641043, 38f17674.

**Pendiente.**
- Cerrar REVIEW-EXTERNAL tras run 2026-10-20 verde.
- Diagnostico WALCL/CP (FLS-COMPONENT-DESIGN).
- RECEIPT-AUDIT-LAG (observacion 4 semanas).
- CRON-SLOTS-OBS y SCHEDULER-DELAYS (2026-11-10).

**Proximo paso.** Run 23:17 UTC (target 2026-10-20). Verificar
run-system success, receipt, receipt_audit con primer LAGGED real.

### 2026-10-20 - Contaminacion fechas + certificacion prod

**Objetivo.** Revisar el scheduled del lunes 19-10. Diagnosticar el fallo
del health check semanal (FAIL cron_slots). Auditar contaminacion de
fechas en outputs/history/*.csv.

**Hecho.**
- Health check (f3efa062): regla cron_slots 3+missing -> 1-4 WARN, 5/5
  FAIL. Issue #6 cerrado.
- Auditoria de fechas: 4 bugs reales + 7 falsos positivos.
  - european_coverage (6d2278c3 + migracion da093a05): _ref_to_date
    delegaba en run_date. 459 filas contaminadas. 1734 -> 1326 tras
    migracion.
  - data_quality rama europea (f02cebd8): age contra now, no contra
    sesion. El fix del 18-10 solo cubrio rama general.
  - monitor_double_counting (14838307 + migracion 5f464729): date=
    datetime.now().isoformat(). 225 -> 123 tras migracion con
    run_timestamp.
  - slpm_history (5ebe1c62): fosil desde 2026-07-17, sin writer activo.
    CSV eliminado + artifact retirado + metrica structural_vs_breadth
    eliminada del monitor.
- Certificacion prod: target 2026-10-19 cubierto por run 39355608824
  (20-10 01:50 UTC, delay 2h33m). Pipeline completo. Receipt emitido.
  receipt_audit: 18 OK + 1 NO_DATE_CHECK, 0 MISSING. data_quality con
  date=sesion (sin findes).

**Commits.** f3efa062, 6d2278c3, da093a05, f02cebd8, 14838307,
5f464729, 5ebe1c62, c98c55f4, 2ba73626.

**Pendiente.**
- Health check lunes 26-10: primera lectura de check_receipt_audit y
  primera aplicacion de la regla nueva de cron_slots.
- Delays del scheduler GitHub (2-7h): observar 2-3 semanas antes de
  considerar mitigaciones (repository_dispatch externo, cambiar cron).
- 3 WARN trimestrales (update_sector_holdings/index/european, fallo
  1-oct): investigacion separada.

**Proximo paso sugerido.** Observar el scheduled del martes 20-10 (ya
cubierto por run 39355608824) y el del miercoles 21-10 con todos los
fixes activos. Confirmar 0 MISSING en receipt_audit y 0 filas no
bursatiles en los 4 CSVs del pipeline.

### 2026-10-18 - Investigacion frescura + auditoria externa + audit post-receipt

**Objetivo.** Investigar y verificar que el sistema se nutre con datos
frescos, fiables y consistentes de todas las fuentes de manera
automatica. Cerrar deudas detectadas. Resolver puntos de auditoria
externa del reporte 2026-10-17.

**Hecho.**

Fase A - Auditoria externa (4 puntos del analista):
- IPS/IPI proxy de mercado, no IPC/PPI: `8eaecb0a`. Texto nuevo en
  `volatility_mte.py` + 2 tests. Los 3 FALSE_POSITIVE (NIPC, SLPM,
  liderazgo) documentados como falsos positivos porque el reporte ya
  los declaraba.

Fase B - Deuda validation/ en workflows:
- `cafc77d8`: quitar validation/ del pytest en ci_tests y daily_run.

Fase C - Deuda data_quality.csv (date no bursatil):
- `4ade7384`: compute_data_quality escribe `date` =
  last_expected_market_date(reference_date). `age` contra sesion.
- `5ef44486`: migracion historica (155 filas, 9 fechas, 593->464).
  Backup, dedup (date, source), validacion bursatil.
- 11 tests nuevos en `test_data_quality_date_sesion_bursatil.py`.

Fase D - Deuda receipt (no detecta CSV incompleto):
- `400e6c9b`: `scripts/post_receipt_audit.py` compara
  `writer_receipts.json` vs `outputs/history/*.csv`. Status
  SKIP/NO_DATE_CHECK/OK/MISSING/CSV_MISSING. No bloquea. Append a
  `receipt_audit.csv`. `daily_run.yml` step `if: always()`.
  `health_check.py::check_receipt_audit()` lee el CSV y WARN si
  hay MISSING en ultimas 7 sesiones.
- 13 tests nuevos.
- `212011f4`: quitar imports unused (pyflakes).

**Commits.** `8eaecb0a`, `cafc77d8`, `4ade7384`, `5ef44486`,
`400e6c9b`, `212011f4`. Rebase sobre `0419c5fa` (bot FRED) en un
punto intermedio.

**Pendiente.**
- Scheduled lunes 2026-10-19 23:17 UTC: primer run con todos los
  fixes activos. Observar `receipt_audit.csv` en artifact.
- Health check lunes 2026-10-26: primera lectura de
  `check_receipt_audit`.
- Deudas aceptadas (no accionables): SSGA sin gate (fetch forzado),
  reporte local en `outputs/report/` (gitignored, auditar via
  artifact), `run.py:48` con `datetime.now(Madrid)` (excepcion
  justificada por `is_session_closed`, no viola contrato porque es
  fecha de ejecucion, no de observacion).

**Proximo paso sugerido.** Verificar scheduled lunes. Si verde,
observar primer `receipt_audit.csv` y confirmar 0 MISSING. Si hay
MISSING, investigar antes del 26-oct.
### 2026-10-18 - GEN-ESTADO-*: cierre de las dos deudas del generador

**Objetivo.** Cerrar los dos items abiertos al cierre de la sesion
anterior (GEN-ESTADO-VALIDATION-VACIO y GEN-ESTADO-TESTS-NOT-FAILCLOSED),
adelantandolos al mismo dia en lugar de esperar al scheduled del lunes.

**Hecho.**
- GEN-ESTADO-VALIDATION-VACIO (72659a9f): retirar validation/ del
  cmd de pytest. Test estatico AST que verifica el cmd del run().
- GEN-ESTADO-TESTS-NOT-FAILCLOSED (cb3253d3): propagar rc de
  section_tests() y section_integridad() a main(). Snapshot se
  escribe siempre; exit code 1 si tests/compileall/pyflakes fallan.
  4 tests hermeticos con tmp_path + monkeypatch.
- Snapshot ESTADO_SISTEMA.md sin cambios (raw 3739).

**Commits.** 72659a9f, cb3253d3.

**Pendiente.**
- Scheduled lunes 2026-10-19 23:17 UTC (certificacion prod v1.1+v1.2).
- C2, H5.3, Wyckoff 5b.X, SOW v6: sin accion antes del lunes.

**Proximo paso sugerido.** Verificar scheduled del lunes. Si verde,
marcar VERIFIED_IN_PRODUCTION los items que corresponda.

### 2026-10-18 - Regenerar ESTADO_SISTEMA + items del generador

**Objetivo.** Cerrar la discrepancia entre ESTADO_SISTEMA.md (3641
passed) y la realidad de la suite (3730 con filtro / 3739 sin filtro),
detectada al arrancar la sesion. Aprovechar para documentar dos deudas
del generador.

**Hecho.**
- Regenerar `ESTADO_SISTEMA.md`: 3641 -> 3739 (pytest sin filtro). La
  cifra es correcta; la 3730 del HANDOFF es la del filtro
  `-m "not integration_real_data"`. Nota aclaratoria anadida a HANDOFF
  (lineas 15 y 118).
- Abrir `GEN-ESTADO-VALIDATION-VACIO`: `generate_estado_sistema.py`
  invoca `pytest tests/ validation/` pero `validation/` no contiene
  `test_*.py`. Ruido, no bug.
- Abrir `GEN-ESTADO-TESTS-NOT-FAILCLOSED`: `section_tests()` no aborta
  con `rc != 0`; publica snapshot aunque la suite falle.
- HANDOFF §1 actualizado: 73 -> 75 items (2 OPEN nuevos).
- CI verde sobre HEAD final.

**Commits.** `2ca11b5a`, `80ae528b`. Push a `origin/main` en
`80ae528b`. CI run `39083891160` success.

**Pendiente.**
- `GEN-ESTADO-VALIDATION-VACIO`: retirar `validation/` o poblarla.
  Post-scheduled 2026-10-19.
- `GEN-ESTADO-TESTS-NOT-FAILCLOSED`: anadir comprobacion `rc != 0`.
  Post-scheduled 2026-10-19.
- Scheduled lunes 2026-10-19 23:17 UTC: primer run con
  `target_session=2026-10-19` (certificacion prod v1.1 + v1.2).

**Proximo paso sugerido.** Esperar scheduled del lunes. Si verde,
marcar items `GEN-ESTADO-*` FIXED en la misma sesion.

### 2026-10-18 - Cierre residuales dictamen + struct en Rankings/Indices

**Objetivo.** Cerrar los 5 residuales del dictamen original (H5, H7,
H10/H19, H11/H12, H16) y publicar struct_score en Rankings Sectoriales e
Indices (opcion B aprobada) para evitar lecturas binarias de fase Wyckoff.

**Hecho.**
- H16 (`9d8d8cf9`): Financial Stress e Inflation Pressure siempre con N/5.
- H7 (`3d8c9689`): ETFs fuera del denominador de cobertura. 305/340=0.897
  -> 305/329=0.927. WARN falso eliminado.
- H5 (`37e1f02e`): pos_change = net_position.diff() en CFTC. Identidad.
- H10/H19 (`e7b12feb`): "Data Conf" -> "Conf. Frescura". Aclara que solo
  expresa frescura, NO cobertura.
- H11/H12 (`caa176d1`): provider lee FILING_DATE. Render publica Fecha
  observacion + presentacion. Dispatch 39074246538 confirmo FILING_DATE
  al 100%.
- Struct (`a226f556` + `22392813`): columnas Struct en Rankings e Indices.
  XLK +0.45 MARKUP, ^NDX +0.30 RANGE a 0.002 del umbral, XLP +0.18 RANGE.
- E2E run.py verificado: los 5 fixes en el reporte real + Struct.

**Commits.** `9d8d8cf9`, `3d8c9689`, `37e1f02e`, `e7b12feb`, `caa176d1`,
`a226f556`, `22392813`.

**Pendiente.**
- Certificacion scheduled lunes 2026-10-19 23:17 UTC.
- H5.3, C2, Wyckoff 5b.X, SOW v6: sin cambios.

**Proximo paso sugerido.** Verificar scheduled del lunes. Comparar reporte
del lunes con el del sabado: cambios esperados (Conf. Frescura, Expansion
en Cross-Module, Struct en Rankings/Indices, N-PORT dos fechas, CFTC Pos
Change identidad) deben ser coherentes con los commits listados.

---

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
