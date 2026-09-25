# ESTADO_DECLARADO - Modulo IAE

Estado declarado de fases, prohibiciones y hallazgos del modulo IAE.
Documento vivo. Se actualiza cuando cambia el estado.

**Actualizado:** 2026-09-24
**HEAD de cierre:** ver `ESTADO_SISTEMA.md` (se regenera con cada commit).
**Estado del sistema:** `ESTADO_SISTEMA.md`.

---

## 1. Fases

| Fase | Estado |
|---|---|
| FA-1 (ingestion + schema) | CERRADA |
| FA-2 (identity + relationships) | CERRADA |
| NIPC (implementacion) | CERRADA |
| Fase A - bloqueantes dictamen v1 (B1, B2, B3, B4) | CERRADA |
| Fase B - criticos dictamen v1 (C1-C18) | CERRADA |
| Fase C - funcionales (C8, C9, C10) | CERRADA |
| Fase D - reenviar al auditor externo | PENDIENTE |
| Fase E - integracion a `run.py` | CERRADA 2026-09-23 |
| Fase F - automatizacion GitHub | CERRADA 2026-09-24 |
| Fase G - automatizacion de mappings (catalogo + crosswalk) | CERRADA 2026-09-24 |

Dictamenes externos aplicados al modulo IAE (todos cerrados):
v2 (8 puntos), v3 (6), v4 (11), v5 (GATE 1 nomenclatura + GATE 2
identidad dual), v6 (auditoria empirica A.1), v7 (cierre final
IAE_MAESTRO). Detalle en §2.

---

## 2. Hallazgos cerrados

- H-05 a H-12, B-01 a B-07: ver historial en git.
- Fase A: B1 reconciliacion radar/complemento/full; B2 cadena forense
  (HEAD + sha256 + pandas + timestamp); B3 validacion externa OpenFIGI
  (227/227 match sobre 243 CUSIPs resolubles por OpenFIGI, 16 sin hit
  no-USA); B4 riesgo PIT declarado (§13.9).
- Fase B: C5 tabla universos, C6 aviso §4, C7 cadena contractual,
  C11 semantica NIPC, C12 frase eliminada, C13 invocadas != verificadas,
  C14 27 listados + 15 ausentes = 42 ficheros (criterio AST), C15 rename crosswalk, C16 nota 14.3.x,
  C18 coherencia numerica.
- Fase C: C8 reporting_dedup diagnostico separado, C9 stats observables,
  C10 coverage_available + coverage_quality.
- Dictamen v2: HEAD unificado via ESTADO_SISTEMA; 83% -> 81%;
  4 -> 6 ficheros; censo 27+15=42; C9/C10 en E2E; OKE via etf_holdings
  TEMPORAL_UNVERIFIED.
- Dictamen v3 (auditor externo, 2026-09-23): 6 puntos cerrados.
  Cobertura real sobre los 759 tests: 92% (antes declarado 81% sobre
  los 557 seleccionados por -k). Contractual vs PROXY separado en
  IAE_MAESTRO §10.7. Umbral 0.95 etiquetado como operativo (gobernanza)
  en §10.5. Fail-closed denom=0 en coverage.py: coverage_quality
  UNAVAILABLE, no PARTIAL (fix + test de regresion). Nota C9 en §12.3
  (E2E no ejercita exclusion; validado por tests con fixture). §5.2
  reformulado sin prediccion. Censo IAE 758 -> 759 tests (42 ficheros).
  Commits: 89b98e9, 97dd12d, 373c75d.
- Dictamen v4 (auditor externo, 2026-09-23): 11 puntos.
  3 bloqueantes cerrados: cobertura 92% reproducible via
  scripts/iae_coverage.py; separadas cobertura catalogo (99,17%) y
  cobertura TARGET construido (100%), con endogeneidad declarada en
  §13.5; §13.7 retitulado a "Alcance de posiciones del NIPC" con
  filtro §10.1 explicito.
  7 importantes cerrados: ref cruzada §10.7 (§10.3 -> §10.5); §13.3
  error historico (valores 20/31/31/55/64/79 vs actuales 87/89/93/91/98/96);
  frase OpenFIGI reformulada (evidencia sobre la muestra, no sobre el
  universo); invariante C9 (n_received = suma disjunta ordenada);
  fail-closed Σw(TARGET_PAIRWISE)=0 explicito en §10.5; reporting_dedup
  35 funciones AST / 64 casos pytest; cabecera con HEAD operativo +
  snapshot de mediciones (§2.1, §12) en 7caa86b.
  1 importante diferido: evidencia externa OpenFIGI no archivada con
  hash/timestamp; registrado como deuda activa en §3.
  Commits: 4e6b012, 7a4c38a, a2c64b5.
- Dictamen v4, punto 7 (evidencia externa OpenFIGI): CERRADO 2026-09-23.
  Script `iae_validate_crosswalk_openfigi.py` versiona los artefactos
  (_input.json, _raw.json, _summary.txt, HASHES_*.txt) con timestamp UTC,
  parametros, hashes y script_version. Fix str/bytes latente en el
  bloque de escritura de HASHES. 5 tests con fixture mock cubren la
  generacion de artefactos (verificado: script_version en input/summary/
  hashes). Commits: 0d5b3ab, 47cba12.
- Dictamen v5 (auditor fresco, 2026-09-23): 15 puntos. GATE 1 cerrado
  (nomenclatura catalog/TARGET, IAE_MAESTRO §1, §13.5). GATE 2
  verificado empiricamente: 0 fragmentacion por shareClassFIGI en la
  configuracion E2E auditada. §13.10 reformulada: dualidad
  `equity:` / `figi:` = capacidad teorica del resolver, no observada
  en E2E con `figi_lookup=None`. Commits: 66557fb, 6869a85.
- Dictamen v6 (auditor externo, 2026-09-23): GATE 2 cerrado
  empiricamente sobre los 4.780.572 units del E2E. Script nuevo
  `scripts/iae_identity_uniqueness_audit.py`. 0 shareClassFIGI
  fragmentadas, 0 uso de rama `figi:*`, delta NIPC = 0 al normalizar.
  §12.5/§12.6 corregidos para reconocer `canonical_security in
  {equity:*, figi:*}`. 4 puntos menores aplicados (§13.1 OpenFIGI
  suavizado, §13.9 titulado ASSUMPTION, §4.3 source_raw + derivation,
  §1 marca de snapshot). Commits: d3e5fa2, bbbd7d2, 0eeb6a2.
- Dictamen v7 (auditor externo, 2026-09-23): cierre final.
  §12.4 corregido (cobertura interna TARGET 1,0 / catalogo declarado
  99,17%). §13.10 retitulado. §12.6 reformulado con conjunto canonico
  explicito. Commits: 0772272, 8b00303.
- Fase F - automatizacion GitHub (2026-09-24): CERRADA.
  F.1 rutas portables (`IAE_OFFICIAL_DIR` env var override, default
  al repo). F.2 descargador Official List 13(f) + versionado Q4 2025
  y Q1 2026. F.3.a doble ruta SEC (`structureddata` /
  `datastandardsinnovation`) + retry 404/408/429. F.3.b orquestador
  `update_sec_13f.py` (`quarter_to_source_period`,
  `latest_published_quarter`, `ensure_quarter`). F.3.c 51 tests.
  F.4 workflow `update_sec_13f.yml` trimestral + cache parquets 13F +
  restore en `daily_run.yml`. Commits: f514295, 0ffeda2, d4c105d,
  feb77d6, 0a0f042, f2082d7.
- F-IAE-HOLIDAY-01 (2026-09-24): RESUELTO.
  Fix de `_fill_holes_respecting_sessions`: tickers USA en festivos
  NYSE (dia laborable) preservan NaN, sin ffill. Antes se propagaban
  valores de tickers UK (LSE abierto en festivos USA) mezclados en
  lotes Yahoo mixtos. Saneamiento del parquet historico: 50 celdas
  limpiadas (5 festivos × 10 tickers USA). Commit: 2423526.
- F-IAE-CRON-01 (2026-09-24): RESUELTO. SUPERSEDED por F-IAE-CRON-02 (2026-09-25, ver abajo).
  Cron de `daily_run.yml` movido de `0 4 * * *` a `0 23 * * *` UTC.
  Motivo: con retraso observado de ~5h, la ejecucion real caia a
  las 09:00 UTC = 11:00 Madrid, dentro de la sesion europea. Con
  23:00 UTC, la ejecucion tipica queda a 04:00 UTC = 05:00/06:00
  Madrid (pre-apertura europea). Margen de retraso tolerado: 8h.
  Commits: 6dec2cf, f396e99.
- F-IAE-CRON-02 (2026-09-25): RESUELTO. Multi-slot de cron con idempotencia.
  Sustituye al cron unico `0 23 * * *` (F-IAE-CRON-01). Razon: el fallo
  del 25-Sep (run 36080921484) demostro que un deadline unico no cubre la
  latencia variable de Yahoo (>5h26m observado a 100% NaN, 16h12m a 0% NaN).
  Solucion: 4 slots (17 23 / 17 3 / 17 7 / 17 11 UTC) con gate pre-pipeline.
  Los 4 apuntan a la misma target_session. Minutos en :17 para evitar la
  franja :00 de alta carga documentada por GitHub.
  Commits: ed0fb07, 5657b1c, 3642038.
- F-IAE-GATE-01 (2026-09-25): RESUELTO. Gate de disponibilidad pre-pipeline.
  `scripts/pipeline_gate.py` decide 4 estados (CURRENT / READY / NOT_READY /
  ERROR) leyendo el manifest + probe fresco de un panel fijo de 20 tickers
  USA. Idempotencia por cobertura (no por last_date). Retry corto (3 intentos,
  sleeps 10/30s) para errores transitorios de red. `scripts/issue_manager.py`
  gestiona Issues: abre o comenta si los 4 slots fallan (schedule + last slot),
  cierra si recuperacion confirmada, nunca abre en workflow_dispatch.
  Commits: ed0fb07, 5657b1c, 3642038.
- Fase G - automatizacion de mappings (2026-09-24): CERRADA.
  Problema diagnostico: el crosswalk CUSIP y el catalogo radar eran
  ficheros estaticos mantenidos a mano. Cada trimestre SEC publica
  CUSIPs nuevos que no entraban al pipeline. La cobertura caia en
  silencio sin aviso.
  G.1 `scripts/regenerate_radar_catalog.py`: regenera el catalogo
  radar desde `stock_prices.parquet` + OpenFIGI (TICKER/US). Wireado
  en `daily_run.yml` tras `run.py`. No-op si no hay tickers nuevos.
  G.2 `scripts/regenerate_cusip_crosswalk.py`: regenera el crosswalk
  CUSIP cruzando filings + catalogo. Acumulativo (preserva historico
  en runners sin cache completa). Wireado en `update_sec_13f.yml`.
  G.3 `update_sec_13f.py --backfill N`: ingesta historica.
  `update_sec_13f.yml` invoca con `--backfill 2`.
  G.4 Cache 13F bump v1 -> v2. Motivo: la cache v1 tenia solo 1
  trimestre; save fallaba al intentar sobreescribir key existente.
  G.5 `workflow_dispatch.inputs.quarter` en `update_sec_13f.yml`
  para lanzamientos manuales en meses fuera de cron.
  G.6 Aviso de cobertura < 90% en el reporte (`catalog_coverage_warning`
  en `iae_section.py` + render en `report/iae.py`).
  G.7 Chequeo defensivo equity-only en el catalogo.
  Commits: 0066a64, 15382d6, e6fac26, 9c15823, c706c0d, f02a495,
  21561ac, 581f3c6, 58d01be, 6eadf9c, 1f93267, 9807e11, b58acac.
- Fixes estructurales (2026-09-24):
  - ROOT via `__file__`: 11 ficheros con `Path(r"D:\Macro_Sectorial")`
    hardcodeado migrados a `Path(__file__).resolve().parent.parent`.
    Bug funcional: en CI (Linux) `DATA_DIR` resolvia a un path
    inexistente y la seccion IAE estaba STALE silenciosamente. Ademas
    bloqueaba la coleccion de tests en CI. Commits: 64b2bce, d2fc7d9.
  - `.gitignore`: eliminada la linea `outputs/` que anulaba las
    excepciones `!outputs/history/` y `!outputs/state/`. El step
    `Commit and push hist/state` fallaba en bash `-e`. Commit: 16d8def.
  - `_MONTHS` en ingles: SEC publica `aug`/`dec`, no `ago`/`dic`.
    El cron trimestral fallaba para Q2/Q4 de cada ano. Q1/Q3
    funcionaban por coincidencia de idioma. Commit: 4c60f8f.
  - Filtro 5.1 (SH + PUTCALL NULL) en el crosswalk. Sin el, CALL/PUT
    se colaban como tickers validos. Commit: e6fac26.
  - `iae_section.py`: campos `stale_reason` (insufficient_quarters /
    official_list_pending) y `catalog_coverage_warning`. El IAE ya no
    enmascara la ausencia de Official List como ERROR. Commits:
    b357870, c706c0d.
  - `regenerate_cusip_crosswalk.py` acumulativo: preserva filas del
    crosswalk actual con `valid_from < cutoff` para no perder cobertura
    historica cuando el runner solo tiene 1 trimestre. Commit: 58d01be.
  - Restauracion del crosswalk degradado tras el primer run en CI:
    commit 6eadf9c.

---

## 3. Deuda activa

- Integracion a `run.py` completada el 2026-09-23 (fase adicional del
  pipeline productivo, + seccion del reporte diario). La integracion a
  `daily_run.yml` queda cubierta porque `daily_run.yml` invoca `run.py`.
- `compute_nipc_contractual`: ya tiene caller productivo desde
  2026-09-24 via `pipeline_contractual.run_contractual_nipc`,
  invocado por `compute_iae_section` desde `run.py`.
- `build_effective_reporting_snapshot` sin callers productivos.
- `scripts/iae_contractual_coverage.py` reproduce §12.3; pendiente
  integrarlo al flujo continuo de validacion.
- SPCX en catalogo radar sin fuente de identidad (emisor privado).
- Cobertura FIGI del 12,2% en INFOTABLE. Un fallback CUSIP -> FIGI
  via OpenFIGI podria ampliar la cobertura, pero no se ha demostrado
  que resuelva exhaustivamente los CUSIPs restantes (ver IAE_MAESTRO
  §13.1 y §12.4).
- Deuda semantica del nombre `NIPC` si se expone en reporte (§5.4).
- Identidad canonica dual (`equity:<TICKER>` / `figi:<FIGI>` en
  `canonical_security`): capacidad del resolver no observada en la
  configuracion E2E auditada. Auditoria empirica A.1 sobre 4.780.572
  units: 0 shareClassFIGI bajo dos canonical_security, 0 uso de la
  rama `figi:*`, delta NIPC = 0 al normalizar por shareClassFIGI.
  Ver IAE_MAESTRO §13.10. Reabrir si se activa `figi_lookup` en algun
  punto del pipeline, si aparece un caso `figi:*` en units o delta,
  o si se modifica el modelo de resolucion de identidad.
- Seccion IAE en produccion: STALE con razon `official_list_pending`.
  SEC no ha publicado `13flist2026q2.txt` en formato TXT (solo PDF
  desde 2026-08-14). Verificado con curl (404). Los parquets 13F
  Q4+Q1+Q2 estan en cache v2; el NIPC Q1->Q2 se calculara
  automaticamente cuando SEC publique el TXT.
- Deuda visible en el REPORTE (no en el pipeline): 3 bugs de render
  detectados tras revision del reporte diario 2026-09-24. RESUELTOS
  2026-09-24 (commits 2fe1c45 + 740be35 + dfd93c4).
  - `Momentum de amplitud`: `delta_1d_ema20 = nan` en 11/11 sectores.
    Fix: `_delta` tolera finde+festivo (days+5).
  - `Representatividad del lider`: 3 bloques sin columna distintiva.
    Fix: filtro a ultima fecha.
  - `Divergencia sector-lideres`: mismo patron que el anterior.
    Fix: filtro a ultima fecha.
- Fiabilidad de metricas sectoriales. RESUELTO 2026-09-24 (commit
  8c6330f, pendiente verificacion CI).
  Causa raiz: `compute_sector_breadth` contaba `n_total` como todos
  los componentes del ETF (78 en XLF, 85 en XLI, etc.), mientras que
  `get_stock_list()` solo descarga los top-20 por weight. Cobertura
  artificialmente baja (26-32% con [BAJA] en 8 de 11 sectores).
  Fix: replicar el cap top-20 en `indicators/sector_breadth.py` via
  nueva constante `TOP_N_SECTOR_COMPONENTS = 20` en
  `config/settings.py`. Los 3 `head(20)` hardcoded de
  `src/stock_data_loader.py` migrados a la constante.
  Efecto esperado: cobertura ~95-100% en 10 sectores, ~90-95% en
  sectores con IPO reciente (XLI con GEV, XLK con SNDK, XLP con
  KVUE). Marca `[BAJA]` solo donde sea real.
  Verificacion en reporte CI pendiente (cron 2026-09-24 23:00 UTC).
- Etiquetas semanticamente enganosas en el reporte. RESUELTAS
  2026-09-24 (commits f4f8003 + 664d839).
  - `Flujo Institucional - Sectores (Proxy)` -> renombrado a
    `Flujo de Mercado - Sectores (Proxy)` (idem Otros Activos).
  - `Acciones Seleccionadas por el Modelo`: anadida nota de criterio
    de seleccion (peso en ETF -> WLS).
- Contexto faltante para analisis (BACKLOG 2026-09-24). RESUELTOS
  2026-09-24 (commits c4b7138 + bb9251b).
  - D1: Rendimiento QQQ sin benchmark. Anadida fila SPY al CSV
    `outputs/history/qqq_returns_yahoo.csv` via generalizacion de
    `scripts/qqq_returns_yahoo.py` a N tickers. El render ya itera.
  - D2: Metricas sin percentil historico. Casos revisados:
    - D2a Institutional Hedge Ratio (2.26): WONT FIX. Bandas
      (<1.2 / 1.2-1.6 / >1.6) ya interpretables; percentil no
      anade valor.
    - D2b MTE Confidence Score: WONT FIX. El reporte declara
      explicitamente "no calibrado"; percentil no tiene sentido.
    - D2c FLOW_CONFIDENCE: FIX. Anadida regla explicita en
      `flows_international.py` + corregido bug latente en
      `flows_secondary.py` (`pos == 3` -> `pos >= 3`, que devolvia
      BAJA cuando las 4 capas estaban alineadas).
  - D3: Rotacion sectorial reciente. FIX: anadida nota semantica
    en `market_context.py` aclarando que delta negativo = mejora
    (rank menor = mejor posicion).
- Redundancia en reporte (BACKLOG 2026-09-24). RESUELTA 2026-09-24
  (commit c4b7138). Las dos tablas SSGA (`render_flujo_spdr` y
  `render_flujo_caracteristicas`) NO son redundantes: la primera
  da dato atomico por ETF, la segunda agrega por sector con
  metricas derivadas. Anadidas notas cruzadas de complementariedad
  en `etf_flows.py`.
- FU-002-bymarket (2026-09-24) - RESUELTO. Manifest de stock_prices
  ampliado con quality.by_market (cobertura por mercado en su propia
  ultima sesion cerrada via is_session_closed de FU-018). Guard
  guard_coverage.py gana exencion condicional cuando la cobertura
  global es baja pero todos los mercados activos cumplen el threshold
  (C-2: solo si status == VALID_WITH_MISSING; C-3: mismo threshold
  que el guard; punto 5: UNKNOWN con n>0 -> INVALID).
  Dictamen auditor externo 2026-09-24: APROBACION CONDICIONADA
  aplicada (3 correcciones materiales + 5 puntos adicionales).
  Politica de schema declarada en PROMPT seccion 11.8: campos aditivos
  backward-compatible, sin bump de schema_version.
  Commits: 88c27d1 (doc), d09f928 (manifest), 62e38d9 (guard), 07104c8 (cierre doc).

- Fechas no bursatiles residuales en `stock_prices.parquet` - RESUELTO
  2026-09-24 (commit 1822ed9). El dry-run de `cleanup_stock_prices_nyse_holidays.py`
  confirmo 0 celdas a limpiar: no hay filas espurias USA en
  festivos NYSE. Las 6 fechas detectadas (MLK, Presidents,
  Memorial, Juneteenth, Independence observed, Labor Day) solo
  tienen datos europeos legitimos. El health check estaba siendo
  demasiado estricto: alertaba por cualquier fila en fecha no
  bursatil NYSE sin distinguir el mercado. Fix: check_non_market_days
  ahora solo alerta si hay tickers US_EQUITY con Close. Los
  europeos operan en festivos USA y sus datos son correctos.
- Bug latente A/D en compute_sector_breadth - RESUELTO 2026-09-24
  (commit fc21671). Detectado al auditar el 22-Sep: 206/313
  tickers USA sin Close ese dia por fallo puntual de Yahoo. El
  calculo daily_ret = close.iloc[-1] - close.iloc[-2] comparaba
  23-Sep contra 21-Sep y lo etiquetaba como movimiento 1d. El
  A/D del reporte mezclaba 107 movimientos reales 1d con 206
  movimientos 2d aparentes. Fix: exigir continuidad temporal via
  previous_market_day (calendario NYSE, universo USA verificado).
  Si hay gap interno, el ticker no contribuye al A/D.
  Verificacion empirica: 107 contribuyen / 206 no contribuyen.
- 22-Sep incompleto marcado como fecha historicamente incompleta.
  RESUELTO 2026-09-24 (commit 5651d2a). Constante
  CONFIRMED_INCOMPLETE_DATES = {2026-09-22} en health_check.py.
  Uso exclusivo en check_coverage_last_5 para no generar WARN en
  cada ejecucion. Otros controles siguen evaluando la fecha.
- Registry de tickers - Consolidacion 2026-09-24 (commit 7025a89).
  `YAHOO_TICKER_MAP` + `normalize_yahoo_ticker(t)` movidos de
  `stock_data_loader.py` y `data_loader.py` (duplicados con riesgo de
  divergencia silenciosa) a `instrument_registry.py` (fuente unica).
  Los dos loaders re-exportan por backward-compat. Fix derivado:
  `get_market` ahora normaliza antes de clasificar por sufijo.
  Bug: `get_market('BRK.B')` devolvia UNKNOWN mientras
  `get_market('BRK-B')` devolvia US_EQUITY. Ahora ambos US_EQUITY.
  Tests: 9 nuevos (test_registry_yahoo_map.py).
  Commits: 7025a89, d1a4676.
- Deuda residual de cobertura unitaria de `run_contractual_nipc`:
  76 stmts no cubiertos por tests con mock (ver IAE_MAESTRO 3.2 y 5.1).

Detalle en `IAE_MAESTRO.md` seccion 5.

---

## 4. Reglas de operacion

- No integracion a produccion sin validacion funcional previa.
- No push a `origin/main` hasta integracion verificada.
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo.
- Toda escritura pasa por `write_artifact_with_manifest`.
- Todo bloque que haga `git commit` debe condicionarse a tests verdes
  (`if ($LASTEXITCODE -eq 0)`).

---

## 5. Documentos vigentes

- `IAE_MAESTRO.md` - referencia unica del modulo.
- `ESTADO_SISTEMA.md` - hechos autogenerados (fuente unica de HEAD).
- `ESTADO_DECLARADO.md` - este documento.

### Scripts publicados

#### Scripts del modulo IAE

- `scripts/iae_test_census.py` - censo AST de tests del modulo.
- `scripts/iae_coverage.py` - cobertura reproducible sobre los 845 tests.
- `scripts/regenerate_radar_catalog.py` - regenera catalogo radar (daily).
- `scripts/regenerate_cusip_crosswalk.py` - regenera crosswalk CUSIP (trimestral).
- `scripts/iae_contractual_coverage.py` - cadena contractual completa.
- `scripts/iae_reconciliation_b1.py` - reconciliacion radar + complemento = full.
- `scripts/iae_validate_crosswalk_openfigi.py` - validacion externa OpenFIGI.
- `scripts/iae_identity_uniqueness_audit.py` - auditoria unicidad por shareClassFIGI.
- `scripts/iae_pipeline.py` - orquestador ingestion + identity.
- `scripts/build_catalog_csvs.py` - construccion de catalogos CSV.
- `scripts/iae_contractual_nipc_e2e.py` - validacion E2E
  `compute_nipc_contractual` vs §12.5 (PASS).
- `scripts/update_sec_13f.py` - orquesta descarga + ingesta SEC 13F
  trimestral + Official List.
- `scripts/download_official_list_13f.py` - descargador Official
  List 13(f) de SEC.
#### Scripts de mantenimiento del radar (fuera del modulo IAE)

- `scripts/health_check.py` - vigilancia semanal del sistema
  (workflows, cache 13F, manifests, cobertura, fechas no
  bursatiles, contaminacion Europa-USA, seccion IAE). Reporta
  via GitHub Issue. Wireado en `.github/workflows/health_check.yml`
  (lunes 07:00 UTC). Commit 8ec488e.
- `scripts/cleanup_stock_prices_nyse_holidays.py` - saneamiento
  puntual del parquet stock_prices (F-IAE-HOLIDAY-01).
- `scripts/guard_coverage.py` - guard pre-commit del pipeline
  (bloquea commit si coverage/status/last_date no cumplen). Con
  exencion condicional FU-002-bymarket (seccion 11.17 del PROMPT).

Documentos historicos (no normativos): `NIPC_CONTRATOS_SEMANTICOS_v1.md`,
`NIPC_COVERAGE_POLICY.md`, `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md`.
Se conservan como documentacion de diseno; no son contratos vigentes.

---

**Nota:** los contratos P60-P70 en `NIPC_CONTRATOS_SEMANTICOS_v1.md`
se mantienen como documentacion historica de como se diseno el modulo.
No son normativos vigentes.
