# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` + `iae/ESTADO_DECLARADO.md`.
**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Rol

Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la sesion (2026-09-28)

    HEAD                 ver docs/auditoria/iae/ESTADO_SISTEMA.md
    Ahead                0 (sincronizado con origin/main)
    Working tree         LIMPIO
    Tests IAE            845 passed (criterio AST, 45 ficheros)
    Suite global         2226 passed + 2 skipped + 0 failed
                         (los 3 test_freshness pasan tras run.py; vuelven
                          a fallar si los parquets llevan >4 dias sin
                          refrescar - ver nota abajo)
    Push                 SI (integrado en produccion)
    Cobertura IAE        90% lineas (reproducible con scripts/iae_coverage.py)
    Auditoria IAE        CERRADA (C2 residual abierto no bloqueante)

Nota. `test_freshness` valida que market_data.parquet y stock_prices.parquet
esten actualizados a la ultima sesion NYSE. Pasan tras un `py run.py` y
vuelven a fallar si pasan >4 dias sin ejecutar el pipeline. Son
ambientales, no bugs. La suite local refleja el estado del repositorio
en el momento del test, no un valor fijo.

Criterio del conteo de tests IAE: ficheros `tests/test_*.py` que importan
`src.institutional_accumulation` (verificado por AST). Reproducible con
`scripts/iae_test_census.py`.

---

## Documentos clave

    docs/auditoria/PROMPT_MAESTRO.md              v7.15
    docs/auditoria/TRANSFER.md                    este documento
    docs/auditoria/README.md                      navegacion

    docs/auditoria/iae/IAE_MAESTRO.md             referencia unica del modulo IAE
    docs/auditoria/iae/ESTADO_DECLARADO.md        estado de fases y deuda
    docs/auditoria/iae/ESTADO_SISTEMA.md          hechos autogenerados
    docs/auditoria/iae/evidence/                  evidencia empirica

Scripts reproducibles del modulo IAE:

    scripts/iae_reconciliation_b1.py              cadena completa delta + NIPC
    scripts/iae_validate_crosswalk_openfigi.py    validacion externa del crosswalk
    scripts/iae_test_census.py                    censo riguroso de tests
    scripts/iae_coverage.py                       cobertura reproducible (44 ficheros)
    scripts/iae_contractual_coverage.py           cadena contractual
                                                  (target -> adapter -> coverage)
    scripts/iae_contractual_nipc_e2e.py           validacion E2E NIPC vs §12.5
    scripts/iae_identity_uniqueness_audit.py      auditoria unicidad shareClassFIGI
    scripts/iae_pipeline.py                       orquestador ingestion + identity
    scripts/regenerate_radar_catalog.py           regenera catalogo radar (daily_run)
    scripts/regenerate_cusip_crosswalk.py         regenera crosswalk CUSIP (trimestral)
    scripts/update_sec_13f.py                     ingesta trimestral SEC 13F + --backfill N
    scripts/download_official_list_13f.py         descarga Official List 13(f)
    scripts/cleanup_stock_prices_nyse_holidays.py saneamiento F-IAE-HOLIDAY-01

Documentos historicos (no normativos): NIPC_CONTRATOS_SEMANTICOS_v1.md,
NIPC_COVERAGE_POLICY.md, INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md.

---

## Estado del modulo IAE en una frase

Integrado, automatizado y verificado end-to-end sobre datos reales.
Con el crosswalk CUSIP regenerado por script (246 filas, 242 tickers):

- Cache 13F en GitHub: 3 trimestres (Q4 2025, Q1 2026, Q2 2026).
- Crosswalk regenerado por `regenerate_cusip_crosswalk.py` en cada
  ingesta trimestral. Acumulativo. Preserva filas manuales.
- Catalogo radar regenerado por `regenerate_radar_catalog.py` tras
  cada `run.py` en `daily_run.yml`. No-op si no hay tickers nuevos.
- Seccion IAE en el reporte diario. DESBLOQUEADA 2026-09-27: SEC
  publico la Official List 13(f) de Q2 2026 con sufijo `-txt`
  (13flist2026q2-txt.txt). El downloader usa URL_EXCEPTIONS + fallback
  automatico 404 -> sufijo `-txt`. Commit 08a6c6a.

NIPC Q1 2026 -> Q2 2026 = 8.254.818.120 (calculado 2026-09-27).
NIPC Q4 2025 -> Q1 2026 = -4.264.449.012 (baseline local reproducible
post-fix H1-B v2.3, commit 72fa824). El valor declarado por el auditor
-4.264.449.932 queda registrado como discrepancia abierta en
golden/current.json (reconciliation_delta = 920,
reconciliation_status = OPEN).

Detalle completo en `iae/IAE_MAESTRO.md`.

---

## Sesion 2026-09-24 (Radar: BACKLOG + FU-002-bymarket + health check)

**Ciclos cerrados (no-IAE):**

1. **BACKLOG del reporte** (bugs de presentacion, no pipeline):
   - C1: _delta en sector_breadth_momentum.py toleraba mal gaps de
     calendario. Fix: days+5. Commit 2fe1c45.
   - C2/C3: render_representatividad_lider y render_divergencia_sector_lideres
     no filtraban a ultima fecha. Commits 740be35 + dfd93c4.
   - B1: 'Flujo Institucional' -> 'Flujo de Mercado' (la metrica es
     FLOW_PROXY). Commit f4f8003.
   - B2: criterio de seleccion de lideres (peso ETF -> WLS). Commit 664d839.
   - D1: benchmark SPY en Rendimiento QQQ. Commit bb9251b.
   - D2c: bug latente FLOW_CONFIDENCE pos==3 -> pos>=3. Commit c4b7138.
   - D3/E1: notas semanticas (rotacion + SSGA). Commit c4b7138.
   - A1: cobertura sectorial contra top-20 real (no ETF completo).
     Commit 8c6330f.

2. **FU-002-bymarket** (manifest + guard, dictamen auditor externo):
   - Manifest stock_prices gana quality.by_market (cobertura por mercado
     en su ultima sesion cerrada via FU-018).
   - Guard_coverage exime condicionalmente cuando todos los mercados
     activos cumplen threshold (C-2: solo con VALID_WITH_MISSING;
     C-3: mismo threshold del guard).
   - Cierra el falso bloqueo de commits en ventana desfasada Europa-USA
     (K-STOCK-PRICES-EOD-01 mitigado).
   - Commits 88c27d1, d09f928, 62e38d9.

3. **Health check semanal** (`health_check.yml` lunes 07:00 UTC):
   - scripts/health_check.py con 7 bloques (workflows, cache 13F,
     manifests, cobertura, fechas no bursatiles, patron EU-USA, IAE).
   - Abre/cierra GitHub Issue con label health-check.
   - Commits 8ec488e, 0464145.

4. **Bug latente A/D en compute_sector_breadth**:
   - Detectado al auditar el 22-Sep (206/313 tickers USA sin Close
     por fallo puntual Yahoo). El calculo diario comparaba 23 vs 21
     sin verificar continuidad. Fix: exigir previous_market_day.
   - Verificacion empirica: 107 contribuyen / 206 no contribuyen.
   - Commit fc21671. 22-Sep marcado en CONFIRMED_INCOMPLETE_DATES.

5. **Consolidacion del registry de tickers**:
   - YAHOO_TICKER_MAP + normalize_yahoo_ticker movidos de
     stock_data_loader.py y data_loader.py (duplicados) a
     instrument_registry.py (fuente unica).
   - get_market ahora normaliza: BRK.B/BF.B -> US_EQUITY (antes UNKNOWN).
   - Commits 7025a89, d1a4676.

Suite acumulada: 1587 -> 1682 passed (+95 tests, 6 bugs latentes
corregidos, 30 commits pusheados).

---

## Sesiones recientes (2026-09-22 / 2026-09-23)

Fase A (bloqueantes del dictamen auditor externo, cerrada):

- B1 - Reconciliacion radar + complemento = full. Verificado con
  `scripts/iae_reconciliation_b1.py`. Split oficial: por `canonical_security`
  (formato `equity:TICKER`). El split por CUSIP no es viable: las filas
  con `observed_security_key=cusip:XXX` son UNRESOLVED_IDENTITY.
- B2 - Cadena forense congelada. sha256 de INFOTABLE Q4/Q1, pandas 2.3.3,
  timestamp UTC en el pie de §12.
- B3 - Validacion externa del crosswalk via OpenFIGI. 227 de 243 CUSIPs
  verificados con shareClassFIGI coincidente, 16 sin hit (prefijos no-USA).
- B4 - Riesgo PIT declarado (§13.9): identidad historica vs pertenencia
  historica al radar.

Fase B (13 criticos, cerrada):

- B.1 - Trazabilidad numerica y honestidad de alcance (C5, C6, C12, C13, C14).
- B.2.1 - Semantica NIPC, cadena contractual, coherencia numerica
  (C7, C11, C16, C18).
- B.2.2 - Rename `cusip_ticker_exceptions.csv` -> `cusip_radar_crosswalk.csv`
  (C15). El string `source` interno se conserva como provenance historica.

Fase C (3 cambios funcionales, cerrada):

- C8 - `reporting_dedup` declarado diagnostico separado, no etapa del
  flujo contractual.
- C9 - `aggregate_positions_by_shareclass_figi` con `return_stats=True`:
  contadores observables (recibidos, verificados, excluidos por status).
- C10 - `coverage_available` (medible) + `coverage_quality`
  (UNAVAILABLE/PARTIAL/COMPLETE con threshold 0.95 sobre las DOS
  dimensiones). `coverage_status` legacy intacto.

Sesion 2026-09-24 (Fases E + F + fixes):

Fase E - Integracion a `run.py` (cerrada):

- E.1 - `scripts/iae_contractual_nipc_e2e.py`: validacion E2E de
  `compute_nipc_contractual` con reconciliacion exacta contra §12.5.
- E.2 - Extraccion del pipeline contractual a
  `src/institutional_accumulation/pipeline_contractual.py`.
- E.3 - `src/pipeline/iae_section.py::compute_iae_section`: detector
  automatico de ultimo par de trimestres + orquestador.
- E.4 - Wire a `run.py` + `src/report/iae.py::render_iae_section`.
  Seccion "Acumulacion Institucional (13F)" en el reporte diario.
  Validation Gate 10/10 intacto.
- E.5 - Documentacion de integracion unidireccional (IAE_MAESTRO §2.3
  y §13.4).

Fase F - Automatizacion GitHub (cerrada):

- F.1 - Rutas portables (`IAE_OFFICIAL_DIR`, default al repo).
- F.2 - Descargador Official List 13(f) + versionado Q4/Q1.
- F.3 - Doble ruta SEC + retry + orquestador trimestral.
- F.4 - Workflow `update_sec_13f.yml` + cache parquets + restore en
  `daily_run.yml`.

Fase G - Automatizacion de mappings (cerrada 2026-09-24):

  Problema diagnostico: el crosswalk CUSIP y el catalogo radar eran
  ficheros estaticos mantenidos a mano. Cada trimestre SEC publica
  CUSIPs nuevos que no entraban al pipeline. La cobertura caia en
  silencio sin aviso.

- G.1 - `scripts/regenerate_radar_catalog.py`: regenera el catalogo
  radar desde `stock_prices.parquet` + OpenFIGI (TICKER/US). Wireado
  en `daily_run.yml` tras `run.py`.
- G.2 - `scripts/regenerate_cusip_crosswalk.py`: regenera el crosswalk
  CUSIP cruzando filings + catalogo. Acumulativo (preserva historico
  en runners sin cache completa). Wireado en `update_sec_13f.yml`.
- G.3 - `update_sec_13f.py --backfill N`: ingesta historica.
  `update_sec_13f.yml` invoca con `--backfill 2`.
- G.4 - Cache 13F bump v1 -> v2. Motivo: la cache v1 tenia solo 1
  trimestre; save fallaba al intentar sobreescribir key existente.
- G.5 - `workflow_dispatch.inputs.quarter` en `update_sec_13f.yml`
  para lanzamientos manuales en meses fuera de cron.
- G.6 - Aviso de cobertura < 90% en el reporte (`catalog_coverage_warning`).
- G.7 - Chequeo defensivo equity-only en el catalogo.

Fixes estructurales (2026-09-24):

- F-IAE-HOLIDAY-01: no ffill USA en festivos NYSE (dia laborable).
  Saneamiento de 50 celdas en `stock_prices.parquet`.
- F-IAE-CRON-01: cron `daily_run.yml` 04:00 -> 23:00 UTC
  (pre-apertura europea). Evita la ventana donde `guard_coverage`
  bloqueaba commits.
- ROOT via `__file__`: 11 ficheros con `Path(r"D:\Macro_Sectorial")`
  hardcodeado migrados a `Path(__file__).resolve().parent.parent`.
  Bug funcional: en CI (Linux) `DATA_DIR` resolvia a un path inexistente
  y la seccion IAE estaba STALE silenciosamente.
- `.gitignore`: eliminada la linea `outputs/` que anulaba las
  excepciones `!outputs/history/` y `!outputs/state/`. El step
  `Commit and push hist/state` fallaba en bash `-e`.
- `_MONTHS` en ingles: SEC publica `aug`/`dec`, no `ago`/`dic`. El
  cron trimestral fallaba para Q2/Q4 de cada ano.
- Filtro 5.1 (SH + PUTCALL NULL) en el crosswalk. Sin el, CALL/PUT
  se colaban como tickers validos.
- `iae_section.py`: campos `stale_reason` (insufficient_quarters /
  official_list_pending) y `catalog_coverage_warning`. El IAE ya no
  enmascara la ausencia de Official List como ERROR.

Estado de la seccion IAE en produccion:

- Cache 13F v2 en GitHub: 3 trimestres.
- Seccion IAE: STALE (`stale_reason=official_list_pending`).
- Causa: SEC no ha publicado `13flist2026q2.txt` (solo PDF desde
  2026-08-14). Verificado con curl (404).
- El mensaje del reporte es honesto: "Datos 13F cargados (2026Q1 -> 2026Q2),
  pero la Official List 13(f) de 2026Q2 aun no ha sido publicada por SEC."
- Cuando SEC publique el TXT, el proximo workflow lo capturara
  automaticamente.

---

## Reglas de operacion

- Push a `origin/main` autorizado tras validacion funcional en CI.
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo.
- No integracion a produccion sin validacion funcional.
- Todo bloque que haga `git commit` debe condicionarse a tests verdes
  (`if ($LASTEXITCODE -eq 0)`). Leccion del commit 977249b.
- Cron productivo de `daily_run.yml`: multi-slot (4 disparos: 17 23 /
  17 3 / 17 7 / 17 11 UTC) con gate pre-pipeline + concurrency queue: max.
  F-IAE-CRON-01 (cron unico 0 23) queda SUPERSEDED por F-IAE-CRON-02
  (2026-09-25). El gate decide en cada slot si correr (READY) o skip
  (CURRENT / NOT_READY / ERROR).
- Dispatch manual de `daily_run.yml`: preferible FUERA de la sesion USA
  abierta (13:30-20:00 UTC). Motivo actual: el gate puede dar READY y
  correr el pipeline con datos intradia USA si el manifest no cubre la
  sesion esperada; en ese caso guard_coverage bloqueara el commit. La
  regla se mantiene por prudencia operativa, no por el mismo motivo que
  antes de F-IAE-CRON-02.
- En here-strings PowerShell que contengan backticks markdown, usar
  `chr(96)` o eliminar los backticks del contenido. PowerShell interpreta
  el backtick como escape y cierra el here-string silenciosamente. Leccion
  del ciclo 2026-09-24 (perdida de 3 operaciones en PROMPT v7.4).
- Antes de commitear patches multi-operacion, verificar que el script
  NO escribe parcialmente si algun assert falla. Corregir el patron para
  que todas las ops se apliquen o ninguna. Leccion del ciclo 2026-09-24.

---

## Comandos de arranque

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py -m pytest tests/ -q --tb=line
    py -m pyflakes src\ scripts\ ; py -m compileall . -q

Esperado:

- HEAD ver docs/auditoria/iae/ESTADO_SISTEMA.md
- ahead 0, behind 0
- working tree limpio
- 2226 passed + 2 skipped + 0 failed (test_freshness pasa tras run.py)
- pyflakes silencio, compileall OK

Censo del modulo IAE (comando aparte, tarda unos segundos):

    py scripts\iae_test_census.py

Esperado: 45 ficheros, 845 tests collected.

---

## Confirmacion esperada

"Confirmado, contexto asimilado."

Estado que reconozco:

- HEAD ver docs/auditoria/iae/ESTADO_SISTEMA.md
- IAE integrado en run.py, automatizado en GitHub, sincronizado con origin/main
- Crosswalk regenerado por script (246 filas, 242 tickers)
- Cache 13F v2 con 3 trimestres en GitHub
- Seccion IAE OPERATIVA 2026-09-27 (Official List Q2 desbloqueada con sufijo -txt)
- Fases A-F cerradas; Fase G (automatizacion de mappings) cerrada 2026-09-24
- Fase D (auditor externo IAE) CERRADA 2026-09-28 con C2 residual abierto
- H1-B: CERRADO con C2 (920) no bloqueante. Baseline local reproducible -4.264.449.012
- H4: CERRADO (golden/current.json ACTIVE_WITH_OPEN_DISCREPANCY)
- H2, H3, H5.1, H5.2, H5.4, H5.5, O1: CERRADOS 2026-09-28
- H5.3: trazabilidad implementada, pendiente verificacion cron nov 2026
- Bugs de render C1-C3 RESUELTOS 2026-09-24 (commits 2fe1c45, 740be35, dfd93c4)
- Cobertura sectorial A1 RESUELTA 2026-09-24 (top-20 real, commit 8c6330f)
- FU-002-bymarket integrado (manifest + guard, commits 88c27d1, d09f928, 62e38d9)
- Health check semanal operativo (lunes 07:00 UTC)
- F-IAE-CRON-02 / F-IAE-GATE-01 integrados en produccion (multi-slot +
  gate pre-pipeline + issue-manager). Verificado en CI real.
- F-IAE-LSE-INTEGRATION integrado en produccion (scraper LSE + override
  parcial de Close). Verificado end-to-end en CI real (run 36208872855,
  applied=20/20 status=OK).
- PROMPT_MAESTRO vigente: v7.15 (cierre H1-B, commit fd1b01b).
- Pendiente: H5.3 en cron nov 2026. C2 (920) pendiente de comando + HEAD del auditor.

Pregunta: "Que hacemos?"

---

## Sesion 2026-09-25 (F-IAE-CRON-02 / F-IAE-GATE-01)

**Problema.** Fallo del run 36080921484 (25-Sep 01:11 UTC) con 262/313
tickers USA sin Close. Analisis forense: Yahoo devolvio la fila de
expected_session con Close=NaN. Dispatch manual 11:57 UTC del mismo dia
paso OK sin cambios de codigo. Ventana de latencia real: >5h26m a 100%
NaN, 16h12m a 0% NaN.

**Diagnostico.** El sistema apostaba a un deadline unico. Si Yahoo no
publicaba en esa ventana, no habia recuperacion ese dia. Mover el cron
(F-IAE-CRON-01 -> F-IAE-CRON-03) es un cambio de parametro, no de clase.

**Solucion.** Multi-slot con gate pre-pipeline. Aprobado por auditor
externo con 2 rondas de dictamen (informe inicial + correcciones).

Ciclos cerrados (3 commits, un subciclo de implementacion por commit):

1. **Commit 1 - ed0fb07.** `scripts/pipeline_gate.py` (182 lineas) +
   `tests/test_pipeline_gate.py` (30 tests iniciales). Decision pura
   CURRENT / READY / NOT_READY / ERROR. Sin tocar produccion.

2. **Commit 2a - 5657b1c.** Extension del gate: CRON_SLOTS,
   resolve_slot_flags, retry corto del probe (3 intentos, sleeps 10/30s).
   +21 tests. 51 tests totales del gate.

3. **Commit 2b - 3642038.** Wiring productivo:
   - `daily_run.yml`: 4 slots (17 23 / 17 3 / 17 7 / 17 11 UTC),
     concurrency queue: max, job gate, run-system condicionado,
     job issue-manager.
   - `scripts/issue_manager.py`: decide_action pura (NOOP / CLOSE /
     ENSURE_FAILURE), validate_target_session pura, execute_action via
     gh CLI (mismo patron que health_check.py). Busqueda de Issue por
     comparacion EXACTA de title en Python.
   - `tests/test_issue_manager.py`: 33 tests (maquina de estados
     completa, execute_action con subprocess mockeado, contrato de
     seguridad GH_TOKEN, drift CRON_SLOTS vs daily_run.yml).

**Verificacion en CI real.** Run 36148143256 (dispatch manual):
- gate -> CURRENT, should_run=false (manifest ya cubre 2026-09-24).
- run-system -> skipped (condicion).
- issue-manager -> CLOSE (no-op sin Issue abierto).
- Overall -> success.

**Rechazos tecnicos durante el ciclo:**
- `issue_manager.js` + `actions/github-script@v9`: rechazado por falta
  de `node` en entorno local (imposibilidad de testear la maquina de
  estados antes del push). Sustituido por `issue_manager.py` + `gh` CLI.
- `queue: max`: adoptado por dictamen auditor. Incompatible con
  cancel-in-progress.
- `issue-manager` con `needs: gate` solo: rechazado por dictamen auditor.
  Añadido `run-system` como dependencia y `if: always()` para asegurar
  que la decision se toma tras conocer el resultado del pipeline.

**Cambios prohibidos en este ciclo (preservados intactos):**
- `guard_coverage.py`.
- `run.py`, pipeline, IAE, manifest FU-002, contratos temporales.

**Lecciones aplicables a ciclos futuros:**
- Un here-string grande de PowerShell con comillas anidadas se corrompe
  silenciosamente. Patron seguro: chunks de ~50 lineas con
  `[System.IO.File]::WriteAllText`/`AppendAllText` y ruta absoluta
  (`Join-Path $PWD ...`). El `$PWD` es obligatorio: el metodo .NET
  usa el CWD del proceso, no el de PowerShell.
- La regex `##` de aqui-strings choca con delimitadores. Verificar con
  `ast.parse` antes de escribir.

---

## Sesion 2026-09-26 (F-IAE-LSE-INTEGRATION)

**Problema.** Los 20 tickers `.L` (FTSE 100) se cubrian exclusivamente
via Yahoo, con el mismo patron de latencia variable documentado en
F-IAE-CRON-02 para USA. El radar marcaba `DATA_ISSUE`
(MISSING_CLOSE_EXPECTED_SESSION) cuando Yahoo aun no habia publicado
el Close del cierre LSE.

**Diagnostico.** No hay proveedor oficial LSE. Euronext/Xetra/BME no
cubren LSE. Yahoo tarda horas en poblar Close. El sistema actual no
tenia segunda fuente para esos 20.

**Solucion.** Scraper LSE privado (`lse-close-scraper`) alimentado via
Refinitiv Widgets. El radar aplica override PARCIAL de Close sobre la
fila ya presente de Yahoo. NO sustituye el registro: Open/High/Low/
Volume siguen viniendo de Yahoo (evita degradar flow_proxy_z, OBV, CMF).

Ciclos cerrados (5 commits, subciclos incrementales):

1. **0313879.** Loader LSE base: `src/external/lse_scraper_loader.py`
   (load_lse_close_for_session) + instrument_registry con refinitiv
   (20 entradas `.L`, BA.L -> BAES.L). +30 tests.

2. **71e32d2.** Subciclo 1: `last_expected_lse_session` (calendario
   LSE) + `build_lse_close_override` + `write_lse_provenance`. +32
   tests.

3. **de30f3b.** Subciclo 2a: provenance con status/reason (dictamen
   D4) + `aplicar_override_close` (solo Close, sin anadir filas).
   +18 tests.

4. **9d5b19e.** Subciclo 2b: integracion en `stock_data_loader.py`
   (`_apply_lse_close_override` tras dedup, antes del manifest).
   +11 tests.

5. **c5ffcd4.** Subciclo 2c: wiring en `daily_run.yml` (2 steps:
   Fetch LSE scraper con PAT fine-grained + Capture SHA) + git add
   de provenance.

**Verificacion en CI real.** Dos runs relevantes:

1. **Dispatch manual 36189310171.** 21:03 UTC, gate dio CURRENT
   (manifest cubria sesion anterior). run-system skipped. Esperado
   por horario; no sirvio para ver el override.

2. **Cron slot 1 36208872855.** 01:36 UTC (26-Sep). gate READY,
   run-system completo (18m16s). Log del step Run Macro Sectorial:
   `[LSE-OVERRIDE] session=2026-09-25 applied=20/20 status=OK`.
   Provenance commiteada con el parquet (commit 9569bf6):
   source_commit=3609bd7..., scraper_available=true, scraper_used=true,
   status=OK, tickers_from_scraper=20, tickers_from_yahoo=0,
   tickers_missing=0.

**Comprobaciones cruzadas post-ciclo (2026-09-26):**
- Close parquet vs scraper: 20/20 match byte-exacto (delta 0.000000).
- OHLCV coherentes (L<=C<=H, L<=O<=H): 20/20.
- Volume preservado (no NaN): 20/20.
- Coherencia reporte vs CSV: 100% (con redondeo 2 decimales).
- Validation Gate: 10/10.
- Divergencia CI vs local en WLS (5 tickers FTSE 100) atribuida a
  revision retrospectiva de Yahoo. No afecta al Close del 25-Sep.
  Documentado como K-LSE-YAHOO-REVISION-01 (PROMPT §12). No es bug.

**Dictamenes externos aplicados:**
- D1: sesion LSE especifica, no la global NYSE.
- D2: override tras dedup (frontera robusta).
- D3: solo DATOS_DIR en config; repo/ref/commit via env del workflow.
- D4: provenance distingue availability/usage con status/reason.
- D5: `persist-credentials: false` en el checkout privado.
- D7: GBX sin conversion (verificado empiricamente vs Yahoo).

**Rechazos tecnicos:**
- `/100` en loader: refutado por comparacion empirica Yahoo vs Refinitiv
  (ambos GBX). Lo detecto el auditor.
- `Volume = NaN` en tickers del scraper: rechazado por auditor. El
  override solo sobrescribe Close, preservando Volume de Yahoo.
- `max(_DATE_END)` como criterio temporal: rechazado. Igualdad exacta
  con `lse_expected_session`, resuelta por FU-018.
- `source_commit` opcional: rechazado (D4). Obligatorio si scraper
  usado.
- Checkout sin `persist-credentials: false`: rechazado (D5).

**Invariante de seguridad:** el radar NO ejecuta codigo del repo
externo. Solo lee sus JSON. El PAT tiene `contents: read` sobre un
unico repositorio.

**Cambios prohibidos en este ciclo (preservados intactos):**
- `run.py`, `guard_coverage.py`, pipeline, IAE, manifest FU-002,
  contratos temporales.

**Lecciones aplicables a ciclos futuros:**
- `git commit -m "..."` con here-string PowerShell: los `$` y comillas
  anidadas rompen el argumento en PowerShell. Patron seguro:
  `[System.IO.File]::WriteAllText` a `_commit_msg.txt` + `git commit -F`.
  `-F` y `-m` no son compatibles entre si.
- `ast.parse()` NO valida YAML. Para YAML, `yaml.safe_load`.
- Cifras en documentacion: contarlas con evidencia (`git log` + `pytest`)
  antes de escribirlas. Un "+83 tests" que deberia ser "+91" es
  detectable cruzando los commits.
- Verificacion del override end-to-end diferida al primer slot de
  produccion donde el gate de READY. Ejecutado con exito el 2026-09-26
  (cron 01:36 UTC, run 36208872855): 20/20 applied.
- Yahoo revisa OHLC historico retrospectivamente. La sesion nueva del
  override coincide byte-exacta (20/20), pero los componentes de WLS
  que dependen de ventanas largas (`wyckoff_score` -> `rws_z`,
  `stability` -> `stab_z`) pueden diferir entre runs. Documentado como
  K-LSE-YAHOO-REVISION-01. No es bug del sistema.
- La funcion `robust_intra()` (indicators/index_leaders.py) normaliza
  cada componente del WLS DENTRO del universo de 15 candidatos. Un
  solo candidato con historico revisado desplaza el z-score de todos.
  Al diagnosticar divergencias entre runs, comprobar primero si
  `rs_z` (Close reciente) coincide. Si coincide y `wls` no, es
  revision historica, no bug.

---

## Sesion 2026-09-26 (auditoria interna PROMPT v7.5 -> v7.6)

**Motivo.** Auditoria interna del PROMPT_MAESTRO por peticion del
usuario. Metodo: cruce del documento con la realidad verificable
(ficheros, workflows, git, pytest). Sin cambios de codigo ni de
arquitectura. Solo correccion documental.

**Resultado.** 15 correcciones aplicadas:

- 7 errores reales:
  * `indicators/darkpool/` no es un paquete, son 4 ficheros sueltos en
    `indicators/`. Corregido en §4.5.
  * `docs/plan/` no existe en disco. Eliminado del arbol §4.2.
  * `audit/` no existe en raiz (es `outputs/audit/`). Movido bajo outputs.
  * `history/`, `state/`, `report/` mal indentados (en raiz, no bajo
    `outputs/`). Corregida indentacion.
  * `providers/`: decia 29, reales 27 ficheros `.py` + `__init__`. El 29
    contaba `__pycache__/`.
  * §4.3: faltaba `iae.py` en `src/report/`.
  * §4.4: faltaba `iae_section.py`. Total ajustado a 17 modulos.

- 1 inconsistencia menor:
  * §10.1: "ver seccion 15" redirige a IAE_MAESTRO. Cambiado a
    "ver IAE_MAESTRO.md seccion 11".

- 3 omisiones nuevas (elementos del ciclo LSE no documentados en §4.2):
  * `src/external/`.
  * `data/lse_close_provenance.json`.
  * `pipeline_gate.py` + `issue_manager.py` en scripts.

- Bump v7.5 -> v7.6 (fecha 2026-09-26) + pie actualizado.

**Verificaciones cruzadas sin hallazgos:** 28/28 ficheros referenciados
existen; 9/9 workflows coinciden en nombre y cron; 64 SHAs citados
(63 OK radar + 1 OK scraper privado); 1857 tests declarados = 1859
collected - 2 skipped; IAE 845 tests coherentes; 10 contratos
temporales; 20 `.L` con refinitiv; 4 funciones publicas del loader LSE.

**Commit:** `5d54ddf` (docs(prompt): v7.5 -> v7.6).

**No propagado a `ESTADO_DECLARADO.md` ni `ESTADO_SISTEMA.md`.** La
auditoria es puramente documental del PROMPT; no cambia hechos del
sistema, no introduce decisiones nuevas, no modifica arquitectura.

---

## Sesion 2026-09-27 (12 ciclos: Cat 1 + MTE + providers + cron + A5-70)

**Resumen.** Sesion de cierre del backlog MEDIA/BAJA pendiente de la
auditoria radar 2026-09-26. 12 ciclos completados. Suite 1857 -> 2184.

**Commits pusheados (sobre 062cd10):**

    8abb6b0  fix(providers): A5-71 except acotado en parse_fecha_es (blackrock x2)
    cf20292  fix(commodities): F3-18 sort_index fuera del bucle preserva in-place
    c576158  test(providers): cubrir parse_fecha_es A5-71 (52 tests)
    9a6ff6a  fix(mte): F2.4-01 + F2.4-02 + F2.4-03 (auditoria radar)
    45124fd  docs(audit): cerrar ciclo Cat 1 (v7.6 -> v7.7)
    e8d68c3  fix(tests): aislar test_indices_intl con tmp_path (side-effect)
    dc16f5c  chore(tests): limpiar 19 warnings pyflakes (F401 + F841 + F541)
    6e5cf03  fix(mte): F2.4-11 except acotado en engine.py + docstring F2.4-13
    aff0afd  fix(mte): F2.4-10 persistir pending + F2.4-12 (ii) reset temporal
    9c07ff8  docs(audit): cerrar ciclo F2.4-10/11/12/13 (v7.7 -> v7.8)
    12b220e  refactor(mte): F2.4-04 eliminar codigo muerto NFCI/OAS
    062cd10  perf(finra): cache local por (endpoint, payload) 30d hit / 24h empty
    03f3b42  ops(ci): probe cron 17 3 + 5o slot defensivo + health-check
    f49277a  feat(health-check): bloque H verifica los 4 cron slots
    9da8b53  fix(health-check): bloque H detecta slot-por-slot (no agregado 24h)
    4049439  refactor(providers): A5-70 extraer _blackrock_base (DAXEX + ISF.L)
    e672f65  docs(audit): cierre A5-70 + rectificacion G7/G8 + K-BLACKROCK-CSV

**Ciclos cerrados:**

1. **Cat 1 quirurgicos.** A5-71 (except acotado en blackrock x2) + F3-18
   (sort_index fuera del bucle) + F2.4-01/02/03 (MTE scoring). 52 tests
   nuevos de parse_fecha_es. Commits 8abb6b0..9a6ff6a + docs 45124fd.

2. **Fix side-effect en test.** test_indices_intl_con_acumulacion
   escribia en outputs/report/ real sin tmp_path. Mismo patron que el CSV
   contaminado con AAA,1.5. Fix: anadir tmp_path + monkeypatch.chdir.
   Commit e8d68c3.

3. **19 warnings pyflakes.** Limpiados (F401 + F841 + F541) en 12 ficheros
   de tests, introducidos por los commits de coverage previos.
   Commit dc16f5c.

4. **F2.4-11 + F2.4-13.** except acotado en engine.py L89; consensus_score
   documentado como WONT FIX (mezcla de escalas intencional). Commit
   6e5cf03.

5. **F2.4-10 + F2.4-12 (ii).** Histeresis adaptativa MTE estaba
   desactivada en produccion: classify_mte escribia via save_scenario
   pero engine.py sobreescribia el state file sin pending. Fix:
   classify_mte ya no escribe; engine.py es el unico writer. Anclado por
   8 tests (5 ramas + hysteresis 2 runs + 2 reset). Golden MTE intacto.
   Commit aff0afd.

6. **F2.4-04.** robust_zscore_series en credit_stress_score era codigo
   muerto (nunca se ejecutaba: nfci_series/credit_oas_series siempre
   None). Eliminadas 182 LOC. Commit 12b220e.

7. **FINRA cache.** compute_darkpool_signals tardaba 68s por 25 requests
   paginadas con sleep(2). Anadida cache local por (endpoint, payload)
   con TTL 30d hit / 24h empty. Segunda llamada: 68s -> 0.04s.
   actions/cache@v4 ya existente en daily_run.yml propaga el beneficio
   a CI. Commit 062cd10.

8. **Cron probe + 5o slot + health-check.** Finding: slot '17 3' no
   habia disparado desde el deploy del multi-slot (25/09). Causa probable:
   GitHub Actions scheduler best-effort. Solucion:
   - _cron_probe.yml: workflow aislado con solo '17 3' para diagnostico.
   - 5o slot '17 5' defensivo.
   - health_check.py bloque H: verifica los slots slot-por-slot.
   Commits 03f3b42, f49277a, 9da8b53.

9. **A5-70.** Refactor blackrock_fund_data + blackrock_isf_fund_data a
   _blackrock_base.py (213 LOC) + 2 wrappers (56 LOC c/u). -153 LOC netas,
   ~90% duplicacion eliminada. 15 gates superados con dictamen de auditor
   externo (GO CONDICIONADO -> CERRADO). 11 tests directos. Manifiesto
   SHA256 versionado en docs/auditoria/a5_70_baseline.json. Commit 4049439.

**Docs:** v7.6 -> v7.7 -> v7.8 -> v7.9.

**Pendientes reales al cierre:**

- **K-BLACKROCK-CSV-STALE-01.** Los CSV blackrock_{dax,isf}_primary_flow.csv
  en git tienen schema anterior al commit a3c2705. Decision politica
  pendiente: (a) regenerar con schema actual, (b) dejar de versionarlos,
  (c) congelar via workflow. El auditor recomienda determinar primero si
  son evidencia historica o derivados locales. No reabre A5-70.

- **_backfill_history (indicators/darkpool_history.py).** yf.download
  ticker a ticker en bucle. 7s residuales en compute_darkpool_signals
  tras FINRA cache. Mismo patron que FINRA: cuello de red, no computo.

- **Cat 4 (20+ providers).** Bloque grande. Requiere Gate 0 read-only
  para filtrar vigentes vs obsoletos (precedente 2026-09-17: 9/9 BAJA
  revisados eran obsoletos).

- **_cron_probe.yml.** En observacion 24-48h. Determinar si '17 3'
  dispara alguna vez de forma aislada.

**Observaciones metodologicas aprendidas en esta sesion:**

- Heredoc PowerShell: acentos y caracteres no-ASCII se manglean al
  procesar here-strings. Patron seguro: placeholders ASCII +
  chr(codepoint) dentro del script Python.
- `py -c "..."` con comillas dobles anidadas rompe el parser. Usar
  fichero Python temporal escrito con [System.IO.File]::WriteAllText.
- Verificacion empirica > auditoria de codigo: F2.4-20/21 (darkpool)
  parecian MEDIA por analisis estatico, pero el cProfile revelo que el
  cuello era FINRA (68s en red), no el codigo de scoring. Los findings
  quedaron retractados.
- No commitear automaticamente. Cada ciclo verificado con
  pyflakes + suite + (si aplica) smoke test funcional. ast.parse solo
  para ficheros Python, nunca para Markdown.

---

**Anexo 2026-09-27 (post-cierre): K-BLACKROCK-CSV-STALE-01 CERRADO.**

Ciclo separado tras el cierre de los 12 ciclos de la sesion.

Gate 0 read-only localizo el writer real de los 4 CSV (dentro de
cada provider, no en flows_primary). Inspeccion de columnas:
flow_zscore_regime en posicion 8 (BlackRock) / 9 (Amundi) en el
writer, vs posicion 10/11 en el CSV commiteado. Enunciado original
del finding ('schema anterior a a3c2705') era incorrecto: el commit
a3c2705 si modifico los CSV, pero el reorder no se propago.

Cierre con reorder byte-preserving de los 3 CSV afectados
(blackrock_dax, blackrock_isf, amundi_lyxi). IWM excluido: su
schema es evolutivo por update_history (concat + drop_duplicates),
21 columnas (13 producidas por el codigo + 8 de legado), no un
desfase.

Verificacion: validate_history_quality.py exit 0; diff 15127
insertions / 15127 deletions; hunk headers identicos; 0 reformateo
de floats; suite 2184 passed + 2 skipped. Commit a0c0ce4.

Hallazgo colateral: validate_history_quality.py (script autonomo
en daily_run.yml:241) SI consume esos CSV, contradiciendo el
enunciado original del finding ('nadie los lee'). Verificado: solo
exige existencia + min_rows=20 + no-NaN en date/nav/shares_outstanding.
No lee flow_zscore_regime.

---

**Anexo 2026-09-27 (post-cierre 2): state.py::save_scenario eliminado.**

Ciclo corto. Deuda diferida del commit aff0afd (F2.4-10). Tras ese
fix, classify_mte dejo de escribir y engine.py paso a ser el unico
writer de mte_state.json. save_scenario (state.py:46) quedo sin uso
productivo.

Gate 0 read-only: 0 usos productivos; unicos consumidores 3 tests
de TestSaveScenario. Ademas era footgun: escribia un subset de 4
campos (schema_version, temporal_contract_version, scenario, pending)
frente a los 12+ del writer inline de engine.py (que anade
effective_date, expected_date, coverage, futures_status, confidence,
msi, ipi). Invocarlo sobrescribia el state file BORRANDO esos campos;
el siguiente load_previous_scenario lo leeria sin errores pero con
state incompleto.

Accion: eliminado de state.py (junto con import os huerfano),
indicators/mte/__init__.py (import + __all__) y
tests/test_mte_state.py (TestSaveScenario, 3 tests).

Verificacion: pyflakes limpio, compileall OK, suite 2181 passed +
2 skipped (-3 tests, esperado). Commit ff782d7.

---

**Anexo 2026-09-27 (post-cierre 3): Cat 4 desmontado + 3 fixes.**

Ciclo de cierre del bloque que PROMPT/TRANSFER agrupaban como
"Cat 4 (20+ providers)". Gate 0 read-only sobre la auditoria
consolidada 2026-09-26 revelo que Cat 4 no es un bloque con entidad:
es un agregado de 3 secciones (FASE 5.6, FASE 5.7, FASE 7), cuyos
hallazgos ya estan mayoritariamente cerrados o subrogados.

Inventario ejecutado:
  - 26 providers clasificados por uso real (imports en src/scripts,
    imports relativos intra-providers, invocacion CLI via workflow).
  - 0 huerfanos reales.
  - 1 falso positivo del Gate 0 inicial: qqq_nport_flow y
    sec_nport_quarters_position_change se invocan como scripts CLI
    desde update_sec_nport.yml, no se importan. Leccion metodologica:
    incluir invocacion CLI en Gate 0 de providers.

3 fixes aplicados (commit 1805766):
  - F5.6-01 fred.is_available: try/except decorativo eliminado.
  - F5.7-01 finra: 2 except desnudos -> tuple tipada
    (requests.RequestException, ValueError, KeyError, TypeError).
  - F7-03 nipc._weighted: dead code eliminado (7 lineas). Logica real
    vive en by_sec.get(k, 0.0).

Fuera de scope: F5.6-02 (fred._download_series) sigue con except
desnudo y ffill ciego entre frecuencias. Es un finding separado.

Estado consolidado de Cat 4:
  - Cerrados: 13 findings.
  - Subrogados / falsos positivos: 2 findings.
  - Backlog diferido: 8 findings (F5.6-02/03/04/06, F5.7-05/14/19,
    F7-01, F7-04). No perseguir por defecto.

Verificacion: pyflakes limpio, compileall OK, suite 2181 + 2 skipped.

---

**Anexo 2026-09-27 (post-cierre 4): cierre de sesion, 12 commits.**

Sesion larga de cierre del backlog de la auditoria consolidada
2026-09-26. 12 commits pusheados. Suite: 2184 -> 2205 + 2 skipped.
10 tests de provider anadidos (cboe, cftc, nport, finra).

Cierres por commit (orden cronologico):
  a0c0ce4, 3aff5be  K-BLACKROCK-CSV-STALE-01 (reorder CSV + docs)
  ff782d7, 3e967ed  state.py::save_scenario eliminado
  1805766, dc6aea2  Cat 4 desmontado + F5.6-01 + F5.7-01 + F7-03
  9cce7dd           F5.6-02b FRED cleanup (DGS2, put_call,
                    get_options_data muerto)
  4fd5d57           F7-04 freshness -> settings.py
  bfe030c           F7-01 sector_correlation_matrix dead end
  743896e           F5.6-03 CboeProvider except tipados
  62d4992           F5.7-14 CFTC thresholds -> settings.py
  b7d1c31           F5.7-19 N-PORT dinamico (trimestres, XML)
  3376bb3           F5.7-05 FINRA get_latest_week memoize
  d84902b           F5.6-06 datetime.now como fecha de ejecucion

Hallazgo metodologico recurrente: ~50% de los findings revisados
estaban subrogados, eran falso positivo, o ya resueltos por ciclos
previos. Refuerza la practica de Gate 0 read-only antes de invertir
ciclos.

Pendiente externo (no accionable internamente):
  - SEC Official List Q2 2026 TXT: sin publicar (curl 404).
  - Fase D (auditor externo IAE): pendiente.

Verificacion: pyflakes limpio, compileall OK, suite 2205 + 2 skipped.

---

**Anexo 2026-09-27 (post-cierre 5): desbloqueo SEC Official List Q2 2026.**

Finding external-dependency: la seccion IAE del reporte llevaba
semanas STALE con stale_reason=official_list_pending. La causa: SEC
habia publicado Q2 2026 con sufijo `-txt` (13flist2026q2-txt.txt),
pero el downloader solo probaba la URL canonica
(13flist2026q2.txt) y obtenia 404.

Gate 0:
  - Verificacion manual: HEAD a URLs canonicas devolvio 403 (rate
    limit de SEC, no bloqueo real). Esperar 60s y repetir.
  - Q1 2026 descarga OK (24641 filas). Q2 2026 404 en la canonica.
  - Q2 2026 SI existe con sufijo `-txt`: 25333 filas, parsea
    correctamente.

Fix (commit 08a6c6a):
  - data/sec_13f/.../URL_EXCEPTIONS: + 2026Q2.
  - Fallback automatico: si 404 en canonica y no esta en excepciones,
    probar sufijo `-txt`. Cubre futuros trimestres sin tocar codigo.
  - data/sec_13f/official_list_13f/13flist_2026Q2.txt versionado.

Verificacion end-to-end (run.py local, exit=0):
  - Seccion IAE del reporte: NIPC Q1 2026 -> Q2 2026 =
    8.254.818.120 (SOLE 5.956.164.211 / DFND 2.328.005.496 /
    OTR -29.351.587). Cobertura catalogo 240/242 (99.17%).
  - Validation Gate 10/10 OK.
  - Secciones ## del reporte: 55 pre, 55 post, identicas.
  - Fix F7-01 verificado: sector_correlation_matrix.csv no regenerado.
  - Fix F5.7-05 verificado: get_latest_week memoizado.
  - Fix F5.7-14 verificado: print CFTC usa las nuevas constantes.

Pendiente externo: Fase D (auditor externo IAE).

Verificacion: pyflakes limpio, compileall OK, suite 2205 + 2 skipped.

---

**Anexo 2026-09-27 (post-cierre 6): auditoria externa del IAE.**

Ciclo de auditoria externa iniciado. Se entrega informe completo al
auditor externo (LLM independiente). Dictamen: APROBADO CON CONDICIONES.

Documentos nuevos creados:
  - docs/auditoria/iae/AUDITORIA_EXTERNA_2026-09-27.md
    Registro consolidado del dictamen + hallazgos H1-A/H1-B/H4/H5.
  - docs/auditoria/iae/TRASPASO_IAE.md
    Documento maestro autocontenido para el asistente entrante.

Hallazgos bloqueantes al cierre:
  - H1-B (clasificacion CALL/PUT incompleta): +53.228.375 de dano sobre
    el NIPC actual. Fix definitivo exige mapping CUSIP->tipo desde
    Official List trimestral.
  - H5.3 (dependencia manual Q2 2026): commit c5e3ee0 con autor humano.

Hallazgos cerrados:
  - H1-A (drift historico del E2E): causa demostrada (regeneracion
    crosswalk Fase G). Prueba forense CUSIP-por-CUSIP.
  - H5.6 (Official List Q2 fail): resuelto con sufijo -txt.

Sin cambios en la logica de calculo productiva del IAE en este ciclo.
El commit 08a6c6a si toco codigo de workflow y scripts de descarga, pero
NO la logica de calculo del NIPC.

Estado del NIPC:
  - -4.317.678.307 = PRE_H1B_OBSERVATION (no baseline).
  - -4.264.449.932 = resultado de mitigacion (regex TITLEOFCLASS).
  - Baseline contractual final: pendiente de cierre H1-B.

Siguiente paso: Gate 0 de H1-B (verificar que las 3 Official List
locales contienen Option Indicator parseable).

Verificacion: pyflakes limpio, compileall OK, suite 2205 + 2 skipped.

---

**Anexo 2026-09-28 (post-cierre): H1-B + H4 + H2 + H3 + H5.1-5.5 + O1 cerrados.**

Sesion de cierre del dictamen externo IAE 2026-09-27. 14 commits
pusheados. Suite: 2205 -> 2226 + 2 skipped.

**Hallazgos cerrados:**

| Hallazgo | Estado final |
|---|---|
| H1-A | CERRADO |
| H1-B | CERRADO con C2 abierto no bloqueante |
| H2 | CERRADO (42 vs 45, reconciliacion 27+17+1=45) |
| H3 | CERRADO (tabla de clasificacion de scripts) |
| H4 | CERRADO (golden/current.json) |
| H5.1 | CERRADO (SEC_13F_QUARTER_LAG_DAYS unificado) |
| H5.2 | CERRADO (retry + Issue automatico) |
| H5.3 | TRAZABILIDAD IMPLEMENTADA - pendiente cron nov 2026 |
| H5.4 | CERRADO (validacion regex YYYYQn) |
| H5.5 | CERRADO (cache key v3 con hash del manifest) |
| H5.6 | CERRADO (historico) |
| H5.7 | HISTORICO |
| O1 | CERRADO (barrido + fix SSHPRNAMT) |

**H1-B v2.3 - resumen tecnico:**

Fix implementado en `sec13f_list.py` (classify_instrument_type),
`operational_universe.py` (_apply_5_3b) y
`delta_shares.py::compute_reported_position_units` (filtro primario
por Official List).

Regla: EQUITY confirmada incluye; OPTION confirmada excluye;
UNRESOLVED no se imputa (mantiene). Word-boundary en regex
(`\b(?:CALL|PUT|OPTION|OPT)\b`).

Baseline local reproducible: `-4.264.449.012` (commit 72fa824).
Referencia externa declarada por auditor: `-4.264.449.932`.
`reconciliation_delta = 920`, `reconciliation_status = OPEN`.
Detalle: INFORME_H1B_RECONCILIACION_FINAL.md, golden/current.json.

**Hallazgo metodologico registrado:**

El primer informe presento el `-4.264.449.932` como si fuera un
golden medido, con "coincidencia dentro del margen de redondeo".
El auditor lo rechazo: no es reproducible. Corregido a "no
reproducido bajo las configuraciones y entorno controlados
actualmente auditados". Este hallazgo metodologico se documenta
como leccion permanente: distinguir siempre entre "valor declarado"
y "valor verificado".

**Commits pusheados:**

    fd1b01b  PROMPT v7.15 + TRASPASO actualizado
    e16737e  cierre H1-B - H4 cerrado + current.json
    4ed2bcf  aclarar que el 932 no es golden reproducible
    528c088  informe reconciliacion final H1-B
    72fa824  O1 - imputacion SSHPRNAMT
    bd8a501  H2 + H3
    7870a57  H5_WORKFLOW_TRIMESTRAL cierre
    c042fef  H5.1-H5.5 workflow
    0308922  informe H1-B post-implementacion
    ce32c77  H1-B v2.3
    54d8435  H1-B v1
    321c385  cierre parcial H1-B + H5 + H4

**Pendientes reales al cierre:**

- H5.3: verificacion en cron real de noviembre 2026.
- C2 (920): comando + HEAD del auditor, o aceptacion definitiva
  de `reconciliation_status = OPEN`.

Sin bloqueantes activos.

---

FIN DE TRANSFER. 2026-09-28.
