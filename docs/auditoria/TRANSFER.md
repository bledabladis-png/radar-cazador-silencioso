# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` + `iae/ESTADO_DECLARADO.md`.
**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Rol

Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la sesion (2026-09-26)

    HEAD                 ver docs/auditoria/iae/ESTADO_SISTEMA.md
    Ahead                0 (sincronizado con origin/main)
    Working tree         LIMPIO
    Tests IAE            845 passed (criterio AST, 45 ficheros)
    Suite global         1857 passed + 2 skipped + 0 failed
                         (los 3 test_freshness pasan tras run.py; vuelven
                          a fallar si los parquets llevan >4 dias sin
                          refrescar - ver nota abajo)
    Push                 SI (integrado en produccion desde 2026-09-26)
    Cobertura IAE        90% lineas (reproducible con scripts/iae_coverage.py)

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

    docs/auditoria/PROMPT_MAESTRO.md              v7.4
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
- Seccion IAE en el reporte diario. Actualmente STALE con razon honesta:
  SEC no ha publicado la Official List 13(f) de Q2 2026 en formato TXT
  (solo PDF desde 2026-08-14). El NIPC se calculara automaticamente
  cuando SEC publique el TXT.

NIPC Q1 2026 -> Q2 2026 pendiente de dato externo. Ultimo calculado:
NIPC Q4 2025 -> Q1 2026 = -4.316.734.936 (referencia historica).

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
- Dispatch manual de `daily_run.yml` solo FUERA de la sesion USA abierta
  (13:30-20:00 UTC). El cron productivo es `0 23 * * *` UTC. Dispatches
  manuales en ventana USA abierta disparan guard_coverage. Leccion de
  F-IAE-CRON-01 (2026-09-24).
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
- 1682 passed + 2 skipped + 0 failed (test_freshness pasa tras run.py)
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
- Seccion IAE STALE por Official List Q2 pendiente de SEC (comportamiento correcto)
- Fases A-F cerradas; Fase G (automatizacion de mappings) cerrada 2026-09-24
- Fase D (auditor externo) PENDIENTE
- Bugs de render C1-C3 RESUELTOS 2026-09-24 (commits 2fe1c45, 740be35, dfd93c4)
- Cobertura sectorial A1 RESUELTA 2026-09-24 (top-20 real, commit 8c6330f)
- FU-002-bymarket integrado (manifest + guard, commits 88c27d1, d09f928, 62e38d9)
- Health check semanal operativo (lunes 07:00 UTC)
- Pendiente: SEC publique Official List Q2 TXT

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

**Verificacion en CI real.** Dispatch manual 36189310171: gate dio
CURRENT (manifest cubria sesion anterior). run-system skipped.
Override no ejecutado en ese run; el codigo esta en produccion y se
activara cuando el gate de READY.

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
  produccion donde el gate de READY (proximo cron con target_session
  = hoy y manifest desactualizado).

---

FIN DE TRANSFER. 2026-09-26.
