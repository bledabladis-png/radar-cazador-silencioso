# TRANSFER - Guia de onboarding

**NO es fuente de estado.**
**Estado vivo:** `iae/ESTADO_SISTEMA.md` + `iae/ESTADO_DECLARADO.md`.
**Referencia unica del modulo IAE:** `iae/IAE_MAESTRO.md`.

---

## Rol

Ingeniero Supervisor del Radar de Rotacion Sectorial (IAE).
Reglas de personalidad y metodo: `PROMPT_MAESTRO.md` secciones 1 y 3.

---

## Estado al cierre de la sesion (2026-09-24)

    HEAD                 ver docs/auditoria/iae/ESTADO_SISTEMA.md
    Ahead                0 (sincronizado con origin/main)
    Working tree         LIMPIO
    Tests IAE            845 passed (criterio AST, 45 ficheros)
    Suite global         1587 passed + 2 skipped + 0 failed
                         (los 3 test_freshness pasan tras run.py; vuelven
                          a fallar si los parquets llevan >4 dias sin
                          refrescar - ver nota abajo)
    Push                 SI (integrado en produccion desde 2026-09-24)
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

    docs/auditoria/PROMPT_MAESTRO.md              v7.2
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
Con el crosswalk CUSIP regenerado por script (245 filas, 242 tickers):

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
- 1587 passed + 2 skipped + 0 failed (test_freshness pasa tras run.py)
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
- Crosswalk regenerado por script (245 filas, 242 tickers)
- Cache 13F v2 con 3 trimestres en GitHub
- Seccion IAE STALE por Official List Q2 pendiente de SEC (comportamiento correcto)
- Fases A-F cerradas; Fase G (automatizacion de mappings) cerrada 2026-09-24
- Fase D (auditor externo) PENDIENTE
- Deuda visible en el reporte: bugs de render + cobertura sectorial baja
- Pendiente: SEC publique Official List Q2 TXT

Pregunta: "Que hacemos?"

---

FIN DE TRANSFER. 2026-09-24.
