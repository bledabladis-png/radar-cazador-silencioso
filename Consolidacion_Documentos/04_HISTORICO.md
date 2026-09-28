# 04 - HISTORICO

**Cronologia de decisiones. NO normativo. NO se pega al arrancar.**
**Para "por que esta asi" este documento. Para "que se hizo" -> 05_BITACORA.md.**
**El detalle granular esta en `git log`.**

---

## 1. PROPOSITO

Este fichero responde a la pregunta: *por que este sistema esta construido asi*. Agrupa decisiones por bloque tematico. No es exhaustivo — el detalle granular esta en los commits. Referencia: cada entrada apunta a commits concretos y al hallazgo o ciclo que la origino.

**No confundir con:**
- `02_ARQUITECTURA.md`: como funciona el sistema hoy (normativo).
- `05_BITACORA.md`: que se hizo en las ultimas sesiones.
- `git log`: cronologia detallada commit a commit.

---

## 2. LINEA TEMPORAL (sep 2026)

Cronologia de las sesiones principales que dieron forma al sistema actual.

### 2026-09-11 a 2026-09-15: Saneamiento base

- Cierre de la primera fase de correcciones A (A1, A2, A3).
- Reconstruccion del calendario NYSE (`market_calendar.py`) y horarios (`market_hours.py`).
- Nacimiento de FU-001 (ffill multi-calendario), FU-002 (validacion circular backup), FU-007 (walk-back de fechas), FU-008 (sector_concentration).
- Introduccion del manifest de artefacto (FU-002).

### 2026-09-16: FU-021-5 contratos temporales

- Ciclo completo de FU-021-5 (20 commits: c1a1df9..7560e12).
- 10 contratos en 5 familias, FSM 5 estados.
- `temporal_meta` como autoridad, `MarketDataBundle` como transporte.
- Este ciclo sento la base de la resolucion temporal actual.

### 2026-09-17: DT1, DT2, DT3 (desmonolitizacion)

- **DT1**: `regimes/sector_regime.py` 827 LOC brutas -> 272 lineas reales. Commit a1406f4.
- **DT2**: `indicators/mte.py` (1964 LOC) -> paquete `indicators/mte/` (5 submodulos). Golden regenerado.
- **DT3**: `indicators/darkpool.py` (286 LOC) -> 4 modulos. Fix normativo `fecha=week_start`.

### 2026-09-18: K-STOCK-PRICES-EOD-01

- Detectado bug latente: commits bloqueados en ventana desfasada Europa-USA.
- Resuelto con `guard_coverage.py` (commit f3114b4).
- Reproducido 2x. Verificado en CI real (run 35376243662).

### 2026-09-22 a 2026-09-23: Fases IAE A/B/C

- **Fase A** (bloqueantes dictamen auditor): B1 reconciliacion radar+complemento, B2 cadena forense, B3 validacion OpenFIGI (227/243 CUSIPs), B4 riesgo PIT declarado.
- **Fase B** (13 criticos): coherencia numerica, semantica NIPC, cadena contractual.
- **Fase C** (3 funcionales): C8 reporting_dedup diagnostico separado, C9 stats observables, C10 coverage_available + coverage_quality.

### 2026-09-24: BACKLOG + FU-002-bymarket + health check

- **Backlog del reporte** (bugs de presentacion, no pipeline): C1 _delta con gaps, C2/C3 filtros de representatividad, B1 Flujo de Mercado, B2 criterio lideres, D1 SPY benchmark, D2c FLOW_CONFIDENCE, D3/E1 notas semanticas, A1 cobertura top-20.
- **FU-002-bymarket**: manifest con cobertura por mercado. Guard con exencion condicional. Cierra falso bloqueo en ventana Europa-USA.
- **Health check semanal** (`health_check.yml`): 7 bloques.
- **Bug latente A/D**: 206/313 tickers USA sin Close el 22-Sep. Fix: exigir continuidad temporal via `previous_market_day`. Commit fc21671.
- **Registry consolidacion**: `YAHOO_TICKER_MAP` + `normalize_yahoo_ticker` a `instrument_registry.py` (fuente unica). Fix derivado: `get_market('BRK.B')` -> `US_EQUITY`.
- **F-IAE-HOLIDAY-01**: tickers USA en festivos NYSE preservan NaN (no ffill).
- **F-IAE-CRON-01**: cron `daily_run.yml` de `0 4 * * *` a `0 23 * * *`. SUPERSEDED por F-IAE-CRON-02.

### 2026-09-25: F-IAE-CRON-02 / F-IAE-GATE-01

- **Causa raiz**: run 36080921484 con 262/313 tickers USA sin Close. Yahoo devolvio la fila de `expected_session` con Close=NaN. Latencia variable (5h26m a 100% NaN, 16h12m a 0% NaN).
- **Solucion**: multi-slot (17 23 / 17 3 / 17 7 / 17 11 UTC) + gate pre-pipeline (`scripts/pipeline_gate.py`) + `scripts/issue_manager.py`.
- Idempotencia por cobertura, no por `last_date`.
- Verificado en CI real (run 36148143256).

### 2026-09-26: F-IAE-LSE-INTEGRATION

- **Problema**: 20 tickers `.L` dependian exclusivamente de Yahoo. Ventana post-cierre LSE con Close=NaN.
- **Solucion**: scraper privado `lse-close-scraper` (Refinitiv Widgets). Override parcial de Close sobre fila ya presente de Yahoo. NO sustituye OHLCV.
- **Invariante:** el radar NO ejecuta codigo del repo externo.
- **Verificacion CI:** run 36208872855. Log: `[LSE-OVERRIDE] session=2026-09-25 applied=20/20 status=OK`.
- **+91 tests** (1766 -> 1857 passed).
- **K-LSE-YAHOO-REVISION-01** (MONITORED): Yahoo revisa OHLC retrospectivamente. No es bug.

### 2026-09-26: Auditoria radar consolidada

- Informe externo con 205 hallazgos: 6 ALTA, 98 MEDIA, 101 BAJA.
- **6 ALTA corregidos** con commits verificados:
  - F3-05 OilPriceAPI 403 futuros (ff95a3d).
  - F3-16 writer spot INVALID (84b2708).
  - F3-17 best-effort enmascarando 401/403/429 (2c8b81f).
  - A5-13 Yahoo fallback devolviendo parquet completo (4808a13).
  - F6-10 tres "lideres" por sector (b5884a1).
  - F6-28 dos "A/D Net" contradictorios (6f4f4b1).
- Fases 0-7 auditadas (consolidado 205 hallazgos).

### 2026-09-27: Ciclos de cierre del backlog

Ciclos completados (12 en la sesion principal):
- **Cat 1**: A5-71, F3-18, F2.4-01/02/03. +66 tests.
- **Fix side-effect** en test_indices_intl.
- **19 warnings pyflakes** limpiados.
- **F2.4-11 / F2.4-13**: except acotado + WONT FIX documentado.
- **F2.4-10 / F2.4-12**: engine.py unico writer del state MTE, histéresis reactivada.
- **F2.4-04**: dead code NFCI/OAS eliminado. -182 LOC.
- **FINRA cache**: 68s -> 0.04s. Commit 062cd10.
- **Cron probe + 5º slot + health-check bloque H**.
- **A5-70**: refactor BlackRock DAXEX + ISF. -153 LOC netas.
- **K-BLACKROCK-CSV-STALE-01**: CERRADO (rectificacion del enunciado).
- **state.py::save_scenario eliminado**: 0 usos productivos.
- **Cat 4 desmontado**: 3 fixes (F5.6-01, F5.7-01, F7-03).
- **F5.6-02b, F7-04, F7-01, F5.6-03, F5.7-14, F5.7-19, F5.7-05, F5.6-06** cerrados.
- **SEC Official List Q2 2026 desbloqueo** (sufijo `-txt`). Commit 08a6c6a.

### 2026-09-28: Cierre masivo de deuda consolidada

- **Sesion IAE** (14 commits): cierre H1-A, H1-B v2/v2.3, H2, H3, H4, H5.1-H5.5, O1. Baseline local reproducible -4.264.449.012. Referencia auditor -4.264.449.932 (delta 920, OPEN).
- **Sesion radar** (30 commits): cierre FASE 3, 5 completas; FASE 7; FASE 2.2; FASE 2.4. ~28 commits de cierre.
- Suite: 2205 -> 2267 passed + 2 skipped.
- **Auditoria interna A1-A5** iniciada y cerrada (nucleo temporal, providers, calculo, reporte, pipeline). Bugs estructurales corregidos: bug en `leaders.py`, check muerto en `mte_confirmation.py`, guards en `indices_intl.py`, SECTOR_ETFS unificado.
- **Refactor documental**: creacion del corpus consolidado (00-05). Borrado del corpus antiguo pendiente.
---

## 3. HALLAZGOS POR BLOQUE TEMATICO

Agrupacion de los cierres mas relevantes por area. El detalle granular esta en `git log`. Los IDs (FU-xxx, K-xxx, F2.4-xx, A5-xx, DT-x, H-x) son de la nomenclatura interna de la auditoria y no se usan ya en el trabajo activo.

### 3.1. Nucleo temporal

- **FU-001** (ffill multi-calendario): RESUELTO 2026-09-15 (38f9ce1 + 8a76380).
- **FU-007 / FU-007-b** (walk-back freshness): RESUELTOS (8c4e111 / 4d5511c). Fix: `_last_market_session(d)` en `market_calendar.py`.
- **FU-016** (desfase 1d entre writers pre-PUBLISH_HOUR): OBSOLETO 2026-09-17.
- **FU-021-3A** (filtro EQUITY_EOD en market_data): RESUELTO correction v2 2026-09-15 (78b7583 + 747cfb1 + fded14b).
- **A1-07** (calendario NYSE solo 2026-2027): CERRADO 2026-09-28. Festivos calculados algoritmicamente (Computus + nth-weekday + observed).
- **A1-08/A1-10** (tz naive en market_calendar): CERRADO. Default `datetime.now(Europe/Madrid)`, normalizacion tz.
- **A1-11** (bucles sin guard): CERRADO. Limite 366 iteraciones + RuntimeError.

### 3.2. Providers

- **FU-014** (Xetra/BME gap >= 2 dias): RESUELTO 2026-09-16 (3105689). Walk-back al siguiente dia bursatil.
- **FU-015** (columnas duplicadas tras retry Yahoo): RESUELTO 2026-09-15 (afcf095).
- **K-DATA-LOADER-01** (cache-hit bypass post-procesado): CERRADO 2026-09-17 (9d77099).
- **A5-13** (Yahoo fallback devolvia parquet completo): CORREGIDO 2026-09-26 (4808a13).
- **A5-15** (router no cubre subset parcial silencioso): CORREGIDO 2026-09-28.
- **A2.2-01** (BME 0/19 en run): CERRADO 2026-09-28. `today = last_expected_market_date(reference_date)`.
- **A2.2-02/A2.2-03** (tz-naive en cobertura/calidad): CERRADO 2026-09-28. Helpers `_ref_to_date`.

### 3.3. Calculo

- **FU-012** (A/D Line acumulada ±1): WONT FIX razonado. El valor es un contador descriptivo.
- **FU-013** (SLPM 0% con n=0): RESUELTO 5d3bc75. Campos se muestran como N/D.
- **A3.1-01** (except desnudo en `liquidity.py`): CERRADO 2026-09-28. Documentado.
- **A3.1-03** (aridad inconsistente de `compute_liquidity_score`): CERRADO. Aridad 4-tuple uniforme.
- **A3.2-01** (SECTOR_ETFS duplicado en 7 indicadores): CERRADO 2026-09-28. Migrados a `MARKET_TICKERS['sectors']`.
- **A3.2-05/A3.2-07**: import duplicado consolidado, mojibake falso positivo rectificado.
- **A3.3-01** (bug real: `robust_zscore` devolvia ndarray en rama mad==0): CERRADO 2026-09-28. Devuelve Series en los 3 paths.
- **A3.3-12** (whitespace inflation en MTE): CERRADO 2026-09-28. -565 lineas.
- **A3.3-13** (test anclado a numero de linea): CERRADO. Busqueda por estructura.

### 3.4. Reporte

- **FU-003** (cosmetico +0.00): OBSOLETO 2026-09-17. FU-003b cubre n=0.
- **FU-003b** (SLPM con n=0): RESUELTO 5d3bc75. Muestra N/D.
- **FU-005** (WARN analisis_lideres.csv): RESUELTO a372028.
- **F6-01** (`MARKET_CLOSED` vs `DATA_PENDING`): CERRADO 2026-09-28.
- **F6-02** (FINRA 26 dias marcado CURRENT): CERRADO. Umbrales ajustados.
- **F6-10** (3 lideres distintos por sector): CORREGIDO 2026-09-26 (b5884a1).
- **F6-28** (2 A/D Net contradictorios): CORREGIDO 2026-09-26 (6f4f4b1).
- **A4-01** (patron NaN en 26 sitios de 7 renders): CERRADO 2026-09-28. Uso sistematico de `_fmt_num`.
- **A4-06** (12 except acotados en 7 renders): CERRADO.

### 3.5. Pipeline y orquestacion

- **A5-01** (dead code en `data_load.py`): CERRADO 2026-09-28.
- **A5-07** (bug estructural en `leaders.py`: `NameError` silenciado en rama INSUFFICIENT_COVERAGE): CERRADO. Indentacion corregida.
- **A5-08** (`'all_signals' in dir()` siempre True): CERRADO. Check real con `is not None and hasattr(columns)`.
- **A5-09/A5-10** (guards faltantes en `indices_intl.py`): CERRADO.
- **A5-11** (`SECTOR_ETFS` duplicado en `engines.py` y `diagnostics.py`): CERRADO. Migrado a config.

### 3.6. IAE

- **FA-1 / FA-2**: ingestion + schema + identity. CERRADAS.
- **Fases A, B, C**: bloqueantes + criticos + funcionales. CERRADAS.
- **Fase D** (auditor externo): CERRADA 2026-09-28. Dictamen APROBADO CON CONDICIONES, luego cerrado con C2 abierto no bloqueante.
- **Fases E, F, G**: integracion a `run.py`, automatizacion GitHub, automatizacion de mappings. CERRADAS.
- **H1-A**: CERRADO. Crosswalk 7caa86b reproduce golden con tolerancia 0.
- **H1-B** (clasificacion CALL/PUT): CERRADO v2.3. Baseline -4.264.449.012.
- **H2** (contadores documentales): CERRADO.
- **H3** (clasificacion scripts): CERRADO.
- **H4** (golden versionado): CERRADO.
- **H5.1-H5.5**: CERRADOS. H5.3 pendiente cron nov 2026.
- **O1** (ausencia de imputacion): CERRADO.

### 3.7. Workflows / CI

- **F-IAE-CRON-01**: cron 04:00 -> 23:00 UTC. SUPERSEDED.
- **F-IAE-CRON-02 / F-IAE-GATE-01**: multi-slot + gate pre-pipeline. Verificado en CI.
- **F-IAE-LSE-INTEGRATION**: scraper LSE. Verificado en CI (run 36208872855).
- **K-CI-CRON-01**: contratos STALE fuera de cron. CERRADO pasivo.

### 3.8. Bugs estructurales recientes (auditoria interna A1-A5)

- **A1**: nucleo temporal. 5 commits. Calendario algoritmico, tz Madrid, guards.
- **A2**: providers. BME walk-back, tz-naive, SSGA fund-flow, retry_call.
- **A3**: calculo. Regimenes, breadth, MTE, darkpool, SLPM.
- **A4**: reporte. Guard NaN en 26 sitios de 7 renders. 12 except acotados.
- **A5**: pipeline. 3 bugs estructurales. ~50 except acotados.

### 3.9. Falsos positivos rectificados

Casos donde un "hallazgo" no era tal y se documento la causa.

- **F2-25**: el fix de continuidad temporal (`fc21671`) estaba aplicado; la auditoria previa lo daba por no propagado.
- **F2-26**: `guard sector_price is not None` es redundante pero no bug.
- **A2.4-08/09**, **A3.2-07**, **A4-04**: 4 falsos positivos de "mojibake" por artefacto de consola (CP850 vs UTF-8).
- **A3.3-09**: `composite` NaN con leader_flows vacios. `_fmt_num` lo maneja correctamente.
- **A3.2-03**: F2-25 duplicaba fc21671 con dominio distinto.
---

## 4. LECCIONES METODOLOGICAS

Las lecciones operativas ya estan destiladas en `01_METODO.md` seccion 8. Aqui se registra **de que caso concreto salio cada una** (util para entender por que existe la regla).

- **"Un here-string no es un archivo."** De la sesion 2026-09-17 (H1 briefing, 95 lineas colgando PowerShell). Confirmado 2026-09-28 al escribir `02_ARQUITECTURA.md` (600 lineas descartadas silenciosamente).
- **"`py -c` con comillas anidadas rompe."** De 3 tropiezos en 2026-09-17.
- **"Backticks en here-string se corrompen."** De 4 tropiezos en 2026-09-17 (H1 briefing).
- **"Verificar codepoints antes de declarar mojibake."** De 4 falsos positivos (A2.4-08/09, A3.2-07, A4-04). `Get-Content` en consola CP850 muestra UTF-8 corrupto. Se verifica con `read_bytes().decode('utf-8')` + `repr()`.
- **"Un test que asume comportamiento generico es un contrato."** De A2.2-3a (Xetra `_open_ws`/`_query_one`) y A5.2 (`RuntimeError` en `sectors_base`).
- **"En el pipeline, `RuntimeError` va SIEMPRE en tuplas de except."** De 3 tests que usan `side_effect=RuntimeError` para simular fallos (A5.2, A5.4).
- **"Antes de unificar una constante, grep de consumidores indirectos via config."** De 2 fallos en A3.2 (`regimes/sector_regime.py`, `cross_asset_context.py` importan de `MARKET_TICKERS['sectors']`, no aparecen buscando literales).
- **"Al regenerar un golden, probe de invariancia ANTES."** De A3.2-b. El generador sintetico acoplaba el drift a la posicion del ticker en config. Si no se hubiera arreglado, cualquier reorden de `MARKET_TICKERS['sectors']` habria roto el golden.
- **"AST check bloqueante para refactors de whitespace."** De A3.3-c-2 (MTE, -565 lineas). Sin AST check, un colapso puede alterar semantica silenciosamente.
- **"El sistema prevalece sobre la documentacion."** Doctrina desde la primera auditoria. Aplicado en A3.2 (config `XLU,XLRE` manda sobre literales), A4 (codigo manda sobre docstrings).
- **"Ver la salida del patch antes de la verificacion."** De A4-b-2-residual y A5.5. Un patch multi-anchor que aborta silenciosamente es comun. Sin ver `exit=` y `[OK]`/`[ABORT]`, la verificacion mide un estado que no es el esperado.
- **"Un falso positivo es una leccion."** Documentarlo en el commit evita repetirlo.
- **"Antes de autorizar un patch sobre Pandas, verificar la semantica exacta del objeto."** Doctrina desde el ciclo C1.

---

## 5. ESTADO DEL CORPUS DOCUMENTAL

Antes del 2026-09-28, el corpus documental era:
- `PROMPT_MAESTRO.md` (85.8 KB, 1572 lineas)
- `TRANSFER.md` (43.1 KB, 990 lineas)
- `ESTADO_DECLARADO.md` (34.5 KB, 627 lineas)
- `ESTADO_SISTEMA.md` (5.0 KB, 121 lineas, auto-generado)
- `IAE_MAESTRO.md` (108.5 KB, 2462 lineas)
- `TRASPASO_IAE.md` (12.7 KB, 366 lineas)
- ~15 documentos historicos (auditorias, informes, disenos)
- 8 `evidence/*/README.md`
- 3 `radar/audits/*.md`

**Total: ~570 KB.**

El 2026-09-28 se inicia el refactor documental. Nuevo corpus:
- `Consolidacion_Documentos/00_ARRANQUE.md` (arranque, se pega).
- `01_METODO.md` (metodo, on-demand).
- `02_ARQUITECTURA.md` (mapa, on-demand).
- `03_IAE.md` (IAE, on-demand).
- `04_HISTORICO.md` (este fichero, no se pega).
- `05_BITACORA.md` (sesiones recientes, no se pega).

**`ESTADO_SISTEMA.md`** se genera en `Consolidacion_Documentos/ESTADO_SISTEMA.md` por `scripts/generate_estado_sistema.py`. Se regenera con cada commit. Se referencia desde `00_ARRANQUE.md`.

**Pendiente:** borrar los ~570 KB del corpus antiguo una vez que los 6 ficheros esten verificados.

---

## 6. REFERENCIAS RAPIDAS

- **Precedente de "conflicto entre doc y codigo":** A3.2-b (config `XLU,XLRE` vs literales `XLRE,XLU`). El codigo gana.
- **Precedente de "falso positivo de auditoria":** F2-25 (fix ya aplicado, la auditoria no lo vio).
- **Precedente de "bug silenciado por except outer":** A5-07 (`leaders.py`, `NameError` en INSUFFICIENT_COVERAGE oculto como "Modulo de lideres omitido").
- **Precedente de "test fragil":** A3.3-13 (test anclado a numero de linea 89 de engine.py).
- **Precedente de "cache freshness vs observacion":** A2.1-01 (mtime de fichero = execution time, no fecha de observacion).
- **Precedente de "schema aditivo backward-compatible":** FU-002, politica 2026-09-24. `quality.by_market` anadido sin bump de `schema_version`.

---

**Fin del historico.**