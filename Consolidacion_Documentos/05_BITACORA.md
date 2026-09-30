# 05 - BITACORA

**Registro de sesiones recientes. NO normativo. NO se pega al arrancar.**
**Para sesiones anteriores, ver `git log`.**
**Se poda por antiguedad: mantener las ultimas 5 sesiones. La 6ª se elimina o se resume en 04_HISTORICO.md.**

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

Al cerrar una sesion nueva, se anade arriba (las mas recientes primero). Si hay mas de 5, la mas antigua se elimina o se resume en el historico.

---

## 3. SESIONES

### 2026-09-30 (tarde, sesion 2) — D5+D6+D6b+D7+D9+D10+D11+D12+D13+D14+D15

**Objetivo.** Continuacion de la sesion tarde. Atacar deudas menores
tras cerrar P1. Algunas eran cosmeticas (BOM, dead code), otras
reales (D10, D11, D13, D6b).

**Hecho.**

- **D5 (corpus desincronizado).** Cinco ficheros con numeros y
  afirmaciones desfasadas: 00_ARRANQUE decia 2340 tests (real 2383),
  corpus v2 (real v3 con 07_RUNBOOK), "10 contratos" (real 9),
  "OilPriceAPI commodities" (eliminada 2026-09-30), 03_IAE decia
  8280 LOC (real 7088), 04_HISTORICO afirmaba erroneamente "H1-A
  reproduce golden con tolerancia 0".

- **D6 + D6b (datetime.now en reporte).** El header del reporte
  usaba datetime.now() sin tz. Violaba 00_ARRANQUE §2. Fix:
  generate_daily_report recibe reference_date, con fallback a
  now(ZoneInfo('Europe/Madrid')). Ademas 6 usos de datetime.now() en
  calculos de "age" (darkpool, freshness, helpers, sentiment). Todos
  propagados a reference_date naive.

- **D9 (concurrency.group desalineado).** update_index_holdings.yml y
  update_european_holdings.yml escriben ambos data/index_holdings.csv
  con grupos distintos. Fix: unificar en update_index_holdings_csv.

- **D10 (traza 13F).** _write_ingest_trace escribia
  ingest_source/ingest_actor en el manifest del trimestre, incluso
  en rama SKIP. El manifest es artefacto de integridad (sha256).
  Fix: append a data/sec_13f/ingest_traces.jsonl (JSONL versionado),
  manifest intacto, outcome explicito (SKIP/INGEST).

- **D11 (tickers residuales en holdings).** Los 4 parsers volcaban
  al CSV todo ticker no vacio: cash ("-"), futuros (IXAU6, XARU6),
  CUSIPs (2602335D), placeholders SSGA (999USDZ92). El consumidor
  filtraba con INVALID_TICKERS fragil. Fix: src/holdings_filter.py
  con regla estructural, aplicada en 4 parsers, INVALID_TICKERS
  retirada, CSV limpiado (526->502 y 2669->2661 filas).

- **D12 (cosmetico).** D7 (volumen Yahoo no consolidado) documentado
  como WONT FIX en 02_ARQUITECTURA §11. BOM UTF-8 retirado de 7
  workflows. Duplicacion de titulo en 05_BITACORA:91 corregida.

- **D13 (macro_conf hardcode).** compute_macro_regime devolvia
  conf = 0.5 siempre. El aviso de header.py (macro_conf < 0.30)
  nunca se activaba. Los otros 3 regimenes SI calculan confianza
  real. Fix: confidence_from_range (misma politica C19).

- **D14 (tests skipped CI).** Verificado: el saliente reportaba
  deuda de cobertura. FALSO: daily_run.yml ya ejecuta los tests de
  frescura en post-run con datos reales. Solo era cosmetica: sin -ra
  no mostraba razones, reason strings mentian ("CI fresco" tambien
  es falso en local), sin marker. Fix: pytest.ini con -ra + marker
  integration_real_data, reasons corregidos, daily_run pre-run
  excluye el marker.

- **D15 (dead code declarado).** Eliminados:
  finra.get_archive_index, finra.get_available_weeks,
  evidence_matrix.save_evidence_matrix, import Path huerfano.

**Commits.** `cb217ac`, `ec1f2da` (snapshot), `5ecafcb` (rango
c193eef..5ecafcb, 8 commits adicionales: D5, D6, D6b, D9, D10, D11,
D12, D13, D14, D15).

**Pendiente.**

- H5.3 verificacion cron nov 2026.
- C2 (920) OPEN. Bloqueado por auditor externo.
- 2 tests skipped por --run-network en test_freshness (opt-in).
- Modulos con cobertura baja: macro_manual_loader (12%),
  european_coverage (12%), pipeline_contractual (27%),
  data_loader (50%).
- Cron trimestral 1-oct-2026: verificar con 07_RUNBOOK.

**Proximo paso sugerido.** Cron 1-oct (04:47, 06:17, 07:17 CEST).
Aplicar 07_RUNBOOK §3. Si algo falla, §4.

---

### 2026-09-30 (tarde) — D4 + D3 + D2 + Runbook + D1: cierre de P1

**Objetivo.** Recibir traspaso del asistente saliente, asimilar contexto,
atacar la deuda abierta por prioridad. P1 tenia tres frentes (D1, D2, D3)
mas D4 (puntual) y una R (runbook del cron trimestral que dispara el 1-oct).

**Hecho.**

- **D4 (lag 13F unificado).** El saliente lo reporto como "falla en dispatch
  manual fuera de {feb,may,ago,nov}". Verificado: los 3 runs X eran
  `cancelled`, no fallos. No habia incidente. Pero al investigar aparecio
  una deuda real: **tres politicas de lag distintas convivian** (doc ~45d,
  cron ~50d, script 60d). El comentario del YAML mentia sobre su propio
  codigo. Fix: eliminar el `case` del YAML, delegar a `--latest`, unificar
  `SEC_13F_QUARTER_LAG_DAYS` a 50d, 4 casos frontera en test para que la
  suite no sea ciega al cambio.

- **D3 (health_check roto por diseno en CI).** El workflow no restaura
  cache; el runner no tiene parquets (no versionados, confirmado con
  `git cat-file` -> `exists on disk, but not in HEAD`). Consecuencia: 2 FAIL
  estructurales en cada run + 9 WARN ruidosos. Fix: `IS_CI` detecta el
  entorno, `parquet` y `iae_section` pasan a SKIP en CI (FAIL/WARN en local),
  `cron_slots` relaja umbral a 0->OK/1-2->WARN/3+->FAIL (delays 2-8h de
  GitHub son ruido, no incidente), workflows trimestrales sin runs -> SKIP.

- **D2 (determinismo vs Yahoo).** El saliente mezclaba dos problemas: A/D Net
  depende de Close, no de Volume; el volumen es D7. `K-LSE-YAHOO-REVISION-01`
  no era huerfano: ya tenia ficha en 02_ARQUITECTURA y 04_HISTORICO como
  MONITORED. Decision de arquitectura (opcion A del usuario): aceptar y
  documentar. `00_ARRANQUE §2` pasa de "Determinista" a "Determinista **dado
  un snapshot del input**". Nuevo check `check_yahoo_revision` compara
  manifest actual vs HEAD por sha256: mismo `last_date` con sha distinto ->
  WARN (revision detectada). 5 tests del helper.

- **R (runbook del cron 1-oct).** Los 4 workflows trimestrales **nunca se
  han disparado por schedule** en historial visible. Todos los runs
  conocidos son `workflow_dispatch`. El cron del 1-oct sera la primera
  ejecucion automatica real. Creado `Consolidacion_Documentos/07_RUNBOOK.md`
  con inventario de crons, verificacion post-cron y contingencias
  (`startup_failure`, conflicto de rebase, cron no disparado).
  Detectado que el traspaso listaba `update_sec_nport` como dia 1; el YAML
  dice dia 20.

- **D1 (comp_breadth + penalizacion dispersion).** El saliente lo describia
  como "tanh_normalize colapsa cuando la serie es constante". Verificado: es
  correcto, 27-61% de filas por sector con `comp_breadth==0`. Pero al
  verificar aparecio un **segundo bug independiente**: `_dispersion =
  std / (|mean|+1e-9)` explotaba cuando la media se aproximaba a cero,
  aniquilando el score de XLU, XLI, XLB. Fix breadth (mapeo directo
  `(breadth-0.5)*2`), fix dispersion (std poblacional `ddof=0` sin dividir
  por media, garantia penalty en [0.5,1]), 5 tests del contrato,
  verificacion triple (verde con fix, rojo con `ddof=1`, verde restaurado),
  golden regenerado una sola vez.

**Commits.** `4d1735b`, `9faa8c6`, `8d6076f`, `d446614`, `e3621cc`, `9d34eba`.

**Pendiente.**

- D5 (corpus desincronizado): 00_ARRANQUE decia 2340 tests (real 2369
  tras esta sesion), `01_METODO` 2297, `02_ARQUITECTURA` "10 contratos"
  (real 9). Sesion documental completa.
- D6 (`datetime.now()` en header del reporte). Viola `00_ARRANQUE §2`.
- D7 (volumen Yahoo no consolidado). WONT FIX razonado, documentar en
  `02_ARQUITECTURA §11`.
- BOM en 7 workflows. Cosmetico. Detectado durante D4.
- `_write_ingest_trace` ensucia manifests productivos incluso en rama
  `[SKIP]`. `Path.write_text` sin newline final en manifests.
- `concurrency.group` desalineado: `update_index_holdings` y
  `update_european_holdings` escriben ambos `data/index_holdings.csv`
  con grupos distintos. Riesgo de conflicto de rebase si coinciden.
- Residuos en `etf_holdings.csv` (ticker `-` y `XASU6` con peso negativo).
- Duplicacion de titulo en `05_BITACORA.md:91` (Integridad parquets).
- `test_sector_regime_dispersion.py` cuenta 5, pero el delta total
  de tests entre D4 y cierre es +5; D2 tambien sumo +5. Cuadra
  2344 -> 2348 (D4) -> 2359 (D3) -> 2364 (D2) -> 2369 (D1).

**Proximo paso sugerido.** Cron trimestral 1-oct-2026 (04:47, 06:17, 07:17
CEST): verificar con el runbook. D5 (corpus) o D6 (datetime.now) para
sesion siguiente.

---

### 2026-09-30 (madrugada) — Baseline IAE vs par vigente: aclaracion

**Objetivo.** Investigar la diferencia entre el NIPC del run manual de
GitHub (8256882557) y el baseline del corpus (-4264449012).

**Hallazgo.** No es bug. El reporte diario calcula el NIPC del par de
trimestres vigente (`_list_available_quarters` toma los 2 ultimos).
Al entrar 2026Q2 (commit c5e3ee0), el par paso de Q4-2025 -> Q1-2026 a
Q1-2026 -> Q2-2026. Reproducido localmente con el par nuevo:
8256882557, identico al CI. El baseline -4264449012 corresponde al par
antiguo, congelado en `current.json`.

**No se toca codigo.** El auto-avance del par es correcto y autonomo.
El pipeline se mantiene independiente del usuario. No se hardcodea
ningun par.

**Corpus actualizado.** 00_ARRANQUE, 02_ARQUITECTURA, 03_IAE,
04_HISTORICO y `current.json` declaran ahora explicitamente que el
baseline es historico (par Q4-2025 -> Q1-2026) y que el run diario
publica el NIPC del par vigente, que cambia con cada trimestre.

**Pendiente.**
- Cron 01:17 CEST (madrugada del 30): valida PENDING BME, EU 5y,
  regularMarketTime de Yahoo.
- Cabo B: revisar si Yahoo actualiza cierre del 29 a las 04:00 CEST.
- C2 (920): sin material del auditor, referido al par Q4->Q1.

**Hallazgos del run manual (2026-09-30, run 36642077307):**
- `update_futures` exit 1. BZ=F/CL=F BLOCKED (403 OilPriceAPI).
  `commodities_futures.parquet` atascado en 21-sep. Señal intencional
  con `continue-on-error: true`.
- Warning 'Cache 13F no disponible' es falso positivo. La cache se
  restauro por restore-key (v2), pero `cache-hit != 'true'` dispara
  la alerta. La primary key v3 aun no existe (se guardara el 20-nov).
- 3 tests skipped en CI (`market_data`, `stock_prices`,
  `european_tickers_recent`) por parquets gitignored. Skip silencioso.
- NIPC del run manual: `8256882557` (par vigente Q1-2026 -> Q2-2026).
  Distinto del baseline `-4264449012` (par Q4-2025 -> Q1-2026, congelado).
  Documentado en `c087cb8`.

**Proximo paso sugerido.** Abordar 3 fixes: (a) condicion de la warning
cache 13F, (b) skipped tests en CI, (c) evaluar sustituto OilPriceAPI
o documentar futuros como BLOCKED de facto con parquet congelado.

### 2026-09-29 (noche) — Integridad parquets + EU 5y + PENDING BME + P65

**Objetivo.** Cerrar el frente de integridad detectado al auditar Ibex 35.

**Hecho.**
- **Guard/gate integridad (2a640f2).** Los .parquet estan gitignored,
  los .manifest.json tracked. Tras cada git pull divergen. Ni guard_coverage
  ni pipeline_gate verificaban sha256 real. Detectado: stock_prices y
  market_data con sha256 divergentes. Fix: ambos calculan sha256 del
  parquet sibling, fail-closed.
- **G-01 paridad (08fe06b).** market_data no truncaba a expected_session
  (stock_prices si). Movido `truncate_to_expected_session` a utils y
  aplicado en data_loader._postprocess_market_data.
- **Regeneracion (1ba5df9).** stock_prices con los 19 .MC que BME no
  habia servido el 28/09 por fallo transitorio.
- **PENDING + EU 5y (4bb54f8).** BME publica T+1. `_compute_by_market`
  marca PENDING (no INVALID) cuando la sesion cerro pero no hay
  observacion. Guard tolera PENDING, no todo-PENDING. Ademas BME y Xetra
  pasan de 430d a 1830d (5y). Euronext topa en el endpoint (~510 sesiones).
- **P65 (f131ce3).** Verificado: `_apply_intra_period_dedup` v1 no emite
  DROP_DUP. Todas las decisiones son KEEP. Conectar P65 no alteraria
  nipc_total. Hipotesis C2 descartada por diseno de codigo. Doc corregida
  (03_IAE.md) + current.json actualizado.

**Commits.** 2a640f2, 08fe06b, 1ba5df9, 4bb54f8, f131ce3.

**Pendiente.**
- Cron 01:17 CEST valida 13F cache, PENDING BME, EU 5y,
  regularMarketTime de Yahoo.
- Cabo B: confirmar si Yahoo revisa cierre del 29 a las 04:00 CEST.
- LSE scraper: cadencia externa + historico en JSON infrautilizado.
- C2 (920): sin material del auditor.

**Proximo paso sugerido.** Verificar el cron de la madrugada.

### 2026-09-29 (noche) — Atomicidad familia 3 (17 sitios) + FU-009-bis

**Frente nuevo: familia 3 (escritura no atomica).** Barrido de
`outputs/history/`: 17 sitios con READ+WRITE del mismo path sin
`.tmp + os.replace`. Si el proceso muere a mitad de `to_csv`, el CSV
original queda truncado y el siguiente run lee menos filas -> perdida
irrecuperable (append+dedup no regenera lo perdido).

Cerrados: european_coverage (1), breadth_metrics (2), engines (1),
flows_primary (1), finalize (1), market_data (2), sectors_base (4),
sector_metrics (5).

**Metodo del test (fuerte, verificado por ambos lados):** simular
`to_csv` que trunca el destino y lanza `OSError`. Verificar que el CSV
original queda intacto. Verificacion triple obligatoria: verde con fix
-> ROJO sin fix (restaurando commit padre) -> verde con fix. Los tests
debiles (verificar patron `.tmp+replace`, no el bug) se retiraron.

**FU-009-bis (append_dedup):** pd.concat con esquemas distintos genera
FutureWarning en pandas 2.x (rompera en 3.0). Fix: excluir columnas
all-NA antes del concat. Detectado via test de atomicidad de engines
con hist de esquema antiguo + new de esquema nuevo.

**Lecciones:**
- `git stash` no stashea lo commiteado. Usar `git checkout <pre> -- <file>`.
- Helpers con guard `if df.empty: return None` no ejecutan el bloque
  de escritura con df vacio. Llamar al helper directo con datos no vacios.
- Hasta 3 falsos verdes detectados y corregidos durante el barrido
  (breadth_metrics, engines, sector_metrics).

**Estado final:** suite 2340 -> 2361 passed + 2 skipped. 0 warnings.
pyflakes/compileall limpios.

### 2026-09-29 (tarde) — Auditoria funcional frente 6 (IAE) + frente 8 (providers) + orquestacion CI

**Objetivo.** Continuar auditoria funcional del sistema tras cerrar A6 + B + C
por la manana. Orden efectivo: orquestacion CI (surgio por un push rechazado),
frente 6 (IAE funcional avanzado), frente 8 (providers).

**Hecho.**

*Orquestacion CI (8 commits).*
- Bug cache 13F v2 <-> v3 desalineada entre daily_run y update_sec_13f.
  La seccion IAE del reporte llevaba en STALE desde el 28. Commit 8b0eb1c.
- Bug issues:write ausente en update_sec_13f. La alerta H5.2 era inoperativa.
  Commit 9129278.
- _update_issue no comprobaba retorno de gh en health_check (mentia).
  Commit 4c15e47. +3 tests (primeros que cubren _run_gh).
- Retry con backoff en git push de los 8 workflows update_*.
  Commits 4297785, e3b0287.
- if:always() en uploads pre-gate de download_failures. Commit 05acb3f.
- env.quarter vacio eliminado en update_sec_nport. Commit 702e2d3.

*Frente 6 - IAE funcional avanzado (14 commits).*
- S-03c: invariante CANONICAL != None tambien en rama crosswalk_internal.
  Commit b0d3ec3. +5 tests.
- PeriodState.sshprnamt: validar finitud (NaN/inf). Commit 474dd49. +4 tests.
- extract_sshprnamt_by_figi: misma familia. Commit 1259542. +3 tests.
- operational_universe: simetria de strip en claves de lookup.
  Commit 88013bf. +3 tests.
- Dedup catalog_p38_adapter <-> period_state. Commit b870b34.
- Dedup build_catalog_csvs <-> catalog_key. Commit 6b32ff0.
- Docstring P65/P66 corregido (contrato implementado desde 33d75cf).
  Commit d11bed2.
- Deudas documentadas: target_universe (de32ea1), timestamps (6e35715).

*Frente 8 - providers (12 commits).*
- downloader: descarga atomica + cache-hit valida ZIP. Commit e56a49b. +3 tests.
- backup_providers._validate_with_cache: validar finitud. Commit ce273eb.
  +4 tests.
- _blackrock_base: flow_pct_assets finitud. Commit 9250f90.
- Cache atomica xetra/bme/euronext (acca9cd) + blackrock_iwm (e197d77).
- Guard 366 iteraciones en 3 bucles is_market_day. Commit 2e83b3e.
- Dedup _fund_flow_utils. Commit 38a3809.
- N-PORT F5.7-20 completado. Commits 58cf809 (backfill + fallo ruidoso),
  2b0bd0c (descarga automatica XML EDGAR en qqq_nport_flow).
- finra dead code documentado. Commit 0f48b62.

*Corpus + coherencia documental (4 commits).*
- 00_ARRANQUE alineado (da61173).
- 02_ARQUITECTURA alineado (a73b2dd).
- 04_HISTORICO: entrada tematica de la sesion (9bdc1c7).
- Este commit (05_BITACORA).

**Frente 7 - indicators (9 commits).**
- call_share finitud (f24e193).
- atómicos: darkpool_history + sector_rank_history (88eb2e5), state_transition
  (46854a8), pcr_history + analisis_lideres (bbf3422).
- slpm_v12 flow_proxy_z=0.0 (894c03c).
- mte/scoring stress finitud (5528ef2).
- fls zscore finitud (7e2c1a3).
- mte/decision consensus finitud (7cb5c7a).

**Commits.** ~50. HEAD final antes de este commit: `675b438`.

**Pendiente.**
- H5.3 (cron nov 2026).
- C2 (920, bloqueado externo).
- Cierre del corpus documental (actualizar 00_ARRANQUE / 02_ARQUITECTURA /
  04_HISTORICO / 05_BITACORA + snapshot).

**Proximo paso sugerido.** Cerrar sesion: actualizar corpus y snapshot.
Los tres frentes funcionales abiertos (6, 7, 8) estan cerrados.

---

### 2026-09-29 — G-01 (truncado stock_prices) + P66-01 (extraccion 14)

**Objetivo.** Diagnosticar el run rojo 36466343234 (slot 17 11, retraso GitHub 7h19m).
El guard abortaba el push por `last_date=2026-09-28 > expected_session=2026-09-25`.
Causa raiz: `stock_prices.parquet` es multi-mercado (US + LSE + Euronext + Xetra),
pero `expected_session` es NYSE-only por construccion. En ventana UE-cerrada +
USA-abierta, el df contiene fila europea con date > expected_session.

**Hecho.**

*G-01 (src/stock_data_loader.py).*
- Truncado del df a `index <= expected_session` solo para el parquet.
- El df en memoria se devuelve intacto: `leaders.py` ya lo trunca por
  `resolve_effective_date` antes de propagar.
- Ningun consumidor aguas abajo usa la fila europea parcial.

*P66-01 (test_p66_contract.py + Consolidacion_Documentos/06_IAE_P65_P66.md).*
- Hallazgo colateral: `cdf47ad` (borrado del corpus antiguo) elimino
  `docs/auditoria/iae/NIPC_CONTRATOS_SEMANTICOS_v1.md`. Los 10 tests
  Capa A de P66 quedaron huerfanos (creados en commits posteriores
  sobre vista obsoleta del repo).
- Extraccion del §14 (P65 + L3 cruzada P66, 313 lineas) a
  `Consolidacion_Documentos/06_IAE_P65_P66.md`.
- Reapuntado CONTRATO en `tests/test_p66_contract.py`.

**Refactor G-01 + tests (ddb4d31).**
- El fix G-01 estaba inline en `download_stock_prices`, sin cobertura.
- Se extrae a `_truncate_to_expected_session(df, expected_session)`
  -> (df_truncado, n_excluidas). Contrato documentado.
- 8 tests unitarios en `test_stock_data_loader_helpers.py`: noop (None,
  vacio, index no-Datetime, index == expected, index < expected),
  caso G-01 (descarta fila post-expected), no mutacion, y test de
  integracion (df truncado al writer + df intacto al caller).
- Suite: 2267 -> 2275 passed + 2 skipped.
- Justificacion de no hacer run end-to-end: el bug solo se manifiesta
  en ventana UE-cerrada + USA-abierta (~16:30-20:00 UTC). Un run
  manual ahora (madrugada UTC) no lo reproduce. Los tests unitarios
  cubren la logica independientemente de la hora y del slot.

**A6.1 (nucleo compartido).** Gate 0 + auditoria de
`src/utils.py` (629 LOC), `src/instrument_registry.py` (863 LOC),
`src/dependency_tracker.py` (112 LOC).

- Gate 0: `outputs/audit/A6_1_1_20260929_011819.txt` (426 lineas).
- **A6.1-01 (MEDIA) CORREGIDO (2f63db7).** `utils.py::_latest_closed_session`
  tenia `except Exception: pass` en el bucle de 30 iteraciones. Si
  `is_trading_session` / `is_session_closed` fallaban sistematicamente
  para un mercado, retornaba None sin traza. `_compute_by_market` lo
  traducia a `status=INVALID` (fail-closed correcto), pero el
  diagnostico se perdia. Fix: contar y reportar el primer error.
  Contrato fail-closed preservado.
- **A6.1-02 (MEDIA) DEUDA DOCUMENTADA, sin patch.** El outer
  `except Exception` de `write_artifact_with_manifest` retorna `{}` y
  deja el parquet previo intacto ante fallo tecnico (dtype, sha, json,
  os.replace). El test `test_artifact_manifest.py:85-103` ancla este
  contrato como intencional. Los 4 callers productivos
  (`data_loader.py:186`, `stock_data_loader.py:865`,
  `cboe_index.py:135`, `futures.py:265/278`) NO comprueban el retorno
  `{}`. El pipeline continua como si la escritura hubiera ocurrido. Fix
  requiere coordinar writer + 4 callers + tests; ROI bajo sin evidencia
  empirica de dano real. Se deja como deuda para bloque C (tests) o
  posterior.
- **Falsos positivos descartados (contra 01_METODO §8.6):**
  - `_latest_closed_session` en utils.py: **no es** patron 6. Compone
    `is_trading_session` + `is_session_closed` de `market_hours.py`.
  - `instrument_registry.py` sin imports top-level: data-driven
    (INSTRUMENTS + 4 funciones puras). `get_instrument_class` llama
    `get_market` internamente (L882).
  - Re-exports `normalize_yahoo_ticker` en data_loader/stock_data_loader:
    validados por `test_registry_yahoo_map.py:27-28`.
  - `dependency_tracker.py:102` `except Exception`: defensivo
    deliberado (import + inspect.signature; fallback a "No disponible").
- **Deuda fuera de A6.1, documentada:**
  `indicators/darkpool_scoring.py:12 def robust_zscore(series)` es
  variante local de 1 arg con contrato distinto al canonico (3 args,
  window=60). A3.4 solo cerro `options.py` y `fls.py`. Reabrir en
  bloque de calculo.

**Commits.** 3f789f1 (G-01), 5a039e6 (P66-01, amend de 6550b13),
ddb4d31 (refactor G-01 + tests), a4095b1 (docs cierre), fabcdae (snapshot),
2f63db7 (A6.1-01), 73758ce (snapshot A6.1-01).

**A6.2 (workflows, en curso).** Gate 0: `outputs/audit/A6_2_1_20260929_012650.txt`
(557 lineas, 10 workflows). Clasificacion:

- **A6.2-02 (BAJA) CORREGIDO (4fbb135).** `_cron_probe.yml` residual
  (introducido en 03f3b42 el 2026-09-27 para diagnosticar el finding
  "slot '17 3' no dispara"). El probe confirmo lo contrario: el cron
  SI dispara, con retrasos variables 2-8h. Finding cerrado con causa
  raiz: no-disparo era falso, es retraso variable de GitHub Actions.
  Probe eliminado. Comentario de `daily_run.yml` actualizado. Tabla
  de `02_ARQUITECTURA` limpiada.
- **A6.2-01 (MEDIA) DEUDA DOCUMENTADA.** 4 workflows comparten cron
  `17 6 UTC` (`update_macro_manual`, `update_qqq_sec_flow`,
  `update_sec_nport`, `update_sec_13f`). Dias 15 ene/jul (2 concurrentes),
  dias 20 ene/abr/jul/oct (2-3 concurrentes). Todos con
  `git pull --rebase` + `git push` sobre `main`. Ventana de race real;
  sin evidencia empirica de dano en 5 meses. Sin patch por ROI bajo.
- **A6.2-03 (MEDIA) DEUDA DOCUMENTADA.** 7 workflows `update_*`
  comparten estructura identica (checkout + setup-python + pip install
  + compileall + script + git add + commit + pull --rebase + push).
  Sin `workflow_call` ni composite action. Refactor masivo no procede
  sin contrato.
- **A6.2-04 (BAJA) DEUDA DOCUMENTADA.** `git pull --rebase` sin retry
  en 7 workflows. Fallo por conflicto → workflow rojo. Combinado con
  A6.2-01, ventana de conflicto.
- **A6.2-05 (MEDIA) CERRADO por A6.2-02.** Los 5 slots de
  `daily_run.yml` disparan con retrasos variables (2-8h observados),
  no con timing fijo. Documentado en el comentario de `daily_run.yml`.
- **Falsos positivos descartados:** `pytest tests/ validation/` en CI
  colecta solo `tests/` (2275); `validation/` no tiene tests pytest,
  sus 7 scripts se invocan en steps dedicados.

**A6.3 (scripts).** Gate 0: `outputs/audit/A6_3_1_20260929_013536.txt`
(822 lineas, 37 scripts + 3 en audit/).

- **A6.3-01 (MEDIA) CORREGIDO (1049b2a).** `update_european_holdings.py`
  y `update_sector_holdings.py` no tenian `if __name__ == "__main__"`.
  Cualquier import disparaba red + escritura (11 GETs a SSGA en sector;
  3 subprocess en european). Footgun real. Refactor a `main()` +
  `if __name__ == "__main__"`.
- **A6.3-02 (BAJA) CORREGIDO (1049b2a).** `subprocess.run(cmd,
  shell=True)` en `update_european_holdings.py:9`. Comando fijo sin
  metacaracteres, sin input externo. Cambiado a lista de args.
- **A6.3-03 (BAJA) DESCARTADO.** Los 20 ORPHAN del Gate no son dead
  code: `parse_*`/`amundi_holdings` se consumen via
  `update_european_holdings.py`; los `iae_*` y `generate_*` son
  auditoria reproducible manual documentada en `03_IAE §5` y
  `02_ARQUITECTURA §8`. El detector del Gate solo mira workflows
  directos.
- **A6.3-04 (BAJA) DESCARTADO.** `scripts/audit/*.py` sin `sys.exit`:
  exploratorios, se invocan manualmente, el retorno no se usa.
- **A6.3-05 (BAJA) DESCARTADO.** `datetime.now()` en 8 scripts es
  log de ejecucion (backup ts, timestamp UTC, doc autogenerada). No
  como fecha de observacion.
- **CLI contract:** 0 issues en 16 invocaciones de workflow contra
  12 scripts con argparse.

**A6.4 (validation + data/validator.py).** Gate 0:
`outputs/audit/A6_4_1_20260929_014123.txt`.

- **A6.4-01 (MEDIA) CORREGIDO.** `validate_history_quality.py` y
  `verify_leader_selection.py` ejecutaban al importar. Refactor a
  `main()` + guard.
- **A6.4-04 (MEDIA) CERRADO por dead code.** `run_all_audits.py`
  (354 LOC) sin caller, ultima modificacion 2026-09-10, invoca
  `['py', ...]` (Windows-only). Eliminado.
- **A6.4-02 (BAJA) WONT FIX razonado.** Los 2 scripts de validacion
  cruzada (`cross_provider_validation`, `qqq_sec_nport_cross_validation`)
  son descriptivos por diseno: escriben CSV a outputs/audit/ como
  artifact. No son gates. El contrato se preserva.
- **A6.4-05 / A6.4-03 (BAJA) falsos positivos.** `verify_leader_selection`
  ya limpiaba el md anterior. `data/validator.py` es solo 2 funciones
  puras.
- **Bug introducido y corregido en el refactor:** dos `import os`
  internos en `verify_leader_selection` se volvieron locales a
  `main()` y rompian refs previas. Eliminados.

**B (IAE) - bloque abierto y cerrado.**

Gate 0: `outputs/audit/B_1_1_20260929_015604.txt` (510 lineas, 36
ficheros, 8202 LOC).

- **B-01 (MEDIA) CORREGIDO (7e7f429).** `TargetUniverse.__post_init__`
  usaba `assert` para 4 invariantes estructurales. Bajo `python -O`
  los asserts se eliminan y las invariantes desaparecen
  silenciosamente. Sustituidos por `raise BuildTargetError` con
  mensaje que incluye tamanos concretos.
- **B-01b (test) (082cc82).** 5 tests cubriendo las 4 invariantes
  + caso bien formado. Coverage target_builder 65% -> 70%.
- **B-02 (docs) (49b471a).** 3 docstrings del modulo IAE actualizados
  al corpus consolidado (`06_IAE_P65_P66.md`). El `.md` original
  fue borrado en `cdf47ad`.
- **B-03 (docs) (0b0bb26 + 9da966f).** Cifras de `03_IAE` corregidas
  tras verificacion: LOC 8178 -> 8202, `35 funciones AST` -> 19 en
  `reporting_dedup.py`, firma completa de `compute_nipc_contractual`,
  censo de tests 845 -> 763 funciones test_ AST (871 pytest).
- **Deudas documentadas (sin patch por ROI bajo):**
  `build_effective_reporting_snapshot` sin caller productivo
  (ya declarada en `03_IAE §10`). C2 (920) OPEN. H5.3 cron nov 2026.

**B-04/05/06/07 + C-07 (auditoria funcional de IAE).**

Tras cerrar B-01/B-02/B-03, se hizo auditoria **funcional** (no solo
estructural): probes sobre parquets 13F reales y sobre la cadena
`run.py -> iae_section -> pipeline_contractual -> security_identity`.

- **C-01 (MEDIA) CORREGIDO (a5405e5).** `_derive_status` con
  `paired_weighted_share_coverage=None` lanzaba `TypeError`
  (`None >= float`).
- **C-04 (BAJA) CORREGIDO (a5405e5).** `_filter_canonical` no
  descartaba SSHPRNAMT negativo.
- **C-05 (MEDIA) CORREGIDO (4ec29d3).** Audit trail con
  `reporting_for_manager_cik == filing_manager_cik` ("A reporta
  para A") en lugar del representado real.
- **B-05 (docs) CORREGIDO (33b8cca).** E2E reconciliaba contra golden
  FROZEN obsoleto (12_5_historic); default pasa a `current.json`,
  `--historic` para el golden. Corregida doc erronea sobre H1-A.
- **B-06 / S-03 (MEDIA) CORREGIDO (ea0c3cb).** `pipeline_contractual`
  pasaba el CSV de equivalence crudo a `resolve_batch_identities`;
  `valid_from` llegaba como str y `_find_active_equivalence` comparaba
  `str <= Timestamp` -> TypeError. Fix: `load_cusip_equivalence`
  (valida + normaliza dtypes). Endurecida invariante CANONICAL ->
  `canonical != None`. Encadenado.
- **B-07 (MEDIA) CORREGIDO (43b0537).** `catalog_manifest.json` con
  2 snapshots abiertos. `target_catalog_as_of` inoperativo
  (CatalogAmbiguous). Cerrado `20260921_01`.
- **C-07 (BAJA) CORREGIDO (3902503).** NBSP en TITLEOFCLASS
  (`CL<NBSP>A`) caia a UNRESOLVED. Normalizado whitespace Unicode.

**Falsos positivos descartados** (con datos reales, sin fix):
- `.upper()` inconsistente en `relationships.classify_token`: el
  dataset SEC usa `NONE` mayusculas de forma consistente (0 casos).
- `operational_universe._apply_5_3b` dead-code parcial: es diseño
  (target excluye CUSIPs fuera de Official List por §5.3).
- `validate_membership` sin caller productivo: deuda documentada.
- `cusip_resolver.resolve_cusip` sin caller productivo: sin efecto.
- Divergencia delta/target (553k vs 240): es diseño del contrato,
  no bug.

**B CERRADO.** Suite: 2275 -> 2294 passed + 2 skipped.

**C (auditoria de tests) CERRADO.**

- Gate 0: 179 ficheros, 28817 LOC, 2115 funciones `test_`, 0 ficheros
  sin asserts. Patrones sospechosos clasificados: todos falsos
  positivos (ramas defensivas, fixtures MagicMock, skips condicionales
  de entorno).
- **C-07 (BAJA) CORREGIDO (3902503).** NBSP en TITLEOFCLASS
  (`CL<NBSP>A`) caia a UNRESOLVED.
- **C-08 (MEDIA) CORREGIDO (09bfa13).** `volatility_regime` mapeaba
  z=NaN a STRESS por cascada de comparaciones. Evidencia empirica:
  1803/1884 STRESS historicos eran falsos positivos. Fix: NaN -> "N/D".
  Coherente con el principio N/D del sistema. Impacto: no afecta a
  ninguna decision del pipeline (macro_regime usa VIX directo); solo
  a la etiqueta del reporte y a `detect_cross_module_conflict`.
- **Cobertura real por modulo**: identificados modulos con cobertura
  baja (`macro_manual_loader` 12%, `european_coverage` 12%,
  `pipeline_contractual` 27%, `data_loader` 50%). Sin bug detectado
  tras inspeccion. Deuda de tests, no de codigo.
- **Tests anclados a numero de linea**: ya cerrado en A3.3-13.

**C CERRADO.** Suite: 2294 -> 2297 passed + 2 skipped.

**A6 CERRADO** (A6.1 + A6.2 + A6.3 + A6.4). Resumen del bloque:

- A6.1 (nucleo compartido): 1 fix (A6.1-01), 1 deuda documentada (A6.1-02).
- A6.2 (workflows): 1 fix (A6.2-02, eliminacion del probe residual),
  3 deudas documentadas (A6.2-01/03/04).
- A6.3 (scripts): 2 fixes (A6.3-01/02), 3 descartes razonados.
- A6.4 (validation): 2 fixes (A6.4-01/04), 1 WONT FIX, 2 falsos positivos.

Suite final: 2297 passed + 2 skipped + 0 failed.

**Commits.** ... 2f63db7 (A6.1-01), 73758ce (snapshot A6.1-01), 4a31d4f
(cierre A6.1), 4fbb135 (A6.2-02), d321802 (cierre A6.2), 1049b2a
(A6.3-01/02), 84bb1ee (cierre A6.3), +A6.4.

**Pendiente.** H5.3 (cron nov 2026). C2 (920).

**Proximo paso sugerido.** Verificar en CI, cuando corra un slot real, que el guard pasa con el parquet nuevo.

---

### 2026-09-28 (tarde/noche) — Auditoria interna A1-A5 del sistema + refactor documental

**Objetivo.** Auditoria linea a linea del sistema por bloques, del nucleo temporal al pipeline. Posteriormente: consolidacion documental (dado que el corpus antiguo era inviable de usar).

**Hecho.**

*Auditoria A1 (nucleo temporal).*
- Calendario NYSE algoritmico (Computus + nth-weekday + observed). Bug: fuera de 2026-2027 los festivos no se reconocian.
- `last_expected_market_date` con tz Madrid. Bug: `datetime.now()` naive en CI (UTC) clasificaba mal.
- Guards de 366 iteraciones en bucles sin red de seguridad.
- Aridad 4-tuple uniforme en `compute_liquidity_score` (antes 3 o 4 segun rama).
- 5 commits. HEAD bab45a4..2c1bb13.

*Auditoria A2 (providers).*
- BME 0/19: `today = last_expected_market_date(reference_date)`. Walk-back a sesion publicada.
- tz-naive en `european_coverage` y `data_quality` (helpers `_ref_to_date`).
- SSGA fund-flow migrado al contrato comun `_fund_flow_utils`.
- `retry_call` sin sleep en el ultimo intento.
- Commits: c1b56e9, 17845c1, 858099e, 64f88cf, 8d0601c, 804dbd8, e3eedc7.

*Auditoria A3 (calculo).*
- Regimenes: aridad 4-tuple, except acotados, alias dead code.
- Breadth: SECTOR_ETFS unificado a `MARKET_TICKERS['sectors']` (7 indicadores + rankings + cross_asset). Fix del generador sintetico del golden (drift por posicion -> drift por alfabetico).
- MTE: whitespace collapse (-565 lineas), except acotados.
- Darkpool: bug real (`robust_zscore` devolvia ndarray en rama `mad==0`).
- SLPM: docstring huerfano, except acotado.
- Commits: 0973fb9, 1fe3bdc, 6ed9c6b, 9b31a9e, a6e2bbd, b723ea8, d3a3002, 1a5730e, 1c2e8d7.

*Auditoria A4 (reporte).*
- Bug observado: `nan` literal en fila 2026-09-14 del reporte (VIX3M/VIX). Fix: guard NaN en 6 campos numericos.
- Barrido del patron: 26 sitios guardados en 7 renders (sentiment, darkpool, leaders, market_context, synthesis, flows_international, rankings).
- 12 except acotados en 7 renders.
- Commits: 64f3331, e4d37f2, 83c1fad, f15ec3a, 9add68b, bbfeb19.

*Auditoria A5 (pipeline).*
- Bug estructural en `leaders.py`: `NameError` silenciado en rama INSUFFICIENT_COVERAGE (`_n_close` fuera del `if`). Corregido indentacion.
- Check muerto en `mte_confirmation.py`: `'all_signals' in dir()` siempre True. Corregido a check real.
- Guards faltantes en `indices_intl.py` (`compute_index_phases` y `download_stock_prices` sin try/except).
- SECTOR_ETFS duplicado en `engines.py` y `diagnostics.py`. Migrado a config.
- ~50 except acotados en 11 ficheros.
- Commits: 59204cd, 28afada, 08d7a23, 6d16bdf, de7f8a3.

*Refactor documental.*
- Nuevo corpus en `Consolidacion_Documentos/`: 00_ARRANQUE, 01_METODO, 02_ARQUITECTURA, 03_IAE, 04_HISTORICO, 05_BITACORA.
- Corpus antiguo (~570 KB) pendiente de borrado.
- Commits: 073b2fe, 3bd00e0, fd6d2c6, c05a02b, e917767.

**Commits de la sesion.** ~20 commits locales + 2 pushes + 4 runs `py run.py` completos.

**Pendiente.**
- Borrar corpus antiguo de `docs/auditoria/`.
- A6 (utils, registry, tracker, workflows, scripts, validation).
- B (IAE completo).
- C (tests).

**Proximo paso sugerido.** Verificar los 6 documentos consolidados (referencias cruzadas, contenido). Borrar corpus antiguo. Retomar A6.

---

## 4. REFERENCIA A SESIONES ANTERIORES

Para sesiones anteriores al 2026-09-24, ver `git log --oneline`. Resumen tematico en `04_HISTORICO.md`.

---

## 5. REGLAS DE PODA

- La bitacora mantiene las ultimas **5 sesiones**.
- Al anadir una nueva, si hay >5, la mas antigua se elimina.
- **Antes de eliminar**, verificar que sus decisiones clave estan en `04_HISTORICO.md`. Si no, resumir primero.

---

**Fin de la bitacora.**