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

- Informe externo con 142 hallazgos: 6 ALTA, 71 MEDIA, 65 BAJA. Informes en `bc656374` (`docs/auditoria/radar/audits/`).
- **6 ALTA corregidos** con commits verificados:
  - F3-05 OilPriceAPI 403 futuros (ff95a3d).
  - F3-16 writer spot INVALID (84b2708).
  - F3-17 best-effort enmascarando 401/403/429 (2c8b81f).
  - A5-13 Yahoo fallback devolviendo parquet completo (4808a13).
  - F6-10 tres "lideres" por sector (b5884a1).
  - F6-28 dos "A/D Net" contradictorios (6f4f4b1).
- Fases 0-7 auditadas (consolidado 142 hallazgos).

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

- **Sesion IAE** (14 commits): cierre H1-A, H1-B v2/v2.3, H2, H3, H4, H5.1-H5.5, O1. Baseline local reproducible -4.264.449.012 (par Q4-2025 -> Q1-2026, congelado). Referencia auditor -4.264.449.932 (delta 920, OPEN).
- **Sesion radar** (30 commits): cierre FASE 3, 5 completas; FASE 7; FASE 2.2; FASE 2.4. ~28 commits de cierre.
- Suite: 2205 -> 2267 passed + 2 skipped.
- **Auditoria interna A1-A5** iniciada y cerrada (nucleo temporal, providers, calculo, reporte, pipeline). Bugs estructurales corregidos: bug en `leaders.py`, check muerto en `mte_confirmation.py`, guards en `indices_intl.py`, SECTOR_ETFS unificado.
- **Refactor documental**: creacion del corpus consolidado (00-05). Borrado del corpus antiguo pendiente.
---

### 2026-09-29: G-01 (truncado stock_prices) + P66-01 (extraccion 14)

- **G-01:** el run 36466343234 (slot 17 11, retraso GitHub 7h19m) fallo
  en `guard_coverage` por `last_date=2026-09-28 > expected_session=2026-09-25`.
  Causa raiz: `stock_prices.parquet` es multi-mercado pero `expected_session`
  es NYSE-only. En ventana UE-cerrada + USA-abierta, el df contiene fila
  europea post-expected. Fix: truncado pre-write a `index <= expected_session`
  en `src/stock_data_loader.py`, solo para el parquet. El df en memoria
  se devuelve intacto. Commit 3f789f1.
- **P66-01:** los 10 tests Capa A de `test_p66_contract.py` quedaron
  huerfanos tras `cdf47ad` (borrado del corpus antiguo). El contrato seguia
  vigente (docstrings de `reporting_dedup.py`, `absence.py`, `timestamps.py`;
  Capa C en xfail esperando GO #40 step 2). Extraccion del 14 (P65 + L3
  cruzada) a `Consolidacion_Documentos/06_IAE_P65_P66.md`. Reapuntado
  CONTRATO. Commit 5a039e6 (amend de 6550b13).
- Suite: 2267 passed + 2 skipped (antes: 2257 + 10 failed por P66).
- **Cronologia del corpus:** nuevo documento `06_IAE_P65_P66.md`. Corpus
  consolidado pasa a v2 (00-06).
- **A6.1 (nucleo compartido):** Gate 0 + auditoria de `src/utils.py`,
  `src/instrument_registry.py`, `src/dependency_tracker.py`. A6.1-01
  (MEDIA, corregido): `_latest_closed_session` con `except: pass` en
  30 iteraciones -> ahora reporta excepciones. A6.1-02 (MEDIA, deuda
  documentada): `write_artifact_with_manifest` retorna `{}` ante fallo
  tecnico y los 4 callers lo ignoran; el contrato actual esta anclado
  por `test_artifact_manifest.py:85-103`. Deuda fuera de A6.1:
  `darkpool_scoring.py::robust_zscore` variante local. Commit 2f63db7.
- **A6.2 (workflows):** Gate 0 + auditoria de los 10 workflows de
  `.github/workflows/`. A6.2-02 (BAJA, corregido): `_cron_probe.yml`
  eliminado tras confirmar que el finding "slot '17 3' no dispara"
  era falso (retrasos 2-8h, no no-disparo). Commit 4fbb135. A6.2-01,
  A6.2-03, A6.2-04 (deuda documentada): colision cron `17 6`, duplicacion
  estructural de 7 workflows `update_*`, `git pull --rebase` sin retry.
- **A6.3 (scripts):** Gate 0 + auditoria de los 37 scripts en
  `scripts/` (34 raiz + 3 audit/). A6.3-01/A6.3-02 (MEDIA+BAJA,
  corregidos): `update_european_holdings` y `update_sector_holdings`
  ejecutaban al importar (sin `if __name__ == "__main__"`). Refactor
  a `main()`. Commit 1049b2a. Sin dead code confirmado. Contrato CLI
  workflow<->argparse verificado: 0 issues.
- **A6.4 (validation):** Gate 0 + auditoria de 6 ficheros en
  `validation/` + `data/validator.py`. A6.4-01 (MEDIA, corregido):
  `validate_history_quality` y `verify_leader_selection` ejecutaban
  al importar. Refactor a `main()`. A6.4-04 (MEDIA, cerrado por dead
  code): `run_all_audits.py` sin caller, eliminado. A6.4-02 (BAJA,
  WONT FIX razonado): scripts de validacion cruzada descriptivos por
  diseno. **A6 (auditoria interna de infraestructura) CERRADO.**
- **B (IAE):** Gate 0 + auditoria de los 36 ficheros / 8280 LOC del
  modulo. B-01 (MEDIA, corregido): `TargetUniverse.__post_init__`
  usaba `assert` para 4 invariantes; bajo `python -O` desaparecian
  silenciosamente. Sustituidos por `raise BuildTargetError`. B-01b
  (5 tests). B-02 (docstrings IAE actualizados al corpus consolidado).
  B-03 (cifras de `03_IAE` corregidas tras verificacion). **B CERRADO.**
- **B-04 (auditoria funcional de IAE):** una vez cerrado el barrido
  estructural, se ejecuto auditoria sobre **datos reales** (parquets
  SEC 13F de 2025Q4, 2026Q1, 2026Q2) y probes sobre la cadena
  productiva `run.py -> iae_section -> pipeline_contractual`. Fixes
  aplicados: C-01 (`_derive_status` con `paired_weighted=None`),
  C-04 (SSHPRNAMT negativo), C-05 (audit trail `reporting_for_manager`
  separado), S-03 (cadena equivalence productiva: `load_cusip_equivalence`
  en `pipeline_contractual` + invariante CANONICAL en `security_identity`),
  C-07 (NBSP en TITLEOFCLASS). C-05 fue detectado por probe sintetico
  sobre `_apply_intra_period_dedup`; los demas, por probes sobre la
  cadena completa.
- **B-05 (E2E contractual):** el script `iae_contractual_nipc_e2e.py`
  fallaba contra el golden `12_5_historic` desde H1-B v2.3 (2026-09-28).
  Correccion: por defecto reconcilia contra `golden/current.json`
  (baseline reproducible con HEAD); flag `--historic` para el golden
  FROZEN. Corregida la afirmacion erronea "PASS con crosswalk 7caa86b".
- **B-07 (B2-PIT):** el `catalog_manifest.json` tenia 2 snapshots con
  `valid_to=null` (violacion del contrato B2-PIT: intervalos sin
  solapamiento). `target_catalog_as_of` devolvia `CatalogAmbiguous`
  para cualquier fecha >= 2026-09-22. Cerrado `20260921_01` con
  `valid_to=2026-09-22`.
- **Falsos positivos descartados** en B-04: inconsistencia `.upper()`
  en `relationships.classify_token` (no se dispara en datos reales);
  `operational_universe._apply_5_3b` dead-code parcial (diseño);
  `validate_membership` sin consumidor productivo (deuda documentada
  en `03_IAE §10`); `cusip_resolver.resolve_cusip` sin caller real.
- **B CERRADO.** Suite: 2275 -> 2294 passed + 2 skipped.
- **C (auditoria de tests):** 179 ficheros en `tests/`, 2115 funciones
  test_ (2280 casos con parametrize). Barrido de tests fantasma
  (0 hallazgos reales), tests anclados a numero de linea (ya cerrado
  en A3.3-13), cobertura real por modulo. C-08 (MEDIA, corregido):
  `regimes/volatility_regime.py` mapeaba NaN a STRESS. Evidencia:
  1803/1884 detecciones historicas de STRESS eran falsos positivos
  por NaN heredado de 90 huecos de VIX x rolling(20) x baseline(756).
  Fix: NaN -> "N/D". Deuda documentada: modulos con cobertura baja
  (`macro_manual_loader` 12%, `european_coverage` 12%,
  `pipeline_contractual` 27% ya documentado). **C CERRADO.** Suite:
  2294 -> 2297 passed + 2 skipped.
- **Cierre sesion 2026-09-29:** 21 commits. G-01 (fix truncado),
  P66-01 (extraccion §14 + reapuntado tests), A6.1-A6.4, B-01/B-02/B-03,
  B-04..B-07 + C-07, C-08. Suite 2267 -> 2297 passed + 2 skipped.
  **Auditoria interna A6 + B + C CERRADA.**

---

### 2026-09-29 (tarde/noche): auditoria funcional frentes 6 (IAE) + 7 (indicators) + 8 (providers) + orquestacion CI

Continuacion de la sesion del mismo dia. Tras cerrar A6 + B + C por la
manana, se abrieron tres frentes nuevos, en este orden: orquestacion CI
(surgio por un push rechazado), frente 6 (IAE funcional avanzado) y
frente 8 (providers). Total: ~35 commits.

**Orquestacion CI (8 commits).**

- **Bug cache 13F (8b0eb1c).** `update_sec_13f.yml` guardaba cache con
  key `13f-processed-v3-<hash de latest_quarter + manifests>`; `daily_run.yml`
  buscaba `v2-<hash de latest_quarter>`. Nunca coincidian. La cache 13F
  nunca se restauraba en daily_run. Consecuencia: `iae_section.py` no
  encontraba DATA_DIR -> status=STALE, stale_reason=insufficient_quarters.
  La seccion IAE del reporte diario llevaba en STALE desde el 28.
  Degradacion silenciosa (solo un ::warning::). Fix: alinear daily_run a v3.
- **Bug issues:write (9129278).** `update_sec_13f.yml` declaraba solo
  `contents: write`. El bloque H5.2 de abrir Issue en fallo ejecuta `gh issue
  list`, `gh issue comment`, etc. Sin `issues:write`, 403 silencioso. La
  alerta H5.2 nunca se habria materializado cuando el cron de noviembre
  fallara. Fix: declarar `issues:write`.
- **Bug _update_issue (4c15e47).** En `health_check.py`, `_update_issue`
  no comprobaba el retorno de `_run_gh` en close/comment/edit. Si `gh`
  fallaba, imprimia 'issue cerrada'/'actualizada' igualmente. Workflow
  verde, sin issue, sin rastro. Fix: comprobar retorno; imprimir WARN
  si gh fallo. 3 tests nuevos (los primeros que cubren _run_gh).
- **Retry en git push (4297785, e3b0287).** 8 workflows con
  `git pull --rebase && git push` sin retry. Colision real 4-6 dias/ano
  (cron `17 6 UTC` compartido por 4 workflows). Fix: bucle 3 intentos
  con backoff, `exit 1` si los 3 fallan (fail-loud, no silent).
- **if:always() en uploads pre-gate (05acb3f).** Cuando `run.py` fallaba
  en validation_gate, los steps post-gate se saltaban por default
  `if:success()`. `download_failures.md` (generado en fase 1) quedaba
  en disco pero inaccesible. Fix: `if:always()` en Upload download
  failures y Check download failures alert.
- **env.quarter vacio (702e2d3).** `update_sec_nport.yml` referenciaba
  `${{ env.quarter }}` sin bloque `env:`. Commit message quedaba
  `'Actualizar N-PORT '` sin trimestre. Fix: texto fijo.

**Frente 6 - IAE funcional avanzado (14 commits).**

- **S-03c (b0d3ec3).** Bug real. `security_identity.py`: la rama
  equivalence tenia guarda S-03b (si `_normalize_canonical` devuelve
  None, degradar a OBSERVED_ONLY). La rama crosswalk_internal no la
  tenia. Con ticker literal `'nan'` string (CSV leido con dtype=str),
  el resolver declaraba CANONICAL con canonical_security=None. Contrato
  del docstring violado. Fix: guarda simetrica + refuerzo de
  `_find_active_crosswalk` para filtrar strings placeholder. 5 tests.
- **PeriodState.sshprnamt (474dd49).** Bug real. Dataclass frozen con
  `__post_init__` validando `>= 0`. Con NaN: `nan < 0` es False, pasa.
  Con inf: `inf < 0` es False, pasa. Fix: `math.isfinite()`. 4 tests.
- **extract_sshprnamt_by_figi (1259542).** Misma familia. Si SSHPRNAMT
  es NaN o inf, contamina la agregacion por FIGI. Fix: math.isfinite
  en el bucle. 3 tests.
- **operational_universe strip (88013bf).** Asimetria. `resolve_eligibility`
  y `resolve_batch_identities` producen claves con `.strip()`. Los
  callers `_apply_5_3`, `_apply_5_3b`, `_apply_5_5` buscan sin strip.
  Con CUSIP con espacios, la busqueda falla silenciosamente (default
  NOT_IN_LIST). 0 CUSIPs con espacios en datos reales; bug latente.
  Fix: `.str.strip()` en los 3 sitios. 3 tests.
- **Dedup catalog_p38_adapter <-> period_state (b870b34).** `_is_feasible`
  reimplementaba `feasible_state` con strings hardcodeados. Fix: delegar
  en la funcion canonica + importar IDENTITY_RESOLVED.
- **Dedup build_catalog_csvs <-> catalog_key (6b32ff0).** `canonical_serialization`
  y `_canonical_value` duplicadas. Fix: importar de catalog_key.
- **Docstring P65/P66 (d11bed2).** El documento decia 'Capa C en xfail
  esperando GO #40'. Falso: implementado desde 33d75cf (2026-09-21).
  Corregido con el expediente recuperado via `git show 7ba266d^`.
- **Deudas documentadas:** target_universe (de32ea1), timestamps (6e35715),
  catalog_pit (sin consumidor productivo, no tocado), reporting_dedup
  (P65/P66 ya documentado).
- Suite: 2297 -> 2336 passed + 2 skipped (14 commits de codigo + 14 de
  tests/corpus, neto +39 tests).

**Frente 8 - providers (12 commits).**

- **downloader atomico (e56a49b).** Descarga directa a zip_path;
  cache-hit sin validar. Si el proceso moria a mitad o el ZIP era
  corrupto, `update_sec_13f.yml` fallaba en CRC sin auto-recuperacion
  (nunca pasa --force). Fix: `.tmp + os.replace` + validar con
  `zipfile.is_zipfile`. 3 tests.
- **backup_providers._validate_with_cache (ce273eb).** Misma familia.
  `diff = abs(new - ref) / abs(ref)` con NaN: `nan > 0.05` es False,
  no entra en REJECTED, devuelve VALIDATED. El validador certificaba
  como 'validado' un dato nunca comparado. Fix: `math.isfinite`. 4 tests.
- **_blackrock_base flow_pct_assets (9250f90).** Si total_net_assets es
  0/NaN/inf, el cociente puede ser inf. inf contamina
  `fund_flow_robust_zscore_with_regime` (`dropna` no elimina inf,
  MAD queda inf, z colapsa a 0). Fix: `where(np.isfinite, np.nan)`.
- **Cache atomica xetra/bme/euronext (acca9cd) + blackrock_iwm (e197d77).**
  Mismo patron que downloader. `_save_cache` escribia directo. Fix:
  `.tmp + replace`. En IWM tambien el CSV historico (mas grave: las
  filas perdidas por CSV truncado desaparecen para siempre).
- **Guard 366 en 3 bucles is_market_day (2e83b3e).** Patron A1-11
  replicado: xetra L405, bme L239, futures L189. Sin guard, un fallo
  sistematico de `is_market_day` cuelga hasta el timeout de CI (90 min).
- **Dedup _fund_flow_utils (38a3809).** `calculate` interno identico a
  `_compute_z_value` (19 lineas). Fix: delegar.
- **N-PORT F5.7-20 completado (58cf809 + 2b0bd0c).** Dos problemas:
  (a) `update_sec_nport_data.py` descargaba 1 solo quarter; `discover_quarters`
  veia 1 -> `len<2` -> return silencioso -> CSV congelado desde 17/08.
  Fix: `--backfill N` + `sys.exit(1)` en `sec_nport_quarters` si <2.
  (b) `qqq_nport_flow.py` buscaba XML en cache gitignored que ningun
  script poblaba en CI. Fix: descarga automatica via submissions API
  + XML renderizado XSL de EDGAR. CSV pasa de 2026Q1 a 2026Q2.
- **falsa pista finra (0f48b62).** `get_archive_index` y
  `get_available_weeks` sin consumidor productivo. Dead code documentado.

**Falsos positivos rectificados (probe sobre datos reales).**

- `sec13f_list.py` con bug activo: descartado. `_aggregate_instrument_type`
  con mix EQUITY+UNKNOWN no se materializa; los 994-1030 CUSIPs multi
  por trimestre son todos OPTION (CALL/PUT).
- `CRON_SLOTS` sin verificacion: falso. `test_issue_manager.py:245`
  valida drift bidireccional contra daily_run.yml.
- `catalog_validator.py` dead code: falso. Contrato P66, integrado en
  `test_p66_pipeline.py` y `test_catalog_p38_adapter.py`.
- `P66 pendiente de implementar`: falso. 31+68 tests pasan, cero xfail.
- `guard_coverage.py` con NaN: falso. `coverage_pct_last` viene de
  division Python con `n_total > 0` guard; NaN imposible.
- `regenerate_cusip_crosswalk` sin secret: falso. No usa OpenFIGI.
- `_load_reference_cache`: solido. Verifica manifest + sha256 + quality.
- `relationships.explode_othermanager_edges`: contrato del docstring
  desalineado con implementacion, pero solo afecta a metricas de conteo
  (807-2120 filas con nombres de manager con coma interna). Docstring
  corregido (2f1da14); sin cambio funcional.

**Frente 7 - indicators/ (9 commits).**

- **call_share (f24e193).** Unica de 7 funciones hermanas en
  `options_metrics.py` sin `np.isfinite`. Con total_volume=inf -> 0.0/NaN
  silenciosos.
- **Escrituras atomicas (88eb2e5, 46854a8, bbf3422).** Cuatro `to_csv`
  directos sobre ficheros historicos: `darkpool_history.csv`,
  `sector_rankings_history.csv`, `pcr_history.csv`, `analisis_lideres.csv`.
  El state file SLPM (`state_transition.py`) tambien: mas grave, porque
  `_load_state` resetea el state a vacio si detecta JSON corrupto. La
  maquina de estados pierde `consecutive_count` y `confirmed_state`
  silenciosamente.
- **slpm_v12 flow_proxy_z=0.0 (894c03c).** `m.get('flow_proxy_z') or
  np.nan` convertia un flow z-score exactamente cero (flujo neutro)
  en NaN. El leader dejaba de contribuir al LIS.
- **mte/scoring stress (5528ef2).** `stress(val)` con `pd.notna(val)`
  como guard: `pd.notna(inf)` es True -> `np.tanh(inf/2)=1.0` ->
  clip 1.0. Retornaba 'estres extremo' silenciosamente.
- **fls _zscore_last_over_lookback (7e2c1a3).** Con `series.iloc[-1]=NaN`
  (hueco en el ultimo dato FRED) o inf, la funcion devolvia NaN/inf
  silenciosamente. Afectaba a los 5 componentes FLS.
- **mte/decision consensus (7cb5c7a).** Bug real. `consensus_score` con
  cualquier input NaN propagaba NaN a `compute_confidence` y de ahi a
  `classify_mte`. El clasificador devolvia `scenario=STAGFLATION
  confidence=nan` sin senal (el `if confidence == 0.0` no captura NaN).
- **mte/engine traceback (88eb2e5).** El `except Exception` outer anadia
  solo mensaje corto. Anadida `traceback.print_exc()` + docstring
  declarando el contrato 'puede devolver None'.

**Lecciones confirmadas.**

- Los bugs reales que aparecen en CI no estan en el codigo local: estan
  en los contratos entre ficheros (dos workflows que comparten cache,
  un script que referencia un modulo inexistente, una condicion que
  solo se materializa en produccion).
- La familia 'invariante declarada en docstring y no enforced en codigo'
  se ha repetido 5 veces: S-03c, PeriodState.sshprnamt,
  extract_sshprnamt_by_figi, _validate_with_cache, flow_pct_assets.
  Cierre sistematico con `math.isfinite` + tests.
- El patron 'fallo silencioso con workflow verde' ha aparecido 4 veces:
  cache 13F, issues:write, _update_issue, update_sector_holdings y
  familia. Cierre con `raise`/`sys.exit(1)`/`if:always()`.

---

### 2026-09-29 (noche): integridad parquets + EU 5y + PENDING BME + P65

- **Integridad parquet vs manifest (2a640f2):** los .parquet estan
  gitignored, los .manifest.json tracked. Tras git pull divergen. Ni
  guard_coverage ni pipeline_gate verificaban sha256. Detectados
  stock_prices (4880... vs 05a0...) y market_data (d077... vs daf6...).
  Fix: sha256 real del parquet sibling, fail-closed.
- **G-01 paridad market_data (08fe06b):** stock_prices truncaba a
  expected_session; market_data no. Movido helper a utils. Aplicado en
  data_loader antes del write_artifact.
- **BME publica T+1 (4bb54f8):** verificado 2026-09-29 23:25 CEST: BME
  devuelve hasta 28/09 mientras Xetra/Euronext/US ya tienen el 29.
  `_compute_by_market` contaba cobertura en 29 para BME -> INVALID ->
  guard aborta. Fix: MARKETS_WITH_PUBLICATION_LAG=('BME',), status
  PENDING. Guard exime PENDING, no todo-PENDING.
- **EU providers a 5y (4bb54f8):** BME/Xetra 430d -> 1830d. Medido:
  1277 filas/0.3s, 1274/0.8s. Euronext topa en endpoint (~510 sesiones,
  verificado con nb_session=1300).
- **P65 decision de diseno (f131ce3):** `_apply_intra_period_dedup` v1
  no emite DROP_DUP. Todas las decisiones KEEP. Conectar P65 no alteraria
  nipc_total. Hipotesis C2 descartada. 03_IAE.md corregido (2 sitios).

### 2026-09-30 (noche): auditoria externa sector_regime + fixes H8/R2/H1 + Fix D

**Contexto.** Auditoria externa del subsistema
`regimes/sector_regime.py::compute_sector_scores` en colaboracion con
un auditor. Tres iteraciones: informe inicial, anexo con hallazgos
nuevos, dictamen definitivo.

**Hallazgos H1-H10.** El expediente completo esta en
`08_AUDITORIA_SECTOR_REGIME.md`. El central (H8) es material: cuando
`close_sector` es NaN, `trend_position` y `breadth` devolvian `-1.0`
(senal bajista maxima fabricada). 1454 filas historicas afectadas.
Acoplamiento grave: cuando solo quedan esos 2 componentes, el sistema
producia `score = -1` + `penalty = 1` (maxima conviccion bajista +
ausencia de descuento por dispersion), derivados de ausencia de datos.

**Decisiones D1-D4.** D1 propagar NaN desde la raiz. D2 umbral
n_valid >= 4. D3 mantener formulas, documentar semanticas. D4
recalcular historico via E2E.

**Fixes ejecutados (6 commits con verificacion triple).**

- **43a2fdb:** cierres excepcionales NYSE (2018-12-05 Bush,
  2025-01-09 Carter). El calendario algoritmico no los reconoce.
- **ee24afc:** `trend_position` propaga NaN via `.mask(close.isna())`.
  Cierra H8 en la raiz.
- **c98c22a:** `compute_sector_scores` resuelve `effective_date` sobre
  los 11 sectores (min_coverage=0.90) antes de `last_scores`. Cierra
  R2 (regla dura: ninguna metrica agregada usa `.iloc[-1]`).
- **e92f61c:** umbral `n_valid >= 4`. Con menos componentes, la
  renormalizacion convertia hasta 0.25 del peso original en el 100%
  del score. Cierra H1.
- **c798857 (Fix D):** `append_dedup` en 5 writers de historico de
  flujo (ssga, amundi, _blackrock_base, qqq_nport_flow, cftc). Bug
  preexistente destapado por el primer run E2E: los writers
  sobrescribian su CSV sin leer el existente, perdiendo filas si el
  proveedor devolvia menos.
- **7b0044d (Fix D2):** normalizar `Date` en ssga antes del
  `append_dedup`. El primer fix D introdujo duplicacion (56093 ->
  112173 filas) porque `append_dedup` normaliza solo `date`
  minuscula y SSGA usa `Date`.

**Verificacion E2E.** Doble run. Primero destapo Fix D, revertido.
Segundo tras Fix D2: sin perdida ni duplicacion, Gate 10/10, NIPC
8256882557, ranking identico pre/post. Suite 2961 + 2 skipped.

**Regla nueva en 01_METODO §8.6:** los tests que verifican "fila
presente" no detectan "conteo correcto". El test de Fix D original
era ciego; el reforzado verifica `count == N` y `duplicated() == 0`.

**Leccion:** el run E2E detecta bugs que los tests unitarios no ven.
Verificar siempre la salida de un run con `numstat` (buscar lineas
perdidas netas) y con conteo de duplicados, no solo con "fila
presente".

### 2026-10-01: auditoria del reporte diario + 6 fixes G/H/K/L/M/N

**Contexto.** Auditoria linea a linea del reporte diario generado el
2026-09-30. Busqueda de NaN, ceros, inconsistencias, bugs silenciosos
y errores en la presentacion. 951 lineas, 70 secciones. Leido completo
en 5 bloques.

**Hallazgos confirmados (todos cerrados).**

- **Bug activo: alerta Price-Flow con texto fijo incorrecto.**
  `alerts.py:31` publicaba literal "Precio fuerte" para cualquier
  `status != ALIGNED`. Con XLU (`PRICE_WEAK_FLOW_SUPPORTIVE`,
  retorno -6.01%) decia "precio fuerte" cuando el detector decia
  "precio debil". Fix F en commit `29f72e8`.
- **Bug activo: 32 tickers europeos sin actualizar.** 13 Euronext +
  19 BME. `_cache_is_fresh` usaba `(ref - last).days <= 1`. Con
  ref=30-sep y cache=29-sep, "fresco". Fix K en commit `22cbcb5`.
- **Bug activo: cache-hit no validaba cobertura.** `stock_data_loader`
  y `data_loader` aceptaban cache si la fecha era la esperada, sin
  mirar cobertura de la ultima fila. Con 89.8% de cobertura y 30-sep,
  el cascade europeo no se ejecutaba. Fix M en commit `c994f01`.
- **Bug activo: 5 writers de fund flow con cache de 23h.** Mismo
  patron que Fix K pero en fund flow. Fix N en commit `b30a918`.

**Falsos positivos descartados.**

- NaN literales: 0 publicados. Los 2 hits son notas metodologicas.
- Encoding tildes: artefacto consola CP850.
- `-0.00` (61 casos): redondeo de floats.
- Celdas vacias Spring/SOS: por diseno (eventos Wyckoff).
- `RS` vs `RS DeltaMed`: metricas distintas con nombres distintos.
- `flow=0` FEZ: shares_outstanding identico 28 vs 29-sep. Real.
- Liquidity Delta 0.000: FRED sin dato nuevo entre runs. Real.
- 3 cifras de tickers (40/220/313): universos distintos.

**Verificacion end-to-end.** 3 runs E2E. Coherencia temporal
restaurada: Sector Breadth publica 30-sep. Cache SSGA avanzada a
29-sep. 12 descargas SSGA + DAX + ISF + IWM + Amundi + CFTC.

**Deuda residual.** 20 tickers LSE sin observacion local. El
scraper externo `lse-close-scraper` tiene el 30-sep; el checkout
local esta congelado en 25-sep. En CI funciona (workflow
`daily_run.yml:139` clona el repo fresco). No es bug.
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
- **Atomicidad familia 3** (escritura no atomica en `outputs/history/`): CERRADO 2026-09-29. 17 sitios con READ+WRITE del mismo path sin `.tmp + os.replace`. Si el proceso muere a mitad de `to_csv`, el CSV original queda truncado y el siguiente run pierde filas. Alta (17): `outputs/history/` append+dedup (1 european_coverage, 2 breadth_metrics, 1 engines, 1 flows_primary, 1 finalize, 2 market_data, 4 sectors_base, 5 sector_metrics). Media (15): regenerables (no leen el CSV antes de escribir). Fix comun: `.tmp + replace`. Tests fuertes por ambos lados (verde con fix, rojo sin fix).
- **FU-009-bis (append_dedup):** columnas all-NA excluidas antes del `pd.concat` (FutureWarning pandas 2.x, rompera en 3.0). CERRADO 2026-09-29.

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
- **H1-A**: CERRADO. Crosswalk 7caa86b NO reproduce golden con HEAD tras H1-B v2.3 + O1. Baseline vigente: `golden/current.json`. Correccion 2026-09-30 (D5): la afirmacion original era erronea.
- **H1-B** (clasificacion CALL/PUT): CERRADO v2.3. Baseline -4.264.449.012 (par Q4-2025 -> Q1-2026, congelado; el run diario usa el par vigente).
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

- **Verificacion triple de tests de bug (2026-09-29 noche):** un test de
  bug NO vale si solo pasa con fix. Debe: verde con fix -> ROJO sin fix
  (restaurando el commit padre del fix via `git checkout <pre> -- <file>`)
  -> verde con fix. Sin el paso intermedio, el test puede estar verde sin
  ejercitar el codigo (falso positivo). Aplicado al barrido familia 3
  (17 sitios de escritura no atomica en `outputs/history/`).
- **`git stash` no stashea lo commiteado:** si el fix ya esta en HEAD, usar
  `git checkout <commit-padre> -- <archivo>` para revertirlo temporalmente.
  Detectado al primer intento de verificacion triple (falso negativo).
- **Guards que esconden el bug en tests:** helpers con `if df_stocks.empty:
  return None` no ejecutan el bloque de escritura si el test pasa df vacio.
  Llamar al helper directamente con datos no vacios. Detectado en
  `sector_metrics`, corregido tras falso verde en el primer intento.

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
- `06_IAE_P65_P66.md` (contrato P65 + P66, on-demand).

**`ESTADO_SISTEMA.md`** se genera en `Consolidacion_Documentos/ESTADO_SISTEMA.md` por `scripts/generate_estado_sistema.py`. Se regenera con cada commit. Se referencia desde `00_ARRANQUE.md`.

**Borrado del corpus antiguo:** ejecutado en `cdf47ad` (2026-09-29). Se conservan `docs/auditoria/iae/evidence/` y `docs/auditoria/iae/golden/`, en uso activo por `03_IAE`.

---

## 6. REFERENCIAS RAPIDAS

- **Precedente de "conflicto entre doc y codigo":** A3.2-b (config `XLU,XLRE` vs literales `XLRE,XLU`). El codigo gana.
- **Precedente de "falso positivo de auditoria":** F2-25 (fix ya aplicado, la auditoria no lo vio).
- **Precedente de "bug silenciado por except outer":** A5-07 (`leaders.py`, `NameError` en INSUFFICIENT_COVERAGE oculto como "Modulo de lideres omitido").
- **Precedente de "test fragil":** A3.3-13 (test anclado a numero de linea 89 de engine.py).
- **Precedente de "cache freshness vs observacion":** A2.1-01 (mtime de fichero = execution time, no fecha de observacion).
- **Precedente de "schema aditivo backward-compatible":** FU-002, politica 2026-09-24. `quality.by_market` anadido sin bump de `schema_version`.

---



---

## Resumen D5-D18 (bitacora podada 2026-10-02)

Extraido de las sesiones 2026-09-30 (tarde, sesiones 2 y 3) al podar
la bitacora. Los commits siguen en `git log`.

### D5 (corpus desincronizado, 2026-09-30)

Cinco ficheros del corpus con numeros/afirmaciones desfasadas:
00_ARRANQUE decia 2340 tests (real 2383), corpus v2 (real v3),
"10 contratos" (real 9), OilPriceAPI commodities (eliminada),
03_IAE decia 8280 LOC (real 7088). Sesion documental completa.

### D6 + D6b (datetime.now en reporte)

El header del reporte usaba `datetime.now()` sin tz. Violaba
00_ARRANQUE §2. Fix: `generate_daily_report` recibe `reference_date`
con fallback a `now(ZoneInfo('Europe/Madrid'))`. Ademas 6 usos de
`datetime.now()` en calculos de "age" propagados a reference_date naive.

### D6b (idem anterior)

### D7 (volumen Yahoo no consolidado, WONT FIX)

Documentado como WONT FIX en 02_ARQUITECTURA §11. Razon: el volumen
Yahoo no consolida correctamente en algunos tickers; afecta
`volume_z` pero el impacto es bajo y el coste de corregir seria alto.

### D9 (concurrency.group desalineado)

`update_index_holdings.yml` y `update_european_holdings.yml`
escriben ambos `data/index_holdings.csv` con grupos distintos.
Fix: unificar en `update_index_holdings_csv`.

### D10 (traza 13F)

`_write_ingest_trace` escribia `ingest_source`/`ingest_actor` en el
manifest del trimestre, incluso en rama SKIP. El manifest es
artefacto de integridad (sha256). Fix: append a
`data/sec_13f/ingest_traces.jsonl` (JSONL versionado), manifest
intacto, outcome explicito (SKIP/INGEST).

### D11 (tickers residuales en holdings)

Los 4 parsers volcaban al CSV todo ticker no vacio: cash ("-"),
futuros (IXAU6, XARU6), CUSIPs (2602335D), placeholders SSGA
(999USDZ92). El consumidor filtraba con INVALID_TICKERS fragil.
Fix: `src/holdings_filter.py` con regla estructural, aplicada en 4
parsers, INVALID_TICKERS retirada, CSV limpiado (526->502 y
2669->2661 filas).

### D12 (cosmetico)

D7 documentado como WONT FIX. BOM UTF-8 retirado de 7 workflows.
Duplicacion de titulo en 05_BITACORA corregida.

### D13 (macro_conf hardcode)

`compute_macro_regime` devolvia `conf = 0.5` siempre. El aviso de
header.py (`macro_conf < 0.30`) nunca se activaba. Los otros 3
regimenes SI calculan confianza real. Fix: `confidence_from_range`
(misma politica C19).

### D14 (tests skipped CI)

El saliente reportaba deuda de cobertura. FALSO: daily_run.yml ya
ejecuta los tests de frescura en post-run con datos reales. Fix
cosmetico: pytest.ini con `-ra` + marker `integration_real_data`.

### D15 (dead code declarado)

Eliminados: `finra.get_archive_index`, `finra.get_available_weeks`,
`evidence_matrix.save_evidence_matrix`, `import Path` huerfano.

### D16 (check_manifest mentia)

`check_manifest` solo verificaba el `.manifest.json`. En CI el
manifest esta versionado, el parquet no (gitignored). Resultado:
`[OK] VALID` sobre un artefacto ausente, mientras parquet daba
`[FAIL]`. Mismo patron que D6/D13/D14: contrato que miente. Fix:
manifest OK + parquet ausente -> SKIP en CI, WARN en local. Manifest
ausente o invalido -> FAIL sin cambio.

### D18 (cobertura baja en 4 modulos)

Documentado en 00_ARRANQUE §5 como "sin bug detectado tras
inspeccion". Los numeros del corpus estaban desfasados: real eran
12/25/27/51%, no 12/12/27/50%.

- `macro_manual_loader`: 12% -> 92%. 7 tests.
- `european_coverage`: 25% -> 98%. 12 tests.
- `pipeline_contractual`: 27% -> 38%. 4 tests. Orquestador
  `run_contractual_nipc` E2E-only por diseno.
- `data_loader`: 51% -> 59%. 38 tests. `download_market_data`
  (red) E2E-only.

Criterio: NO inflar cobertura con mocks del orquestador.


### Frente B (caracterizacion de macro_regime)

S-03-deep cerro `regimes/macro_regime.py` como CERRADO SIN CAMBIOS
(205 LOC, 12 ramas alcanzables, 1 BAJA + 6 INFO WONT FIX). La
auditoria detecto una asimetria de cobertura: `sector_regime.py`
tiene golden + 5 suites; `macro_regime.py` solo D13 (confianza) y
D22 (helpers). Las 12 ramas del regime y `compute_macro_score`
quedaban sin test directo.

Decision 2026-10-03: caracterizar el modulo con monkeypatch de
`compute_macro_signals`/`compute_macro_score` en el namespace de
`regimes.macro_regime`. Descartado df_market sintetico completo
(20+ columnas fragiles; el autor de `test_macro_regime_confidence.py`
ya lo rechazo) y descartado refactor-extract (contradice el
dictamen S-03-deep; requiere contrato firmado). Sin golden: los 12
casos parametrizados son el contrato.

Resultado: 27 tests (12 ramas + 5 precedencia + 1 contrato de
ramas inalcanzables sin fundamentales + 5 de score + 4 de cadena de
fallback de volatilidad). Dos hallazgos cerrados:

- **MR-2** (BAJA, WONT FIX condicional): confirmado con test
  explicito. Sin `^VIX` ni `vol_regime_score`, `all_signals` sale
  sin columna `volatility` y `compute_macro_regime` lanzaria
  KeyError.
- **MR-8** (BAJA, nuevo): el comentario de `compute_macro_score`
  afirmaba renormalizacion entre niveles; el codigo solo la hace
  entre componentes del mismo nivel. Comentario alineado con codigo.

Commits: 7f13e6c, 6a60842, 3dc0c1a. Contrato: los tests anclan el
comportamiento actual, no el declarado. Coherente con "el sistema
prevalece sobre la documentacion".

### Auditoria linea a linea wyckoff_v1.py (v1.8)

El dictamen externo `38_dictamen_final_5bX_v3.md` (2026-10-03) firmo
el paquete contractual de 5b.X v3, pero declaro expresamente en su
seccion 5 que no habia inspeccionado fisicamente el diff
`d324346..d0874f4`. La firma se emitio sobre la evidencia presentada,
no como certificacion de inspeccion visual.

Auditoria 2026-10-03 cierra ese hueco sobre
`indicators/wyckoff_v1.py` (497 LOC, contrato v1.8). Metodo:
barrido de los 13 patrones de `01b_PATRONES.md` + revision linea a
linea + cotejo I1-I36 contra tests.

Resultados: 4 BAJA (3 dead code eliminadas, 1 trazabilidad de
invariantes). Cero ALTA, cero MEDIA. 3 INFO WONT FIX razonado.
Cotejo invariantes: 36/36 por nombre. Suite sin regresion.

Estado operativo confirmado: el pipeline del 3-oct ejecuto el
modulo LEGACY, no v1.8. Coherente con el plan de migracion (Frente
D bloqueado hasta cierre 5b.X, prohibicion vigente del dictamen §3).
No es bug ni regresion: es el estado esperado del proyecto.

Audit doc: `docs/auditoria/wyckoff/40_auditoria_v18_linea.md`.
Commits: `036c4ef`, `de7d729`, `6717bf9`.

### Walk-forward historico de SOW (expediente 41)

El plan v3 (§15, cdec27d) introdujo el walk-forward historico como
metodo de validacion alternativo a 5b.X, con el objetivo de desbloquear
la migracion del modulo Wyckoff sin esperar 12 meses. El sistema es
experimental y el legacy no discrimina fases.

Se descargo historico adicional 2015-2021 desde Yahoo (no productivo,
`data/stock_prices_extended.parquet`) para ampliar la ventana.

Test confirmatorio (split 2021-2023 / 2024-2026): FAIL. La candidata
SOW no pasa el criterio predefinido. Periodo A sin evidencia, Periodo
B positivo.

Analisis exploratorio anual (post-hoc, declarado): heterogeneidad
marcada. 3 anos con IC95 positivo (2020, 2025, 2026), 0 con IC95
negativo, 9 sin evidencia. Hallazgo etiquetado como HIPOTESIS de
dependencia de regimen, no como conclusion. Dictamen externo aclara:
no demuestra "SOW = detector de estres sistemico", no refuta
overfitting reciente, no autoriza filtro VIX/drawdown ni recalibracion.

Candidata SOW: CONGELADA / NO PRODUCTION. Config `WYCKOFF_SOW_* = None`
intacto. Fail-closed intacto. 5b.X (2027) sigue siendo juez final.

Expediente: `docs/auditoria/wyckoff/41_walk_forward_sow.md`.
Commits: cdec27d, 1a08884.

### Analisis SOW: walk-forward + grid 240 (expedientes 41-43)

Derivado del walk-forward de calibracion (expediente 41). Se ejecuto
un grid de 240 combinaciones calibrado SOLO sobre TRAIN 2015-2020 y
validado sobre TEST 2021-2026 (OOS limpio, sin ver el tramo de
calibracion).

Hallazgo principal: la region M=5, X_ATR >= 0.50 del grid generaliza
mucho mejor que la candidata congelada v1.9. La ganadora TRAIN cae
en percentil 1.2 de las 240 combinaciones ordenadas por lift en TEST.
La v1.9 cae en percentil 20.4.

Expediente 42 propuso C2 como v2.0. Dictamen externo posterior
corrigio la interpretacion: C2 fue seleccionada sobre TEST, no
tiene evidencia OOS limpia.

Interpretacion consolidada (expediente 43):

- **C1** (N=40 M=5 X_ATR=1.00 Y_VOL=1.10): seleccionada solo con
  TRAIN, TEST lift +0.103 IC95 [+0.018, +0.192]. **Evidencia OOS
  historica limpia.** No activada.
- **C2** (N=60 M=5 X_ATR=0.50 Y_VOL=1.10): top-1 del grid sobre TEST.
  IC95 [+0.043, +0.170] exploratorio, no confirmatorio. Congelada
  como candidata exploratoria. No validada.
- **C0/v1.9**: intacta. 5b.X v3 solo aplica a C0.

Config SOW sigue None / fail-closed. 5b.X (2027) sigue siendo juez
final. Nuevo protocolo + nuevo power analysis requeridos para llevar
C1 o C2 a validacion oficial.

Expedientes: `41_walk_forward_sow.md`, `42_candidata_c2.md`
(interpretacion corregida), `43_estado_sow_tras_walkforward.md`
(referencia consolidada).
Commits: acd3632, 0bb893d, este cierre.

### Verificacion E2E del reorder D3 y fix de frescura Evidence Matrix

Sesion 2026-10-05. Cierre del Hueco 1 declarado por el traspaso:
E2E real en main tras el merge del reorder D3.

E2E ejecutado en main (HEAD 46291c9). Snapshot pre -> run.py ->
snapshot post. Resultados: exit=0, Gate 10/10, 52 secciones ##
identicas, 1021 lineas identicas, diff textual de 6 lineas
(3 timestamps + 3 de tabla de frescura). El reorder D3 no rompe el
reporte. Commits del reorder: 0dd363d, 1503fe4.

Hallazgo lateral: la fila `Evidence Matrix` de la seccion
`## Calidad, frescura y cobertura de datos` reportaba un run de
desfase. Causa: `compute_data_quality` (via `compute_market_data`)
leia `evidence_matrix.csv` antes de que `finalize` lo reescribiera.
Bug preexistente, cosmetico (BAJA), ajeno a D3.

Decision estructural: diferir `compute_data_quality` a despues de
`finalize` en el pipeline. Orden actual:
... finalize -> data_quality -> IAE -> report. Parametro nuevo
`skip_data_quality` en `compute_market_data` con default `False`
(preserva comportamiento legacy para tests existentes). Commit
538f2aa.

Deuda anotada: pandas `to_csv` escribe CRLF en Windows; `.gitattributes`
declara `eol=lf` para csv/json. Cada `run.py` marca ~25 ficheros como
modificados en `outputs/history` y `outputs/state`. Convencion actual:
commit diario `Daily hist/state`. Fix de raiz pendiente:
`lineterminator='\n'` en writers.

Anomalias del diff PRE/POST resueltas: (A) `analisis_lideres*.csv`
generados por el run, no perdidos; (C) 3 CSVs encogen por cambio de
`coverage 0.998...` -> `1.0`, no por reescritura retroactiva.

Commits: 538f2aa, 7aa9e68.

### Limpieza estructural del repo y fix rs_mom NaN

Sesion 2026-10-06. Dos frentes independientes ejecutados en el mismo dia.

**Limpieza estructural (~810 MB liberados).**

Se retiro material obsoleto tras inventario con contraste de consumidores:
- `outputs/audit/`: 29 snapshots de verificaciones cerradas (137 MB).
- `data/cache/`: cache HTTP regenerable (71 MB).
- `data/sec_13f/raw/`: TSV crudos 13F (600 MB); `processed/` intacto.
- `docs/automatica/`: 22 `.md` auto-generados. Retirados y anadido a
  `.gitignore`; regenerables via `scripts/generate_docs.py`.
- `docs/auditoria/iae/evidence/`, `docs/auditoria/daily_run_gate_*`, 3
  scripts Wyckoff sin referencias.

El expediente Wyckoff intermedio (42 ficheros, contratos v1.0-v1.7,
protocolos 5b.2-5b.4-bis) se archivo en `docs/auditoria/wyckoff/_archivo/`
con README explicativo. Se conserva por trazabilidad de auditoria externa
para 5b.X (2027). 15 ficheros vigentes quedan en la raiz del expediente.

4 ramas locales obsoletas (`backup-pre-rebase-20260912`, `gold-standard-v1/v2/v3`)
se convirtieron en tags `_archivo/*` y se eliminaron las ramas locales.
Los commits siguen accesibles por tag.

**Fix rs_mom NaN (commit 34c8029).**

Detectado en el run CI 37406727238 (2026-10-06 05:01 UTC): `RS Mom = nan%`
en las 20 filas de `analisis_lideres.csv`, con `n_valid_momentum=0` en
`sector_concentration.csv`. En local funcionaba por casualidad (NaN residual
no caia en posicion -21).

Causa raiz: `rs = close.loc[common] / price_etf.loc[common]` puede contener
NaN residuales por festivos USA (Labor Day, Thanksgiving, MLK, Presidents)
presentes en el indice comun de close y price_etf. `np.log(rs).diff(20).iloc[-1]`
propaga el NaN al ultimo valor. La formula original no hacia `dropna()` ni
validaba longitud; `index_leaders.py` ya tenia el fix, `stock_leader.py` y
`sector_breadth.py` no.

Impacto en cascada: `rs_z` (peso 25% del WLS) se anulaba, y el `WLS`
calculaba sin componente de momentum. Ademas, f-string directo en
`stock_leader.py:196` imprimia `nan%` en vez de `N/D`.

Fix:
- `indicators/stock_leader.py`: `rs.dropna()` + guard `len(rs) >= 21`,
  reemplazo del f-string por `_fmt_num`.
- `indicators/sector_breadth.py`: mismo guard, mismo patron.
- `tests/test_stock_leader_rs_mom_nan_guard.py`: 4 tests de regresion.

Verificado E2E: `analisis_lideres.csv` 20/20 rs_mom validos,
`sector_concentration.csv` n_valid_momentum=14-15 por sector, reporte
con 0 `nan%`.

**Fix cobertura europea (commits cc867fd, 5299e4c).**

Dos bugs encadenados que provocaban 3 SIN_DATOS perpetuos en Xetra
(`DB1.DE`, `HEI.DE`, `MUV2.DE`). Sintoma: `european_coverage.md` con
48 OK + 3 SIN_DATOS en todos los runs.

Bug 1 (cc867fd): `get_stock_list()` construia el universo de descarga
con 3 fuentes (`etf_holdings.csv`, `index_holdings.csv`,
`radar_target_catalog.csv`). Los 3 tickers estaban en
`config/xetra_ticker_map.csv` pero no aparecian en ninguna: el top-20
del DAX por peso los excluye, y no estaban en el catalogo radar. Fix:
anadir una 4a fuente = `supported_tickers()` de los 3 providers
europeos. Razon: el mapa del provider es la fuente de verdad de que
europeos rastrear.

Bug 2 (5299e4c): al anadir los 3 tickers al universo de descarga, el
parquet `stock_prices.parquet` seguia aceptandose como cache-hit
porque la validacion solo miraba fecha y cobertura de columnas
existentes. Razonamiento: `_last_row_coverage_ok` mide sobre las
columnas del df, no sobre `all_tickers`. Los 3 tickers no eran
columnas, no contaban en el denominador, cobertura seguia siendo
"OK". Fix: mover `get_stock_list()` antes del cache-check y validar
que el parquet contiene todos los tickers esperados. Si falta alguno,
forzar descarga.

Efecto colateral observado: la limpieza de `data/cache/` hizo visible
el bug 2. Sin la limpieza, los 3 tickers tenian cache previa y no se
notaba el problema. Con la limpieza, la cache se recreo solo para los
16 tickers que si estaban en `get_stock_list()` antes del fix 1.

Verificado E2E: `european_coverage` 48 OK + 3 SIN_DATOS -> 51 OK,
0 SIN_DATOS. Cache Xetra con `DB1_DE.csv`, `HEI_DE.csv`,
`MUV2_DE.csv` (1277 filas, 2021-10-04 a 2026-10-06).

Leccion registrada: una metrica que devuelve NaN puede ser sintoma de
huecos residuales en los datos, no necesariamente de ausencia. `dropna()`
antes de operaciones tipo `diff()` sobre series con calendarios parciales.

Leccion registrada (cache-hit): los contratos de cache-hit deben
validar contra el universo declarado, no contra el subconjunto ya
presente. Un parquet con N columnas al 100% no implica que cubra el
universo esperado de N+k tickers.

Commits: 34c8029 (fix rs_mom), cc867fd + 5299e4c (fix europeo),
48a9ee6 + c578ee1 + 15a7391 + cb1d8d0 (Daily hist/state), 826694b +
f847377 (limpieza). Tags: `_archivo/backup-pre-rebase-20260912`,
`_archivo/gold-standard-v1`, `_archivo/gold-standard-v2`,
`_archivo/gold-standard-v3`.

### F6-1b (2026-10-07): cierre de la divergencia FINRA

F6-02 (2026-09-28) ajusto los umbrales de frescura FINRA de (30, 45, 60)
a (20, 28, 45) tras detectar que 26d se marcaba como CURRENT, fuera del
rango regulatorio documentado (2-4 semanas = 14-28d).

El fix se aplico en `settings.FRESHNESS_FINRA` y en
`helpers._classify_finra_freshness`, pero dejo `indicators/data_quality.py`
con los valores antiguos hardcoded. Resultado: dos contratos paralelos
que se contradician visiblemente en el reporte diario (FINRA RECENT en
Data Freshness, CURRENT en Calidad, misma fuente y edad).

F6-1b elimina el hardcode. `data_quality.py` importa `FRESHNESS_FINRA`
de settings. Una unica fuente de verdad. Commit bdaa0326.

No se amplia a SEC/CFTC/FRED: sus clasificadores siguen hardcoded en
`data_quality.py` pero coinciden con settings hoy. Mismo patron de fix
si aparece otra divergencia.

Leccion: cuando un valor se centraliza en settings, buscar TODOS los
sitios que lo usan. Un fix parcial puede dejar dos verdades paralelas
que solo se descubren al cruzar el output. En este caso, F6-02 paso
desapercibido durante 9 dias porque cada test miraba solo su funcion.

## Resumen D1-D4 + D19-D40 + cierre 10-01 (bitacora podada 2026-10-08)

Extraido de las sesiones 2026-09-30 (madrugada, tarde, noche sesiones
5-8) y 2026-10-01 (madrugada, tarde, cierre) al podar la bitacora.
Los commits siguen en `git log`.

### Cierre P1: D1-D4 + Runbook (2026-09-30 tarde)

- **D4 (lag 13F unificado).** Tres politicas de lag convivian (doc ~45d,
  cron ~50d, script 60d). El comentario del YAML mentia sobre su propio
  codigo. Fix: eliminar el case del YAML, delegar a `--latest`, unificar
  `SEC_13F_QUARTER_LAG_DAYS` a 50d, 4 tests frontera.
- **D3 (health_check en CI).** 2 FAIL estructurales por parquets
  gitignored. Fix: `IS_CI` detecta entorno, `parquet` y `iae_section`
  pasan a SKIP en CI, `cron_slots` relaja umbral, workflows trimestrales
  sin runs -> SKIP.
- **D2 (determinismo vs Yahoo).** `00_ARRANQUE 2` pasa de "Determinista"
  a "Determinista **dado un snapshot del input**". Nuevo check
  `check_yahoo_revision`: compara manifest actual vs HEAD por sha256;
  mismo `last_date` con sha distinto -> WARN.
- **D1 (comp_breadth + penalizacion dispersion).** `tanh_normalize`
  colapsa con series constantes (27-61% de filas con `comp_breadth==0`).
  Segundo bug: `_dispersion = std/(|mean|+1e-9)` explotaba con media
  proxima a 0. Fix: mapeo directo `(breadth-0.5)*2` + std poblacional
  `ddof=0` sin dividir por media.
- **R (runbook cron 1-oct).** Los 4 workflows trimestrales nunca se
  dispararon por schedule. Creado `07_RUNBOOK.md`. Detectado que el
  traspaso listaba `update_sec_nport` dia 1; el YAML dice dia 20.

### Barrido de cobertura D19-D40 (2026-09-30 noche)

Cierre de la deuda P3 (cobertura de tests). Regla de parada:
ROI < 1 -> NO-DEUDA. Resultado: 2297 -> 2945 tests (+648). Detalle
granular en `git log`.

- **D19.** check_yahoo_revision verificado en produccion, sin commit.
- **D20.** utils.py 74% -> 88% (detect_cross_module_conflict,
  _confidence_range_row, _try_cleanup).
- **D21.** stock_data_loader NO-DEUDA: 110 sin cubrir son
  download_stock_prices (red) + _apply_lse_close_override (scraper
  externo). ROI < 1.
- **D22.** credit 19% -> 100%, breadth 22% -> 100%, macro_fundamental
  13% -> 97%.
- **D23 (barrido datetime.now).** 64 llamadas reales; 5 en pipeline
  productivo (mte_confirmation, validation_gate x2, flows_secondary,
  fundamental_signals). Fix: `reference_date=None` propagado desde
  `compute_all_regimes` y `run.py`.
- **D24.** Verificacion E2E. Gate 10/10, NIPC 8256882557.
- **D25-D28.** fls 7% -> 100%, index_phase 11% -> 92%,
  commodity_market_correlation 10% -> 90%, index_leaders 12% -> 70%,
  darkpool_history 8% -> 88%, breadth_equity 76% -> 84%,
  evidence_matrix 78% -> 94%, mte/decision 78% -> 88%,
  mte/engine 79% -> 95%.
- **Falsos positivos corregidos (verificacion triple):** D26
  (assert all sobre dict vacio), D27 (fixture ya ordenada), D23
  (reference_date = hoy).
- **D29.** sector_context 63% -> 100%, sectorial 67% -> 100%,
  synthesis 66% -> 96%. 19 tests.
- **D30.** volatility_regime 33% -> 100%, tactical_engine 14% -> 97%,
  structural_engine 21% -> 100%.
- **D31.** state_machine 12% -> 100%, slpm_v12 38% -> 93%. 38 tests.
- **D32.** sector_leader_divergence 16% -> 79%,
  sector_flow_characteristics 35% -> 100%. 21 tests.
- **D33.** options_metrics 65% -> 88%, rs_internal 28% -> 98%,
  vol_metrics 15% -> 100%, options.py 67% -> 69%. 33 tests.
- **D34.** options_metrics 88% -> 93%, index_leaders 70% -> 87%.
- **D35.** health_check 52% -> 84%. 31 tests. No cubre `main()`.
- **D36 (bug real).** `_generate_coverage_table` repetia
  `pd.Timestamp(pcr_data['last_date'])` sin proteccion (el padre ya
  tenia try/except). Con last_date no parseable, el reporte entero
  caia con DateParseError. Fix en ambos bloques.
- **D37.** pipeline_gate, guard_coverage, issue_manager cubiertos.
- **D38.** download_official_list_13f 27% -> 99% (34 tests, bug F5.6-X
  fallback 404), regenerate_cusip_crosswalk 69% -> 99% (12 tests, bug
  `str(NaN or "")` = "nan"), qqq_returns_yahoo 63% -> 98% (6 tests).
  `iae_pipeline` descartado (E2E-only por diseno).
- **D39.** Dead config CI (`daily_run.yml` con `git add` de 4 rutas
  inexistentes). Fosiles `02_ARQUITECTURA 11`. Poda bitacora 5 -> 10 +
  dia. Criterio fino E2E en `01_METODO 4`. Skips opt-in formalizados.
- **D40 (auditoria workflows).** 9 workflows reales (no 10).
  `03_IAE` decia 286 LOC update_sec_13f (real 373). Riesgo:
  `update_macro_manual` (17 6 UTC) coincide con `european` a 06:17
  (groups distintos, no serializados, ambos pushean a main).

### Incidente cron trimestral + sys.path (2026-10-01 tarde)

- **Los 3 workflows trimestrales fallaron en el primer schedule real**
  (`ModuleNotFoundError: No module named 'src'`). `python scripts/X.py`
  pone `scripts/` en sys.path[0], no la raiz. Fix `1bead8a`.
  `sys.path.insert` en 4 scripts (patron ya en 18). Los 2 ultimos
  tenian cuerpo al importar: refactor a `main()` + guard.
- **Fix `27248d3` (desalineacion SSGA).** `get_state_street_holdings`
  con entradas invalidas del Excel desalineaba las 3 listas.
- **Tests nuevos.** `test_scripts_sys_path.py` (runpy desde cwd=scripts),
  `test_update_index_holdings_ssga.py` (Excel sintetico).
- **Observacion.** Gate `_manifest_satisfies` exige sha256 del parquet
  gitignored. En CI, rama CURRENT inalcanzable -> siempre READY.

### Completion Receipt v1 + WLS NaN (2026-10-01 cierre)

- **Diagnostico externo del gate** (`daily_run_gate_discrepancia.md`).
  Bug: `_manifest_satisfies` con sha256 de parquet gitignored ->
  gate inalcanzable en CI. Los 4 slots del 01-oct ejecutaron pipeline
  completo por esto.
- **Dictamen externo** (`daily_run_gate_dictamen.md`). Recomendacion E:
  separar integridad del artefacto (manifest <-> parquet <-> sha256,
  sin cambios) de idempotencia del workflow (Completion Receipt en
  GitHub Actions Artifacts). 8 invariantes I1-I8.
- **Contrato v1** (`daily_run_gate_contrato_v1.md`). Aprobado
  2026-10-01.
- **Implementacion (6 commits).** `e3c82fb` (find_completion_receipt,
  write_completion_receipt.py, 10 tests I1-I8), `e440fec` (diagnostico
  wls/tickers/len), `feee18d` (RECEIPT_FILENAME = basename artifact),
  `f5e817f` (GH_TOKEN en env del step Run gate).
- **Fix WLS NaN** (`2f956ed` + `e819952`).
  `compute_wls_for_index::robust_intra` usaba `np.median(np.abs())`
  que NO ignora NaN. Con 1 ticker NaN (SPCX en QQQ), toda la columna
  `rws_z/stab_z` -> NaN. Fix: `np.nanmedian` + guard `pd.isna(mad)`.
- **Fix gitignore** (`1c0ab0c`). `completion_receipt.json` y
  `validation_gate_result.json` en `.gitignore`.
- **Verificacion CI.** Run `36904431677` verde (artifact
  `completion-receipt-2026-09-30`, 462 bytes). Run `36909383320`:
  `state=CURRENT`, `run-system=skipped`. Contrato v1 confirmado en
  produccion.

**Pendiente heredado:** test rojo `test_build_catalog_csvs::test_idempotencia`
(resuelto en sesion Catalog PIT del 10-02). Deudas latentes `np.median`
en `stock_leader.py:59` y `options.py:46`.

### Baseline IAE vs par vigente (2026-09-30 madrugada)

No es bug. El reporte diario calcula el NIPC del par **vigente**
(`_list_available_quarters` toma los 2 ultimos). Al entrar 2026Q2
(`c5e3ee0`), el par paso de Q4-2025 -> Q1-2026 a Q1-2026 -> Q2-2026.
Reproducido local: 8256882557 (identico al CI). El baseline
-4264449012 corresponde al par antiguo, congelado en `current.json`.
No se toca codigo. Corpus actualizado (`c087cb8`).

**Hallazgos del run manual 36642077307:** `update_futures` exit 1
(BZ=F/CL=F BLOCKED 403 OilPriceAPI). Warning 'Cache 13F' es falso
positivo (restore-key v2 vs primary v3). 3 tests skipped en CI por
parquets gitignored.

---

### Regeneracion de sector_wyckoff_distribution (2026-10-08)

**Contexto.** Migracion de los 5 consumidores de Wyckoff del legacy
(`indicators/wyckoff.py`) a v1.8 (`indicators/wyckoff_v1.py`). Al
migrar se detecto que el CSV historico `sector_wyckoff_distribution.csv`
contiene dos problemas:

1. **Warm-up artefactual.** Las primeras fechas (2026-09-08 a ~09-15)
   escribieron `20 RANGE` en todos los sectores. Causa: el pipeline
   arranco sin datos suficientes -> `wyckoff_score` devolvia NaN ->
   legacy caia en `return "RANGE"`. No eran clasificaciones, eran
   ausencia de clasificacion.
   Dos ramas silenciosas del legacy contribuian: `len(close) < 200 ->
   INSUFFICIENT_DATA` y `score_clean.empty -> RANGE`. La segunda
   etiquetaba como fase valida una ausencia de datos.
2. **Mezcla de clasificadores.** Las fechas hasta 10-07 se calcularon
   con legacy; desde 10-08 con v1.8. Un CSV con dos algoritmos
   distintos es incoherente.

**Decision.** Regenerar el CSV completo con v1.8 core (4 fases).
Script: `scripts/regenerate_wyckoff_distribution.py`. Recorre las 20
fechas unicas del CSV, aplica `classify_wyckoff_phase(df, ticker,
as_of=fecha)` con v1.8, reconstruye la distribucion de fases por
sector.

**Limitacion reconocida.** El parquet actual esta revisado por Yahoo.
Las fechas pasadas se calculan con `as_of=fecha` sobre el OHLC
revisado. No es exactamente lo que v1.8 habria dicho el dia. La
diferencia es ruido menor (las fases se calculan sobre ventanas de
200+ dias y los precios apenas cambian).

**Preservacion.** El CSV legacy completo queda en:
- Git history: commit `089f819a`.
- `outputs/audit/wyckoff_legacy_REAL_20261008.csv` (copia para
  referencia).

**Efecto medido.** 0/220 filas coinciden con legacy. Es esperado:
legacy clasifica por umbral de score; v1.8 por conjuncion
estructural. El cambio en las primeras fechas es correccion de
warm-up, no reescritura arbitraria.

**Relacion con 13_comparativa_legacy_v1.md.** Esa comparativa mide
una foto (un dia, 316 tickers) con legacy vs v1.8. Este cambio
aplica el mismo criterio al historico del CSV.

### Auditoria externa radar 2026-09-26: cifra corregida (2026-10-08)

Correccion 2026-10-08: la cifra "205 hallazgos (6 ALTA / 98 MEDIA / 101 BAJA)" que circulaba en `00_ARRANQUE` y en este historico era erronea. Los informes originales existen en git (commit `bc656374`):

- `docs/auditoria/radar/audits/AUDITORIA_2026-09-26.md` (FASE 0-3)
- `docs/auditoria/radar/audits/AUDITORIA_FASE_5_2026-09-26.md` (FASE 5)
- `docs/auditoria/radar/audits/AUDITORIA_CONSOLIDADA_2026-09-26.md` (consolidado + addendum FASE 2.2/2.4)

Cifra real (tabla por fases + addendum en la consolidada): **142 hallazgos (6 ALTA / 71 MEDIA / 65 BAJA)**.

Los 3 ficheros fueron borrados de `main` en `cdf47ad2` (limpieza de corpus antiguo). Recuperables desde `bc656374`. Los 6 ALTA se corrigieron (trazables en git log pre-2026-10-01). **No son deuda huerfana**: los informes existen.

Accion: cifras corregidas en `00_ARRANQUE` (commit 634d44d4) y en este historico. Sin accion tecnica adicional.

### Archivo de sow-v4-validation: WONT FIX razonado (2026-10-08)

La rama `sow-v4-validation` fue archivada como tag `_archivo/sow-v4-validation-20261008` (2026-10-08). Contiene 88 ficheros unicos frente a `main`: 15 scripts del paquete `scripts/sow_v4/` + `scripts/sow_v19_bootstrap*.py`, y 4 artefactos del protocolo (`SOW_v19_PROTOCOL.json`, `.freeze.json`, `49_sow_v19_resultado.md`, `50_sow_v19_bootstrap_ops.md`). El resto son el expediente wyckoff antiguo ya movido a `_archivo/`.

**WONT FIX razonado.** Verificado 2026-10-08: ningun fichero de `main` referencia `scripts/sow_v4/*`, `sow_v19`, `SOW_v19_PROTOCOL.json`, `49_*` ni `50_*`. Cero referencias colgantes.

Los scripts implementan el protocolo v4.1 / v5, descartado tras el walk-forward del 2026-10-04 (47_estado_sow_tras_v5_grounded.md: `decision.estado = MUESTRA INSUFICIENTE`, `N_eval=2/8`). El protocolo vigente es v6 (borrador `48_sow_protocol_v6.md` en `main`, pendiente firma externa, sin implementacion hasta firma).

Portar el material del tag crearia huerfanos (implementacion de un protocolo descartado, sin consumidor). El tag preserva el material integro y recuperable si el auditor externo lo pide.

Nota adicional: los ficheros 46 y 47 del expediente SOW fueron recuperados selectivamente a `main` en `e528da19` (2026-10-05).

### Archivo de gold-standard-v4 (2026-10-08)

La rama `gold-standard-v4` (HEAD 8afce786, 2026-10-05) fue archivada como tag `_archivo/gold-standard-v4-superseded-20261008` y borrada. Superseded por `main`:

- Migracion Wyckoff v1.8 rehecha con commits 2fd6c449/35663000/f782b9d4 (plan seccion 6 de main corrigio sector_breadth -> sector_regime).
- Fix D aplicado (5 providers LF).
- Fix NaN stock_leader (2026-10-06).
- Manifests sec_13f mas nuevos.

Delta positivo no portado: nota explicativa RS en `stock_leader.py` ("*RS = RS Level (precio accion / precio sector)..."). Recuperable desde el tag si se necesita cherry-pick.

Hallazgo colateral corregido el mismo dia: la evidencia IAE de `docs/auditoria/iae/evidence/` (8 directorios, 53 ficheros) referenciada en `03_IAE.md` seccion 6 no existia en `main`. Portada desde el tag antes de archivar (commit a356ae1d). Tabla de conteos corregida de 67 -> 53.

### Verificacion completa del sistema + 4 fixes (2026-10-08)

Sesion de verificacion end-to-end bajo peticion: revisar el estado real del sistema (no solo documentacion). Hallazgos y fixes:

**Hallazgos corregidos.**

- **check_iae_section (health_check).** Buscaba `STALE` en todo el reporte. El STALE del N-PORT en Data Freshness disparaba un falso WARN cuando la seccion IAE estaba limpia. Fix: aislar el check al bloque de la seccion (`## Acumulacion Institucional (13F)`). Commit c09a268c + 4 tests.
- **save_regime_history (pipeline).** Escribia fila con la fecha natural de FRED (iorb.csv) cuando el pipeline resolvia `effective=sesion NYSE anterior`. Fila huerfana con `date=2026-10-08` y datos de 2026-10-07. Rompia R1 (cobertura no declarada). Fix: parametro `effective_date` propagado desde `temporal_meta['by_contract']['EQUITY_EOD']['effective_date']`. Commit d495dc1e + 3 tests.
- **check_workflows (health_check).** (a) Filtraba `--event schedule`, manteniendo WARN permanente sobre workflows trimestrales ya corregidos el 1-oct. (b) Trataba `cancelled`/`timed_out`/`startup_failure`/`skipped` como OK. Fix: mirar ultimo run de cualquier evento + WARN para conclusiones no-exito. Commit 8e8826f5 + 5 tests.
- **etf_holdings.csv / index_holdings.csv sin vigilancia.** Deuda reconocida en 58936778. Sin manifest ni check. Si un workflow trimestral falla parcialmente, el CSV queda con datos mixtos sin senal. Fix: `check_holdings_csvs` (ETFs esperados, min tickers/ETF, edad mtime <= 120d). Commit 0017bf4f + 6 tests.

**Workflows cron trimestrales (frente cerrado, sin fix nuevo).** Los 3 schedule del 1-oct (update_sector_holdings, update_index_holdings, update_european_holdings) fallaron por dos causas raiz ya corregidas el mismo dia: `sys.path` sin insertar (1bead8ad) y arrays desalineados en get_state_street_holdings por tickers invalidos SSGA (27248d35). Los workflow_dispatch posteriores pasan. Proximo schedule: 1-ene-2027.

**Verificacion acumulada.** 3196 -> 3203 tests. Gate 10/10. Pyflakes limpio. Corpus actualizado. Commits: a356ae1d, 203b61c7, c09a268c, d495dc1e, 8e8826f5, 0017bf4f, 6027a943.
**Fin del historico.**