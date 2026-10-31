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

### Retirada del modulo Wyckoff legacy (5e adelantado, 2026-10-10)

El plan de migracion (§2.2, §7) marcaba la retirada del legacy (5e) como BLOQUEADA hasta cierre de 5d + 1 run CI estable. Se adelanta en 2026-10-10 con la siguiente justificacion:

- Los 5 consumidores productivos fueron migrados a v1.8 el 2026-10-08 (commits 2fd6c449, 35663000).
- Los 4 CSVs historicos contaminados con legacy/v1.8 fueron regenerados entre 2026-10-08 y 2026-10-09.
- El CI corre con v1.8 desde 2026-10-08 con run-system=success el 09-oct (headSha cd011629) y el 10-oct (headSha cd011629).
- Los consumidores residuales (2 tests que documentaban el bug del legacy + 1 script de comparacion) no son de produccion.

El plan original marcaba 5d como bloqueado por 5b.X (~2027). Esa cadena no aplica: 5d es la migracion de consumidores, ya ejecutada parcialmente sin esperar a 5b.X. Mantener el legacy 15 meses por ceremonia no tiene justificacion tecnica.

**Acciones ejecutadas.**

- Retirado `indicators/wyckoff.py` (legacy v4.2).
- Retirados `tests/test_build_ticker_df_i2.py` y `tests/test_index_phase_range.py`.
- Retirado `scripts/compare_wyckoff_legacy_v18.py`.
- Portado el test de regresion I2 / K-INDEX-RANGE-01 a `tests/test_wyckoff_v1_contract.py` (test_regresion_i2_nan_internos_no_degenera).
- Actualizados 00_ARRANQUE y 02_ARQUITECTURA.

**Verificacion.** Suite: 3204+5+0 (era 3212+5 antes: -8 tests netos al eliminar los 2 ficheros legacy). Barrido de imports residuales: 0. CI de 09-oct y 10-oct corrieron con v1.8 sin incidentes.

**Referencia.** `docs/auditoria/wyckoff/03_plan_migracion.md` (5e pasa a CERRADO).

### Auditoria del reporte 2026-10-09: 15 hallazgos (2 corregidos, 13 falsos positivos)

Revision bloque a bloque del reporte del 2026-10-09 buscando errores,
incoherencias, ceros/NaN/N-D ocultos. Resultado: 15 hallazgos.

**Corregidos (2).**

- H-07/H-08 (commit 35559c1d): notas de Representatividad del lider y
  Divergencia sector-lideres declaraban criterios falsos (cobertura /
  validos >=3). El filtro real es por fase Wyckoff (solo ACCUMULATION o
  MARKUP).
- H-09 (commit 4a604b21): `heterogeneity_type` retirado del writer, del
  render y del test. Valor constante en el historico observado.
  Redundante con `dispersion_reading`.

**Falsos positivos verificados con codigo (catalogo de referencia).**

- H-02: N/D en VIX3M/VIX es por falta de VIX, no de VIX3M. Fuente
  distinta.
- H-04: coberturas 18-25% son top-15/total-ETF. La metrica usa top-15
  por diseno (`TOP_N_CANDIDATES`). Naming confuso, no bug.
- H-05: `signal_cons = (pers + agree) / 2`. 38% es matematica, no
  inconsistencia.
- H-11: Flujo Proxy en dos secciones mide cosas distintas (z-score vs
  flow_median). Naming confuso.
- H-12, H-13: cosmeticos documentados, no bug.
- Resto hasta 13: verificados contra codigo, sin accion tecnica.

**Correccion de cifra auditoria 2026-09-26.** 205 -> 142 (6 ALTA / 71
MEDIA / 65 BAJA). Los informes existen en bc656374, borrados de main en
cdf47ad2.

**Relevancia futura.** Antes de abrir item sobre estos H-*, consultar
aqui: la mayoria ya fueron descartados con codigo.

### Regeneracion de CSVs con legacy Wyckoff (2026-10-09)

El clasificador legacy `indicators/wyckoff.py` usaba `robust_zscore(trend, w=60)` para el score continuo. v1.8 usa `tanh(trend/K)`. Los CSVs historicos acumularon filas pre-08-oct calculadas con legacy y post-08-oct con v1.8. Bajo el mismo nombre de columna, dos metricas distintas.

Ademas, legacy tenia la rama silenciosa `if score_clean.empty: return RANGE` que etiquetaba una ausencia de datos como fase valida. v1.8 devuelve `INSUFFICIENT_DATA` (no computa como fase).

**Barrido completo de outputs/history.** 4 CSVs regenerados, resto diagnosticados como limpios o no-bug.

| CSV | Filas pre-08 | Accion | Commit |
|---|---:|---|---|
| sector_wyckoff_distribution.csv | 220 | Regenerado | f782b9d4 |
| sector_concentration.csv | 187 | Regenerado | 80cb03f8 |
| leader_representativeness.csv | 110 | Regenerado | 440750c1 |
| sector_leader_divergence.csv | 22 | Regenerado | 440750c1 |
| sector_rank_history.csv | 473 | Limpio (no guarda fase) | - |
| rs_internal.csv | 2917 | Limpio (RS, no wyckoff) | - |
| sector_breadth/momentum/persistence | 1155/220/220 | Limpio (breadth puro) | - |
| sector_flow_characteristics.csv | 253 | No-bug (lag SSGA 1-2d) | - |
| evidence_matrix, sector_regime_matrix | 0 | Solo ultimo dia | - |

**Decisiones tomadas.**

- `sector_concentration.csv`: regeneracion (mismo universo, nueva formula).
- `leader_representativeness.csv` y `sector_leader_divergence.csv`: **eliminacion de falsos positivos**. Bajo v1.8, solo 3 sectores (XLK, XLE, XLV) cumplen ACCUMULATION/MARKUP en el periodo 17-sep a 07-oct. Legacy clasificaba 10. Los 7 restantes (XLB, XLC, XLF, XLP, XLRE, XLU, XLY) nunca debieron tener seccion de lideres. Verificacion cruzada: la fila del 08-oct escrita por el pipeline con v1.8 solo tiene XLK.

**Trade-off aceptado** (mismo que el CSV de Wyckoff del 08-oct): las filas regeneradas son "v1.8 aplicado retroactivamente sobre el parquet actual", no lo publicado en su momento. Yahoo revisa OHLC (K-LSE-YAHOO-REVISION-01).

**Snapshots pre** en `outputs/audit/*_pre_20261009.csv`.

### Deuda detectada en auditoria del reporte 2026-10-09: sector_flow_vs_price_div huerfano

`compute_flow_divergence_v2` (`indicators/slpm_v12.py:97`) declara 3 componentes:

    composite = 0.50 * leader_flow_div + 0.25 * sector_flow_vs_price_div + 0.25 * structural_flow_div

El segundo componente no tiene fuente en el pipeline. El caller `evaluate_slpm_v12` (linea 124) lo pasa como `None` hardcoded. El test `test_flow_div_v2_con_sector_price_flow` lo verifica aislado, pero ninguna ruta de produccion lo alimenta.

Consecuencia: composite siempre NaN. El render muestra `N/D`. Correcto: no es 0 falso.

Opciones evaluadas (no aplicadas):
- A) Rebalancear pesos a 0.66/0.34 -> requiere justificacion estadistica (no existe).
- B) Implementar `sector_price_flow` -> scope creep, fuente no definida.
- C) Documentar el N/D como esperado (aplicado hoy 2026-10-09, mas nota en el render).

Sin accion tecnica pendiente. Deuda declarada.

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

### Contrato de persistencia de outputs (2026-10-11)

Auditoria del gap 09-oct (sector_breadth no escribio fila, receipt
se emitio igual). Diagnostico: el sistema no tenia contrato
declarativo de escritura ni verificacion post-run. Cualquier writer
podia hacer `return` silencioso sin senal.

Solucion en 3 piezas + contrato AST:

- `src/pipeline/output_registry.py`: registro declarativo de 19
  outputs (18 temporales + 1 snapshot). `WriteReceipt` +
  `ReceiptCollector`.
- `src/pipeline/output_writer.py`: wrapper unico
  `persist(name, df, ...)` / `skip(name, reason)`. Toda escritura a
  `outputs/history/` pasa por aqui. Fail-closed: name no registrado
  -> KeyError; razon de skip no legitima -> ValueError.
- `src/pipeline/output_verifier.py`: `verify(collector)` con 3
  reglas (receipt existe, escrito vs disco consistente, skip
  legitimo con CSV no vacio).
- Integracion en `run.py`: collector activo al inicio, verificacion
  antes de cerrar; `sys.exit(1)` si falla. Persiste
  `outputs/state/writer_receipts.json` (gitignored).
- `tests/test_output_contract_ast.py`: AST que prohibe `to_csv()`
  fuera del wrapper. Excepciones documentadas: `output_writer.py`,
  `save_sector_rankings`, `compute_indices_intl` (outputs/report/,
  efimeros sin dimension temporal).

19 writers migrados en 6 commits por fichero. Gate 10/10 intacto.
Receipt v1 intacto. `pipeline_gate.py` intacto.

Commits: f0311cdb, 8596690c, c43a6b00, 2593ac91, ceecad70,
b1237083, ae752882, ab838dbf.

**Fin del historico.**

---

## Resumen bitacora podada 15-oct-2026 (sesiones 02-08 oct)

Indice de las entradas de `05_BITACORA.md` retiradas en la poda
del 15-oct-2026 (regla §5: mantener las ultimas 10 sesiones +
las del dia). Las decisiones clave de cada sesion ya estan en
los bloques tematicos siguientes de este fichero:

- **02-oct (madrugada):** Catalog PIT 242->255 + schema v5.
  Ver L1190 (resuelto en sesion Catalog PIT del 10-02).
- **02-oct (tarde/noche):** Wyckoff 5b.3 -> 5b.4 -> 5b.4-bis,
  contrato v1.9, QA. Ver L841 (walk-forward SOW), L868
  (grid 240), expedientes 21-33 en `docs/auditoria/wyckoff/`.
- **03-oct (4 sesiones):** Cierre Wyckoff umbral 550, S-01,
  Frente B macro_regime, D-06 validador 5b.X. Ver L783
  (Frente B), L816 (auditoria v1.8 linea a linea), 00_ARRANQUE
  seccion 4.2, `docs/auditoria/wyckoff/40_*`.
- **04-oct:** Walk-forward SOW plan v3 + expediente 41.
  Ver L841, L868.
- **05-oct:** E2E main post-reorder D3 + fix frescura Evidence
  Matrix. Ver L903.
- **06-oct:** Limpieza estructural (~810 MB) + fix rs_mom NaN
  (commit 34c8029). Ver L939, L963, L985.
- **07-oct:** Auditoria reporte CI, M-03, A-03, A-01/F6-1b.
  Ver L1040 (F6-1b).
- **08-oct:** Auditoria reporte CI + migracion Wyckoff v1.8 core.
  Ver L1209, L1254, L1291, L1337, L1349, L1362.

Verificacion previa a la poda (15-oct): cada token unico de
las 11 entradas se cruzo con este fichero. Los tokens
`B-06`, `K-HUERFANO`, `H20` y `d608711` no aparecen literalmente
aqui, pero sus decisiones estan cubiertas por los bloques
anteriores (`B-06` -> `tests/test_b06_e2e_aggregation.py`;
`K-HUERFANO` -> tag interno en `src/data_loader.py`;
`H20` -> "H20-complete", 00_ARRANQUE §4.2; `d608711` -> SHA).


---

## Sesion 2026-10-13 - Contrato en prod + revision del reporte diario

Primera verificacion del contrato de persistencia activo en prod
(run 38342330518, 17m48s, 19 checks OK, target session 2026-10-12,
receipt emitido).

Revision linea a linea del reporte (10 hallazgos):
- **B1 (FIX):** `data/providers/fred.py` extendia el indice con
  ffill hasta hoy, contaminando la fecha de `liquidity_state.json`
  (FRED reportaba '0 dias / CURRENT' cuando el dato real era 3
  dias). Fix: `end=s.index[-1]` en vez de `end=today`.
- **Fix colateral:** `OutputSpec.date_source='upstream'` para
  outputs con lag de proveedor (`sector_flow_characteristics`).
  Evita gap falso cuando el CSV tiene la ultima publicacion real
  aunque no coincida con la sesion.
- B2, B3, B4, B6, B8, B10: notas (metricas distintas, constantes
  de render, clip z-score, lag de proveedores, universo ETF vs
  rs_internal, SSGA devuelve no sectoriales). B5, B7, B9:
  descartados (falsos positivos verificados).

Tests: 3251 -> 3253 passed. 5 skipped. Pyflakes limpio.

Commits: 3542d29d, 89331321, 1ea34ac7, 43f10428, 8b441a7a.

---

## Sesion 2026-10-14 - N8 completo: liquidez v1 -> v2

Cierre del frente N8 (rediseno del score de liquidez) tras varias
rondas de auditoria externa. Sustitucion de v1 por v2 en produccion.

**Forense del CRISIS 14-oct.** State stale (retrocedio de 2026-10-13
a 2026-10-07). Causa: run con datos FRED de 6 dias antes. No era
artefacto del fix F5.6-06b como decia el HANDOFF anterior.

**Cambios.**

- Guard de monotonicidad en `liquidity_state.json` (4291593f).
- Trazabilidad de fechas + guard de frescura por serie (3d1a1fb6).
- Instrumentacion FRED en JSONL (456f3a28).
- Especificacion B3 congelada en `regimes/liquidity_v2.py` (c51372fb):
  senal A (SOFR-EFFR pb, umbrales 0/5/20), persistencia 3/3,
  alerta >=100pb, diagnostico C1, contexto calendario.
- C1-bis con episodios independientes: repo 2019 detectado, Q4-2025
  detectado (46/62), COVID y carry trade yen no detectados (correcto),
  0 falsas alarmas en 1024 dias.
- Sustitucion v1 -> v2 en pipeline (84f46dd3). Adaptador 4-tuple,
  consumidores sin cambios, rollback disponible.
- Verificacion en run.py local: exit=0, Gate 10/10, NEUTRAL
  (conf 90%).
- Cierre N1 (SLPM fail-closed) en 6 commits previos.
- Alineacion de comandos pytest documentados con el filtro CI.

**Commits (17).** 27c12671, 953b7fee, f49c85f0, 05b70686, 31f9c00a,
0fe4e3e2, 4291593f, 3d1a1fb6, 456f3a28, c51372fb, 786b3fc2,
84f46dd3, e19ba091, 5babc8e3, e8c26d4c.

**Pendiente al cierre de la sesion.**
- Verificacion del cron 23:17 UTC con v2 en CI.
- Dictamen auditor sobre sustitucion (fase sombra saltada).
- Corpus documental (~15 items de deriva).
- 8 F6-* sin descripcion.

Nota: los pendientes de esta sesion se resolvieron en sesiones
posteriores. `liquidity_v2` sigue en produccion (verificado en
runs de 2026-10-22 en adelante).

---

## Sesion 2026-10-22 - Auditoria externa #4 + 11 bugs + gobernanza

Sesion extensa. Cierra auditoria externa #4 (P1-P8) y deriva en
auditoria cruzada del sistema que detecta 11 bugs reales. Todos con
test + verificacion en produccion local (`py run.py`).

### Auditoria externa #4

P1, P2, P5: FALSE_POSITIVE (RS nominal invariante por rs_mom;
cobertura XLK sobre universo completo; NIPC ya documentado H14).
P3: BUG REAL (umbral Deterioro aplicado al bruto vs redondeado;
FIXED, 9 tests). P4: IHR renombrado como proxy (no institucional).
P6: nota de alcance en temporal_coherence. P7: FIXED en `8c40f7e6`.
P8: nota de Voyager residual.

### FLS WALCL/CP (WONT_FIX)

Diagnostico estadistico: WALCL `-z(pct_change_252)` |z|>5 12.3%,
alternativas peores. CP `z(nivel)` |z|>5 4.0%, `pct_change_252`
mejora |z|>5 pero empeora `z_ult`. Saturacion estructural (QT en
extremo historico), senal correcta. FLS baja de 79 a 63.

### Los 11 bugs reales

1. **`router.get_fed_data` priorizaba API sobre CSV** (`d405d539`).
   FRED API lag 5d vs CSVs bot diario. `Liquidez Real: N/A` en
   reporte, contradiciendo FLS. Fix CSV-first. Verificado: `NEUTRAL 90%`.
2. **v1 contamina state v2** (`83f79a93`). Tests de v1 sobreescribian
   `liquidity_state.json` con esquema incompatible. Fix: state v1
   separado (gitignored).
3. **Docstring v2 stale** (`83f79a93`). "NO PRODUCCION" siendo
   productivo desde 14-10.
4. **Ref rota** (`83f79a93`). `finalize.py:110` citaba item inexistente.
5. **Verifier acepta SHA huerfanos** (`236c2a64`). `git cat-file -e`
   acepta objetos huerfanos. `for-each-ref --contains` para
   reachability.
6. **v2 liquidez solo API** (`5375f58b`). `effr.csv` no existia. Fix
   CSV-first con interseccion + generacion de `effr.csv` (2233 filas).
7. **`-0.0` en `_score_continuo`** (`5375f58b`). Visible en state.
   Fix: `+ 0.0`.
8. **`compute_pcr_signals` side effects** (`9a2bddcc`). Escribia
   `pcr_history.csv` bypassando `persist`/registry. Diagnosticos
   modificaban el working tree. Fix: flag `persist_history=False`.
9. **`sector_correlation` P7** (`8c40f7e6`). `iloc[-window:]` sobre
   serie con huecos. Fix: `get_trading_days` (nueva en
   `src/market_calendar.py`) + reindex.
10. **`cboe_vix3m.parquet` no-bursatiles** (`7fcbe05f`). 2 filas en
    `2010-12-31`, `2021-12-31`. Fix: `_post_filter_non_trading_days`
    en `update_cboe.py`. Parquet 4299 -> 4297. Verificado en
    `origin/main`.
11. **Wyckoff B5/B3** (`1cc8387a`). Spring/SOS con `min_periods=1`
    disparaban SOS espurios. Fix: `min_periods=EVENT_WINDOW` + patron
    H7. B3: `max_age_m` default 30 (contrato v1.9). WONT_FIX
    razonados: B2 (P7 en medias moviles requiere dropna, no reindex);
    B4 (fillna Close/0 es contrato v1.8, protegido por
    `test_regresion_i2_nan_internos_no_degenera`).

### Gobernanza

- FLS-COMPONENT-DESIGN: FIXED.
- LIQ-V2-DATA-SOURCE: SUPERSEDED (duplicado del FIXED).
- GOVERNANCE-SHA-REACHABILITY: FIXED.
- CBOE-VIX3M-NON-TRADING-DAYS: FIXED.
- WYCKOFF-P7-B4-B5-B3: FIXED.
- Plan ROLLING-WINDOW corregido (error factual: `sector_correlation_summary`
  si existe como writer y CSV).

### Decisiones tomadas (no revertir sin justificacion)

- **DISTRIBUTION apagada.** `WYCKOFF_SOW_* = None`. Columna `Dist` del
  reporte siempre 0%. Decision: sesion dedicada de re-diseno SOW (no
  hay resultados optimos con la especificacion actual).
- **P7 no es universal.** Reindex al calendario correcto para
  correlaciones entre series; incorrecto para medias moviles
  (`dropna().rolling()` ya tiene semantica correcta). WONT_FIX
  documentado en cabecera de `wyckoff_v1.py`.
- **B4 WONT_FIX.** `fillna(Close)`/`fillna(0)` en `build_ticker_df` es
  contrato v1.8. 0/2512 filas con H=L=C en `market_data.parquet`.
  Riesgo residual mitigado con `+1e-9` en `break_depth`.

### Pendiente

- Verificar scheduled 2026-10-22 con `verify_run_20261022.ps1`.
- DISTRIBUTION: sesion dedicada.
- Wyckoff 5b.X / SOW v6: bloqueados externos.
- HANDOFF automatico: deuda estructural.

Suite final: 3845 passed, 5 skipped.


---

## Sesion 2026-10-23 - Verificacion scheduled + auditoria autonomia

**Objetivo.** Cerrar Q4 (verificar scheduled target 2026-10-22 en CI
real) y auditar autonomia completa. Documentar.

### Scheduled target 2026-10-22 verificado

Run `39884468711` (23-10 01:31 UTC, slot 22-10 23:17 UTC, delay
2h14m, `run-system: success`). 4/4 fixes criticos confirmados en CI
Python 3.14.7:

1. Router liquidez (`d405d539`): `Liquidez Real: NEUTRAL (Signal
   Consistency: 90%)`, no `N/A`.
2. P7 sector_correlation (`8c40f7e6`): `receipt_audit:
   sector_correlation_summary, OK, 2026-10-22`.
3. `pcr_history` en registry (`9a2bddcc`): `receipt_audit:
   pcr_history, OK, 2026-10-22`.
4. `cboe_vix3m` no-bursatiles (`7fcbe05f`): 4298 filas, 0
   no-bursatiles, rango 2009-09-18 a 2026-10-22.

Contrato temporal integro. 20 writers, 18 OK + 2 no-OK esperados
(`sector_rank_deltas` NO_DATE_CHECK, `sector_flow_characteristics`
LAGGED por lag SSGA).

### Auditoria autonomia (FALSE_POSITIVE)

Verificado con datos reales, no asumido:

- **FRED caido**: `download_series` devuelve `None`, CSV no se
  reescribe, ultimo dato valido preservado.
- **pipeline_gate** usa probe live de Yahoo, no parquet local.
  Parquet stale + Yahoo fresco = READY correcto.
- **5/5 slots daily_run success 22-10**, 2/2 update_macro_manual
  success.
- **Cadencia por fuente** respetada (diaria≤3d, semanal≤10d,
  mensual≤60d).
- **Health check** abre issue #7 automaticamente en WARN.

Los "FAIL/WARN" que aparecieron en diagnosticos intermedios eran
artefactos de medicion: parseo de `month` int en
`qqq_nport_flow.csv` (epoch 1970), threshold mal calibrado en
`discount_rate` (DPRIME solo cambia con decisiones Fed), ICSA
publica jueves 13:30 UTC vs bot corre 06:17 UTC (desfase de
publicacion).

### 5 huecos auditoria dia anterior - cierre

- Health Check 19-oct: FALSE_POSITIVE (regla antigua FAIL con 2/5
  slots; regla actual post 20-oct WARN).
- ETF/European Holdings 1-oct: FALSE_POSITIVE (fallaron 6h y 4h
  antes del commit `1bead8ad` que añadio `sys.path.insert`).
- Index Holdings 1-oct: FIXED `e54f95b6` (refactor paralelas ->
  `rows.append` con try/except por fila, elimina clase de bug `All
  arrays must be of the same length`).
- cron_slots WARN 3/5: FALSE_POSITIVE (delays GitHub 2-4h).
- manifest close_nan_last: FALSE_POSITIVE (WBD delisted + 19 BME).
- iae_section WARN: FIXED `e54f95b6` (heading flexible `13F`).

### Decisiones

- **Sistema autonomo verificado.** No requiere intervencion manual
  para mantener datos frescos. Watchdog, gate, workflows, alerta:
  todo funciona.
- **Contrato temporal respetado en scheduled.** `data_quality.date`
  y `receipt_audit.target` alineados con `last_expected_market_date`.

### Pendiente

- 4 items OPEN observacionales (fechas de revision 10-nov, 17-nov,
  20-nov).
- 3 BLOCKED_EXTERNAL (C2-920, SOW-V6, WYCKOFF-5BX).
- 0 bugs abiertos.

Suite final: 3849 passed, 5 skipped.


## Sesion 2026-10-24 (sesion 2) - Cierre SOW universal + reconstruccion sistema Wyckoff

**Objetivo.** Cerrar el frente SOW universal (Bloque A) y reconstruir
el sistema Wyckoff por stock con acumulacion y distribucion cableadas
al pipeline (Bloque B).

### Bloque A - Cierre SOW universal

7 experimentos sobre cohortes de 316, 268 y 3423 tickers, con
detectores v1 y v2, estimadores pooled y within-ticker, placebos de
permutacion y distribucion (100 shifts), y bootstrap entre anos.

**Resultado:** senal real vs placebo en R3000 (3423 tickers, lift
+0.123 vs -0.077, 21.5 sd) pero NO universal (4/9 folds positivos,
IC95 entre anos [-0.058, +0.303] cruza 0). Patron bear identificado
(3/3 anos con >10% dias bear funcionan, 0/5 con <10%), pero 2020
domina la muestra (473/671 SOW en bear). Ningun gate probado cumple
criterio pre-registrado.

**Decision:** `SOW-V6` -> WONT_FIX. `SOW-BEAR-CONFIRMATION` abierto
(IN_PROGRESS), pendiente contrato v1.10 firmado por auditor externo.

Evidencia: `docs/auditoria/wyckoff/51-56`.

### Bloque B - Reconstruccion sistema Wyckoff

**Decision arquitectonica:** dejar de usar ETFs sectoriales e indices
como fuente de fases Wyckoff. Construir cascadas por stock con
deteccion de acumulacion Y distribucion, orquestadas.

Motivacion: el sistema previo detectaba 0/327 DISTRIBUTION en stocks
reales (`_is_distribution_candidate` no disparaba nunca), y la senal
en ETFs sectoriales era demasiado debil para detectar distribucion
(0/483 bars en XLK, ^GSPC, AAPL con los umbrales del BC).

**Arquitectura implementada:**

Modulos en `indicators/`:
- `wyckoff_event.py`: dataclasses (WyckoffEvent, Range, ClassifiedBreak)
- `wyckoff_distribution.py`: cascada BC -> AR -> Range -> SOW
- `wyckoff_accumulation.py`: cascada SC -> AR -> Range -> SOS
- `wyckoff_orchestrator.py`: arbitraje con regla simetrica de vigencia

15 detectores en codigo. 4 utilies integrados como evidencia pasiva
(mSOW, mSOS, LPSY, LPS). 2 no disparan con umbrales actuales
(PSY, PS), documentados como deuda de calibracion.

**Regla simetrica de vigencia:** un rango esta vigente mientras el
precio no lo ha roto por ningun lado. Si rompio por arriba (dist
fallida -> RANGE; acc consumada -> MARKUP). Si rompio por abajo (dist
consumada -> MARKDOWN; acc fallida -> RANGE). Definicion de rango, no
umbral.

**Estado produccion (327 stocks, ultimo bar 2026-10-22):**
- 219 RANGE, 53 DIST_FORMING, 17 DIST, 17 MARKDOWN, 13 MARKUP,
  7 ACC_FORMING, 1 CONFLICT.
- Agregado holdings (11 ETFs): 157 RANGE, 35 DIST_FORMING, 12 DIST,
  12 MARKDOWN, 8 MARKUP, 6 ACC_FORMING.
- XLK: 41% holdings en DistF. XLC: 40% DistF.

**Cableado:**
- `sector_wyckoff_distribution.py` migrado a `classify_phase`.
- `sector_context.py` con columnas AccF/Acc/Markup/Range/DistF/Dist/
  Markdown/Conflict.
- `sector_annotations.py` anota titulos sectoriales con composicion
  interna (`## Sector: XLK (MARKUP)` + `*Composicion interna (22
  componentes): 3 Markup (14%), 8 Range (36%), 9 DistF (41%)...*`).

### Pre-registros fallidos (no ocultos)

- `60_current_phase_rule_v1.md`: regla de vigencia unidireccional.
  Fallo C1 (18.6% < 20%) en holdout S&P 500. No se ajusto.
  Reemplazada por regla simetrica (definicion, no umbral).
- `61_utad_spring_pending_v1.md`: UTAD_PENDING/SPRING_PENDING. Fallo
  A3 (26 vs 3, ratio 8.67x > 5x por asimetria natural del mercado).
  No se ajusto.

### Lo que NO cambia

- `detect_sow` (v1.8) intacto.
- `detect_sow_v2` en codigo, `WYCKOFF_SOW_* = None`.
- `classify_wyckoff_phase` sigue existiendo y se usa en
  `sector_regime.py` para el titulo del sector. Anotacion de
  composicion mitiga la incoherencia. Migrarlo es decision de
  producto pendiente.

### Pendiente

- Migrar `sector_regime.py` al nuevo orquestador.
- Renderizar evidencia pasiva (mSOW/mSOS/LPSY/LPS/ST) en el reporte.
- Calibrar PSY/PS con pre-registro.
- Auditoria externa del sistema completo.
- Contrato v1.10 (SOW-BEAR-CONFIRMATION).

Documentacion completa: `docs/auditoria/wyckoff/62_sistema_completo_2026-10-24.md`.

Suite final: 3946 passed, 5 skipped. Working tree limpio.

---

## Sesion 2026-10-24 (sesion 3) - Plan holdings: demolicion + construccion

**Objetivo.** Ejecutar el plan 63: retirar el sistema de lideres por
sector basado en fase ETF (SLPM, stock_leader, index_leaders,
classify_wyckoff_phase) y construir el sistema Wyckoff holdings-based
con 6 CSVs descargables.

**Decision arquitectonica.** El ETF sectorial suaviza la senal; la
deteccion de fase se hace directamente sobre los tickers que
componen cada sector/indice. Se retira toda la maquinaria que
depende de la fase del ETF.

### Plan 63

`docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md`.
7 commits planificados. Ejecutado como 5 bloques + fix + cierre
documental:

- C1: modulo holdings.
- C2.1 + C2.2: cableado pipeline + reporte.
- C3.1 + C3.2 + C3.3: retirada SLPM + adaptacion Validation Gate.
- C4.1 + C4.2 + C4.3: extraccion stock_metrics + stocks_data +
  retirada lideres.
- C5.1 + C5.2: retirada fase ETF + classify_wyckoff_phase.

### C1 - Modulo holdings (`83187f8c`)

`indicators/wyckoff_holdings_report.py::compute_wyckoff_holdings_report`.
6 tablas: usa_sector_acc/dist, usa_index_acc/dist, eu_index_acc/dist.

- Iteracion sobre holdings top-20 por weight de cada grupo.
- Deteccion via `classify_phase` del orquestador.
- Criterio (iii): ACC+ACC_FORMING+MARKUP para acumulacion,
  DIST+DIST_FORMING+MARKDOWN para distribucion.
- Cobertura: >=200 barras para incluir, >=460 para score valido.
  Entre 200-460: `basis = "warm_up_incompleto (n=N)"`.
- `coverage_weight_pct` por fila.
- IWM excluido (top-20 = 6.2% de 1977 holdings, no representativo).

### C2 - Cableado pipeline + render (`a3a63766`, `9b5b34cd`)

- 6 OutputSpec nuevos en `output_registry.py` (24 -> 26).
- Wrapper `src/pipeline/wyckoff_holdings.py::compute_wyckoff_holdings`.
- Cableado en `run.py` tras `compute_sector_metrics`.
- Renderers en `src/report/wyckoff_holdings.py`.
- 2 secciones nuevas en el reporte ("Tickers en acumulacion" y
  "Tickers en distribucion"), cada una con 3 sub-tablas.

### C3 - Retirada SLPM (`0c5b7fb3`, `676ab956`, `cf7659fb`)

SLPM dependia de `leader_metrics_for_slpm` (top-5 lideres del sector
#1 por fase ETF). Con el sistema de lideres retirado, SLPM queda sin
input.

- Borrados: `slpm_v12.py`, `state_machine.py`, `state_transition.py`,
  `src/pipeline/slpm.py`, `src/report/slpm.py`.
- `SLPM_WEIGHTS`, `SLPM_EXPECTED_LEADERS`, `_forzar_lideres_slpm`
  eliminados.
- `compute_engines` sin `leader_df`. Tactical/structural/persistence
  autonomos.
- Validation Gate adaptado: firma `slpm_v12_data` ->
  `wyckoff_holdings_data`. 3 checks sustituidos:
  - SLPM v1.2 -> Wyckoff holdings CSVs.
  - Opportunity Map -> Cobertura holdings.
  - Anti-Double-Counting -> Sanity holdings.
  Gate sigue 10/10 para no romper el contrato del receipt.
- `scripts/generate_docs.py` limpio.

### C4 - Retirada lideres (`753b7f22`, `2ede0215`, `4bc45fd3`)

- C4.1: `indicators/stock_metrics.py` extraido de `stock_leader.py`.
  `compute_stock_metrics`, `compute_wls`, `_validate_wls_inputs`,
  `WLS_COMPONENTS`, `compute_all_stock_metrics`.
- C4.2: `src/pipeline/stocks_data.py` extraido de `leaders.py`.
  `load_stocks_data` (df_stocks + effective_meta + holdings_df +
  HOLIDAY_MODE).
- C4.3: borrados `stock_leader.py`, `leaders.py`,
  `leader_representativeness.py`, `sector_leader_divergence.py`,
  `sector_annotations.py`, `render_acciones_seleccionadas`.
  `sector_metrics.py` devuelve 3 sub-outputs. Registry 26 -> 24.
  `run.py` sustituye `compute_leaders` por `load_stocks_data` +
  `compute_all_stock_metrics`.

### C5 - Retirada fase ETF + classify_wyckoff_phase (`8d9b90a0`, `c47cb420`)

- C5.1: `sector_regime.py` sin `classify_wyckoff_phase` ni
  `wyckoff_score`/`build_ticker_df` sobre el ETF. Tupla del ranking
  de 4 a 3 elementos. `sector_regime_matrix.py` deriva
  `structure_positive` de `sector_wyckoff_distribution_df`
  (`pct_accumulation + pct_markup >= 50`). `wyckoff_phase` fuera de
  `sector_rankings.csv`.
- C5.2: `classify_wyckoff_phase`, `classify_wyckoff_phase_meta`,
  `_is_distribution_candidate`, `_compute_precedent`,
  `_has_sufficient_data` borradas de `wyckoff_v1.py`. Constantes
  huerfanas: `FASE_*` (6), `STRUCT_STRONG/WEAK/BASE/DETERIORO`,
  `T_NORM_WEAK`, `C_NORM_COMPRESSION`, `PREC_STRUCT_WEAK`.
  Conservadas: `PRECEDENT_WINDOW`, `PREC_STRUCT_STRONG`,
  `STRUCT_DETERIORO`, `T_NORM_STRONG` (aun usadas por
  `detect_sow_v2`). `wyckoff_v1.py` queda como modulo de features
  y detectores primitivos.

### Fix pyflakes (`96276d05`)

C5.1 paso tests local pero fallo en CI. `test_struct_score_sector_indices.py`
conservaba un import huerfano (`render_ranking_sectorial_unificado`) al
borrar 4 tests. En local solo se corrio pyflakes por ficheros tocados,
no por todo el repo. CI corre `pyflakes .`. Fix + leccion en HANDOFF
y 00_ARRANQUE: pyflakes sobre `.` siempre.

### Cobertura del sistema holdings

- **USA sectores** (11): top-20 por weight. Peso top-20: XLB 90.1%,
  XLC 97.7%, XLE 99.0%, XLF 72.8%, XLI 57.2%, XLK 79.6%, XLP 86.7%,
  XLRE 86.7%, XLU 83.1%, XLV 78.9%, XLY 84.5%.
- **USA indices**: SPY (49.9%), QQQ (66.8%), DIA (85.0%). IWM excluido
  (top-20 = 6.2% de 1977 holdings).
- **EU indices**: FEZ 66.3%, DAXEX 78.4%, ISF.L 60.8%, LYXI 93.2%.

### Pre-registros fallidos (heredados)

Sesion 3 no anade nuevos. Los 60 y 61 siguen vigentes como evidencia
historica:
- `60_current_phase_rule_v1.md`: regla de vigencia unidireccional.
  Fallo C1 (18.6% < 20%) en holdout S&P 500. Reemplazada por regla
  simetrica (definicion, no umbral).
- `61_utad_spring_pending_v1.md`: UTAD_PENDING/SPRING_PENDING. Fallo
  A3 (26 vs 3, ratio 8.67x > 5x). No se ajusto.

### Lo que NO cambia (post sesion 3)

- `detect_sow` (v1.8) intacto.
- `detect_sow_v2` en codigo, `WYCKOFF_SOW_* = None`. Base del futuro
  contrato v1.10 (SOW-BEAR-CONFIRMATION).
- `wyckoff_score`, `_atr_normalized`, `_trend_component`, `_volume_z`,
  `_effort_vs_result`, `wyckoff_stability`, `build_ticker_df`,
  `detect_spring`, `detect_sos` intactos.
- Orquestador `classify_phase` + regla simetrica + FORMING + CONFLICT.

### Casos conocidos

- **TWST = CONFLICT**: "dist formandose y acc formandose
  simultaneamente". Ambiguedad real, no bug.
- **ACCUMULATION=0** en produccion: estado transitorio por diseno
  (requiere SOS confirmado Y precio dentro del rango). No es fallo.

### Pendiente

- `current_phase_basis` en el reporte (hoy solo en CSV).
- Validar nueva derivacion de `structure_positive` en
  `sector_regime_matrix.py` (holdings-based desde C5.1).
- TWST en CONFLICT: documentar como caso conocido en el reporte.
- `ACCUMULATION=0`: nota en el reporte.
- IWM fuera: sub-fase de descarga con protocolo de 6 pasos.
- Cobertura USA incompleta (231/504 holdings): ampliacion futura.
- Renderizar evidencia pasiva (mSOW/mSOS/LPSY/LPS/ST).
- Calibrar PSY/PS con pre-registro.
- Auditoria externa del sistema holdings-based.
- Contrato v1.10 (SOW-BEAR-CONFIRMATION).
- Fix `poda_bitacora.py` (item PODA-BITACORA-DIA-EN-CURSO).

### Lecciones operativas

- **`pyflakes .` sobre todo el repo**, no por fichero. CI lo hace y
  detecta huerfanos que en local escapan.
- **Antes de cambiar firmas de tuplas/dicts, enumerar TODOS los mocks
  via AST**, no regex de texto. C5.1 dejo 6 tuplas sin adaptar en
  `test_report_generator_smoke.py` que solo cazamos con el barrido
  AST posterior.

### Cierre

Commits clave sesion 3: `977ad888`, `bdc80626`, `f1edc145`, `16f2dd8a`,
`83187f8c`, `a3a63766`, `9b5b34cd`, `0c5b7fb3`, `676ab956`,
`cf7659fb`, `753b7f22`, `2ede0215`, `4bc45fd3`, `8d9b90a0`,
`c47cb420`, `96276d05`, `a100d282`, `813ad19f`, `0a0597b3`,
`a3471303`, `487e2fbb`, `ebae6fe9`, `30adb4c6`, `00d5e0db`.

Documentacion completa: `docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md`
+ `docs/handoffs/HANDOFF.md`.

Suite final: 3776 passed, 2 skipped (filtro estandar). Sin filtro:
3790 passed, 2 skipped. Working tree limpio. Writers: 24.

### C4.4 - Retirada sistema legacy de indices (2026-10-24, continuacion)

Verificacion con pipeline real revelo que C4.4 se habia saltado en la
primera parte de la sesion: el reporte contenia 2 secciones del
sistema legacy de indices (`Indices (USA + Europa) - Fases Wyckoff` y
`- Oportunidades de Acumulacion y Markup`) con mojibake en el titulo
(`## Ãndices`), coexistiendo con el sistema holdings-based.

**Borrados**: `indicators/index_phase.py`, `indicators/index_leaders.py`,
`src/pipeline/indices_intl.py`, `config/index_tickers.py` (solo lo
usaban los anteriores), `render_indices_internacionales` de
`src/report/synthesis.py`, 12 ficheros de test.

**Adaptados**: `run.py` (sin `compute_indices_intl`),
`src/report_generator.py` (sin los 3 kwargs ni el render),
`src/report/synthesis.py` (sin `INDEX_CONFIG`, `_fmt_num`,
`FLOW_ZSCORE_WINDOW`), `tests/test_output_contract_ast.py`,
`tests/test_render_modules_low_coverage_d29.py`,
`scripts/generate_docs.py`.

**REGISTRO**: 4 items FIXED (REP-AUDIT-20261018-STRUCT,
REPORT-STRESS-LABEL, FLOW-RECONCILIATION, STRUCT-COLUMN) referenciaban
`test_struct_score_sector_indices.py` (borrado en C4.4). Quitado el
test, anotada la retirada en evidencia. Fix historico vigente.

Commit `f7fdbeba`.

### Bug de rotacion sectorial (2026-10-24, continuacion)

`sector_rank_history.py:54` hacia
`for (ticker, name, score, fase) in enumerate(...)` pero C5.1 dejo la
tupla del ranking en 3 elementos. El `except` de `sectors_base.py:56`
lo capturaba (`Rotacion sectorial omitida: not enough values to unpack
(expected 4, got 3)`) y skipeaba `sector_rank_deltas` con NO_DATA. La
seccion "Rotacion sectorial reciente" del reporte llevaba desde el run
del 10:44 sin actualizarse (bug silenciado desde C5.1).

Fix `8819ba70`: unpacking 4 -> 3, tests adaptados,
`traceback.print_exc()` anadido en el `except` para que futuros fallos
no se silencien. Verificado en run real: rotacion calculada, seccion
presente en el reporte con 11 sectores.

### Leccion: auditar el reporte en 2 dimensiones

La auditoria por headers `^## ` no detecto 11 menciones textuales a
SLPM/`slpm_v12.py`/`stock_leader.py`/LIS/State Machine en notas y en
la seccion Anti-Double-Counting Audit. Esas menciones siguen en el
reporte, aunque el codigo ya no tiene el sistema. Se abre item
`C4.5-SLPM-STRINGS-RESIDUALES` para limpiarlas.

La verificacion completa del reporte debe incluir:
- Headers `^## `.
- Menciones textuales a modulos/componentes retirados.
- Mojibake (`Ã`, `â€`).
- Conteo de N/D.
- Presencia de las secciones clave.

### Estado al cierre de la sesion extendida

- Suite: 3705 passed, 2 skipped (filtro).
- Sin filtro: 3719 passed, 2 skipped.
- Working tree limpio.
- Cero secciones legacy de indices.
- Cero mojibake.
- 6 CSVs de holdings con fecha del ultimo run (23:19).
- Gate 10/10, 24 writers OK.
- Rotacion sectorial calculada.

**Deudas nuevas abiertas**:
- `C4.5-SLPM-STRINGS-RESIDUALES`.
- `STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE`.
- `VERIFY-LEADER-SELECTION-ROTO`.

**Siguiente paso**: C4.5.

## Sesion 2026-10-25 - Cierre VERIFY-LEADER-SELECTION-ROTO

Commit `37bca52d` cierra el item `VERIFY-LEADER-SELECTION-ROTO`.

**Diagnostico confirmado.** `validation/verify_leader_selection.py`
importaba `config.index_tickers` (borrado en C4.4) y leia
`analisis_lideres*.csv` (no generados desde C4.3). El step
"Verify leader selection" del `daily_run.yml` habria fallado en el
primer `run-system` real tras C4.4. Los runs de 25-10 (sabado)
estaban en skipped por gate, no por exito del step.

**Retirado.**
- `git rm validation/verify_leader_selection.py`.
- Step "Verify leader selection" de `daily_run.yml` (lineas 202-203).
- Artefactos `analisis_lideres*.csv` (lineas 226-227) y
  `slpm_state.json` (linea 239) del workflow.
- `git rm outputs/state/slpm_state.json` (fosil sin productor desde
  C3.1; ultimo escritor `slpm_v12.py` retirado en sesion 3).
- Comentario `data_quality.py:233` ("sigue activo via slpm_v12.py").
- Bloque `# SLPM COVERAGE` huerfano en `settings.py:87-90`.
- Fallback muerto en `generate_docs.py:214` (nunca se ejecutaba:
  `extract_docstring` siempre encuentra el docstring real de
  `compute_structural_score`).
- Referencia a `verify_leader_selection` en comentario `run.py:62`.

**Verificacion.** pyflakes ., compileall, suite 3705/2,
`verifica_gobernanza [OK]`, `git diff --check` limpio, Y4 cero
residuales de `verify_leader_selection`/`config.index_tickers`/
`SLPM COVERAGE` en codigo productivo.

**Pendiente del mismo frente (va a C4.5).** `src/report/helpers.py:136`
lee `analisis_lideres.csv` en `try/except FileNotFoundError` (fallback
`n_acciones=0`, defensivo); `run.py:64` limpia los CSVs al inicio
(best-effort). Ambas inocuas, se limpian con C4.5.

## Sesion 2026-10-25 (continuacion) - Cierre C4.5-SLPM-STRINGS-RESIDUALES

Commit `c1e4af2e` cierra el item `C4.5-SLPM-STRINGS-RESIDUALES`.

**Diagnostico.** El grep inicial (sesion 3) habia identificado 3
notas (sectorial.py, rankings.py x2). Al auditar con grep expandido
post-Commit A aparecieron 9 ficheros productivos con menciones
residuales, incluidos 3 no listados en el HANDOFF:
`src/report/helpers.py` (lectura `analisis_lideres.csv`),
`run.py` (limpieza CSVs de lideres) y `report_generator.py:169`
(bloque comentado huerfano).

**Retirado.**
- `src/dependency_tracker.py` v1.2 -> v1.3:
  `DIRECT_DEPENDENCIES` sin `slpm_v12.py`/`stock_leader.py`/
  `Leader Breadth`, `liquidity.py` -> `liquidity_v2.py`, anadido
  `WLS -> sector_concentration.py`. `INDIRECT_DEPENDENCIES` sin SLPM
  ni LIS, anadido WLS. `EVIDENCE_CLASSIFICATION` sin Leader Breadth.
  Nota 3 niveles sin leader breadth.
- `sectorial.py:54` sin SLPM.
- `rankings.py:90,145` sin SLPM.
- `engines.py:2` docstring sin forzar lideres SLPM.
- `rs_internal.py:6` docstring sin State Machine ni stock_leader.py.
- `momentum.py:73` sin index_leaders/stock_leader.
- `report_generator.py:169` bloque `# C3 - LIDERES POR SECTOR`
  huerfano eliminado.
- `helpers.py` bloque Acciones lideres (analisis_lideres.csv)
  eliminado.
- `run.py` bloque limpieza CSVs de lideres eliminado.

**Verificacion.** Run real 14.4 min, exit 0, gate 10/10,
24 writers OK. A1: 42 headers sin secciones legacy. A2: cero
menciones SLPM/slpm_v12/stock_leader/Leader Breadth/leader_breadth/
analisis_lideres/index_phase/index_leaders/classify_wyckoff_phase.
A4: 8 secciones clave presentes. A5: Anti-Double-Counting Audit
con WLS y liquidity_v2.py. A6: cero "Acciones lideres".

**Nueva deuda detectada.** `REPORT-ENCODING-MOJIBAKE-ADC` (OPEN):
el bloque de clasificacion de evidencia del Anti-Double-Counting
tiene mojibake preexistente ("ClasificaciÃ³n", "CrÃ©dito"). El
chequeo A3 del pipeline (Select-String "Ã|â€" sobre el .md) no lo
detecto por problemas del propio PowerShell al filtrar. Afecta a
~14 filas de la tabla de clasificacion. Abierto en REGISTRO con
alcance: rastrear donde se rompe la cadena (probablemente el
escritor del reporte), fix, test de contrato y extension del
chequeo A3.

## Sesion 2026-10-25 (continuacion 2) - Cierre STRUCTURAL-ENGINE-LEADER-BREADTH-ZOMBIE

Commit `7679d3fb` cierra el item.

**Diagnostico.** En Q1.6 de la auditoria inicial se confirmo que
el cuerpo de `compute_structural_score` NO usa `leader_breadth`.
`STRUCTURAL_WEIGHTS` tiene solo 3 claves (rs_structural 0.50,
flow_structure 0.30, persistence 0.20, suma 1.0). El parametro era
residual de la retirada de `compute_leader_breadth_v2` en C3.1.
Cosmetico puro: no sesga el calculo, no hay offset. Frente abierto
en sesion 3 como prioridad 2 por precaucion; degradado a prioridad 3
tras verificacion.

**Retirado.**
- `leader_breadth=0.5` de la firma de
  `regimes/structural_engine.py::compute_structural_score`.

**Verificacion.** AST de 10 llamadores (9 en tests + engines.py);
todos args=2 o kwargs explicitos sin `leader_breadth`. Docstring ya
correcto. 28 tests focalizados verdes, 3705/2 suite completa,
pyflakes ., compileall, `git diff --check` limpio.

**Estado de la lista de deudas de sesion 3.** C4.5 FIXED,
VERIFY-LEADER-SELECTION-ROTO FIXED, STRUCTURAL-ENGINE-LEADER-
BREADTH-ZOMBIE FIXED. Quedan CURRENT-PHASE-BASIS-CSV,
ACCUMULATION-ZERO, VALIDATION-GATE-NO-STANDALONE,
IWM-HOLDINGS-EXPANSION, SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE,
PODA-BITACORA-DIA-EN-CURSO (todos documentales u observacionales,
sin codigo pendiente en produccion salvo REPORT-ENCODING-MOJIBAKE-ADC
nueva deuda).

## Sesion 2026-10-25 (continuacion 3) - Fase 1: schema versioning outputs

Commit `c9a848b5` (codigo) + `a88e85da` (CSV regenerado) cierran el
incidente de schema mezclado detectado en auditoria.

**Diagnostico.** El commit `415487ff` (sesion 3, 24-oct) anadio
`ACCUMULATION_FORMING`, `DISTRIBUTION_FORMING` y `CONFLICT` al schema
de `sector_wyckoff_distribution.csv` sin bumpear version. El CSV
quedo con 330 filas v1 + 11 filas v2 mezcladas. `persist()` hacia
`append_dedup` ciego, gate no detectaba mismatch, `data_quality` no
reportaba schema. Detectado 2026-10-25.

**Implementado.** `OutputSpec.schema_version` (default 1,
`sector_wyckoff_distribution=2`). `persist()` detecta schema via
`.schema` lateral; si difiere, COPIAR el CSV a `.v{old}.csv` (backup),
escribir nuevo SIN appendear (evita mezclar). Gate check 11. Tests
nuevos. `.gitignore` blinda `.schema` y `.v*.csv` como metadata local.

**Verificacion.** Suite 3711/2. Run real 15 min, gate 11/11 (antes 10).
CSV nuevo 11 filas v2, backup `.v1.csv` 342 filas (untracked).

**Pendiente.** Fases 2-6 (data_quality schema_v, regeneracion
historica, unificacion de estructura, structure_score, descarga).
Ver HANDOFF seccion 4.1.

## Sesion 2026-10-25 (continuacion 4) - Fase 4: unificacion estructura

Commit `9960cdac` cierra `WYCKOFF-ESTRUCTURA-UNIFICADA`.

**Diagnostico.** `sector_wyckoff_distribution` iteraba TODOS los holdings
con umbral 60 barras. Los 6 CSVs usaban top-20 + 200 barras. Divergian
en XLC/XLK/XLP. `structure_positive` heredaba la violacion. Contrato doc
63 §2.6 dice top-20 + 200 barras.

**Implementado.** `compute_wyckoff_snapshot`: una pasada de
`classify_phase`, DataFrame long con 14 columnas (12 del contrato §4 +
`n_bars` + `evaluable`). `sector_wyckoff_distribution` v3.0 consume el
snapshot. 6 CSVs como proyecciones. `run.py` calcula una vez, pasa a
los 2 consumidores. `phase_since` ISO. Test de coherencia en CI.

**Verificacion.** Suite 3717/2. Run real 14.4 min, gate 11/11,
`Snapshot Wyckoff holdings: 357 tickers evaluados`. Coherencia agregado
vs CSVs en los 11 sectores. `structure_positive` 0/11 (baseline ya era
0/11).

**Incidente hotfix.** Fixture de tests borraba `.schema` reales,
sobrescribia `.v1.csv` (342 filas). Recuperado desde `7d223b63`. Fix:
`persist` no sobrescribe backup existente, fixture con mtime cutoff.

**Pendiente.** Fase 4b (renombrar `compute_sector_scores`), Fase 5
(`structure_score`), Fase 6 (descarga).

## Sesion 2026-10-25 (continuacion 5) - Fase 2: schema_v en frescura

Fase 2 de 6. Cierra la asimetria de Fase 1: `.schema` existia pero el
reporte no lo publicaba. `_read_schema_version()` lee `<path>.schema`,
`compute_data_quality` anade `schema_v` por fila, `render_calidad_datos`
publica columna Schema entre Cobertura y Notas. Fuentes sin `.schema`
(parquet, state JSON, europeas) -> v1 implicito. Nota al pie con
semantica y referencia a doc 64.

Suite 3717/2. Run real verificado (columna visible: `v2` para
sector_wyckoff_distribution, `v1` para el resto).

Pendiente: Fase 3 (regeneracion historica), Fase 4b (renombrar
compute_sector_scores), Fase 5 (structure_score), Fase 6 (descarga).

## Sesion 2026-10-25 (continuacion 6) - Fase 4b: rename compute_sector_scores

Fase 4b de 6. `compute_sector_scores` -> `compute_sector_rank_score`.
El score es un continuo de 6 componentes (rs_mom_20/50/126, trend,
vol_inv, breadth) sobre el ETF. NO es estructura Wyckoff. El nombre
era ambiguo: confundia con `STRUCTURAL_WEIGHTS` del motor
`structural_engine` y con el ranking "Score" del reporte. Ambos
aparecen como columnas distintas en `Rankings Sectoriales` pero
miden cosas distintas.

31 reemplazos en 7 ficheros (1 en `sector_regime.py`, 2 en
`sectors_base.py`, 28 en tests). Sin cambio de contrato. Suite
3717/2. Run real verificado.

Pendiente: Fase 3 (regeneracion historica, diferida), Fase 5
(structure_score con pre-registro), Fase 6 (descarga unificada,
bloqueada por falta de 4 ficheros).

## Sesion 2026-10-25 (continuacion 7) - Fase 3: regeneracion historica

`scripts/regenerate_wyckoff_history.py` regenera el historico completo de
`sector_wyckoff_distribution`: 11946 filas, 1086 fechas (2022-08-11 ->
2026-10-23), 11 sectores. Paralelizado (12 workers), 49.8 min, checkpoint
cada 25 tickers. Validacion obligatoria contra el `.csv` activo: 11 filas,
20 columnas identicas en 2026-10-23.

`.v1.csv` (342 filas, 330 v1 + 11 v2 mix): backup no-comparable,
gitignored. `.csv.v1_mixed` eliminado.

**Hallazgo critico para Fase 5.** `structure_positive = (pct_acc +
pct_markup) >= 50` da 0/11946 en todo el historico. `pct_range` domina
(media por sector 47.4%-91.6%). El umbral es estructuralmente
inalcanzable. No es calibracion: la metrica no discrimina. Fase 5
requiere rediseño como score continuo, con pre-registro previo.
`SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE` confirmado roto empiricamente.

Doc 66. Pendiente: Fase 5 (rediseño), Fase 6 (descarga).

## Sesion 2026-10-25 (sesion 10) - Dictamen externo + cierres

Dictamen externo sobre el modulo Wyckoff: NO APROBABLE como sistema
de señales, APROBABLE CONDICIONADO como herramienta descriptiva.
9 hallazgos aceptados. Doc 68.

**Cerrados end-to-end:**
- PIT holdings: snapshot append-only diario (holdings_snapshots.csv).
  etf_holdings.csv/index_holdings.csv estaticos sin fecha; regenerar
  historico con composicion 2026Q4 introduce sesgo de supervivencia.
  Fix: hash SHA256 de tickers ordenados por (source_file, etf),
  comparar con ultimo snapshot, escribir si cambia. daily_run + 3
  scripts update. 14 tests.
- H-03: FORMING invalidaba si el precio rompia el rango.
  wyckoff_orchestrator._dist_derived_phase y _acc_derived_phase.
  Historico 1086 fechas regenerado (script con 12 workers, checkpoint,
  validacion contra .csv activo). 52.3% filas cambian solo FORMING y
  RANGE; confirmadas 0. 9 tests.
- H-06: stock_metrics.py usaba tanh(score_median/MAD) para stability
  (conviccion direccional, arrastra t_norm). Cambio a
  wyckoff_stability = 1-2*tanh(MAD/K). 43% signos opuestos. 3 tests.
- structure_ratio continua + n_estructuras en sector_regime_matrix
  (Bloque 2). structure_positive binaria (0/11946 en el universo
  correcto) retirada. Render con Struct y N est + nota H-01.
- sector_regime_matrix migrada a schema v2: .csv -> .v1.csv (91 lineas
  historico) + fresh start (12 lineas).
- Gate check 11: migracion pendiente (spec > CSV) es WARN, regresion
  (spec < CSV) es BLOCKED. Antes ambos BLOCKED.
- Bug Xetra: 'datetime.date' object has no attribute 'date'. Helper
  _as_date(x) normaliza date/datetime/Timestamp/string/None.
  4 usos directos corregidos. 19 tickers DE recuperados. 10 tests.

**Abiertos (DICT3-*):**
- DICT3-H04-INTERVALS: ICs por cohorte anual, solapamiento temporal,
  correlacion cross-sector.
- DICT3-H05-FILLNA-FUTURE: bar_flags en build_ticker_df (calidad por
  fila) sin romper contrato v1.8.
- DICT3-TOP20-COBERTURA: universo dinamico 80-90% peso (XLI 57.2%,
  SPY 49.9% actuales).
- DICT3-REACCUMULATION: detector de re-acumulacion de continuacion en
  tendencia alcista. Frente nuevo con pre-registro propio.

**SECTOR-REGIME-MATRIX-STRUCTURE-POSITIVE -> SUPERSEDED** por Bloque 2.

20+ commits. CI verde en cada push. Al cierre: 3755 passed, 2 skipped.

## Resumen bitacora podada 28-oct-2026 (sesiones 16-oct)

Resumen de las 3 sesiones del 2026-10-16, podadas de
05_BITACORA.md el 2026-10-28 al exceder la retencion
(keep=10 + dia en curso). Contenido completo recuperable
con `git show <SHA>:Consolidacion_Documentos/05_BITACORA.md`
para cualquier SHA anterior al commit de poda.

### 2026-10-16 - Auditoria CI + dictamen externo + fases 1-7 + gobernanza

**Objetivo.** (a) Auditar el workflow `daily_run.yml`. (b) Recibir dictamen externo sobre los hallazgos. (c) Implementar las 7 fases del dictamen en 3 bloques: 1-3, 4-7, cierre documental.

**Commits.** 8269d671, fd42010d, c8a16f06, 2dda7986, 083cfd39, 3e8d2828, 433da326, 7b790b12, 66179b90, 529f4ed2e1ab037940e88f65449cd82f2080a615.

### 2026-10-16 (cierre del dia) - Auditoria del reporte + dictamen + enmienda v1.1

**Objetivo.** Tras la sesion 15-oct y la primera del 16-oct, auditar el reporte diario del run 38834597307 (16-oct 01:20 UTC). Detectados 10 hallazgos (5 criticos + C2 + 4 menores). Elevado informe al auditor externo. Recibido segundo dictamen. Aplicadas las fases aprobadas por el auditor.

### 2026-10-16 (ultima tanda) - Pinning Ubuntu + fix resolve_target_session

**Objetivo.** Cerrar dos hallazgos descubiertos al verificar el cron: pinning del runner de CI y el contrato de `resolve_target_session`.

**Commits.** 7c913031, e3eb2fe0.


## Resumen bitacora podada 30-oct-2026 (sesiones 17-oct)

### 2026-10-17 - Sostenibilidad: 10 capas de fixes al reporte + auditoria externa

**Objetivo.** Tras auditoria externa del reporte del 2026-10-17, aplicar
10 capas de fixes con tests contractuales. Eliminar textos categoricos
sin soporte, literales desincronizados de config, clasificacion de
regimenes por substring accidental, naming enganoso del CSV.

**Decisiones estructurales.**
- Clasificacion exacta de regimenes: CSV financial_regime +
  real_liq_regime (antes substring accidental). Dictamen Gate 0.5
  aprobado con correcciones (ba4fca3a, a92f52e9, 5d21b45d).
- Eliminado alias legacy liquidity_regime del CSV + script migracion
  idempotente (f2dec91f, aa82a549).
- Renombrado parametros liquidity_regime -> financial_regime en
  render_alerts/regimenes/sintesis y generate_daily_report (a6adeacd).
- Literales ventana/umbral desde config (0d58c5ae).
- Bandas HYG/LQD 0.5->0.75 (781da474), umbral materialidad 1% en
  divergencia precio-flujo (9b3d5b33).
- local_run_id determinista `local-<8hex>` (c2d6b604).
- A/D Line declara recalculo por run + universo (85b79cd9).
- Flow proxy N/D para tickers sin volumen (FX, DX-Y, SPGSCI) (09f32770).
- 9 items nuevos FIXED en REGISTRO (86963d56).
- Issue #5 cerrado: update_sec_13f cancelled = FALSE_POSITIVE,
  cron_slots 3/5 = WONT_FIX.

**Commits.** 0d58c5ae, 781da474, 9b3d5b33, c2d6b604, 85b79cd9,
ba4fca3a, a92f52e9, 5d21b45d, f2dec91f, aa82a549, a6adeacd, 86963d56,
09f32770.

**Pendiente al cierre.** Certificacion scheduled lunes 2026-10-19 23:17
UTC (target_session nuevo).

### 2026-10-17 - Fix CI (pyyaml, fetch-depth) + verificacion en produccion enmienda v1.1

**Objetivo.** Diagnosticar y corregir los fallos de CI de los runs
23:17 UTC (16-oct, exit 2) y 01:19 UTC (17-oct, exit 1). Verificar
la enmienda v1.1 en produccion.

**Decisiones estructurales.**
- P0-A pyyaml: commit 529f4ed2 introdujo `import yaml` en 4 ficheros
  sin anadir pyyaml a requirements.txt. En local pasaba por dependencia
  transitiva de optuna; en CI exit 2. Fix 4be5a90c (pyyaml==6.0.3).
- P0-B shallow clone: test_registro_real_pasa verifica existencia de
  37 SHAs contra git. actions/checkout@v4 hace fetch-depth:1. Fix
  bd30aecb (fetch-depth: 0 en run-system).
- Prevencion: tests/test_imports_declarados.py (import vs requirements,
  30bb02c4) y tests/test_workflow_fetch_depth.py (fetch-depth: 0 en
  todo job con pytest, 0c862502). .github/workflows/ci_tests.yml corre
  suite completa en cada push a main (042033a4).
- Bug fix confirmado con probe: _collect_failed_contracts duplicaba
  failed_contracts si un manifest generaba N razones BLOCKED (8e96531c).
- 12 items VERIFIED_IN_CI_PENDING -> VERIFIED_IN_PRODUCTION (4e2822b5).
- 00_ARRANQUE.md: eliminada cifra literal ~3388 (drift con
  ESTADO_SISTEMA). Fix 8ec3fbd7.
- HANDOFF reescrito con el ciclo 17-oct (33550d82).

**Commits.** 4be5a90c, bd30aecb, 4e2822b5, 33550d82, 8ec3fbd7,
b63df200, ccff8fca, c91cb76d, a792fa91, d8c42736, 5ef37d34, 81f8b90f,
93df396f, 3aa67a31, 8e96531c, 30bb02c4, 0c862502, 3c92cdfd, 019adbdc,
042033a4, 161964a3.

**Verificacion.** Run 38994345784 success (02:17 UTC 17-oct): receipt
COMPLETED, contract_validation_status: OK, manifest_coverage_pct: 0.9757,
guard_coverage: OK, validation_gate 10/10, pipeline_conclusion: success.

**Pendiente al cierre.** Verificacion completa de la enmienda v1.1 desde
cero: primer run con target_session nuevo (lunes 19-oct 23:17 UTC).

## 2026-10-30 - Ticker health autonomo + WBD + 15 expansions

**Objetivo.** Resolver dos problemas detectados en sesion 17: (a) 15
tickers fuera de top-20 sin datos (XLF 78.40%, XLV 81.43%, XLU 83.31%,
XLY 84.62% de cobertura efectiva); (b) WBD con n_bars=1 sin diagnostico
claro. Convertir el descubrimiento manual (via artifact p1_h02) en
sistema autonomo.

**Decision estructural.** Eventos corporativos (delisting, fusion,
adquisicion) requieren clasificacion explicita, no tolerancia
implicita. Diseno de 5 capas: (1) data/mappings/ticker_health.yaml
como fuente canonica de estados por ticker (ACTIVE/DELISTED/STALE/
TRUNCATED_IPO); (2) src/ticker_health.py como API unica de consulta;
(3) consumidores filtran DELISTED (get_stock_list, p1_h02,
breadth_metrics via parquet); (4) check_ticker_health WARN si ticker
ACTIVE stale no clasificado, dispara issue; (5) REGISTRO + reabrir_si
documentan cada clasificacion. Invariante: ningun ticker ACTIVE pasa
10 sesiones sin datos sin que el operador lo sepa.

**WBD diagnostico final.** Warner Bros. Discovery adquirida por
Paramount Skydance (cierre 2026-10-06, ultima cotizacion 2026-10-05
a USD 30.95). Yahoo solo sirve esa barra en WBD y todos los
cross-listings (WBD.MX, J5A.SG, J5A.DU, J5A.HM, J5A.HA). SKYD es
entidad fusionada distinta (marketCap 11B vs 77.6B, sin continuidad
de CUSIP). No hay fix tecnico: el activo no existe. SSGA/BlackRock
retiraran WBD de sus holdings en el rebalanceo trimestral.

**15 tickers fuera de top-20.** Causa raiz: TOP_N_SECTOR_COMPONENTS=20
en get_stock_list() + head(20). Ninguna de las 5 fuentes del universo
(etf_holdings top-20, index_holdings top-20, radar_target_catalog,
providers EU, universe_expansions) los capturaba. Anadidos a
universe_expansions.yaml: XLF 9 (MCO KKR AON AJG TFC ALL APO MET
AFL), XLV 4 (BSX HCA COR CAH, MRNA ya estaba), XLU 1 (CNP), XLY 1
(NKE). Mecanismo existente, sin tocar codigo.

**Lecciones operativas.**
- 4 trampas de PowerShell pisadas 1+ vez: Find -Filter "HANDOFF*.md"
  excluye .txt; ReadAllBytes con ruta relativa resuelve CWD .NET
  (C:\WINDOWS\system32), no PS; @() + ConvertFrom-Json no
  desenvuelve Object[] en PS 5.1, Where-Object no filtra;
  Select-String -SimpleMatch desactiva regex (\b se busca literal).
- Ancla temporal: insertar HEALTH_STALE_SESSIONS antes de
  CONFIRMED_INCOMPLETE_DATES desplazo el ancla
  HEALTH_CHECK_CONFIRMED_INCOMPLETE en ANCLAS.yaml:37 de la linea
  72 a la 81. Solucion: reordenar F4 DESPUES de
  CONFIRMED_INCOMPLETE_DATES. Amend + force-with-lease.
- Predecir el error: la comprobacion "cobertura efectiva del p1_h02"
  != "peso de holdings del CSV" casi rompe el yaml de expansions.
  El test contractual detecto el error antes del commit.

**Commits.** c879df71, fc6232d8, 38bc1c42, 90e269ff, 78ce37a2.
Suite 4062 passed, 2 skipped al cierre.

## 2026-10-30 (sesion 19) - Cierre deuda del reporte 41037457375

**Objetivo.** Leer el reporte del primer scheduled verde post-fix
(41037457375 sobre 89b6e0a) y cerrar cada hallazgo, no documentarlo.
La sesion 18 habia cerrado WBD y los 15 tickers de expansions. Esta
sesion cierra 9 hallazgos adicionales que el reporte del run
41037457375 exponia pero que solo eran visibles al abrirlo.

**Metodo.** Leer el reporte seccion por seccion (1173 lineas, 40
secciones) + abrir los CSVs del artifact (sector_rankings.csv,
european_coverage.md, mte_state.json, liquidity_state.json,
macro_regime.csv, cross_provider_validation.csv). Cada hallazgo:

1. Diagnostico en codigo (leer el renderer, el state, los tests).
2. Distinguir bug real de deuda de semantica / observabilidad.
3. Fix + test nuevo o adaptado.
4. Suite completa + gobernanza antes de commitear.
5. Si CI rojo, diagnostico. No commitear con suite roja (regla §3).

**Hallazgos y decisiones.**

- #3 gate 10/10 hardcoded con 11 checks. Bug de contrato. N/N
  dinamico en 3 sitios. 32 tests. (090a3439)
- N1 mte_state.json con futures_status STALE oculto en el reporte.
  Linea explicita. La confidence no se degrada (es decision de
  scoring, no de reporte). (1fb791ab)
- N3 liquidity_state.json con severidad_cruda TENSION_ELEVADA y
  diff_bp 6.0 ocultos. Linea explicita. (1fb791ab)
- N2 cross_provider_validation.py que solo consultaba Yahoo con 2
  tickers hardcoded, con 2 env vars sin uso. Deuda muerta disfrazada
  de validacion. Eliminado con step y upload. Las otras 5 API keys
  (POLYGON, ALPHA_VANTAGE, TWELVE_DATA, FINNHUB, FMP) tienen
  consumidores reales (DataRouter + BackupProvider). No son deuda.
  (fac7d5f1)
- #4 WARN catalogo 83.97% con coverage_quality=COMPLETE. WARN
  perpetuo por deuda estructural (catalogo daily vs 13F trimestral).
  INFO si COMPLETE, WARN solo si regresion. DELISTED (WBD) fuera del
  denominador. Mismo criterio que F2 y F3. (55a63833)
- #2 "Fase Wyckoff" renombrado a "Fase dominante". Es moda de
  holdings, no fase del ETF (esa se retiro en sesion 3). Los 11
  sectores salian RANGE, coherente con pct_range 70-95%. (2c2a1c3e)
- A Concentracion medía medianas sobre 4-5 tickers (Cob 18-25%) sin
  declarar cuantos. Columna "N valid" expuesta. (2c2a1c3e)
- B Sector Breadth publicaba NH=0/NL=0 sin declarar n_valid_nhnl.
  Columna "N valid NH/NL" expuesta. (2c2a1c3e)
- H -0.00 en XLE 60d FX. Ruido de coma flotante con signo visible.
  _fmt_signed (FU-003a). (71371872)
- E Liderazgo relativo doble en 55/55. Tautologico por diseno (top-5
  por retorno siempre bate). WONT_FIX razonado: la nota de sesgo ya
  esta en el reporte, cambiar el criterio es decision de producto.

**Lecciones.**
- Un WARN que siempre esta encendido y no requiere accion degrada
  la senal de los WARN reales. Dos casos (catalogo, Fase Wyckoff).
- Una columna con nombre heredado puede mentir. Dos casos (Fase
  Wyckoff, Liderazgo doble). Si el nombre sugiere X pero el valor es
  Y, renombrar o documentar en la nota.
- Metricas sobre universo minimo (4-5 tickers) sin declararlo no son
  honestas. La columna "N valid" cierra la ambiguedad.
- Un script que solo consulta el proveedor principal no es "cross-
  provider". Si el nombre miente, eliminar es mas honesto que
  ampliarlo.

**Commits.** 090a3439, 1fb791ab, fac7d5f1, 55a63833, 2c2a1c3e,
71371872 + este docs. Suite 4097 passed, 2 skipped al cierre.

## 2026-10-30 (sesion 19b) - Contrato bidireccional MTE

**Objetivo.** Cerrar la clase de bug 1fb791ab, no solo la
instancia. El bug original no era que faltara futures_status en el
return; era que state y return eran fuentes divergentes sin
contrato que lo impidiera. Cualquier campo nuevo que se anadiera
solo al state (o solo al return) repetia el ciclo.

**Metodo.**
1. Diagnostico del bug real: re-run local del run 41037457375
   (post 1fb791ab) dio 7/8. Fallo [4]: state tenia futures_status=
   STALE, render_mte no lo pintaba. Rastreo: render_mte lee
   mte_result.get('futures_status'), return de _compute_mte no
   incluia la clave. Bug.
2. Reconocer el patron: test unitario mockeado verde, integracion
   real no ejercitada. La misma clase que el commit 1fb791ab
   original (render anadido, return no exponia la clave).
3. Cerrar la instancia: futures_status en return (5637d45e) + test
   de integracion real.
4. Identificar la siguiente instancia: pending. State tenia
   pending=RECESSION, reporte no lo comunicaba. Mismo patron.
5. Cerrar la instancia pending: pending en return + render + 4
   tests (5b96192f, db1e0a0f).
6. Cerrar la CLASE: AST guard bidireccional (fc6f63c4). No basta
   con que render lea solo lo que return expone (downstream). Hay
   que garantizar que state publica solo lo que return expone, o
   esta en whitelist justificada (upstream).

**Decision estructural.** AST guard bidireccional como contrato
preventivo entre modulos que comparten un dict. Aplicable a otros
contratos del sistema (data_quality, evidence_matrix, mte_state,
liquidity_state, sector_regime_matrix). Patron:

- Modulo A produce dict con claves K_A.
- Modulo B consume K_B via A_output.get(X).
- Contrato: K_B subset K_A (downstream) + K_A - whitelist subset K_B
  (upstream) + whitelist justificada por test.

Sin el upstream, un campo funcional puede vivir en el state durante
meses (visible al operador que lee el JSON, invisible al reporte).
Caso real: futures_status, desde 2026-09-30 (R5: Yahoo futures)
hasta 2026-10-30.

**Lecciones (tambien en 01c_LECCIONES).**
- Test unitario con dict mockeado NO valida integracion real. El
  fix 1fb791ab parecia correcto por test; el run real lo detecto.
- AST guard no solo previene: define el contrato. Si el contrato
  cambia (nuevo campo, nueva direccion), el test falla y obliga a
  revisar la relacion completa.
- Whitelist explicita con test anti-creep: si alguien anade un
  campo a la whitelist para evitar el test, falla
  test_whitelist_state_only_justificada y obliga a justificar.

**Commits.** 5637d45e, 5b96192f, db1e0a0f, fc6f63c4 + este docs.
Suite 4108 passed, 2 skipped.

## 2026-10-30 (sesion 20) - Contrato dict framework: cerrar clase en codigo

**Objetivo.** Convertir el parche de la clase 1fb791ab en framework
preventivo. La clase no es "un campo falta en un dict"; es "modulo A
produce dict, modulo B lo consume via .get(X), A no publica X, B
obtiene None silenciosamente". Test unitario con dict mockeado
verde; integracion real no ejercita. Leccion repetida 3 veces en
octubre (mte futures_status, mte pending, confirmation mte_scenario).

**Decision estructural.** Framework en 4 capas:
1. Helper AST (tests/_dict_contract.py): producer_return_keys,
   producer_state_keys, producer_assign_dict_keys,
   producer_subscript_keys, producer_all_return_keys,
   consumer_read_keys. 2 assert: downstream (render lee X => X en
   return) y upstream (state publica X => X en return).
2. Contrato por par: un fichero de test por productor principal
   (MTE, liquidity, IAE, darkpool, cross_module) + un fichero
   combinado (flows + confirmation) donde el productor muta el
   dict dinamicamente o hay composicion.
3. Schema constante cuando el AST no llega: flow_synthesis
   (subindice + update dinamico) y confirmation_data (subindice +
   update + composicion via compute_mte_confirmation).
4. Manifiesto + heuristico: DICT_CONTRACT_PAIRS declara los 9
   pares cubiertos. test_heuristico_render_dict_detecta_nuevos_
   pares enumera todos los render_* con clasificador df_attrs
   (distingue DataFrame de dict real). Falla si aparece un dict
   real sin manifiesto ni whitelist.

**Descubrimientos del inventario.**
- 32 render_* con dict-like. 10 DataFrames disfrazados (regex
  inicial los contaba como dict; df.empty es Attribute, no Call).
  2 dicts reales cubiertos en este plan. 2 en whitelist. 2
  consumidores del mismo productor.
- Bug real de la clase: render_confirmation leia
  confirmation_data.get('mte_scenario') pero ningun productor lo
  escribia. La rama RECESSION CAPITULATION estaba muerta. Fix en
  compute_mte_confirmation. Detectado por el inventario, no por
  run. Si el heuristico hubiera existido, se habria detectado
  antes.
- Whitelist justificada + test anti-creep: si alguien anade un
  par a la whitelist para evitar el test, falla
  test_whitelist_justificada y obliga a justificar.

**Lecciones.**
- Una clase de bug repetida 3 veces en un mes no se resuelve con
  parches. Se resuelve con un contrato en el codigo que lo haga
  visible en CI.
- El AST no lo ve todo: mutation por subindice, .update(dict),
  composicion entre funciones. Schema constante explicito donde
  AST no llega.
- Clasificador df_attrs: distinguir DataFrame de dict real es
  heuristica, no certeza. Pero reduce falsos positivos de 10 a 0.
- Whitelist con razon + test de conteo: la unica defensa contra
  whitelist creep.

**Commits.** d100ba01, f364ad2e, 053db86c, 457742f7, 4056c033,
e8a2a4ba + este docs. Suite 4133 passed, 2 skipped.

## 2026-10-31 (sesion 21) - Coherencia analisis vs descarga: 2 bugs estructurales

**Objetivo.** Auditar el reporte del run local 20261030_224230 en busca
de errores, inconsistencias o bugs silenciosos no detectados por tests.
Foco especial: el gap n_valid_return20=33 < n_valid_rs=34 en XLF tras
el fix O2 (1cb15d52).

**Metodo.**
1. Run local + snapshot pre/post + diff contra baseline (post_verify19b).
2. Auditoria automatizada del reporte: formatos sospechosos, cambios
   esperados, cambios condicionales, secciones faltantes, contradicciones
   cruzadas, N/D regresiones.
3. Auditoria fina contra CSVs fuente: n_valid_* vs CSV, JSON raw, gates.
4. Cruce de gaps: n_valid_return20 vs n_valid_rs por sector.

**Hallazgos.**
- O2 (1cb15d52): el cap head(TOP_N_CANDIDATES=15) en compute_all_stock_
  metrics contradecia el universo de descarga (universe_expansions +
  get_stock_list). Los 9 tickers XLF anadidos en c879df71 (sesion 19)
  estaban descargados (1253 barras) y en rs_internal (33 XLF) pero
  fuera de stock_metrics (15 XLF). n_valid_rs de XLF = 14 cuando hay 34.
- BRK.B (e04b53ff): O2 expuso incoherencia interna. holdings usa BRK.B,
  parquet BRK-B. WYCKOFF-TICKER-NORMALIZATION (2026-10-27) resuelto
  en wyckoff_holdings_report y stock_metrics, pero no en rs_internal
  ni sector_concentration. El gap ret20 (33) vs rs (34) delataba el
  bug: BRK.B (peso 12.32% de XLF) caia silenciosamente en
  _compute_concentration._get_series.

**Decision estructural.** "Coherencia analisis = descarga" como
principio: si el sistema descarga un ticker (parquet), debe analizarlo
(metricas por sector); si lo excluye, debe ser por decision explicita
(delisted, guard defensivo). Un cap rigido por numero (top-15) es
incompatible con universe_expansions. Sustituido por universo dinamico
(holdings presentes en df_stocks.Close) con guard MAX_CANDIDATES_PER_
SECTOR=100 para ETFs gigantes (IWM 1977).

**Verificacion empirica.**
- Run local 20261030_230703: O2 aplicado. XLF n_valid_rs 14->34, XLV
  14->26, XLU 15->22, XLY 15->22, XLI 42->50, XLK 33->39.
- Run local 20261031_003442: BRK aplicado. Gap ret20 vs rs = 0 en los
  11 sectores. XLF ret20=34, rs=34.

**Lecciones.**
- Un fix de datos (c879df71) requiere auditar todos sus consumidores.
  Llego a descarga y rs_internal, no a stock_metrics. Sintoma: cobertura
  distinta segun la metrica.
- Los gaps entre metricas del mismo universo delatan bugs silenciosos.
  n_valid_return20 vs n_valid_rs diferentes en el mismo sector = un
  ticker se pierde en una ruta y no en otra.
- El AST guard del DICT-CONTRACT-FRAMEWORK no detecto BRK.B porque el
  dict se construye via lookup individual (get_col, _get_series) sin
  claves literales. El gap ret20 vs rs lo expuso.

**Commits.** 1cb15d52 (O2), e04b53ff (BRK) + este docs. Suite 4137
passed, 2 skipped.

## Resumen bitacora podada 31-oct-2026 (4 sesiones 2026-10-18)

### 2026-10-18 - Investigacion frescura + auditoria externa + audit post-receipt

**Objetivo.** Verificar que el sistema se nutre con datos frescos de
forma automatica. Cerrar deudas detectadas y puntos de auditoria
externa del reporte 2026-10-17.

**Decisiones.**
- IPS/IPI son proxy de mercado (no IPC/PPI): texto nuevo en
  volatility_mte.py + 2 tests (8eaecb0a). 3 falsos positivos
  documentados (NIPC, SLPM, liderazgo).
- Deuda validation/ en workflows: quitar validation/ del pytest en
  ci_tests y daily_run (cafc77d8).
- data_quality.csv con date = sesion bursatil (4ade7384 + 5ef44486):
  migracion historica 155 filas en 9 fechas, 593->464 filas. 11 tests
  nuevos en test_data_quality_date_sesion_bursatil.py.
- Audit post-receipt no bloqueante: scripts/post_receipt_audit.py
  compara writer_receipts.json vs outputs/history/*.csv.

**Commits.** 8eaecb0a, cafc77d8, 4ade7384, 5ef44486.

### 2026-10-18 - GEN-ESTADO-*: cierre de las dos deudas del generador

**Objetivo.** Cerrar GEN-ESTADO-VALIDATION-VACIO y GEN-ESTADO-TESTS-
NOT-FAILCLOSED, adelantandolos al mismo dia.

**Decisiones.**
- GEN-ESTADO-VALIDATION-VACIO (72659a9f): retirar validation/ del cmd
  de pytest. Test AST que verifica el cmd del run().
- GEN-ESTADO-TESTS-NOT-FAILCLOSED (cb3253d3): propagar rc de
  section_tests() y section_integridad() a main(). Snapshot se
  escribe siempre; exit code 1 si tests/compileall/pyflakes fallan.
  4 tests hermeticos con tmp_path + monkeypatch.

**Commits.** 72659a9f, cb3253d3.

### 2026-10-18 - Regenerar ESTADO_SISTEMA + items del generador

**Objetivo.** Cerrar la discrepancia entre ESTADO_SISTEMA.md (3641
passed) y la realidad de la suite (3730 con filtro / 3739 sin filtro).
Aprovechar para documentar dos deudas del generador.

**Decisiones.**
- Regenerar ESTADO_SISTEMA.md: 3641 -> 3739 (pytest sin filtro). La
  cifra 3730 del HANDOFF es la del filtro `-m "not integration_real_data"`.
  Nota aclaratoria anadida a HANDOFF.
- Abrir GEN-ESTADO-VALIDATION-VACIO y GEN-ESTADO-TESTS-NOT-FAILCLOSED
  (cerrados en la sesion siguiente).
- HANDOFF §1 actualizado: 73 -> 75 items (2 OPEN nuevos).

**Commits.** 2ca11b5a, 80ae528b. CI run 39083891160 success.

### 2026-10-18 - Cierre residuales dictamen + struct en Rankings/Indices

**Objetivo.** Cerrar los 5 residuales del dictamen original (H5, H7,
H10/H19, H11/H12, H16) y publicar struct_score en Rankings Sectoriales
e Indices (opcion B aprobada) para evitar lecturas binarias de fase
Wyckoff.

**Decisiones.**
- H16 (9d8d8cf9): Financial Stress e Inflation Pressure siempre con N/5.
- H7 (3d8c9689): ETFs fuera del denominador de cobertura. 305/340=0.897
  -> 305/329=0.927. WARN falso eliminado.
- H5 (37e1f02e): pos_change = net_position.diff() en CFTC. Identidad.
- H10/H19 (e7b12feb): "Data Conf" -> "Conf. Frescura". Aclara que solo
  expresa frescura, NO cobertura.
- H11/H12 (caa176d1): provider lee FILING_DATE. Render publica Fecha
  observacion + presentacion. Dispatch 39074246538 confirmo FILING_DATE
  al 100%.
- Struct (a226f556 + 22392813): columnas Struct en Rankings e Indices.
  XLK +0.45 MARKUP, ^NDX +0.30 RANGE a 0.002 del umbral, XLP +0.18 RANGE.
- E2E run.py verificado: los 5 fixes en el reporte real + Struct.

**Commits.** 9d8d8cf9, 3d8c9689, 37e1f02e, e7b12feb, caa176d1,
a226f556, 22392813.

