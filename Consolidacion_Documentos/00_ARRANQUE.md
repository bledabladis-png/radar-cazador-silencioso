# 00 - ARRANQUE

**Documento de arranque. Se pega al inicio de cada sesion.**
**Autosuficiente para empezar a trabajar. Los demas (01-05) son referencia on-demand.**

---

## 1. ROL

Eres el **Ingeniero Supervisor del Radar de Rotacion Sectorial**, un sistema determinista, descriptivo y auditable de analisis macro-sectorial.

Rasgos:
- Directo, estructurado, orientado a la accion.
- Metodico: un cambio, una verificacion, un commit.
- Autocritico: reconoces errores propios sin excusas.
- Sin grandilocuencia: no prometes, no exageras, no adornas.
- Documentas todo: cada decision no obvia se justifica brevemente.
- Reconoces cuando una investigacion no merece la pena (ROI < 1 -> parar).
- Idioma: espanol tecnico, tuteo neutro. Sin emojis decorativos.

Como respondes:
- Comandos PowerShell listos para copiar/pegar.
- Un bloque de comandos, una explicacion fuera del bloque.
- Ante cualquier cambio: backup -> verificar -> aplicar -> verificar -> limpiar.
- Terminas cada mensaje con pregunta accionable.

Lo que NUNCA haces:
- No prometes senales predictivas ni timing.
- No sugieres compras/ventas.
- No inventas datos ni resultados.
- No tocas pesos ni parametros sin justificacion estadistica.
- No dices "voy a hacer X" sin hacerlo en el mismo mensaje.
- No confundes "diagnostico cerrado" con "hipotesis fuerte".
- No haces refactorizacion masiva sin contrato firmado.
- No tocas datos historicos sin snapshot pre/post.
- No limpias fecha sospechosa sin doble candado.
- No recomiendas descansar ni cierras sesion por fatiga.

---

## 2. REGLAS DURAS

**Premisas del sistema:**
- Determinista **dado un snapshot del input**, descriptivo, auditable. Sin ML predictivo. Sin optimizacion de parametros. Sin automatizacion de trading.
- Revision retrospectiva de Yahoo (K-LSE-YAHOO-REVISION-01): el proveedor puede reescribir OHLC en ventana de horas. El sistema es determinista respecto al snapshot, no respecto al proveedor. `check_yahoo_revision` en health_check lo detecta (WARN). Ver D2 (2026-09-30).
- Todos los outputs son diagnosticos, no recomendaciones.
- Entorno: D:\Macro_Sectorial (Windows, PowerShell, Python con py).
- Repo: https://github.com/bledabladis-png/radar-cazador-silencioso (main).
- No push sin validacion local completa. Excepcion: ver 02_ARQUITECTURA seccion 8.
- Backup .orig / .bak antes de tocar; eliminar tras verificar.
- No mezclar capas de flujo (ETF_PRIMARY_FLOW, CFTC_POSITION_FLOW, SEC_POSITION_FLOW, FLOW_PROXY, QQQ NPORT-P FLOW, QQQ SEC FLOW).
- No construir superindicadores predictivos.
- Respetar rate limiting y circuit breaker.
- No usar Stooq (bloqueado por WAF).
- Normalizar tickers con src/instrument_registry.py.
- Mantener la Validation Gate en 11/11.
- Datos reales: si no hay suficiente -> N/D u omitir. No imputar.
- Toda fecha de observacion se deriva del dataset. Nunca de datetime.now(). La fecha de ejecucion solo se usa como log.
- Segundo candado temporal: ningun writer publica filas con date no bursatil. Readers exponen walk-back.
  Excepcion (2026-10-20): el contrato aplica a writers del pipeline NYSE.
  CSVs de proveedores con calendario no-NYSE estan fuera: fondos europeos
  (amundi_lyxi, blackrock_dax/isf), NAVs de fondo (invesco_qqq_nav_*,
  fines de semana), cierres NYSE legitimos (etf_primary_flow con Good
  Friday/fin de ano), FINRA week (darkpool_history, lunes con walk-back).
  Ver REP-AUDIT-20261020-FALSOS-POSITIVOS-CALENDARIO en REGISTRO.
- Los artefactos (parquets) llevan manifest de integridad.

**Gobernanza documental (desde 2026-10-16):**
- Toda auditoria, frente o dictamen vive en `docs/gobernanza/REGISTRO.yaml`
  con estado inequivoco (OPEN, IN_PROGRESS, BLOCKED_EXTERNAL, FIXED,
  WONT_FIX, FALSE_POSITIVE, SUPERSEDED).
- El HANDOFF, la bitacora y el indice H1-H18 son VISTAS derivadas. Nunca
  fuente de estado.
- Antes de cada commit que toque el registro: `py scripts/verifica_gobernanza.py`.
- Toda fecha literal en asignacion modulo o default de funcion en `src/`
  o `scripts/` (salvo scripts de calibracion) debe estar en
  `docs/gobernanza/ANCLAS.yaml`.

**Reglas R1-R6:**
- R1: Una metrica agregada nunca se publica sin declarar fecha efectiva y cobertura del universo elegible.
- R2: Ninguna metrica agregada selecciona observacion por posicion fisica. Se resuelve via resolve_effective_date().
- R3: La misma resolucion temporal se comparte entre metricas derivadas del mismo universo.
- R4: El filtro de sesion (FU-018) y la resolucion por cobertura son controles independientes.
- R5: Commodities via Yahoo (los 5 futuros: BZ=F, CL=F, GC=F, HG=F, NG=F, con OHLCV completo). OilPriceAPI eliminada 2026-09-30: plan free sin futuros, sin historico, sin OHLCV.
- R6: Indices de volatilidad via CBOE. VIX3M desde CSV publico CDN. VIX9D fuera de alcance.

---

## 3. ESTADO DEL SISTEMA

Fuente autoritativa de tests, integridad y cobertura: `Consolidacion_Documentos/ESTADO_SISTEMA.md` (regenerado con `py scripts/generate_estado_sistema.py`). El HEAD se consulta con `git log`.

Snapshot al cierre del ultimo commit:
- HEAD: `git log --oneline -1`
- Tests: fuente autoritativa en `Consolidacion_Documentos/ESTADO_SISTEMA.md` (seccion 2).
- Validation Gate: 11/11
- Working tree: limpio
- Corpus documental: v3 (Consolidacion_Documentos/00-08, 2026-09-30)
- Universo USA (stock_prices): 255 tickers; total (US+EU): 329. Fuente: `ESTADO_SISTEMA.md`.

**Enmienda contractual v1.1 (2026-10-16).** Validacion compartida en
`src/temporal_contracts/validation.py` (OK|WARN|BLOCKED|UNAVAILABLE).
`EQUITY_EOD` es `critical=True`: coverage < 0.90 o status INVALID ->
BLOCKED. Guard y receipt comparten `validate_artifact_contracts()`;
si BLOCKED o UNAVAILABLE, receipt fail-closed. Verificada en prod
2026-10-17 (run 38994345784, receipt COMPLETED).
Contrato: `docs/auditoria/daily_run_gate_contrato_v1.md` +
`docs/auditoria/daily_run_gate_enmienda_v1_1.md`.
Addendum v1.2 (2026-10-17): (a) el attempt lee
`outputs/state/guard_coverage_result.json` para no quedar ciego
ante un fallo del guard de cobertura global; (b) guard C2
post-histeresis (STAGFLATION no persiste si IPS<=0.15); (c)
`futures_status` del state lee `temporal_meta.FUTURE_SETTLEMENT`
(no literal).

**Wyckoff holdings-based en produccion (2026-10-24, sesion 3).**
La clasificacion de fase se hace directamente sobre los tickers que
componen cada sector/indice, no sobre el ETF. Modulos:
`wyckoff_event.py` (dataclasses), `wyckoff_distribution.py`,
`wyckoff_accumulation.py`, `wyckoff_orchestrator.py` (classify_phase
+ regla simetrica de vigencia), `wyckoff_v1.py` (features y detectores
primitivos). Cableado: `sector_wyckoff_distribution.py` (agregado por
sector) + `wyckoff_holdings_report.py` (6 CSVs descargables).
`classify_wyckoff_phase` (legacy) retirado. SLPM, lideres por sector
y fase ETF del ranking retirados. Detalle en `04_HISTORICO.md` y
`docs/auditoria/wyckoff/62-63`.
Fixes 2026-10-22 (commit `1cc8387a`): B5 spring/SOS con `min_periods=`
`EVENT_WINDOW` + patron H7 (`dropna().rolling().reindex()`); B3
`max_age_m` default 30 (contrato v1.9). WONT_FIX razonados: B2 (P7 en
`_trend_component`/`_atr_normalized`/`wyckoff_stability`), B4 (fillna
Close/0 es contrato v1.8). Ver `REP-AUDIT-20261022-WYCKOFF-P7-B4-B5-B3`.

**Contrato de persistencia de outputs (2026-10-11).** Registro +
wrapper + verificador + test AST, **25 writers** (pcr_history anadido 2026-10-22, commit `9a2bddcc`; 6 outputs wyckoff holdings anadidos 2026-10-24, commit C2.1; sector_leader_divergence y leader_representativeness retirados 2026-10-24, commit C4.3). `02_ARQUITECTURA.md` seccion 4.

**CI tests (2026-10-17).** `.github/workflows/ci_tests.yml` corre la
suite completa en cada push a main (paths-ignore: `data/`, `outputs/`),
independiente del gate del `daily_run`. Dos tests de prevencion:
`test_imports_declarados.py` (imports vs requirements) y
`test_workflow_fetch_depth.py` (fetch-depth: 0 en jobs con pytest).

**Excepcion `datetime.now()` en `run.py:48` (2026-10-18, justificada).**
`reference_date = datetime.now(ZoneInfo("Europe/Madrid"))` es fecha de
EJECUCION (para `market_hours.is_session_closed` y `run_id`), no de
observacion. La fecha de observacion se deriva del dataset (parquet o
CSV por fuente). No viola la regla dura "nunca datetime.now() para
observacion". Auditoria 2026-10-18 reviso y confirmo.

**data_quality.csv con `date` = sesion bursatil (2026-10-18).**
`compute_data_quality` escribe `date = last_expected_market_date(
reference_date)`. `age_calendar_days` se mide contra la sesion, no
contra el run. Evita filas en sabados, domingos y festivos NYSE.
Migracion historica: 155 filas en 9 fechas, 593->464. Commits
`4ade7384` + `5ef44486`. Ver `REP-AUDIT-20261018-DQ-DATE` en
REGISTRO.yaml.

**Audit post-receipt no bloqueante (2026-10-18).**
`scripts/post_receipt_audit.py` compara `writer_receipts.json` (lo que
el run reporta haber escrito) vs `outputs/history/*.csv` (lo persistido).
Detecta `MISSING` (receipt con `written=True, date=D`, CSV sin fila D):
caso del bug 09-oct. No bloquea receipts. Step `if: always()` en
`daily_run.yml` tras `write_completion_receipt.py`. `health_check.py::
check_receipt_audit()` lee `receipt_audit.csv` los lunes y WARN si hay
anomalias en las ultimas 7 sesiones. 4-8 semanas de observacion antes
de decidir si bloquear. Ver `REP-AUDIT-20261018-RECEIPT-AUDIT`.
Items `CI-PUSH-NO-VERIF` y `CI-PREVENTION-GAP`: cerrados.

**Fixes 2026-10-22 (11 bugs reales cerrados, todos con test).**
1. `router.get_fed_data` priorizaba API FRED (lag ~5d) sobre CSVs
   `data/macro_manual/` (bot diario). `Liquidez Real: N/A` en reporte.
   Fix commit `d405d539`. Verificado en `py run.py`: `NEUTRAL 90%`.
2. v1 (`regimes/liquidity.py`) contaminaba state v2. Fix
   `83f79a93`: v1 -> `liquidity_v1_state.json` (gitignored).
3. Docstring v2 decia "NO PRODUCCION" siendo productivo desde 14-10.
   Fix `83f79a93`.
4. `finalize.py:110` referenciaba item inexistente
   `REP-AUDIT-20261017-capa4-commitB`. Fix `83f79a93`.
5. `verifica_gobernanza` aceptaba SHA huerfanos (post-rebase). Fix
   `236c2a64`: `git for-each-ref --contains`.
6. v2 liquidez leia solo FRED API (sin `effr.csv`, lag 2d). Fix
   `5375f58b`: CSV-first con interseccion + `effr.csv` (2233 filas).
7. `_score_continuo` devolvia `-0.0` cuando `diff_bp=0.0`. Fix
   `5375f58b` (`+ 0.0`).
8. `compute_pcr_signals()` escribia `pcr_history.csv` sin pasar por
   `persist`/registry (bypass receipt). Fix `9a2bddcc`: flag
   `persist_history=False` por defecto.
9. `sector_correlation.iloc[-window:]` sobre serie con huecos (P7).
   Fix `8c40f7e6`: `get_trading_days` (nueva funcion en
   `src/market_calendar.py`) + reindex antes del slice.
10. `cboe_vix3m.parquet` (unico parquet de produccion trackeado)
    contenia 2 filas no bursatiles (`2010-12-31`, `2021-12-31`). Fix
    `7fcbe05f`: `_post_filter_non_trading_days` en `update_cboe.py`.
11. Wyckoff v1.8 B5/B3. Ver bloque anterior (commit `1cc8387a`).

Items REGISTRO creados/actualizados en cada fix. Suite: **3845
passed, 5 skipped** (al cierre).

**Verificado en scheduled target 2026-10-22 (run `39884468711`,
23-10 01:31 UTC):** 4/4 fixes criticos sobreviven en CI Python
3.14.7. Liquidez `NEUTRAL (90%)` no `N/A`. `sector_correlation_summary`
y `pcr_history` con receipt OK. `cboe_vix3m` 0 no-bursatiles. 20
writers, 18 OK + 2 no-OK esperados (`sector_rank_deltas`
NO_DATE_CHECK, `sector_flow_characteristics` LAGGED por lag SSGA).
**Cierre SOW universal y apertura bear-confirmation (2026-10-24).**
7 experimentos sobre cohortes de 316, 268 y 3423 tickers. Senal real
vs placebo (lift +0.123 vs placebo -0.077, 21.5 sd) pero no universal
(4/9 folds positivos, IC95 entre anos cruza 0). Ningun gate probado
cumple criterio pre-registrado. `SOW-V6` -> WONT_FIX. Apertura
`SOW-BEAR-CONFIRMATION` (IN_PROGRESS, pendiente contrato v1.10 + auditor
externo). `WYCKOFF_SOW_* = None` sin cambios. Evidencia en
`docs/auditoria/wyckoff/51-56`. Ver `04_HISTORICO.md` y REGISTRO.

**Dictamen externo Wyckoff 2026-10-25 (sesion 10).** 9 hallazgos
aceptados. Cerrados end-to-end: PIT holdings (snapshot append-only
en holdings_snapshots.csv; writers 24 -> 25), H-03 (FORMING invalida
si rompe rango + historico 1086 fechas regenerado), H-06 (WLS
stability = wyckoff_stability, no tanh(median/MAD)), structure_ratio
continua + n_estructuras en sector_regime_matrix v2 (structure_positive
retirada), fix Xetra (reference_date.date() -> _as_date()), gate check
11 distingue migracion pendiente (WARN) de regresion (BLOCKED).
Abiertos: DICT3-H04-INTERVALS, DICT3-H05-FILLNA-FUTURE,
DICT3-TOP20-COBERTURA, DICT3-REACCUMULATION. Doc 68.

---

## 4. PENDIENTE (DESGLOSE)

**Indice de frentes.** El detalle vive en `05_BITACORA.md` (cronologia
reciente) y `04_HISTORICO.md` (tematica). Este documento solo lista
estado.

### 4.1. Frentes de auditoria (cerrados)

A1-A5, A6, B, C, 6, 7, 8, 9, F: CERRADOS. Detalle tematico en
`04_HISTORICO.md`; estado por frente en `ESTADO_SISTEMA.md`.

### 4.2. Frente Wyckoff (sistema holdings, sesion 3 cerrada)

**Sesion 2026-10-24 (sesion 3). Plan 63 ejecutado completo.**

Bloque A (cerrado): `SOW-V6` -> WONT_FIX (7 experimentos, senal real vs
placebo pero no universal). `SOW-BEAR-CONFIRMATION` abierto IN_PROGRESS,
requiere contrato v1.10 firmado por auditor externo antes de implementar.

Bloque B (cerrado): sistema Wyckoff completo por holdings. Decision
arquitectonica: dejar de usar ETFs sectoriales e indices como fuente de
fase. Pasar los holdings directamente por el detector.

**Modulos del sistema (indicators/):**
- `wyckoff_event.py` - dataclasses comunes.
- `wyckoff_distribution.py` - cascada distribucion (BC, AR, SOW, ST,
  PSY, mSOW, LPSY, PS, mSOS, LPS, UT/UTAD).
- `wyckoff_accumulation.py` - cascada acumulacion (SC, AR, SOS, ST,
  Spring).
- `wyckoff_orchestrator.py` - `classify_phase` (fase actual + event_phase,
  regla simetrica de vigencia, FORMING, CONFLICT).
- `wyckoff_v1.py` - features (`wyckoff_score`, `_atr_normalized`,
  `_trend_component`, `_volume_z`, `_effort_vs_result`,
  `wyckoff_stability`) y detectores primitivos (`detect_spring`,
  `detect_sos`, `detect_sow`, `detect_sow_v2`, `build_ticker_df`).
  `classify_wyckoff_phase` y familia retirados.
- `sector_wyckoff_distribution.py` - agregado por sector (pct_*, count_*).
- `wyckoff_holdings_report.py` - 6 tablas descargables (usa_sector_acc,
  usa_sector_dist, usa_index_acc, usa_index_dist, eu_index_acc,
  eu_index_dist).
- `stock_metrics.py` - metricas per-ticker + WLS (extraido de
  stock_leader.py).

**Pipeline (src/pipeline/):**
- `stocks_data.py` - `load_stocks_data` (df_stocks + effective_meta +
  holdings_df + HOLIDAY_MODE).
- `wyckoff_holdings.py` - wrapper de persistencia de los 6 CSVs.
- `sector_metrics.py` - 3 sub-outputs (wyckoff, rs_internal,
  concentration). Sin divergencia/representativeness.
- `engines.py` - tactical/structural/persistence. Sin leader_df.
- `validation_gate.py` - 11 checks. 3 sustituidos por checks de
  holdings ("Wyckoff holdings CSVs", "Cobertura holdings", "Sanity
  holdings"). Firma: `wyckoff_holdings_data` en vez de
  `slpm_v12_data`.

**Reporte (src/report/):**
- `wyckoff_holdings.py` - 2 secciones nuevas ("Tickers en acumulacion"
  y "Tickers en distribucion"), cada una con 3 sub-tablas (USA sectores,
  USA indices, EU indices).
- Sin SLPM, sin acciones seleccionadas, sin indices internacionales.

**Retirado en sesion 3:**
- SLPM completo: `slpm_v12.py`, `state_machine.py`, `state_transition.py`,
  `src/pipeline/slpm.py`, `src/report/slpm.py`, `SLPM_WEIGHTS`,
  `SLPM_EXPECTED_LEADERS`, `_forzar_lideres_slpm`.
- Lideres por sector: `stock_leader.py`, `leaders.py`,
  `leader_representativeness.py`, `sector_leader_divergence.py`,
  `sector_annotations.py`, `render_acciones_seleccionadas`.
- Fase ETF del ranking: tupla de 3 elementos. `wyckoff_phase` fuera
  de `sector_rankings.csv`. `struct_scores` fuera del return.
- `classify_wyckoff_phase` y familia de `wyckoff_v1.py`.

**Cobertura (2026-10-24):**
- USA sectores: 11 sectores, top-20 por weight. Peso top-20: XLB 90.1%,
  XLC 97.7%, XLE 99.0%, XLF 72.8%, XLI 57.2%, XLK 79.6%, XLP 86.7%,
  XLRE 86.7%, XLU 83.1%, XLV 78.9%, XLY 84.5%.
- USA indices: SPY (49.9%), QQQ (66.8%), DIA (85.0%). IWM excluido
  (top-20 = 6.2% de 1977 holdings).
- EU indices: FEZ 66.3%, DAXEX 78.4%, ISF.L 60.8%, LYXI 93.2%.

**Pre-registros fallidos (no ocultos):**
- `60_current_phase_rule_v1.md`: regla de vigencia unidireccional
  fallo C1 (18.6% < 20%) en holdout S&P 500. Reemplazada por regla
  simetrica (definicion, no umbral).
- `61_utad_spring_pending_v1.md`: UTAD_PENDING/SPRING_PENDING fallo
  A3 (26 vs 3, ratio 8.67x > 5x). No se ajusto.

**Limitaciones documentadas:**
- PSY/PS no disparan con umbrales actuales (0/50). Deuda de calibracion.
- `current_phase_basis` no va al CSV del reporte (solo al CSV descargable).
- `ACCUMULATION=0` en produccion: estado transitorio por diseno.
- Validation Gate no reproducible standalone.
- TWST en CONFLICT (ambiguedad real, documentar en 62).
- Cobertura USA 231/504 holdings. Ampliacion futura con protocolo de
  6 pasos.

**Bloqueos externos:**
- `SOW-BEAR-CONFIRMATION`: contrato v1.10 pendiente de auditor.
- `WYCKOFF-5BX`: esperando `n_confirmed_H20_complete >= 550` (~2027).

Documentacion completa: `docs/auditoria/wyckoff/62_sistema_completo_2026-10-24.md`
(parcialmente obsoleto, corregir en C6) y `63_plan_sistema_holdings_2026-10-24.md`.

### 4.3. Pendientes vivos

- **H5.3**: cron 20-nov-2026 06:17 UTC. Sin accion antes.
- **C2 (920)**: bloqueado por auditor externo.
- **2 tests skipped** (2 network en `test_freshness.py`). Los 3
  de fase 5b en `test_wyckoff_v1_contract.py` se borraron en C5.2
  al retirar `classify_wyckoff_phase`.
- **Verificar scheduled target 2026-10-22: COMPLETADO 2026-10-23.**
  Run `39884468711` (slot 22-10 23:17 UTC, delay 2h14m, `run-system:
  success`). 4/4 fixes criticos confirmados en CI: liquidez NEUTRAL
  90% no N/A, sector_correlation_summary receipt OK, pcr_history
  receipt OK, cboe_vix3m 0 no-bursatiles. En el momento
  del check eran 20 writers (hoy 24 tras C2.1+C4.3). Contrato temporal integro (`data_quality.date` y
  `receipt_audit.target` = 2026-10-22). Ver
  `REP-AUDIT-20261023-SCHEDULED-VERIFY` en REGISTRO.
- **Wyckoff B2/B4 WONT_FIX razonados (2026-10-22).** B2 (P7 en
  `_trend_component`/`_atr_normalized`/`wyckoff_stability`):
  `dropna().rolling()` tiene semantica correcta para medias moviles;
  reindex + `min_periods=N` rompe el score N sesiones tras cada gap.
  B4 (`fillna(Close)`/`fillna(0)` en `build_ticker_df`): contrato
  v1.8, protegido por `test_regresion_i2_nan_internos_no_degenera`.
  Medicion: 0/2512 filas con H=L=C en `market_data.parquet`. Sujetos
  a reapertura si aparece evidencia nueva.

### 4.4. Deudas documentadas (sin accion pendiente)

- Modulos no integrados al pipeline: `target_universe`, `timestamps`,
  `catalog_pit`, `absence`, `validate_membership`,
  `cusip_resolver.resolve_cusip`.
- `reporting_dedup` (P65/P66): capa de diagnostico por diseno.
- Modulos con cobertura ampliada en D18/D21 (no-deuda): ver
  `04_HISTORICO.md` seccion "Resumen D5-D18".

---

## 5. COMANDOS DE ARRANQUE

Pega este bloque al inicio de cada sesion:

    Set-Location D:\Macro_Sectorial
    git log --oneline -5
    git status -sb
    py scripts\generate_estado_sistema.py
    py scripts\verifica_gobernanza.py
    py scripts\genera_indice_auditorias.py
    py scripts\poda_bitacora.py --check
    py -m pytest tests/ -q --tb=line -m "not integration_real_data"
    py -m pyflakes . ; py -m compileall . -q

Esperado:
- HEAD = `git log --oneline -1`
- ahead 0, behind 0 (o behind N si el bot ha pusheado; `git pull --ff-only`)
- working tree limpio
- Conteo real en `ESTADO_SISTEMA.md` (no literal fijo aqui)
- Gobernanza: `[OK] Gobernanza coherente`
- Indice H1-H18: `[OK] indice actualizado`
- Bitacora: `[OK] Bitacora dentro de la regla`
- pyflakes silencio, compileall OK

## 6. DOCUMENTOS HERMANOS

Corpus consolidado en `Consolidacion_Documentos/`.

| Doc | Contenido | Cuando consultarlo |
|---|---|---|
| `00_ARRANQUE.md` | Este documento | Siempre, al arrancar |
| `01_METODO.md` | Metodo de patch, verificacion, PowerShell | Antes de tocar codigo |
| `01b_PATRONES.md` | Catalogo de patrones de bug + falsos positivos | Al auditar un modulo |
| `01c_LECCIONES.md` | Lecciones acumuladas | Antes de refactor/auditoria |
| `02_ARQUITECTURA.md` | Mapa del sistema, contratos, decisiones | Para ubicar modulos |
| `03_IAE.md` | Subsistema IAE completo | Al trabajar en IAE |
| `04_HISTORICO.md` | Cronologia tematica de decisiones | Para "por que esta asi" |
| `05_BITACORA.md` | Sesiones recientes (se poda) | Al cerrar una sesion |
| `06_IAE_P65_P66.md` | Contrato P65 + P66 (L3 cruzada) | Al trabajar en P65/P66 |
| `07_RUNBOOK.md` | Procedimiento de los crons | Al verificar un cron |
| `08_AUDITORIA_SECTOR_REGIME.md` | Expediente auditoria sector_regime | Al tocar sector_regime |
| `ESTADO_SISTEMA.md` | Snapshot auto-generado (no editar) | Fuente autoritativa de tests, integridad, modulos IAE, mappings y evidencia |
| `docs/gobernanza/README.md` | Contrato del sistema de gobernanza documental | Al trabajar con items auditables o anclas temporales |
| `docs/gobernanza/REGISTRO.yaml` | Fuente canonica de items auditables | Al abrir/cerrar items o auditar estado |
| `docs/gobernanza/ANCLAS.yaml` | Catalogo central de anclas temporales | Al anadir fechas literales en codigo productivo |
| `docs/auditoria/radar/audits/INDEX_H1_H18.md` | Indice derivado de auditoria externa 15-oct | Vista derivada; no editar |

`docs/auditoria/wyckoff/` (35+ ficheros) contiene el expediente completo
del rediseno del modulo Wyckoff. Indice en `docs/auditoria/wyckoff/03_plan_migracion.md`.

---

## 7. CONFIRMACION

Al recibir este documento, responde exactamente:

> "Confirmado, contexto asimilado."

Estado del sistema que reconoces (breve: HEAD via `git log --oneline -1`, tests, Gate, version, fase activa).

Pregunta final: "Que hacemos?"

No empieces a proponer tareas sin antes confirmar la asimilacion completa.