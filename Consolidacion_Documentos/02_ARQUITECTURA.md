# 02 - ARQUITECTURA

**Referencia on-demand. Se consulta para ubicar modulos, entender contratos, o resolver "por que esta asi".**
**No se pega al arrancar. El arranque es 00_ARRANQUE.md.**

---

## 1. DIAGRAMA DE FLUJO

    OHLCV -> Regimenes -> Motores -> Indicadores -> Scores -> Reporte Markdown
                                                                   |
                                                        Validation Gate 10/10
                                                                   |
                                                        Side effects + IAE

Sistema determinista **dado un snapshot del input**, descriptivo, auditable. Sin ML predictivo. Sin optimizacion de parametros. Sin automatizacion de trading. Todos los outputs son diagnosticos. La revision retrospectiva de Yahoo (K-LSE-YAHOO-REVISION-01) puede alterar el snapshot; el determinismo es respecto al snapshot, no al proveedor.

---

## 2. ESTRUCTURA REAL DE DIRECTORIOS

Verificada el 2026-09-28 en la auditoria interna A0.

    D:\Macro_Sectorial
    +-- run.py                      (231 LOC)
    +-- config/                     (11 ficheros)
    |   +-- settings.py, tickers.py, weights.py, index_tickers.py, __init__.py
    |   +-- bme_ticker_map.csv, euronext_ticker_map.csv, xetra_ticker_map.csv
    |   +-- instrument_exclusions.csv, market_close_exceptions.csv, market_close_regular.csv
    +-- regimes/                    (8 ficheros)
    |   +-- financial_conditions.py, liquidity.py, macro_regime.py, sector_regime.py
    |   +-- structural_engine.py, tactical_engine.py, volatility_regime.py, __init__.py
    +-- indicators/                 (50 ficheros)
    |   +-- breadth.py, breadth_core.py, breadth_equity.py
    |   +-- commodity_market_correlation.py, credit.py
    |   +-- cross_asset.py, cross_asset_context.py
    |   +-- darkpool.py, darkpool_history.py, darkpool_io.py, darkpool_scoring.py
    |   +-- data_quality.py, evidence_matrix.py, fls.py
    |   +-- index_leaders.py, index_phase.py
    |   +-- leader_representativeness.py, macro_fundamental.py, momentum.py
    |   +-- options.py, options_metrics.py, persistence.py
    |   +-- price_flow_divergence.py, rs_internal.py
    |   +-- sector_breadth.py, sector_breadth_momentum.py, sector_concentration.py
    |   +-- sector_correlation.py, sector_dispersion.py, sector_flow_characteristics.py
    |   +-- sector_leader_divergence.py, sector_rank_history.py, sector_regime_matrix.py
    |   +-- sector_wyckoff_distribution.py, signal_agreement.py
    |   +-- slpm_v12.py, state_machine.py, state_transition.py
    |   +-- stock_leader.py, trend.py
    |   +-- volatility.py, volatility_structure.py, vol_metrics.py
    |   +-- wyckoff.py      (legacy v4.2, sin consumidores productivos desde 2026-10-08)
    |   +-- wyckoff_v1.py   (v1.8 core, productivo desde 2026-10-08)
    |   +-- __init__.py
    |   +-- mte/ (paquete: __init__ + engine + state + scoring + decision)
    +-- src/
    |   +-- cboe_merge.py, commodities_merge.py
    |   +-- data_loader.py, dependency_tracker.py
    |   +-- effective_date.py, european_coverage.py
    |   +-- instrument_registry.py, macro_manual_loader.py
    |   +-- market_calendar.py, market_hours.py
    |   +-- report_generator.py, stock_data_loader.py, utils.py, __init__.py
    |   +-- external/               (2 ficheros: lse_scraper_loader.py + __init__.py)
    |   +-- report/                 (20 ficheros)
    |   |   +-- helpers.py, header.py, freshness.py, alerts.py, breadth.py
    |   |   +-- sectorial.py, leaders.py, rankings.py, slpm.py, sentiment.py
    |   |   +-- etf_flows.py, market_context.py, sector_context.py
    |   |   +-- flows_international.py, volatility_mte.py, confirmation.py
    |   |   +-- darkpool.py, synthesis.py, iae.py, __init__.py
    |   +-- pipeline/               (18 ficheros: 17 modulos + __init__)
    |   |   +-- data_load.py, regimes.py, sectors_base.py
    |   |   +-- flows_primary.py, flows_secondary.py
    |   |   +-- leaders.py, sector_metrics.py, breadth_metrics.py
    |   |   +-- engines.py, slpm.py, diagnostics.py, market_data.py
    |   |   +-- mte_confirmation.py, indices_intl.py
    |   |   +-- validation_gate.py, finalize.py, iae_section.py
    |   +-- temporal_contracts/     (12 ficheros: 10 contratos + base + consolidate)
    |   |   +-- base.py, registry.py, _common.py, consolidate.py, __init__.py
    |   |   +-- equity_eod.py, index_eod.py, volatility_index.py, rate_yield.py
    |   |   +-- future_settlement.py, fx_daily_cut.py, spot_commodity.py
    |   +-- institutional_accumulation/   (36 ficheros, 8280 LOC)
    |       +-- aggregation/, identity/, sec_13f/ (subpaquetes)
    |       +-- absence.py, catalog_pit.py, operational_universe.py
    |       +-- pipeline_contractual.py, security_type.py
    |       +-- temporal_validity.py, timestamps.py
    +-- data/
    |   +-- providers/              (26 ficheros)
    |   +-- macro_manual/           (12 CSVs FRED)
    |   +-- mappings/               (catalogos + snapshots + crosswalk)
    |   +-- sec_13f/                (raw, processed, official_list, manifests)
    |   +-- external/               (lse_close/)
    |   +-- cache/                  (finra, amundi, blackrock)
    |   +-- market_data.parquet (gitignored) + manifest
    |   +-- stock_prices.parquet (gitignored) + manifest
    |   +-- commodities_spot.parquet + manifest (tracked)
    |   +-- commodities_futures.parquet + manifest (tracked)
    |   +-- cboe_vix3m.parquet + manifest (tracked)
    |   +-- etf_holdings.csv, index_holdings.csv
    |   +-- lse_close_provenance.json
    +-- scripts/                    (34 activos + 3 en scripts/audit/)
    +-- validation/                 (7 ficheros)
    +-- tests/                      (182 ficheros locales, 192 tracked)
    +-- docs/
    |   +-- automatica/             (22 .md auto-generados)
    |   +-- auditoria/              (parcialmente conservada: iae/evidence + iae/golden en uso)
    +-- outputs/
    |   +-- history/                (versionado)
    |   +-- state/                  (versionado: mte_state.json, slpm_state.json, liquidity_state.json)
    |   +-- report/                 (NO versionado)
    |   +-- audit/                  (NO versionado)
    +-- .github/workflows/          (9 workflows)

**Nota de auditoria:** cifras verificadas en la auditoria A0 (2026-09-28). Cifra de workflows corregida 2026-09-30 (10 -> 9): se elimino `_cron_probe.yml` en A6.2-02, commit 4fbb135.
---

## 3. CAPAS DE FLUJO (SEPARADAS, NUNCA SE MEZCLAN)

| Capa | Frecuencia | Formula / Fuente |
|---|---|---|
| ETF_PRIMARY_FLOW | diaria | dSharesOutstanding * NAV |
| CFTC_POSITION_FLOW | semanal | CFTC TFF |
| SEC_POSITION_FLOW | trimestral | SEC EDGAR N-PORT |
| QQQ NPORT-P FLOW | trimestral | SEC N-PORT Item B.6 |
| FLOW_PROXY | diaria | 0.30*flow_smooth + 0.35*obv_z + 0.35*cmf_z |
| FLOW_SYNTHESIS | diaria | Concordancia de signos |

NUNCA se mezclan. NUNCA se construye superindicador.

---

### Nota: retirada del modulo Wyckoff legacy (2026-10-10)

El fichero `indicators/wyckoff.py` (legacy v4.2) fue retirado el 2026-10-10. Todos los consumidores migraron a `indicators/wyckoff_v1.py` (v1.8) en la migracion del 2026-10-07/08 (commits 2fd6c449, 35663000, f782b9d4). Los 5 CSVs historicos afectados por la mezcla legacy/v1.8 fueron regenerados entre 2026-10-08 y 2026-10-09.

Tests de regresion I2 / K-INDEX-RANGE-01 portados a `test_wyckoff_v1_contract.py` (test_regresion_i2_nan_internos_no_degenera). El script `compare_wyckoff_legacy_v18.py` (comparativa legacy vs v1.8) fue retirado junto al legacy: sin legacy no tiene objeto.

El contrato v1.8 esta en `docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md`. La secuencia completa (5a-5e) en `docs/auditoria/wyckoff/03_plan_migracion.md`.

## 4. CONTRATO DE MODULOS

- `src/report/*.py`: funciones `render_*(...) -> list[str]`. Sin side effects. Devuelve lista de lineas markdown.
- `src/pipeline/*.py`: funciones `compute_*(...) -> dict`. Encapsulan logica + side effects (escritura a disco).
- `indicators/*.py`: funciones puras. NO deben usar `datetime.now()` como fecha de observacion.

**Excepciones legitimas al prefijo `compute_*` en pipeline:** `load_all_data`, `run_validation_gate`, `save_regime_history`, `save_sector_rankings`, `generate_european_coverage`.

**Fuentes unicas declaradas:**
- `src/market_calendar.py` es la fuente unica de utilidades temporales NYSE.
- `src/market_hours.py` es la fuente unica de horarios de cierre y elegibilidad EOD.
- `src/effective_date.py::resolve_effective_date` es un resolutor por cobertura (no consulta calendario).
- `src/temporal_contracts/` es la fuente unica de contratos temporales (10 contratos, 5 familias). FSM: PENDING | OK | STALE | INSUFFICIENT | BLOCKED.
- `src/instrument_registry.py` expone `get_market` (calendario) y `get_instrument_class` (clase economica). Disjuntos. Fuente unica de `YAHOO_TICKER_MAP` + `normalize_yahoo_ticker(t)`.
- `src/utils.py::write_artifact_with_manifest` es la fuente unica de escritura de parquet + manifest.
- `src/utils.py::robust_zscore` es el canonico para precios / RS (window=60, ffill(limit=3), clip +-5).
- `data/providers/_fund_flow_utils.py::fund_flow_robust_zscore_with_regime` es el canonico para fund-flow (window=120, min_periods=20, clip +-5, MAD==0 & s==median -> 0.0, MAD==0 & s!=median -> NaN). NO fusionar con el canonico de precios.
- `src/institutional_accumulation/` es el modulo IAE. Detalle en `03_IAE.md`.
- `config/tickers.py::MARKET_TICKERS['sectors']` es la fuente unica de los 11 ETFs sectoriales (`['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']`). Cualquier modulo que necesite la lista, la importa - no la duplica.

---

## 5. MODELO TEMPORAL

Tres capas independientes que NO deben confundirse:

**FU-018 - Elegibilidad EOD:** `is_session_closed(market, session_date, reference_date)`. Determina si una vela intradia puede considerarse EOD. Independiente de la cobertura.

**FU-020 - Resolucion por cobertura:** `resolve_effective_date(prices, eligible_tickers, min_coverage)`. Devuelve la fecha mas reciente cuya cobertura alcanza el minimo. NO consulta calendario. Independiente de FU-018.

**FU-021-5 - Contratos temporales:** 9 contratos en 5 familias (EQUITY_EOD, INDEX_EOD_*, VOLATILITY_INDEX, RATE_YIELD, FUTURE_SETTLEMENT, FX_DAILY_CUT). SPOT_COMMODITY absorbido en FUTURE_SETTLEMENT el 2026-09-30 al eliminar OilPriceAPI. Cada uno declara `max_lag_days`, `min_coverage`, `session_calendar`, `settlement_semantics`. `build_temporal_meta` consolida los 9 en un dict. `temporal_meta` es la autoridad; `df.attrs['temporal_meta']` es solo espejo auxiliar.

**Reglas asociadas:**
- **R1:** toda metrica agregada declara fecha efectiva + cobertura del universo elegible.
- **R2:** nunca se selecciona fecha por posicion fisica (`.iloc[-1]`). Siempre `resolve_effective_date`.
- **R3:** metricas derivadas del mismo universo comparten resolucion.
- **R4:** filtro de sesion y resolucion por cobertura son controles independientes, no intercambiables.
---

## 6. REGIMENES Y SCORES

**Financial Conditions (4 componentes):** VIX (invertido), HYG/LQD (ratio), DXY (invertido), curva TNX-FVX. Pesos en `config/settings.py`. Confidence = clip(1 - (max-min)/2, 0, 1). Ventana 60.

**Liquidity Score (5 componentes):** SOFR, WALCL, RRP, Discount Rate, Commercial Paper. Confidence por std de senales.

**Volatility Regime:** VIX (70%) + termino VIX3M-VIX (30%).

**Macro Regime (12 senales):** Critical (0.60) + Important (0.30) + Contextual (0.10).

**Sector Regime:** `0.25*rs_mom_20 + 0.15*rs_mom_50 + 0.10*rs_mom_126 + 0.15*trend + 0.15*vol_inv + 0.20*breadth`. Aplica `SECTOR_DISPERSION_PENALTY=0.5` (dispersion = std(ddof=0) de componentes en [-1,+1] -> penalty en [0.5,1]) y renormaliza por `valid_weight_sum`. **Umbral publicable (D2, 2026-09-30): `n_valid >= 4` -> score NaN si menos.** Con 4 componentes, peso minimo representado 0.55. Detalle en `08_AUDITORIA_SECTOR_REGIME.md`.

**Tactical Score (5 componentes):** RS20 / Flow / Mom20 / Breadth20 / Aceleracion. Pesos en `config/weights.py::TACTICAL_WEIGHTS`.

**Structural Score (3 componentes):** RS multi-ventana / Flow structure / Persistence. Pesos en `STRUCTURAL_WEIGHTS`.

**SLPM v1.2:** Leader Breadth + LIS + Effective Breadth. State Machine: CONFIRMED | EMERGING | STRUCTURAL_DECAY | LOST | UNRESOLVED | TACTICAL_CORRECTION (declarado pero inalcanzable, WONT FIX razonado A3.3-05). Con n=0, los campos se muestran como N/D.

**Robust Z-Score canonico (precios/RS):**

    def robust_zscore(series, window=60, min_periods=None):
        if min_periods is None:
            min_periods = max(window // 3, 10)
        s = series.ffill(limit=3)
        median = s.rolling(window, min_periods=min_periods).median()
        mad = (s - median).abs().rolling(window, min_periods=min_periods).median()
        z = (s - median) / (1.4826 * mad + 1e-9)
        return z.clip(-5, 5)

**Flow Proxy:** `0.30*flow_smooth + 0.35*obv_z + 0.35*cmf_z`, con `obv.diff()`.

---

## 7. PIPELINE (12 FASES + IAE)

Orden ejecutado por `run.py::main()`:

| Fase | Modulo | Que hace |
|---|---|---|
| 0 | run.py | `reference_date` (Europe/Madrid, tz-aware) + `run_id` (uno por ejecucion) |
| 1 | pipeline/data_load.py | Descarga + validacion + macro manual |
| 2 | pipeline/regimes.py | 4 regimenes (financial, liquidity, volatility, macro) |
| 3 | pipeline/sectors_base.py | Rankings sectoriales + rotacion + dispersion + correlacion + cross-asset + breadth |
| 4 | pipeline/flows_primary.py | SSGA + BlackRock (DAXEX, ISF, IWM) + Amundi + QQQ SEC + CFTC |
| 5 | pipeline/flows_secondary.py | Sintesis + N-PORT + QQQ Yahoo + QQQ NPORT-P |
| 6a | pipeline/leaders.py | df_stocks + lideres sectoriales |
| 6b | pipeline/sector_metrics.py | Divergencia + Wyckoff + RS interno + concentracion + representatividad |
| 6c | pipeline/breadth_metrics.py | Sector Breadth & Health + Momentum amplitud |
| 8a | pipeline/engines.py | Forzar lideres SLPM + tactical/structural + persistence |
| 8b | pipeline/slpm.py | SLPM v1.2 State Machine |
| 9a | pipeline/diagnostics.py | Directional Agreement + Price-Flow + Shock Sensitivity |
| 9b | pipeline/market_data.py | PCR + Dark Pools + Volatilidad estructural + Calidad datos |
| 10a | pipeline/mte_confirmation.py | MTE + Cross-Module Conflict + Confirmation Data |
| 10b | pipeline/indices_intl.py | Indices internacionales (fases Wyckoff + lideres) |
| 11 | pipeline/validation_gate.py | Validation Gate 10/10 + audit double-counting |
| 12 | pipeline/finalize.py | Matrices finales + side effects + cobertura europea |
| IAE | pipeline/iae_section.py | NIPC contractual (13F) |

`reference_date` y `run_id` se resuelven UNA vez en `main()` y viajan como parametros.

Si `validation_gate['passed'] == False` -> `sys.exit(1)`.

**Convencion de serializacion temporal:** toda fecha de observacion se deriva del dataset. `datetime.now()` solo se usa como (a) `reference_date` en main; (b) execution time para definir rangos de query HTTP; (c) edad de ficheros de cache. Nunca como fecha de observacion.
---

## 8. WORKFLOWS GITHUB ACTIONS

| Workflow | Cron | Proposito |
|---|---|---|
| daily_run.yml | 17 23 / 17 3 / 17 5 / 17 7 / 17 11 UTC (5 slots) | Run multi-slot con gate pre-pipeline (F-IAE-CRON-02) + validacion + push de outputs |
| update_macro_manual.yml | 17 6 * * * | FRED auto |
| update_european_holdings.yml | 17 5 1 1,4,7,10 * | Holdings europeos |
| update_index_holdings.yml | 17 4 1 1,4,7,10 * | SPY/DIA/QQQ/IWM |
| update_qqq_sec_flow.yml | 17 6 15 1,7 * | QQQ SEC flow |
| update_sec_nport.yml | 17 6 20 1,4,7,10 * | N-PORT |
| update_sec_13f.yml | 17 6 20 2,5,8,11 * | SEC 13F trimestral + cache IAE |
| update_sector_holdings.yml | 47 2 1 1,4,7,10 * | Holdings sectoriales |
| health_check.yml | 47 6 * * 1 | Vigilancia semanal (7 bloques) |

**Notas:**
- `daily_run.yml` aplica `git fetch + pull --rebase` antes de cualquier push local.
- Multi-slot (4 disparos) con idempotencia por cobertura y gate pre-pipeline (`scripts/pipeline_gate.py`).
- Issue-manager (`scripts/issue_manager.py`) abre/comenta/cierra Issues si los 4 slots fallan.
- F-IAE-LSE-INTEGRATION: 2 steps antes de "Run Macro Sectorial" (fetch LSE scraper + capture SHA).
- Fase G: step "Regenerar catalogo radar (IAE)" tras run.py; `update_sec_13f.yml` gana step "Regenerar crosswalk CUSIP".

**Clasificacion de workflows por validacion local (2026-10-08):**

- **Nucleo (`daily_run.yml`)**: `pytest tests/ validation/` + `run.py` + 5 validaciones post-run (cross-validation, freshness, history quality, double counting, guard coverage). Cumple "no push sin validacion local completa".
- **Datos externos (5)**: `update_macro_manual`, `update_european_holdings`, `update_index_holdings`, `update_sector_holdings`, `update_sec_13f`. Scope acotado a `data/`. Unica validacion: compileall. Excepcion razonada: no dependen de outputs del radar ni afectan determinismo.
- **Outputs derivados (2)**: `update_qqq_sec_flow`, `update_sec_nport`. Escriben en `outputs/history/`. Validacion: compileall + tests contractuales de funciones puras (`tests/test_nport_sec_helpers.py`, `tests/test_provider_nport.py`) + los tests que hacen parseo XML/TSV. Queda fuera el HTTP puro (descarga SEC) por ROI < 1. Excepcion de gate local mantenida.
- **Lectura (`health_check.yml`)**: sin push.

---

## 9. VALIDACION Y TESTS

**Suite completa:** 2340 passed + 2 skipped + 0 failed. Incluye 777 funciones test_ (AST) / 885 tests collected (pytest) del modulo IAE (45 ficheros).

**Los 3 tests `test_freshness`** (ambientales) pasan tras un `py run.py` que refresca los parquets; vuelven a fallar si pasan >4 dias sin ejecutar el pipeline.

**Validation Gate (10/10):**
1. SLPM v1.2 (sin errores de validacion).
2. PCR Total (no NaN).
3. Dark Pool medio (no NaN).
4. MTE MSI/IPI (no NaN).
5. Rangos Tactical/Structural en [-1, +1].
6. Opportunity Map consistente.
7. Freshness Dark Pool.
8. Freshness PCR.
9. Config pesos (`validate_weights()`).
10. Anti-Double-Counting (LIS fuera de State Machine).

Si el gate falla, `run.py` aborta con `sys.exit(1)`.
---

## 10. DECISIONES ARQUITECTONICAS VIGENTES

**Generales:**
- Europa primero. Cascada europea con Yahoo como fallback solo si el europeo falla.
- Opcion A estricta: sin fallback Yahoo si falla europeo.
- Pipeline lideres 15 -> 5.
- Confianza por rango: `1 - (max - min) / 2`.
- `[BAJA]` informativa si cobertura < 70%.
- Sin superindicadores.
- Sin datos ficticios: `N/D` antes que imputar.
- BOM: leer con `utf-8-sig` cuando aplica.
- Local-first refactors.
- Deteccion por contenido > por indices.
- `compute_*` en pipeline, `render_*` en report.

**Integridad de sesion (C1):**
- `ffill(limit=3)` NO imputa sobre sesion NYSE.
- `_fill_holes_respecting_sessions(df, reference_date)`.
- `_log_yahoo_raw_diagnostics(df_raw, reference_date, batch_label)`.

**Temporalidad de breadth (C2):**
- Fecha de observacion != `Timestamp.now()`.
- `compute_sector_breadth(..., as_of_date=None)`: `as_of_date` explicito debe ser sesion NYSE; `None` -> `df_stocks.index[-1]`.
- Preservacion de ultima observacion valida (`_load_latest_valid_breadth_snapshot`).

**Presentacion A/D (C3):**
- `_fmt_ad_net(advances, declines, ad_net, fmt)`:
  - Invalidos -> `N/D`.
  - `advances + declines == 0` -> `N/D`.
  - `ad_net == 0` con `advances = declines > 0` -> `+0`.

**Writers historicos (C4-code):**
- `_observation_date_from_df(df, col=None)`: `None`/vacio -> `None`; index no-DatetimeIndex -> `None`.
- Writers con `reference_date` obligatorio: `compute_sector_dispersion`, `compute_sector_concentration`, `compute_leader_representativeness`.
- `save_regime_history(...)`: candado `is_market_day(obs_date)`.

**Saneamiento historico (C4-data):**
- Doble candado: `is_market_day(date) == False AND date in CONFIRMED_B2_DATES`.
- 6 fechas B2: 2026-05-25, 06-19, 07-03, 09-06, 09-07, 09-12.

**FU-002 - Manifest de artefacto:**
- Todo parquet producido por el pipeline lleva `<parquet>.manifest.json` con schema_version, artifact, producer, content, quality.
- `quality.status`: `INVALID` si `temporal_contract` declarado Y `last_date > expected_session`; `INVALID` si `pct_dup_last > 0.5`; `VALID_WITH_MISSING` si `close_nan_last > 0`; `VALID` en otro caso.
- Escritura atomica: `.tmp.<run_id>` + `os.replace`.
- FSM del reader: `VALID` / `INVALID` / `UNAVAILABLE`. Combinacion conservadora.
- Politica de schema: campos adicionales son backward-compatible. `schema_version` solo se incrementa ante cambio incompatible (eliminacion de campo, cambio de tipo, cambio de semantica).

**FU-021-3A - Filtro EOD en market_data:**
- `_filter_non_eod_equity` aplica FU-018 solo al universo EQUITY_EOD (539 tickers) antes de `resolve_effective_date`.
- `function_lag` = `resolve_effective_date.lag_days`. Puede ser 0 tras filtro EOD.
- `reference_lag` = `(reference_date.date() - effective_date.date()).days`.

**FU-021-5 - Contratos temporales de df_market:**
- 10 contratos, 5 familias, FSM 5 estados.
- `temporal_meta` es autoridad; `df.attrs` es espejo auxiliar.
- `MarketDataBundle` es el transporte.
- `global_last_date` = max(effective_date) de contratos OK/STALE. Nunca sustituye fechas por contrato.

**FU-021-3C-bis - Commodities:**
- SPOT (GC=F, HG=F, NG=F) via OilPriceAPI. `settlement_semantics=spot_reference`.
- FUTURE_SETTLEMENT (BZ=F, CL=F) BLOCKED de facto (403 "Feature access required" desde 2026-09-22).
**F-IAE-CRON-02 - Multi-slot:**
- 4 slots de cron (17 23 / 17 3 / 17 7 / 17 11 UTC) con gate pre-pipeline.
- Idempotencia por cobertura (no por `last_date`).
- `target_session` = sesion bursatil objetivo.
- Retry corto (3 intentos, sleeps 10/30s) para errores de red.
- `concurrency` con `queue: max`.

**F-IAE-LSE-INTEGRATION:**
- 20 tickers .L via scraper privado `lse-close-scraper` (Refinitiv Widgets).
- Override PARCIAL de Close sobre fila ya presente de Yahoo. NO sustituye OHLCV.
- Provenance en `data/lse_close_provenance.json` (versionado).
- El radar NO ejecuta codigo del repo externo, solo lee sus JSON.

**A5-11 - Constante SECTOR_ETFS unificada:**
- `config/tickers.py::MARKET_TICKERS['sectors']` es la fuente unica (ver Seccion 4).

---

## 11. LIMITACIONES CONOCIDAS ACTIVAS

**N-PORT con retraso SEC (~60d):** aceptado por diseno.

**Dark Pool con retraso FINRA (2-4 sem):** marcado ARCHIVAL si `age > 14d` (excluido de MTE).

**Volumen Yahoo no consolidado intra-dia (D7, WONT FIX):** Yahoo publica el volumen del ultimo dia a ritmos distintos por ticker. Una consulta a las 05:59 UTC puede devolver 2632 para NG=F y otra a las 12:00 devuelve 212563 para el mismo dia. Afecta a `OBV.diff()` y `CMF(20)` -> componente del Flow Proxy (65% del peso). **No es bug del pipeline**, es limite de la fuente. Mitigacion: el reporte se ejecuta con `reference_date` derivado del dataset y los candidatos de flujo son ventanas largas (60 sesiones), donde la ultima fila pesa poco. Fix real requeriria consultar Yahoo a hora fija (posterior a consolidacion), fuera del alcance del sistema (depende de proveedor). Verificado 2026-09-30.

**~31 tickers con DATA ISSUE:** esperado (IPOs recientes).

**Confidence sensible a N componentes:** documentado (C19).




**H8 - NaN convertido en senal bajista (CERRADO 2026-09-30):** cuando `close_sector` es NaN, `trend_position` y `breadth` devolvian `-1.0` (senal bajista maxima fabricada). 1454 filas historicas afectadas. Cuando solo quedaban esos 2 componentes, el sistema producia `score=-1` + `penalty=1`. Fix: `.mask(close.isna())` en `trend_position` (commit `ee24afc`); mismo tratamiento inline para `breadth`. Contrato: `missing observation != bearish observation`. Detalle en `08_AUDITORIA_SECTOR_REGIME.md`.

**Cierres excepcionales NYSE (CERRADO 2026-09-30):** el calendario algoritmico no reconoce cierres no recurrentes. Anadidos `2018-12-05` (Bush) y `2025-01-09` (Carter) a `_EXCEPTIONAL_CLOSURES` en `src/market_calendar.py`. Commit `43a2fdb`. Solo cierres posteriores al inicio del parquet (2016-09-29).

**Fix D - Perdida de filas en historico de flujo (CERRADO 2026-09-30):** 5 providers (`ssga_fund_data`, `amundi_fund_data`, `_blackrock_base`, `qqq_nport_flow`, `cftc_data`) escribian su historico sobrescribiendo el CSV sin leer el existente. Si el proveedor o su cache devolvia menos filas, se perdian permanentemente. Fix: `append_dedup(hist, nuevo, key)` antes de escribir. Commit `c798857`. Fix D2 (`7b0044d`): normalizar `Date` en ssga (usa `Date` mayuscula, `append_dedup` normaliza solo `date` minuscula). Test: `test_flows_primary_preserve_history.py`.

**K-LSE-YAHOO-REVISION-01 (WATCHED con check activo):** Yahoo revisa OHLC historico retrospectivamente. Afecta al WLS (dependiente de ventanas largas). No es bug: el determinismo del sistema es respecto al snapshot del input, no al proveedor (00_ARRANQUE §2, D2 2026-09-30). Detectado via `check_yahoo_revision` (health_check): compara manifest actual vs HEAD por sha256.

**`as_of_date` en `qqq_returns_yahoo.csv` es timestamp, no fecha del dato (WONT FIX razonado):** La columna mezcla `latest_date` (dataset-derived) con `datetime.now()` (reloj de ejecucion, sin tz). El nombre es enganoso. Verificado 2026-09-30: unico consumidor (`flows_international.py:152-159`) prefiere `effectiveDate` (dataset-derived); el fallback trunca `as_of_date` con `split(" ")[0]`, por lo que la hora nunca se publica. La parte publicada es `latest_date`, dataset-derived. No viola regla dura 00_ARRANQUE 2: el campo es log. WONT FIX: renombrar a `run_timestamp` seria correcto pero ROI < 1 (regenerar CSV, actualizar test, sin cambio funcional).

**Fund-flow: descarga siempre, cache solo fallback (CERRADO 2026-10-01):** los 5 writers de fund flow (`ssga_fund_data`, `amundi_fund_data`, `_blackrock_base`, `blackrock_iwm_fund_data`, `cftc_data`) usaban `mtime < 23h -> cache-hit`. Si la fuente publica despues del run, el siguiente run acepta cache de 22h y no refresca. Fix: descarga siempre, cache como fallback si la descarga falla (WARN con mtime). Commit `b30a918`. Coste medido: 13s los 5 providers.

**Cache-hit valida cobertura, no solo fecha (CERRADO 2026-10-01):** `stock_data_loader.py` y `data_loader.py` aceptaban cache si `_df_last >= _last_exp`, sin mirar cobertura de la ultima fila. Un parquet del 30-sep con 89.8% de cobertura se aceptaba. El cascade europeo no se ejecutaba. Fix: helper `_last_row_coverage_ok` en `src/utils.py`, gate 90%. Commit `c994f01`.

**Cache freshness europea unificada (CERRADO 2026-10-01):** Euronext y BME usaban `(ref - last).days <= 1`. Con ref=30-sep y cache=29-sep, "fresco". 13 Euronext + 19 BME no refrescaban. Fix: helper `_cache_freshness.py::european_cache_is_fresh`. Euronext, BME y Xetra (refactor) lo usan. Commit `22cbcb5`.

**LSE scraper (dependencia externa, best-effort):** el repo `bledabladis-png/lse-close-scraper` publica 20 tickers LSE diarios con cron propio. `daily_run.yml:139` lo clona fresco en cada run (CI). Localmente el checkout puede quedar congelado si no se refresca manualmente. No es bug del radar. Si el scraper no corre, esos 20 tickers quedan sin observacion hasta el siguiente run (Yahoo no cubre `.L` con close real, solo ADR).
**KHC / lote parcial:** deteccion anadida en `download_market_data`. Monitorizacion activa, sin retry.

**H5.3 - Trazabilidad cron trimestral:** implementada, pendiente verificacion en cron de noviembre 2026.

**C2 (920) - Discrepancia H1-B:** `-4.264.449.012` (local reproducible) vs `-4.264.449.932` (referencia auditor). `reconciliation_status = OPEN`. **Par contractual: Q4-2025 -> Q1-2026** (congelado en `current.json`). El reporte diario publica el NIPC del par vigente (auto-avance a los 2 ultimos quarters disponibles en `data/sec_13f/processed/`). Cuando entra un trimestre nuevo, el numero del reporte cambia: no es comparable con este baseline, ni con C2. Ejemplo verificado 2026-09-29: con Q1-2026 -> Q2-2026 el NIPC total es `8256882557`, reproducible.

---

## 12. PROVIDERS (RESUMEN)

**Yahoo:** 262 tickers (equity + indices + FX + rates). Fuente unica de equity USA.

**Euronext (13 .PA/.AS/.MI):** AES-256-CBC + EVP_BytesToKey + MD5.

**Xetra (19 .DE):** WebSocket MDS + JWT (~175s).

**BME (19 .MC):** REST JSON sin auth.

**LSE (20 .L):** scraper privado (Refinitiv Widgets) + override parcial de Close.

**OilPriceAPI:** 3 commodities spot (GC=F, HG=F, NG=F). Budget 3 requests/dia.

**CBOE:** opciones diarias (PCR) + indice ^VIX3M via CSV publico.

**CFTC:** TFF semanal.

**FINRA:** Dark Pools ATS semanal.

**FRED:** liquidez y yields.

**Fund data:** SSGA (SPDR), BlackRock (DAXEX, ISF.L, IWM), Amundi (LYXI).

**SEC:** N-PORT (trimestral), 13F (trimestral), QQQ SEC flow.

---

## 13. COMANDOS DE VERIFICACION

    # Estado repo
    git status -sb
    git log --oneline -10

    # Validacion
    py -m compileall . -q
    py -m pyflakes src scripts regimes indicators config data\providers validation
    py -m pytest tests/ validation/ -q --tb=short

    # Run completo
    py run.py

    # Estado del sistema (regenerar snapshot)
    py scripts/generate_estado_sistema.py

    # Manifest de artefactos
    Get-ChildItem data\*.manifest.json | Select-Object Name, LastWriteTime

---

## 14. ALCANCE DEL SISTEMA

**Incluye:** regimenes macro y sectoriales, scores tacticos y estructurales, SLPM v1.2, MTE, dark pools, opciones, flows primarios y secundarios, IAE (institucional 13F), cobertura europea, cobertura internacional (indices).

**No incluye:** trading, recomendaciones, timing, ML predictivo, optimizacion de parametros, construccion de superindicadores.

---

## 15. AUDITORIA INTERNA (estado)

| Bloque | Superficie | Estado |
|---|---|---|
| A1 | nucleo temporal | CERRADO |
| A2 | providers | CERRADO |
| A3 | calculo | CERRADO |
| A4 | reporte | CERRADO |
| A5 | pipeline | CERRADO |
| A6 | utils / registry / tracker / workflows / scripts / validation | CERRADO |
| B | IAE | CERRADO |
| C | tests | CERRADO |
| F6 | IAE funcional avanzado (security_identity, period_state, catalog_key, operational_universe, catalog_p38_adapter) | CERRADO |
| F8 | providers (yahoo, router, fred, polygon, cftc, finra, backup, downloader, xetra, bme, euronext, blackrock, fund_flow_utils, nport) | CERRADO |
| F7 | indicators/ (50 ficheros, 6684 LOC) | CERRADO |


---

## 16. UNIVERSOS DE TICKERS

Cuatro cifras distintas de "universo" conviven en el sistema. Cada una responde a un contrato diferente. No son contradictorias: son vistas distintas de la misma cadena.

| Universo | Tamano | Fuente | Consumer |
|---|---:|---|---|
| `get_usa_tickers()` | 220 | `etf_holdings.csv`, top-20 por weight y sector | metricas intra-sectoriales (SLPM, breadth, stock_leader) |
| `_ticker_list()` (`data_loader`) | 504 | `etf_holdings.csv` completo + `MARKET_TICKERS` | descarga. Despues se filtra con `_is_equity_ticker` al leer Close. |
| `darkpool_io` | 504 + `MARKET_TICKERS` | `etf_holdings.csv` + `config/tickers.py` | concentracion ATS. Reporte declara `n=534` explicito. |
| `get_stock_list()` | 329 | 4 fuentes: top-20 sector + top-20 indices + `radar_target_catalog` + europeos de providers | descarga completa del pipeline |
| Parquet `stock_prices` USA | 255 | filtrado `_is_equity_ticker` sobre la descarga | metricas del radar |
| Parquet `stock_prices` total | 329 | descarga completa (USA + EU) | analisis cross-market |
| `radar_target_catalog.csv` | 255 | OpenFIGI + curacion | IAE crosswalk CUSIP-ticker |

**Relacion:** 220 subset 255 subset 329. El CSV fuente (`etf_holdings.csv`) tiene 504; cada consumer corta (o no) segun su contrato.

**De donde vienen los 35 tickers entre 220 y 255:** son componentes del top-20 de indices USA (Russell 2000 sobre todo) que no caen en el top-20 de su sector ETF. Estan en `index_holdings.csv` y en `radar_target_catalog.csv`. Ejemplos: BX, CASY, CMG, CRM, ES, FOX, GD, JXN, PNC, PRAX, WY.

**Regla derivada:** un consumer que lee `etf_holdings.csv` o `index_holdings.csv` directamente debe declarar en su docstring si corta a top-N o usa el pool completo. Un cambio silencioso de uno a otro altera el universo de la metrica sin senal.

Verificado 2026-10-08 (sesion de verificacion del sistema). Sin bug. Ningun huerfano. Cifras cuadran entre manifest, parquet y catalogo.
