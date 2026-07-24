# ÍNDICE TÉCNICO – RADAR DE ROTACIÓN SECTORIAL v3.15

**Actualizado:** 2026-07-24 (post Auditoría Maestra)
**Propósito:** Mapa completo de archivos, funciones y dependencias para mantenimiento.

---

## 1. NÚCLEO DEL SISTEMA (raíz)

| Archivo | Tipo | Descripción |
|--------|------|-------------|
| `run.py` | Orquestador | Punto de entrada principal. Ejecuta todos los módulos en orden y genera el reporte diario. |
| `config/settings.py` | Configuración | Ventanas temporales, caché, umbrales de calidad, freshness, SLPM_EXPECTED_LEADERS, MTE_STATE_FILE. |
| `config/tickers.py` | Configuración | Universo de activos (sectores, bonos, crédito, divisas, commodities, factores). Validación `validate_sector_universe()`. |
| `config/weights.py` | Configuración | Pesos de todos los scores (Macro, Sector, SLPM, Tactical, Structural). `SECTOR_DISPERSION_PENALTY`. `validate_weights()`. |
| `requirements.txt` | Dependencias | Librerías Python requeridas. |
| `INDICE_TECNICO_v3.15.md` | Documentación | Este archivo. |

---

## 2. REGÍMENES (`regimes/`)

| Archivo | Descripción |
|--------|-------------|
| `financial_conditions.py` | Score de condiciones financieras (VIX, crédito HYG/LQD, dólar DXY, curva 10Y-2Y). Pesos: 0.40/0.30/0.15/0.15. Signo crédito corregido (+tanh). |
| `liquidity.py` | Liquidez real (WALCL, SOFR, RRPONTSYD). Media de señales disponibles. |
| `volatility_regime.py` | Régimen de volatilidad basado en VIX. Z-Score robusto sobre vol realizada 20d vs mediana 3Y. |
| `macro_regime.py` | Clasificación del régimen macro (11 categorías). Reglas de precedencia determinista. Delega en `scores/macro_scores.py`. |
| `sector_regime.py` | Ranking sectorial (RS momentum 20/50/126, Trend, Volatilidad, Breadth, Wyckoff). Penalización por dispersión (`SECTOR_DISPERSION_PENALTY`). |
| `tactical_engine.py` | Tactical Score (RS20 30%, Momentum20 25%, Flow 20%, Breadth20 15%, Aceleración 10%). |
| `structural_engine.py` | Structural Score (RS multi-ventana 35%, Leader Breadth 25%, Flow Structure 20%, Persistence 20%). |

---

## 3. INDICADORES (`indicators/`)

| Archivo | Descripción |
|--------|-------------|
| `momentum.py` | `compute_returns()`, `momentum_score()` (Sharpe-like 63d), `compute_flow_proxy()` (30% ret×vol + 35% OBV + 35% CMF), `compute_price_momentum()`. |
| `trend.py` | `trend_position()`: posición respecto a EMAs 20/50/100/200. Devuelve [-1, +1]. |
| `breadth.py` | Amplitud sectorial: % sectores sobre EMA20/50/200. NH/NL 52 semanas vía `breadth_core.py`. |
| `breadth_core.py` | Funciones compartidas: `compute_new_highs_lows()` con shift(1), `validate_coverage()`. |
| `breadth_equity.py` | Advance/Decline sobre acciones líderes. Optimizado con `pd.concat` (sin PerformanceWarning). |
| `persistence.py` | Prevalencia direccional: % observaciones > umbral en ventana lookback. Retorna `None` si datos insuficientes. |
| `signal_agreement.py` | Directional Agreement: % de señales que apuntan en la misma dirección (solo signo, no magnitud). |
| `price_flow_divergence.py` | Detección de divergencias precio vs Flow Proxy. Umbrales ±5% precio, ±0.10 flow. |
| `commodity_market_correlation.py` | Correlación sector-commodities/mercado (Pearson 126d). Contexto, no predicción. |
| `wyckoff.py` | `wyckoff_score()`, `classify_wyckoff_phase()`, `detect_spring()`, `detect_sos()`. |
| `volatility.py` | `volatility_regime()` (z-score robusto), `atr()`, `beta()`. |
| `vol_metrics.py` | Volatilidad realizada (RV21, RV60) y VRP Proxy (Implied-Realized Volatility Spread). |
| `cross_asset.py` | Ratios cross-asset con tendencia (SPY/TLT, HYG/LQD, etc.). |
| `credit.py` | Señales de crédito corporativo. |
| `macro_fundamental.py` | Señales macro fundamentales desde `data/macro_manual/`. |
| `fls.py` | Financial Liquidity Stress: indicador compuesto de estrés de liquidez. |
| `stock_leader.py` | Wyckoff Leadership Score (WLS): identifica líderes por sector (RS momentum 35%, Flow 30%, Wyckoff 25%, Persistencia 10%). |
| `slpm_v12.py` | SLPM v1.2 activo: auditoría estructural del liderazgo del sector #1. Incluye Leader Breadth 2.0, LIS, Flow Divergence 2.0, State Machine, histéresis. |
| `state_machine.py` | State Machine centralizada (6 estados jerárquicos + UNRESOLVED). `classify_leadership_state()` con `coverage` y `data_quality`. |
| `state_transition.py` | Histéresis temporal para SLPM. Requiere 2 ejecuciones para EMERGING↔CONFIRMED. |
| `structural_leadership.py` | SLPM v1.0 Legacy. Desactivado en `run.py` (conservado como referencia histórica). |
| `mte.py` | Market Transition Engine: infiere escenario macro (CRISIS, RECESSION, STAGFLATION, etc.) vía SRS, SHS, CLS, IPS. Guarda `mte_state.json`. |
| `options.py` | OMS v2.0: orquestación PCR. Z-Score robusto 252d con estados FULL/PARTIAL/INSUFFICIENT. |
| `options_metrics.py` | Métricas puras de opciones: PCRs, IHR, Put/Call Share. Validación `np.isfinite()`. |
| `darkpool.py` | Dark Pools: % volumen ATS vía FINRA. Z-Score con ventanas 13/26/52/104 semanas. Backfill incremental. |

---

## 4. CAPA DE DATOS (`src/`, `data/`)

| Archivo | Descripción |
|--------|-------------|
| `src/data_loader.py` | Descarga de datos de mercado (yahoo, stooq). Cache 23h. |
| `src/stock_data_loader.py` | Descarga de precios de acciones líderes. Cache 23h. |
| `src/macro_manual_loader.py` | Carga CSVs en `data/macro_manual/` y los une por fecha. |
| `src/report_generator.py` | Generación del reporte diario en Markdown (`outputs/reporte_diario.md`). |
| `src/utils.py` | `robust_zscore()`, `get_col()`, `tanh_normalize()`, `detect_cross_module_conflict()`. |
| `src/dependency_tracker.py` | Matriz Anti-Double-Counting: dependencias directas, indirectas y canal WLS→LIS. |
| `src/wyckoff_agreement.py` | Acuerdo Wyckoff en sección de líderes sectoriales. |
| `data/market_data.csv` | Datos OHLCV cacheados de todos los tickers del universo. |
| `data/stock_prices.csv` | Precios de acciones líderes (producción). |
| `data/stock_prices_historical.csv` | Histórico completo de acciones líderes para auditoría (descarga incremental). |
| `data/etf_holdings.csv` | Holdings de los 11 ETFs sectoriales. |
| `data/macro_manual/` | CSVs de datos macro: walcl, sofr, rrpp, inflacion, empleo, 10y3m, etc. |
| `data/providers/` | Proveedores de datos: yahoo, cboe, finra, fred, polygon, stooq, router. |

---

## 5. VALIDACIÓN (`validation/`)

| Archivo | Descripción |
|--------|-------------|
| `integration_check.py` | Verificación de integración entre módulos. |
| `macro_regime_validation.py` | Validación del régimen macro. |
| `sector_rankings_validation.py` | Validación de rankings sectoriales. |
| `breadth_validation.py` | Validación de amplitud. |
| `breadth_equity_validation.py` | Validación de breadth sobre acciones. |
| `momentum_flow_validation.py` | Validación de momentum y flow. |
| `wyckoff_validation.py` | Validación de puntuaciones Wyckoff. |
| `oms_darkpool_validation.py` | Validación de OMS y Dark Pools. |
| `mte_validation.py` | Validación del MTE. |
| `mte_validation_v11.py` | Validación MTE v1.1. |
| `slpm_validation_v11.py` | Validación SLPM v1.1. |
| `module_correlation.py` | Correlación entre módulos. |
| `feature_corr.py` | Correlación entre features. |
| `feature_importance.py` | Importancia de features. |
| `vif_analysis.py` | Análisis VIF (multicolinealidad). |
| `bootstrap_montecarlo.py` | Bootstrap y Montecarlo. |
| `rolling_walkforward.py` | Walkforward analysis. |
| `temporal_stability.py` | Estabilidad temporal de señales. |
| `transition_matrix.py` | Matriz de transición de regímenes. |
| `lead_lag.py` | Análisis lead-lag. |
| `event_analysis.py` | Análisis de eventos. |
| `information_coef.py` | Information Coefficient. |
| `deflated_sharpe.py` | Deflated Sharpe Ratio. |
| `purged_cv.py` | Purged Cross-Validation. |
| `jackknife.py` | Jackknife resampling. |
| `sensitivity_heatmap.py` | Heatmap de sensibilidad. |
| `sensitivity_noise.py` | Sensibilidad al ruido. |
| `calibration_explorer.py` | Explorador de calibración. |
| `cls_comparison.py` | Comparación CLS. |
| `data_audit.py` | Auditoría de datos. |
| `turnover.py` | Análisis de turnover. |
| `phase2_validate_new_assets.py` | Validación de nuevos activos (Fase 2). |

---

## 6. TESTS (`tests/`)

| Archivo | Descripción |
|--------|-------------|
| `test_regimes.py` | Tests de módulos de régimen. |
| `test_utils.py` | Tests de utilidades. |
| `test_validator.py` | Tests del validador de datos. |

---

## 7. OTROS ARCHIVOS

| Archivo | Descripción |
|--------|-------------|
| `.github/workflows/daily_run.yml` | GitHub Actions: ejecución diaria automática. |
| `setup_macro_data.py` | Descarga inicial de datos FRED. |
| `build_objective_benchmark.py` | Construcción de benchmark objetivo. |
| `backtest_v2.py` | Backtest histórico bajo demanda. |
| `audit_slpm_independence.py` | Auditoría de independencia SLPM vs SRS. |
| `update_stock_history.py` | Descarga incremental del histórico de acciones para auditoría. |

---

## 8. MODIFICACIONES DE LA AUDITORÍA MAESTRA (24/07/2026)

| Cambio | Archivo(s) |
|--------|------------|
| `SECTOR_DISPERSION_PENALTY` externalizado | `config/weights.py`, `regimes/sector_regime.py` |
| Cobertura SLPM corregida (n/expected_leaders) | `indicators/slpm_v12.py`, `config/settings.py` |
| "Breadth vs Price" → "Sector Flow vs Price" | `indicators/slpm_v12.py`, `src/report_generator.py` |
| State Machine + coverage + data_quality | `indicators/state_machine.py`, `indicators/slpm_v12.py` |
| `reason_code` en State Machine | `indicators/state_machine.py` |
| Histéresis temporal (nuevo módulo) | `indicators/state_transition.py` |
| Escritura automática `mte_state.json` | `indicators/mte.py`, `config/settings.py` |
| SLPM legacy desactivado | `run.py` |
| Mensaje Dark Pool mejorado | `indicators/darkpool.py` |
| Optimización `breadth_equity.py` (pd.concat) | `indicators/breadth_equity.py` |
| Anti-Double-Counting: WLS→LIS documentado | `src/dependency_tracker.py` |
| Persistence: `None` en lugar de `0.5` | `indicators/persistence.py`, `run.py`, `indicators/slpm_v12.py`, `src/report_generator.py` |
| MTE: `MIXED` si confianza 0% | `indicators/mte.py` |
| Etiqueta "Liderazgo táctico no confirmado" | `src/report_generator.py` |
| "Signal Agreement" → "Directional Agreement" + nota | `run.py`, `src/report_generator.py` |
| Documentación CRISIS/RECESSION en MTE | `indicators/mte.py` |
| Validación `np.isfinite()` en opciones | `indicators/options_metrics.py` |
| Estados FULL/PARTIAL/INSUFFICIENT en OMS | `indicators/options.py` |
| Documentación Proxy VRP en vol_metrics | `indicators/vol_metrics.py` |
| Docstring prevalencia direccional en persistence | `indicators/persistence.py` |
| Hallazgo Cap3D: Breadth↔LIS (+0.838) documentado | `src/dependency_tracker.py` |
| Hallazgo Cap4: mezcla horizontes temporales documentado | `indicators/slpm_v12.py` |

---

## 9. ARCHIVOS DE AUDITORÍA GENERADOS

| Archivo | Contenido |
|--------|-----------|
| `outputs/corr_pearson_señales.csv` | Capa 3A: Matriz Pearson entre RS20, Flow, Momentum, Trend |
| `outputs/corr_slpm_v12_full.csv` | Capa 3D: Matriz Spearman entre outputs del SLPM v1.2 activo |
| `outputs/auditoria_maestra_v3.15.md` | Informe consolidado de la Auditoría Maestra (4 capas) |
| `data/stock_prices_historical.csv` | Histórico de acciones para auditorías futuras |
