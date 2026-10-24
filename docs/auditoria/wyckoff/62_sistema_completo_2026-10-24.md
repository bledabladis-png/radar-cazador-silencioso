# Sistema Wyckoff completo — mapa tecnico

**Fecha:** 2026-10-24
**Commit de referencia:** ver `git log --oneline -1`
**Estado:** IMPLEMENTADO + CABLEADO al pipeline.
**Tests:** 3776 passed (suite completa) + 2 skipped.

## §1 Decision arquitectonica

Sustituir el sistema previo (classify_wyckoff_phase sobre ETFs sectoriales
e indices) por cascadas por stock con deteccion de acumulacion y
distribucion, orquestadas.

Motivacion: el sistema previo detectaba 0/327 DISTRIBUTION en stocks
reales (el gate de distribucion no disparaba nunca), y para los
ETFs sectoriales la senal era demasiado debil. La cascada nueva detecta
eventos reales con evidencia y fase actual derivada.

**Estado (2026-10-24, tras sesion 3):** el sistema esta
IMPLEMENTADO + CABLEADO. `classify_wyckoff_phase` y su familia
retirados de `wyckoff_v1.py` (C5.2). El sistema previo (lideres
por sector, SLPM, fase ETF del ranking) tambien retirado (C3 y C4).

## §2 Modulos

### `indicators/wyckoff_event.py`

Dataclasses comunes. No contiene detectores.

- `WyckoffEvent`: evento atomico (BC, AR, SOW, SC, SOS, ST, UT, Spring,
  PSY, PS, mSOW, mSOS, LPSY, LPS).
  - Campos: ticker, event_type, event_date, available_at, status,
    range_id, evidence.
  - Status: CONFIRMED, CANDIDATE, NOT_OBSERVED, NOT_EVALUABLE,
    INVALIDATED, REJECTED, EXPIRED.
- `Range`: estructura de rango con soporte, resistencia, tolerancia,
  state, range_type, fechas de disponibilidad.
- `ClassifiedBreak`: salida de una cascada. Campos clave:
  - `classification`: estado de la cascada
  - `structural_state`: RANGE_ACTIVE / RANGE_BROKEN / NOT_EVALUABLE
  - `break_direction`: UP / DOWN
  - `range_type`: DISTRIBUTION_CANDIDATE / ACCUMULATION_CANDIDATE /
    UNCLASSIFIED
  - `trend_direction`: UP / DOWN / NONE
  - `sow_status`: estado del SOW/SOS
  - `coherence_error`: si range_type y trend son incoherentes
  - `range_active_at`, `resolution_event_date`: fechas
  - `range_support`, `range_resistance`, `range_tol_abs`: niveles del
    rango (necesarios para regla simetrica de vigencia)

### `indicators/wyckoff_distribution.py`

Cascada de distribucion. Detectores:

- `detect_bc(df, ticker, scan_window)`: Buying Climax. High del
  lookback + volumen alto + volatilidad alta + tendencia alcista.
- `detect_ar(df, ticker, bc_event)`: Automatic Reaction. Caida post-BC
  que fija el soporte del rango.
- `build_range_from_bc_ar`: construye Range desde BC+AR confirmados.
- `detect_sow_in_range(df, ticker, range, as_of)`: SOW anclado al
  rango. CANDIDATE -> CONFIRMED (3 cierres bajo soporte) / REJECTED /
  EXPIRED.
- `detect_trend_context(df, ticker, as_of)`: t_norm promedio en
  ventana previa. UP / DOWN / NONE.
- `classify_distribution_at(df, ticker, as_of)`: cascada completa.
- `classify_distribution(df, ticker)`: wrapper (as_of = ultimo bar).

Detectores auxiliares (evidencia pasiva, NO cambian classification):
- `detect_st_dist`: Secondary Test. Retest del BC con rechazo.
- `detect_psy`: Preliminary Supply. Techo temprano antes del BC.
- `detect_msow`: Minor SOW. SOW con umbrales reducidos.
- `detect_lpsy`: Last Point of Supply. Rebote debil post-SOW.
- `detect_false_breakout_up`: UT/UTAD. Falsa ruptura arriba.
- `detect_ps`, `detect_msos`, `detect_lps`: simetricos de
  PSY/mSOW/LPSY en tendencia bajista.

### `indicators/wyckoff_accumulation.py`

Cascada simetrica de acumulacion. Detectores:

- `detect_sc`: Selling Climax. Simetrico a BC.
- `detect_ar_accumulation`: Automatic Rally. Simetrico a AR.
- `build_range_from_sc_ar`: Range desde SC+AR.
- `detect_sos_in_range`: SOS anclado al rango.
- `classify_accumulation_at`, `classify_accumulation`: cascada completa.

Detectores auxiliares (evidencia pasiva):
- `detect_st_acc`: Secondary Test simetrico.
- `detect_false_breakout_down`: Spring.

### `indicators/wyckoff_orchestrator.py`

Orquesta las dos cascadas y emite una sola fase por stock y fecha.

Funciones:
- `classify_phase_at(df, ticker, as_of)`: corre ambas cascadas hasta
  as_of, aplica regla simetrica de vigencia, emite FinalPhase.
- `classify_phase(df, ticker)`: wrapper.

Estados de `phase` (fase ACTUAL):
- `DISTRIBUTION`: dist confirmada, precio dentro del rango
- `MARKDOWN`: dist confirmada, precio < support - tol (consumada)
- `ACCUMULATION`: acc confirmada, precio dentro del rango
- `MARKUP`: acc confirmada, precio > resistance + tol (consumada)
- `DISTRIBUTION_FORMING`: rango activo con contexto dist, sin SOW
- `ACCUMULATION_FORMING`: rango activo con contexto acc, sin SOS
- `RANGE`: dist/acc fallidas, o sin cascada activa
- `CONFLICT`: dos estructuras compitiendo (ambas dentro, o misma fecha
  de resolucion)
- `INSUFFICIENT_DATA`: falta de datos real (no "no BC/SC")

Estados de `event_phase` (fase historica, sin vigencia por precio):
- Mismos nombres, pero sin FORMING ni diferencia por precio.

FinalPhase tambien lleva:
- `phase_since`: fecha de resolucion del episodio ganador
- `current_phase_basis`: justificacion textual (ej. "dist vigente:
  price 98 dentro del rango")

## §3 Regla simetrica de vigencia

Un rango esta vigente mientras el precio NO lo ha roto por NINGUN lado:

- Si price > resistance + tol: dist fallida -> RANGE; acc consumada
  -> MARKUP.
- Si price < support - tol: dist consumada -> MARKDOWN; acc fallida
  -> RANGE.
- Si price dentro: dist/acc vigente.

**No es un umbral estadistico. Es definicion de rango.**

Esto resuelve el problema de las "distribuciones fantasma": cascadas
que ocurrieron hace meses y que el sistema seguia etiquetando como
fase actual.

## §4 Cableado al pipeline

### `indicators/sector_wyckoff_distribution.py`

Migrado de `classify_wyckoff_phase` (legacy) a `classify_phase`
(nuevo orquestador). Agrega por sector los holdings.

Columnas del CSV `outputs/history/sector_wyckoff_distribution.csv`:
- `date`, `sector`, `n_total`, `n_valid_wyckoff`, `n_insufficient_wyckoff`
- `coverage_wyckoff`
- `count_*` y `pct_*` para cada fase en VALID_PHASES:
  ACCUMULATION, ACCUMULATION_FORMING, MARKUP, RANGE, DISTRIBUTION,
  DISTRIBUTION_FORMING, MARKDOWN, CONFLICT

### `src/report/sector_context.py`

`render_wyckoff_sectorial` renderiza la tabla del reporte con columnas:
Sector | AccF | Acc | Markup | Range | DistF | Dist | Markdown |
Conflict | N | Cob Wyckoff.

Notas al pie explican FORMING y Conflict.

### `src/report/wyckoff_holdings.py` (nuevo 2026-10-24)

Renderiza 2 secciones en el reporte:
- **Tickers en acumulacion**: 3 sub-tablas (USA sectores, USA indices,
  EU indices). Incluye ACC, ACC_FORMING, MARKUP.
- **Tickers en distribucion**: 3 sub-tablas. Incluye DIST, DIST_FORMING,
  MARKDOWN.

Columnas: Group | Ticker | Fase | Desde | Peso % | Cob % | Close.
Los 6 CSVs estan en `outputs/history/` con receipt.

### `run.py`

`load_stocks_data` + `compute_all_stock_metrics` sustituyen a
`compute_leaders`. `compute_wyckoff_holdings(df_stocks, holdings_df)`
genera los 6 CSVs. `wyckoff_holdings_data` se pasa a
`generate_daily_report`.

**Retirado en sesion 3:** `src/report/sector_annotations.py` (la
anotacion de titulos sectoriales vivia de `leader_lines`; sin lideres,
no aplica).

## §5 Lo que NO cambia

- `detect_sow` (v1.8) intacto: sigue siendo el detector de breakdown
  con volumen, con su API historica.
- `detect_sow_v2` (gate distribution) sigue en codigo, `WYCKOFF_SOW_*`
  sigue en `None`. Base del futuro contrato v1.10 (SOW-BEAR-CONFIRMATION).
- `wyckoff_score`, `_atr_normalized`, `_trend_component`, `_volume_z`,
  `_effort_vs_result`, `wyckoff_stability`, `build_ticker_df`,
  `detect_spring`, `detect_sos` intactos.
- **`classify_wyckoff_phase` retirado** (C5.2). La clasificacion vive
  en `wyckoff_orchestrator.classify_phase`. `sector_regime.py` dejo de
  usar fase ETF en C5.1; la tupla del ranking es de 3 elementos.

## §6 Pre-registros fallidos

### `60_current_phase_rule_v1.md`

Regla de vigencia unidireccional (solo recuperacion, no consumacion).
Fallo C1 en holdout S&P 500: 18.6% invalidados < 20% minimo. **No se
ajusto.**

Se reemplazo por la regla simetrica (dos lados). La regla simetrica NO
es un pre-registro: es definicion de rango. El pre-registro v1 media
una hipotesis sobre un lado; el simetrico mide los dos porque es lo
que significa "rango vigente".

### `61_utad_spring_pending_v1.md`

UTAD_PENDING y SPRING_PENDING como fases intermedias. Fallo A3
(asimetria): 26 UTAD_PENDING vs 3 SPRING_PENDING, ratio 8.67x > 5x.
**No se ajusto.** Los detectores quedan en codigo, sin cambiar
classification.

**Diagnostico:** la asimetria es del mercado (S&P 500 alcista tiene mas
intentos de ruptura arriba rechazados que abajo). No es bug tecnico.
Un futuro pre-registro v2 tendria que aceptar la asimetria como dato
de mercado, no como fallo del detector.

## §7 Estado en produccion (2026-10-22)

327 stocks, ultimo bar:

| Fase actual | N |
|---|---|
| RANGE | 219 |
| DISTRIBUTION_FORMING | 53 |
| DISTRIBUTION | 17 |
| MARKDOWN | 17 |
| MARKUP | 13 |
| ACCUMULATION_FORMING | 7 |
| CONFLICT | 1 |

Agregado holdings (11 ETFs, top-20 cada uno):

| Fase actual | N |
|---|---|
| RANGE | 157 |
| DISTRIBUTION_FORMING | 35 |
| DISTRIBUTION | 12 |
| MARKDOWN | 12 |
| MARKUP | 8 |
| ACCUMULATION_FORMING | 6 |

XLK: 41% de sus holdings en DISTRIBUTION_FORMING (presion interna
fuerte).

**Casos conocidos (2026-10-24):**

- **TWST = CONFLICT.** Razon: "dist formandose y acc formandose
  simultaneamente". No es falso positivo, es ambiguedad real (dos
  estructuras compitiendo). Documentado como caso conocido.
- **`ACCUMULATION=0`** en produccion. Estado transitorio por diseno:
  requiere SOS confirmado Y precio dentro del rango. Dura sesiones,
  no semanas. No es bug.

**Cobertura (2026-10-24, tras C1):**
- USA sectores: 11, top-20 por weight (peso top-20 entre 57.2% y
  99.0%).
- USA indices: SPY, QQQ, DIA. IWM excluido (top-20 = 6.2% de 1977
  holdings, no representativo).
- EU indices: FEZ, DAXEX, ISF.L, LYXI (100% cobertura de holdings).

## §8 Pendiente (2026-10-24)

- Calibrar PSY/PS con pre-registro formal (actualmente 0/50 disparos).
- Renderizar evidencia pasiva (`msow`, `msos`, `lpsy`, `lps`, `st_dist`,
  `st_acc`) en el reporte.
- `current_phase_basis` en el reporte (hoy solo en CSV).
- Validar la nueva derivacion de `structure_positive` en
  `sector_regime_matrix.py` (holdings-based desde C5.1).
- TWST en CONFLICT: documentar como caso conocido en el reporte.
- `ACCUMULATION=0`: nota en el reporte.
- IWM fuera: sub-fase de descarga con protocolo de 6 pasos.
- Cobertura USA incompleta (231/504 holdings): ampliacion futura.
- Auditoria externa del sistema completo.
- Contrato v1.10 para SOW-BEAR-CONFIRMATION.

**Retirado del pendiente (2026-10-24):**
- ~~Migrar `sector_regime.py` al nuevo orquestador~~: hecho en C5.1.
- ~~`classify_wyckoff_phase` en `sector_regime.py`~~: retirado en C5.1.

## §9 Como extender el sistema

Para añadir un detector nuevo:

1. Definirlo en el modulo correspondiente (dist o acc).
2. Firma: `detect_X(df, ticker, ...) -> WyckoffEvent`.
3. Estados: CONFIRMED / NOT_OBSERVED / NOT_EVALUABLE (+ CANDIDATE /
   REJECTED / EXPIRED si aplica).
4. Fechas: `event_date` (cuando ocurrio) y `available_at` (cuando se
   pudo saber).
5. Tests en `tests/test_wyckoff_*.py`.
6. Si es evidencia pasiva (no cambia classification): añadirlo a
   `_enrich_dist_evidence` o `_enrich_acc_evidence`.
7. Si cambia la fase actual: modificar `_decide_current_phase` con
   pre-registro previo escrito.