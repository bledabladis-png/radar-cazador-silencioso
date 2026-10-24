# Sistema Wyckoff completo — mapa tecnico

**Fecha:** 2026-10-24
**Commit de referencia:** ver `git log --oneline -1`
**Estado:** IMPLEMENTADO + CABLEADO al pipeline.
**Tests:** 3946 passed (suite completa) + 5 skipped.

## §1 Decision arquitectonica

Sustituir el sistema previo (classify_wyckoff_phase sobre ETFs sectoriales
e indices) por cascadas por stock con deteccion de acumulacion y
distribucion, orquestadas.

Motivacion: el sistema previo detectaba 0/327 DISTRIBUTION en stocks
reales (`_is_distribution_candidate` no disparaba nunca), y para los
ETFs sectoriales la senal era demasiado debil. La cascada nueva detecta
eventos reales con evidencia y fase actual derivada.

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

### `src/report/sector_annotations.py`

`annotate_sector_titles_with_composition(leader_lines, dist_df)`:
postprocesa las lineas del reporte generadas por `stock_leader.py`.
Para cada `## Sector: XLK (MARKUP)`, inserta una linea de composicion
interna:

    *Composicion interna (22 componentes): 3 Markup (14%), 8 Range
    (36%), 9 DistF (41%), 1 Dist (5%), 1 Markdown (5%)*

Idempotente. No modifica los datos que producen las secciones, solo
añade contexto textual.

### `run.py`

En L~280, antes de `generate_daily_report`, anota `leader_lines` con
`annotate_sector_titles_with_composition`.

## §5 Lo que NO cambia

- `detect_sow` (v1.8) intacto: sigue siendo el detector de breakdown
  con volumen, con su API historica.
- `detect_sow_v2` (gate distribution) sigue en codigo, `WYCKOFF_SOW_*`
  sigue en `None`.
- `classify_wyckoff_phase` sigue existiendo y se usa en
  `sector_regime.py` (rankings) para el titulo del sector. Migrarlo
  es decision de producto pendiente.

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
fuerte). La anotacion en el titulo sectorial lo hace visible.

## §8 Pendiente

- Migrar `sector_regime.py` al nuevo orquestador.
- Renderizar evidencia pasiva (`msow`, `msos`, `lpsy`, `lps`, `st_dist`,
  `st_acc`) en el reporte.
- Calibrar PSY/PS con pre-registro formal (actualmente 0/50 disparos).
- Auditoria externa del sistema completo.
- Contrato v1.10 para SOW-BEAR-CONFIRMATION.

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