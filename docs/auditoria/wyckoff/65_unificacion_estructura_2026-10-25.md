# 65 - Unificacion de estructura sectorial (Fase 4)

**Fecha:** 2026-10-25
**Commits:** `9960cdac` (codigo) + `d4f2a8e2` (datos, ver git log)
**Estado:** CERRADA. Fase 4 de 6.

## 1. Motivacion

Cuatro caminos de "estructura sectorial" coexistian con universos y
umbrales distintos:

| Camino | Universo | Umbral |
|---|---|---|
| `sector_wyckoff_distribution` | TODOS los holdings | 60 barras |
| `wyckoff_holdings_report` (6 CSVs) | top-20 | 200 barras |
| `sector_regime_matrix.structure_positive` | derivado del agregado violado | (acc+markup)>=50 |
| `sector_regime.compute_sector_scores` | 11 ETFs (score continuo) | 6 componentes |

Consecuencia verificada: en el run del 23-10, XLC, XLK y XLP daban
`count_*` distintos entre agregado y CSVs. Sumar los CSVs no reproducia
el reporte.

## 2. Contrato aplicado (doc 63 §2.6)

- Universo: **top-20 por peso** del grupo (sector USA / indice USA / indice EU).
- Barras minimas: **>=200** para incluir un ticker (contrato v1.1).
- Barras validas: **>=460** para score completo (MA200 + robust_zscore(200,min_periods=60)).
- Entre 200-460: `evaluable=False`, `basis="warm_up_incompleto"`.

## 3. Cambios

### 3.1 Snapshot unico (`compute_wyckoff_snapshot`)

`indicators/wyckoff_holdings_report.py` gana `compute_wyckoff_snapshot`.
Devuelve DataFrame long con TODOS los tickers evaluados (top-20 por
grupo, >=200 barras). Una sola pasada de `classify_phase`. Columnas:
12 del contrato doc 63 §4 + `n_bars` + `evaluable` = 14.

`_evaluate_holding`:
- `phase_since` formateado como ISO date (`YYYY-MM-DD`) o vacio (antes
  salia como `2026-03-26 00:00:00`, violando §4).
- `n_bars` y `evaluable` anadidos.
- `evaluable=False` si 200<=n_bars<460.

### 3.2 Agregado alineado (`sector_wyckoff_distribution`)

v2.0 -> v3.0. **Consume el snapshot** (o lo calcula si no se pasa).
Universo pasa de "todos los holdings + 60 barras" a "top-20 + 200
barras". Fin de la divergencia con los 6 CSVs.

### 3.3 CSVs como proyecciones

`compute_wyckoff_holdings_report(snapshot=...)` proyecta el snapshot
en los 6 CSVs via `_tables_from_snapshot`. No calculan por su cuenta.

### 3.4 Cableado

`run.py` calcula el snapshot **una vez** antes de `sector_metrics` y lo
pasa a los 2 consumidores (`compute_sector_metrics` y
`compute_wyckoff_holdings`). Ambos wrappers aceptan `snapshot` opcional
(retrocompat tests).

### 3.5 `structure_positive` sin cambios

`sector_regime_matrix.structure_positive = (pct_accumulation +
pct_markup) >= 50` sigue igual. Cambia el universo del agregado, no la
formula. El umbral `>=50` no se toca en Fase 4 (va a Fase 5).

### 3.6 `compute_sector_scores` (ranking) sin cambios

Es un score continuo del ETF (tendencia/compresion/volatilidad/breadth).
No es estructura Wyckoff. Renombrarlo va a Fase 4b separada.

## 4. Test de coherencia (critico)

`tests/test_wyckoff_holdings_coherence.py` + `test_sector_wyckoff_distribution.py::test_coherence_snapshot_vs_csvs`:
- Snapshot con phases mixtas para 11 sectores.
- Agrega por sector, proyecta a ACC/DIST.
- Verifica que `count_*` del agregado == cuenta del snapshot por fase.
- Verifica que ACC y DIST filtrados coinciden.

Si CI verde, la divergencia XLC/XLK/XLP no se puede reproducir. Es el
test que el saliente marco como "mas importante que el cambio de
codigo".

## 5. Diff baseline (evidencia)

Run 2026-10-23 con el nuevo universo (top-20 + 200 barras):

| sector | n_total | n_valid | pct_acc | pct_markup | pct_acc+markup | sp |
|---|---|---|---|---|---|---|
| XLK | 20 | 20 | 0.0 | 10.0 | 10.0 | False |
| XLF | 19 | 19 | 0.0 | 0.0 | 0.0 | False |
| XLV | 20 | 20 | 0.0 | 10.0 | 10.0 | False |
| XLE | 20 | 20 | 0.0 | 5.0 | 5.0 | False |
| XLY | 20 | 20 | 0.0 | 0.0 | 0.0 | False |
| XLP | 20 | 20 | 0.0 | 5.0 | 5.0 | False |
| XLI | 20 | 20 | 0.0 | 0.0 | 0.0 | False |
| XLB | 20 | 20 | 0.0 | 5.0 | 5.0 | False |
| XLU | 20 | 20 | 0.0 | 0.0 | 0.0 | False |
| XLRE | 20 | 20 | 0.0 | 0.0 | 0.0 | False |
| XLC | 19 | 19 | 0.0 | 0.0 | 0.0 | False |

`structure_positive=True: 0/11` (baseline ya era 0/11 con universo
violado). El umbral `>=50` sigue sin cumplirse. Confirma que la metrica
necesita rediseño en Fase 5, no el universo.

## 6. Verificacion

- Suite: 3717 passed, 2 skipped.
- Focalizados: 37 passed (incluye coherencia, snapshot, persist endurecido).
- Run real 14.4 min, gate 11/11, 24 writers.
- Log: `Snapshot Wyckoff holdings: 357 tickers evaluados.`
- Coherencia agregado vs CSVs: `acc+dist = n_valid - RANGE - CONFLICT`
  en los 11 sectores.

## 7. Pendiente

- Fase 2: `data_quality.SOURCES` con schema_v.
- Fase 3: regeneracion historica.
- Fase 4b: renombrar `compute_sector_scores` (es trend, no estructura).
- Fase 5: `structure_score` continuo con datos coherentes.
- Fase 6: descarga unificada.

## 8. Incidente hotfix 2026-10-25

Durante la verificacion se detecto que la fixture `_cleanup_schema_files`
de `tests/conftest.py` borraba `.schema` reales de produccion. Un run
posterior sobrescribio `.v1.csv` (342 lineas originales) con 11 filas
del run actual. Recuperado desde `7d223b63`. Fix:
- `persist()` no sobrescribe `.v{old}.csv` si ya existe con >1 linea.
- `_cleanup_schema_files` solo borra `.schema` creados por el test (mtime).
