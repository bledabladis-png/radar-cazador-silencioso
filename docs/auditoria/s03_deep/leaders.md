# S-03-deep - src/pipeline/leaders.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 9673c3f
- **Fichero:** `src/pipeline/leaders.py` (108 LOC, 5.5 KB)
- **mtime:** 2026-09-28
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (LD-1). Cero ALTA, cero MEDIA.

## 1. Alcance

108 LOC. Fase 6a del pipeline (refactor C2-7a). Carga `df_stocks`,
valida cobertura efectiva, resuelve `HOLIDAY_MODE` y genera
`leader_lines` + `leader_df` + `full_metrics_df`.

Funcion publica: `compute_leaders(df_market, sector_results,
reference_date=None, run_id=None, temporal_meta=None)`.

Consumidor: `run.py:125`, con `reference_date`, `run_id` y
`temporal_meta`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| LD-1 | BAJA | L15-23 (docstring) | Docstring declaraba `holiday_mode` como key del return; el return real devuelve `df_stocks_effective_meta`. `run.py:126-131` lee las 6 keys reales, ninguna es `holiday_mode`. Desalineo documental. | **FIX APLICADO** (commit 8f12e16). |
| LD-1b | INFO | `tests/test_pipeline_sectors_leaders.py:8-12` | Comentario del test afirmaba "el docstring solo menciona 5 [keys]"; en realidad mencionaba 6 (una erronea). | **FIX APLICADO** (mismo commit). |
| LD-2 | INFO | L79 | `pd.read_csv('data/etf_holdings.csv')` sin try/except. Si falla, aborta todo el bloque (incluido df_stocks ya valido). | WONT FIX: sin holdings no hay lideres. |
| LD-3 | INFO | L84-88 | `else: pass` redundante tras `if HOLIDAY_MODE`. | WONT FIX: cosmetico. |
| LD-4 | INFO | L76-97 | Doble `if HOLIDAY_MODE:` consecutivo. | WONT FIX: cosmetico. |
| LD-5 | INFO | L103-105 | `except Exception` + `traceback.print_exc()` + `print(e)`. Contrato defensivo. | WONT FIX. |
| LD-6 | INFO | L91 | `sector_results['ranking']` sin guard. | WONT FIX: contrato upstream. |
| LD-7 | INFO | L104-105 | `traceback.print_exc()` + `print(e)` duplican informacion. | WONT FIX: cosmetico. |

## 3. Contraste con tests

`tests/test_pipeline_sectors_leaders.py` (4 tests de leaders):
- `test_leaders_contract_6_keys`: contrato real de 6 keys.
- `test_leaders_sin_df_stocks_devuelve_todo_none`: sin df_stocks ->
  leader_lines/leader_df/full_metrics_df None.
- `test_leaders_holiday_mode_cuando_cobertura_baja`: <50% Close
  validos en ultima fila -> HOLIDAY_MODE -> df_stocks=None.
- `test_leaders_insufficient_coverage_omite_df`: `resolve_effective_date`
  con status != OK -> df_stocks=None.

No cubren:
- LD-1: el docstring no es verificado por test.
- LD-2: fallo de lectura de `etf_holdings.csv` no se simula.
- LD-4: HOLIDAY_MODE no se expone directamente; se infiere por
  df_stocks=None.

## 4. Falsos positivos descartados

- A5-07 (2026-09-28) ya corrigio el bug estructural: `_close_cols`
  quedaba fuera del `if df_stocks is not None`. El codigo actual
  mantiene el guard correcto. No es bug residual.
- `resolve_effective_date(..., min_coverage=0.90)` con
  `_eligible` de columnas Close: correcto segun FU-020.
- `else: pass` (LD-3) es cosmetico, no bug: la rama `else` solo
  comenta que se conserva `df_stocks`.

## 5. Patron observado

`leaders.py` cierra el bloque de ficheros de `src/pipeline/` con
mtime 2026-09-28. La mayoria de los ultimos 3 ficheros auditados
(`sectors_base`, `flows_primary`, `finalize`, `leaders`) siguen el
patron: try/except defensivo con contrato claro, hallazgos
cosmeticos o documentales, sin deuda tecnica activa.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 1 BAJA corregida en commit
`8f12e16` (docstring + comentario del test). 7 INFO WONT FIX
razonado.

`leaders.py` queda auditado. Sin acciones pendientes.

## 7. Proximo

Continuar cierre de `src/pipeline/` (12 auditados + este = 13 de 18):
- `market_data.py` (101), `diagnostics.py` (91), `indices_intl.py` (69),
  `regimes.py` (63), `data_load.py` (47), `slpm.py` (33),
  `sectors_base.py` y `sector_metrics.py` ya hechos.
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
