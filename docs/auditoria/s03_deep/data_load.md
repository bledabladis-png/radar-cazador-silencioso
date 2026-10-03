# S-03-deep - src/pipeline/data_load.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 65d46e7
- **Fichero:** `src/pipeline/data_load.py` (47 LOC, 2.1 KB)
- **mtime:** 2026-09-28
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (DL-1). Cero ALTA, cero MEDIA.

## 1. Alcance

47 LOC. Fase 1 del pipeline (refactor C2-3). Descarga mercado, valida
y carga datos macro manuales.

Funcion publica: `load_all_data(reference_date=None, run_id=None)`.
Devuelve dict con 5 keys o `None` si fallo critico.

Consumidor: `run.py:65`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| DL-1 | BAJA | docstring | Declaraba 4 keys; el return real tiene 5 (añade `temporal_meta`, de `df_market.attrs`). `run.py:70` lo lee con `.get('temporal_meta', {})`. Tercer caso del mismo patron en S-03-deep (tras LD-1 de leaders.py e II-1 de indices_intl.py). | **FIX APLICADO** (commit efe2388). |
| DL-2 | INFO | L~45 | `len(valid) < 5` hardcoded. Sin constante en config.settings. | WONT FIX: semantica clara + test explicito. |
| DL-3 | INFO | funcion | Sin try/except por bloque; fail-loud coherente con el docstring ("Devuelve None si hay fallo critico"). | WONT FIX: diseno. |
| DL-4 | INFO | return | `df_market.attrs.get('temporal_meta', {})`. | WONT FIX: pandas 2.x estandar. |

## 3. Contraste con tests

`tests/test_pipeline_data_load_indices.py` (bloque data_load, 7 tests):
- `test_load_all_data_contrato_5_keys`: contrato de 5 keys.
- `test_load_all_data_df_market_none_devuelve_none`.
- `test_load_all_data_df_market_empty_devuelve_none`.
- `test_load_all_data_pocos_tickers_devuelve_none`: `len(valid) < 5`.
- `test_load_all_data_imprime_issues`: propagacion de issues.
- `test_load_all_data_propaga_reference_date_y_run_id`.
- `test_load_all_data_temporal_meta_de_attrs`: `df.attrs` -> dict.

No cubren:
- DL-1: el docstring no se verifica por test (el contrato real si).

## 4. Falsos positivos descartados

- A5-01 (2026-09-28): la comprobacion duplicada de `df_market` ya
  fue eliminada. El codigo actual tiene un solo guard. Correcto.
- FU-021-5 Fase 9: `trim_to_last_valid_date` retirado. La
  responsabilidad esta en `download_market_data`. Correcto.
- Fail-loud (sin try/except): coherente con el contrato declarado.
- `valid` como lista vs `len` como int: `validate_market_data`
  devuelve `(list, dict)`. Correcto.

## 5. Patron confirmado: docstring con keys desactualizadas tras refactor

Tres casos en S-03-deep:
- LD-1 (`leaders.py`): docstring `holiday_mode` vs return `df_stocks_effective_meta`.
- II-1 (`indices_intl.py`): docstring con `index_data`, return sin ella.
- DL-1 (`data_load.py`): docstring con 4 keys, return con 5.

Patron consistente: los refactors C2 reasignan el return pero no
actualizan el docstring. Candidato a busqueda activa sobre los
ficheros de `src/pipeline/`, `src/report/` y `regimes/` que aun no
se han auditado (queda 1 de src/pipeline/, 19 de src/report/,
8 de regimes/).

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 1 BAJA corregida en commit
`efe2388` (docstring). 3 INFO WONT FIX razonado.

`data_load.py` queda auditado.

## 7. Proximo

Cierre de `src/pipeline/` (17 auditados + este = 18 de 18 con
`slpm.py` y `__init__.py` pendientes):
- `slpm.py` (33), `__init__.py` (2).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
