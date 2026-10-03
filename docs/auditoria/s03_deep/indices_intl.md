# S-03-deep - src/pipeline/indices_intl.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 1b7cb7b
- **Fichero:** `src/pipeline/indices_intl.py` (69 LOC, 3.6 KB)
- **mtime:** 2026-09-28
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (II-1, II-2). Cero ALTA, cero MEDIA.

## 1. Alcance

69 LOC. Fase 10b del pipeline (refactor C2-10b). Calcula fases
Wyckoff de indices internacionales y selecciona lideres para los
indices en acumulacion/markup.

Funcion publica: `compute_indices_intl(df_market, reference_date=None,
run_id=None, temporal_meta=None)`. Devuelve dict con 2 keys.

Consumidor: `run.py:188`, con `reference_date`, `run_id` y
`temporal_meta`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| II-1 | BAJA | docstring | Declaraba 3 keys (`index_phases`, `index_data`, `index_leaders`); el return real tiene 2. `_index_data` se descarta con prefijo `_` de la llamada a `compute_index_phases`. `run.py:189-190` lee 2. | **FIX APLICADO** (commit del fix, S-03-deep). |
| II-2 | BAJA | L67-69 | CSV `analisis_lideres_internacionales.csv` escrito con `to_csv` directo, sin `.tmp` + `.replace()`. Familia 3 (32 sitios) exige atomicidad. | **FIX APLICADO** (mismo commit). |
| II-3 | INFO | varios | Asimetria de excepts entre bloques (sin ImportError). Coherente: no hay imports tardios en este fichero. | WONT FIX. |
| II-4 | INFO | L14 | `_index_data` descartado pero mencionado en docstring. | Cubierto por II-1. |
| II-5 | INFO | L53 | `select_index_leaders(None, df_stocks, [nombre], ...)` con `None` como primer arg. | WONT FIX: contrato downstream. |

## 3. Contraste con tests

`tests/test_pipeline_data_load_indices.py` (bloque indices_intl):
- `test_indices_intl_contrato_2_keys`: contrato de 2 keys.
- `test_indices_intl_sin_acumulacion_no_descarga`: si ningun indice
  esta en ACCUMULATION/MARKUP, no se descarga.
- `test_indices_intl_con_acumulacion_descarga_y_selecciona`: flujo
  completo.
- `test_indices_intl_errores_por_indice_no_abortan`: fallo de un
  indice no aborta el resto.
- `test_indices_intl_csv_falla_no_rompe`: fallo de to_csv no propaga.
- `test_indices_intl_csv_escrito`: CSV escrito con columna `indice`.

`tests/test_fu_c19_indices_intl.py`:
- `test_c19_propagates_reference_date_and_run_id`: FU-C19,
  propagacion a `download_stock_prices`.
- `test_c19_signature_accepts_kwargs`: firma.
- `test_c19_source_contains_propagation`: regresion estructural.

No cubren:
- II-1: el docstring no es verificado por test.
- II-2: el test `test_indices_intl_csv_falla_no_rompe` verifica que
  un fallo no propaga, pero no la atomicidad (que el CSV original
  quede intacto).

## 4. Falsos positivos descartados

- A5-09 (2026-09-28) y A5-10 (2026-09-28): guards ya aplicados en
  compute_index_phases y download_stock_prices. No hay bug residual.
- FU-C19: la propagacion de reference_date y run_id es correcta y
  esta testada.
- Ausencia de ImportError en excepts: coherente, no hay imports
  tardios en el fichero.

## 5. Patron observado

`indices_intl.py` es el segundo fichero de S-03-deep con docstring
desalineado (el primero fue leaders.py LD-1). Ambos tenian un
argumento descartado con prefijo `_` que aparecia en docstring.
Patron: cuando se refactoriza un fichero, los args descartados
tienden a quedar en el docstring. Candidato a busqueda activa.

Tambien es la primera incidencia de atomicidad CSV fuera del bloque
D-05/D-04. La familia 3 (32 sitios) no cubre todos los writers
historicos.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 2 BAJA corregidas (II-1
docstring, II-2 atomicidad CSV) en el commit del fix. 3 INFO
WONT FIX razonado.

`indices_intl.py` queda auditado.

## 7. Proximo

Cierre de `src/pipeline/` (15 auditados + este = 16 de 18):
- `regimes.py` (63), `data_load.py` (47), `slpm.py` (33),
  `__init__.py` (2).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
