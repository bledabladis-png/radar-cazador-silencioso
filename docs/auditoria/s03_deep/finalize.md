# S-03-deep - src/pipeline/finalize.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** b3302f8
- **Fichero:** `src/pipeline/finalize.py` (128 LOC, 6.2 KB)
- **mtime:** 2026-09-29
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. Alcance

128 LOC. Fase 12 del pipeline (refactor C2-12). 4 funciones publicas:
- `compute_final_matrices`: 2 matrices (regimen + evidencia).
- `save_regime_history`: persistencia del regimen macro actual.
- `save_sector_rankings`: persistencia del ranking de sectores.
- `generate_european_coverage`: reporte cobertura europea.

Consumidor: `run.py:220, 278-282`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FN-1 | INFO | save_regime_history / save_sector_rankings | Sin try/except; la excepcion se propaga. Fail-loud intencional. | WONT FIX: test `test_save_regime_history_fallo_to_csv_no_pierde_filas` lo verifica con `pytest.raises(OSError)`. |
| FN-2 | INFO | L~85 | `macro_score.iloc[-1]` sin guard. | WONT FIX: contrato de run.py (pasa Series). |
| FN-3 | INFO | L~88 | `dtype=str` en lectura + `astype(str)` en escritura. | WONT FIX: formato de persistencia legible. |
| FN-4 | INFO | L~78-79 | `os.path.exists` + `os.path` mezclado con `Path`. | WONT FIX: cosmetico. |
| FN-5 | INFO | to_csv | Sin `encoding='utf-8'` en `save_regime_history` y `save_sector_rankings` (compute_final_matrices si lo pone). | WONT FIX: contenido en ASCII (regimes, tickers). |
| FN-6 | INFO | save_sector_rankings | `['ranking']` sin guard. | WONT FIX: contrato de sectors_base. |
| FN-7 | INFO | L~50, L~80 | `except` sin `RuntimeError` (otros bloques del pipeline si lo incluyen). | WONT FIX. |
| FN-8 | INFO | L~100 | `pd.concat` sin sort previo a `drop_duplicates(keep='last')`. | WONT FIX: CSV gestionado ordenado. |

## 3. Contraste con tests

`tests/test_pipeline_sector_metrics_mte_finalize.py` (bloque finalize):
- `test_final_matrices_contract`: contrato 2 keys.
- `test_save_regime_history_escribe_csv`: escritura + verificacion.
- `test_save_regime_history_fallo_to_csv_no_pierde_filas`: la
  excepcion se propaga (`pytest.raises(OSError)`) y el CSV queda
  intacto (atomicidad familia 3).
- `test_save_sector_rankings_escribe_csv` (nombre en el pegado; el
  contenido pertenece a `test_c4_code.py`): auditoria C4 de
  residuales `Timestamp.now()` en un listado de targets que incluye
  `src/pipeline/finalize.py`.

No cubren:
- FN-4/FN-5: encoding y os.path no se testean.
- FN-8: orden del concat.

## 4. Falsos positivos descartados

- Ausencia de try/except en `save_regime_history` y
  `save_sector_rankings`: es **intencional** (fail-loud). El caller
  (`run.py:278-282`) decide. Test lo verifica.
- `compute_final_matrices` con `ImportError` en la tupla: coherente
  con las importaciones tardias de `indicators.sector_regime_matrix`
  y `indicators.evidence_matrix`.
- Doble candado en `save_regime_history`: obs_date no-None + NYSE.
  Patron B5-followup (2026-09-12). Correcto.

## 5. Patron observado

`finalize.py` es el primer fichero de S-03-deep donde las excepciones
no se capturan dentro del modulo (dos funciones fail-loud). Refleja
una decision explicita: los side effects finales deben abortar si
fallan. Coherente con el rol del modulo (ultima fase antes del
reporte).

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 8 INFO,
todos WONT FIX razonado.

`finalize.py` queda auditado. Sin acciones pendientes.

## 7. Proximo

Continuar cierre de `src/pipeline/` (10 auditados + este = 11 de 18):
- `leaders.py` (108), `market_data.py` (101), `diagnostics.py` (91),
  `indices_intl.py` (69), `regimes.py` (63).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
