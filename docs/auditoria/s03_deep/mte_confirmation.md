# S-03-deep - src/pipeline/mte_confirmation.py

- **Auditoria:** 2026-10-03
- **HEAD:** dee7edf
- **Fichero:** `src/pipeline/mte_confirmation.py` (146 LOC, 7.7 KB)
- **mtime:** 2026-10-03 (modificado en D-05, commit e0aacda)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

146 LOC. 3 funciones privadas + 1 publica:
- `_compute_mte`: calcula MTE con frescura de darkpool.
- `_compute_cross_module`: conflicto cross-module.
- `_compute_confirmation`: T10Y3M, vol_metrics, cross_asset,
  fls, advance_decline.
- `compute_mte_confirmation`: orquestador.

Extraido de `run.py` (refactor C2-10a). Ultimo cambio funcional:
D-05 (2026-10-03, commit e0aacda) normaliza `reference_date` a
naive antes de calcular edades (mismo patron que validation_gate).

Consumidor: `run.py` (raiz del repo).

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| C1 | INFO | L53 | `except Exception` en `_compute_mte`. Contrato explicito en `engine.py`: `compute_mte` puede devolver None, el consumidor lo maneja. Intencional. | WONT FIX: contrato deliberado. |
| C2 | INFO | L51 | `except (ValueError, TypeError, OSError): pass` en freshness check. | WONT FIX: cubierto por test D-05. |
| C3 | BAJA | L77 | `pd.read_csv('data/macro_manual/10y3m.csv')` con ruta relativa. | WONT FIX: funciona desde raiz. |
| C4 | INFO | L112 | `fls_data['fls_normalized']` sin `.get()`. | WONT FIX: capturado. |
| C5 | INFO | L33-34 | `cred_signal`/`vol_signal` pueden ser Series. | WONT FIX: `_get_last` maneja ambos. |

## 3. Contraste con tests

`tests/test_pipeline_sector_metrics_mte_finalize.py` (varios tests):
- `test_mte_confirmation_contract_4_keys`: contrato de las 3 claves.
- `test_mte_confirmation_pasa_scenario`: propagacion del scenario.
- `test_compute_mte_tz_aware_no_silencia_archival`: D-05, tz-aware
  debe reportar ARCHIVAL.
- `test_compute_mte_confirmation_propaga_reference_date`: D-05,
  `reference_date` llega a `_compute_mte`.

Adicionales en `tests/test_mte_engine.py`, `test_mte_scoring.py`,
`test_mte_state.py`, `test_mte_classify_mte.py`, `test_indicators_low_coverage_d28.py`,
`test_mte_scoring_characterization.py`.

No cubren:
- C1: el except amplio no se testea directamente.
- C2: el `pass` del chequeo freshness no se testea con week malformado.
- C3: los tests usan `_setup_tmp`, no la ruta relativa real.
- C5: el test usa escalares (0.0), no Series.

## 4. Falsos positivos descartados

- **`indicators/mte.py` no existe**: es un paquete `indicators/mte/`,
  con `__init__.py` que reexporta `compute_mte` de `.engine`. El import
  funciona, la suite esta verde.
- **`compute_mte` con `except Exception` interno**: contrato documentado
  en docstring del engine ("defensa en profundidad" + "trazabilidad via
  traceback en stdout"). No es bug.
- **`patched` `indicators.mte.compute_mte`**: funciona por la
  reexportacion; los tests mockean el simbolo publico.

## 5. Reclasificaciones respecto al primer pase

Informe inicial: C1 MEDIA, C2 MEDIA, C5 MEDIA. Tras contraste:

- C1: MEDIA -> INFO (contrato intencional documentado).
- C2: MEDIA -> INFO (cubierto por test D-05).
- C5: MEDIA -> INFO (`_get_last` maneja Series y escalar).
- C3: BAJA (sin cambio).
- C4: INFO (sin cambio).

Patron: igual que validation_gate.py. Tres falsas alarmas por no haber
verificado primero el contrato del modulo y los tests.

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. `mte_confirmation.py`
queda auditado post-D-05. Sin acciones pendientes.

## 7. Proximo

Siguiente fichero con mtime 10-03 en `src/pipeline/`:
`breadth_metrics.py` (156 LOC, 8 KB).
