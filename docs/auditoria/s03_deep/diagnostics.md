# S-03-deep - src/pipeline/diagnostics.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 4579c64
- **Fichero:** `src/pipeline/diagnostics.py` (91 LOC, 4.8 KB)
- **mtime:** 2026-09-28
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

91 LOC. Fase 9a del pipeline (refactor C2-9a). 3 sub-calculos:
- Directional Agreement (signal agreement por sector).
- Price-Flow Divergence.
- Shock Sensitivity (commodity-market correlation).

Funcion publica: `compute_diagnostics(df_market, tactical_scores,
structural_scores, sector_flow_rank, temporal_meta=None)`.
Devuelve dict con 4 keys.

Consumidor: `run.py:161`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| DI-1 | BAJA | _compute_price_flow_divergence except | Fallback `{status: ALIGNED, message: ''}` para 11 sectores. Confunde "modulo fallo" con "sin divergencias". Asimetria con _compute_directional_agreement, que usa 0.5 / '50% MIXED' (neutro explicito). | WONT FIX: sin evidencia de confusion en prod. Fix requeriria nuevo status (UNKNOWN/UNAVAILABLE) + cambio en consumidor. |
| DI-2 | INFO | 3 except | `except Exception` sin ImportError especifico (solo _compute_shock_sensitivity importa dentro del try, y Exception ya cubre ImportError). | WONT FIX. |
| DI-3 | INFO | price_flow / shock | Sin try/except por sector. Fallo en uno anula todos. Coherente con directional_agreement. | WONT FIX. |
| DI-4 | INFO | price_flow | `name = sector_etf` redundante. | WONT FIX: cosmetico. |
| DI-5 | INFO | shock | Sin `reference_date`. Solo `temporal_meta`. | WONT FIX: contrato downstream. |
| DI-6 | INFO | directional | `close_spy` con `^GSPC` hardcoded. | WONT FIX: benchmark canonico. |

## 3. Contraste con tests

`tests/test_pipeline_diagnostics_engines.py` (bloque diagnostics):
- `test_diagnostics_contract_4_keys`: contrato 4 keys.
- `test_directional_agreement_degradacion_sin_df`: get_col con
  KeyError -> 11 sectores con 0.5 / '50% MIXED'.
- `test_price_flow_divergence_devuelve_11_sectores`: 11 keys con
  deteccion mockeada a ALIGNED.
- `test_shock_sensitivity_degradacion_sin_df`: RuntimeError ->
  11 sectores con `{}`.

No cubren:
- DI-1: el fallback `ALIGNED` del except exterior no se testea;
  solo el fallback individual de `get_col` (que es el inner).

## 4. Falsos positivos descartados

- `except Exception` en los 3 helpers: patron recurrente del
  pipeline (contrato defensivo). No es bug.
- La asimetria directional vs price_flow en el fallback: tecnica,
  no error. Documentada en DI-1 como decision deliberada (aunque
  discutible).
- Persistencia en este fichero: ninguna. Solo calculo en memoria.
  Correcto.

## 5. Patron observado

`diagnostics.py` es el primer fichero de `src/pipeline/` auditado
que no escribe CSV: solo calcula y devuelve. Los 12 previos hacian
persistencia con `.tmp` + `.replace()`. La ausencia de disco
descarta toda la familia de bugs 3 (atomicidad).

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. 1 BAJA (DI-1) WONT FIX
razonado; 5 INFO WONT FIX.

`diagnostics.py` queda auditado.

## 7. Proximo

Cierre de `src/pipeline/` (14 auditados + este = 15 de 18):
- `indices_intl.py` (69), `regimes.py` (63), `data_load.py` (47),
  `slpm.py` (33), `__init__.py` (2).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
