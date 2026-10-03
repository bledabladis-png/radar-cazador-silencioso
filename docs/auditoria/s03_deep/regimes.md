# S-03-deep - src/pipeline/regimes.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** d99d781
- **Fichero:** `src/pipeline/regimes.py` (63 LOC, 2.9 KB)
- **mtime:** 2026-09-30
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. Alcance

63 LOC. Fase 2 del pipeline (refactor C2-4). Orquesta los 4 regimenes:
- Financial Conditions.
- Liquidez real (FRED).
- Volatilidad (VIX).
- Macro.

Funcion publica: `compute_all_regimes(df_market, df_macro_manual,
temporal_meta=None, reference_date=None)`. Devuelve dict con 14 keys.

Consumidor: `run.py:72`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| RG-1 | INFO | funcion | Sin try/except por bloque. Downstream autodegrada (volatility_regime C-08 -> N/D; liquidity A3.1-03 -> 4-tuple con None). Fail-loud delegado. | WONT FIX: diseno + tests. |
| RG-2 | INFO | L~47 | `except KeyError` estrecho en VIX. Coherente con test `test_regimes_sin_vix_usa_volatilidad_plana`. | WONT FIX. |
| RG-3 | INFO | `liq_conf` | Nombre ambiguo (viene de `compute_financial_conditions`, no de liquidez real). | WONT FIX: naming heredado. |
| RG-4 | INFO | firma | `temporal_meta`/`reference_date` solo a macro. Firmas downstream no los piden. | WONT FIX. |
| RG-5 | INFO | prints | `:.0%` con `conf` eventualmente None. Diseno A3.1-03 garantiza coherencia. | WONT FIX. |
| RG-6 | INFO | return | Sin `liq_regime` (no declarado, no consumido). | WONT FIX. |

## 3. Contraste con tests

`tests/test_pipeline_slpm_regimes.py` (bloque regimes):
- `test_regimes_devuelve_14_keys`: contrato de 14 keys.
- `test_regimes_degradacion_liq_real_none`: `compute_real_liquidity`
  con `(None, None, None, None)` -> `real_liq_score=None`,
  `real_liq_regime='N/A'`, `real_liq_conf=0.0`.
- `test_regimes_sin_vix_usa_volatilidad_plana`: `get_col` con
  `KeyError` -> Series vacia pasada a `compute_volatility_regime`.
- `test_regimes_propaga_temporal_meta_a_macro`: propagacion de
  `temporal_meta` a `compute_macro_regime`.

No cubren:
- RG-1: fallo de `compute_financial_conditions` o
  `compute_macro_regime` no se simula.
- RG-5: `conf` como None no se testea.

## 4. Falsos positivos descartados

- Ausencia de try/except por bloque: diseno explicito. Los
  downstream degradan internamente (C-08, A3.1-03). El unico
  except externo (VIX KeyError) esta justificado.
- `liq_conf` naming: cosmetico, no bug. Documentado en RG-3.
- `temporal_meta` solo a macro: coherente con las firmas
  (`compute_financial_conditions(df)`, `compute_liquidity_score()`,
  `compute_volatility_regime(returns)`).
- 14 keys reales vs 14 declaradas en docstring: alineadas.

## 5. Patron observado

`regimes.py` sigue el patron de los ultimos 4 ficheros de
`src/pipeline/`: sin hallazgos materiales. La zona refactorizada
en C2 esta limpia. Los hallazgos corregibles (LD-1, II-1, II-2,
MD-2) aparecen en:
- Docstrings desalineados con return real (LD-1, II-1).
- Atomicidad CSV fuera del bloque familiar (II-2).
- Except sin ImportError en bloques con import tardio (MD-2).

Ninguno de esos tres patrones aparece aqui.

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 6 INFO,
todos WONT FIX razonado.

`regimes.py` queda auditado.

## 7. Proximo

Cierre de `src/pipeline/` (16 auditados + este = 17 de 18):
- `data_load.py` (47), `slpm.py` (33), `__init__.py` (2).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
