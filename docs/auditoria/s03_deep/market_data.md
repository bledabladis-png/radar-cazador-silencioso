# S-03-deep - src/pipeline/market_data.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** debb05a
- **Fichero:** `src/pipeline/market_data.py` (101 LOC, 4.8 KB)
- **mtime:** 2026-09-29
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (MD-2). Cero ALTA, cero MEDIA.

## 1. Alcance

101 LOC. Fase 9b del pipeline (refactor C2-9b). Orquestador de 4
subsistemas: PCR (options), Dark Pools (FINRA ATS), Volatilidad
estructural, Calidad de datos.

Funcion publica: `compute_market_data(df_market, df_stocks=None,
temporal_meta=None, reference_date=None)`. Devuelve dict con 4 keys.

Consumidor: `run.py:167`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| MD-1 | INFO | firma | `temporal_meta` no se usa ni se propaga a los 4 helpers. Consumidor lo pasa. | WONT FIX: contrato de API. |
| MD-2 | BAJA | 4 bloques except | `except` sin `ImportError`. Los 4 helpers importan `indicators.*` dentro del try. Si un modulo deja de importar, la excepcion propaga y `compute_market_data` aborta, contradiciendo el mensaje "Modulo X omitido". `finalize.py` si captura ImportError. | **FIX APLICADO** (commit 2238a3f). |
| MD-3 | INFO | _compute_vol_structure | `pcr_data.get('z_score', np.nan)` asume dict-like. | WONT FIX: contrato de compute_pcr_signals. |
| MD-4 | INFO | 3 to_csv | Sin `encoding='utf-8'` (volatility_structure, data_quality). Contenido en ASCII. | WONT FIX. |
| MD-5 | INFO | varias | Rutas relativas `outputs/history/*.csv`. | WONT FIX: CWD raiz. |
| MD-6 | INFO | _compute_pcr | Sin parametros; los otros 3 helpers si reciben args. | WONT FIX: compute_pcr_signals() no los pide. |
| MD-7 | INFO | _compute_data_quality | No recibe pcr_data ni otros outputs previos. | WONT FIX: contrato propio. |

## 3. Contraste con tests

- `tests/test_pipeline_market_data_atomic.py`: atomicidad familia 3
  en `_compute_vol_structure` y `_compute_data_quality` (to_csv a
  mitad -> CSV original intacto). Contrato de 4 keys de
  `compute_market_data`. `_compute_pcr` con RuntimeError -> None.
  `_compute_darkpool` propaga df_market/df_stocks como kwargs.
  `_compute_vol_structure`/`_compute_data_quality` con df vacio -> None.
- `tests/test_pipeline_market_data_flows.py`: tests de flows_secondary
  (los 4 keys del contrato; separado).

No cubren:
- MD-2: fallo de import en `indicators.*` no se simula. El fix
  previene el escenario pero no hay test que lo verifique.
- MD-1: uso real de temporal_meta (no lo usa).

## 4. Falsos positivos descartados

- `except` sin `RuntimeError` (MD-2 original): **si lo incluye**.
  El problema era solo `ImportError`. Verificado contra codigo.
- Atomicidad via `.tmp` + `.replace()`: correcta, con test.
- Patron try/except por helper: coherente con el resto del pipeline
  (A5.5).
- `vol_structure_df = None` cuando df vacio: coherente con el patron
  de los demas modulos.

## 5. Patron observado

`market_data.py` es el primer fichero de S-03-deep donde el hallazgo
no es cosmetica ni docstring, sino **contrato de excepciones**. El
patron (import tardio + except sin ImportError) puede repetirse en
otros ficheros del pipeline. Candidato a busqueda activa en la
proxima sesion.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 1 BAJA corregida en commit
`2238a3f` (ImportError en las 4 tuplas). 6 INFO WONT FIX razonado.

`market_data.py` queda auditado.

## 7. Proximo

Continuar cierre de `src/pipeline/` (13 auditados + este = 14 de 18):
- `diagnostics.py` (91), `indices_intl.py` (69), `regimes.py` (63),
  `data_load.py` (47), `slpm.py` (33), `__init__.py` (2).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
