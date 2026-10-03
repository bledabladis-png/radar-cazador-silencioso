# S-03-deep - src/pipeline/flows_primary.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 560cdce
- **Fichero:** `src/pipeline/flows_primary.py` (141 LOC, 7.1 KB)
- **mtime:** 2026-09-29
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. Alcance

141 LOC. Fase 4 del pipeline (refactor C2-6a). Orquestador de 7
providers externos (SSGA, BlackRock DAXEX/ISF/IWM, Amundi LYXI, QQQ
SEC, CFTC) + 1 indicador derivado (Sector Flow Characteristics).

Funcion publica: `compute_flows_primary(df_market, temporal_meta=None)`.
Devuelve dict con 8 keys.

Consumidor: `run.py:100`, con `temporal_meta=temporal_meta`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FP-1 | INFO | L30 | `temporal_meta` no se usa en el cuerpo. Consumidor (run.py:100) lo pasa. | WONT FIX: contrato de API. |
| FP-2 | INFO | L45 | `compute_sector_flow_characteristics` lee `outputs/history/etf_primary_flow.csv`, escrito por `get_etf_primary_flow_data` (SSGA, `ssga_fund_data.py:17`, fix D 2026-09-30 append_dedup). Patron reader-after-writer dentro del mismo pipeline. | WONT FIX: diseno + test. |
| FP-3 | INFO | varias | Rutas relativas `outputs/history/*.csv`. | WONT FIX: CWD raiz. |
| FP-4 | INFO | varias | Except tuplas distintas: bloque 1 (7 providers) sin `EmptyDataError`; bloque 2 (sector_flow_char) con. Coherente con naturaleza (lectura CSV vs provider). | WONT FIX. |

## 3. Contraste con tests

`tests/test_pipeline_flows_primary.py` (7 tests):
- `test_flows_primary_contract_8_keys`: 8 keys siempre.
- `test_flows_primary_propaga_df_de_cada_provider`: 8 propagaciones.
- `test_flows_primary_fallo_dax_no_rompe_otros`: fallo individual
  no bloquea los demas.
- `test_flows_primary_empty_df_se_convierte_a_none`: df.empty -> None.
- `test_flows_primary_sector_char_no_se_calcula_sin_etf_flow`:
  compute_sector_flow_characteristics no se invoca si
  etf_primary_flow_data es None.
- `test_sector_flow_characteristics_fallo_to_csv_no_pierde_filas`:
  atomicidad familia 3.

No cubren:
- FP-1: uso real de temporal_meta (no lo usa).
- FP-2: el writer del CSV queda mockeado (los tests usan DataFrames
  directos).

## 4. Falsos positivos descartados

- `retry_call(get_*_primary_flow)` sin args extra: `retry_call(f,
  *args, retries=3, backoff=2, **kwargs)` en `retry_utils.py:7` con
  defaults sanos.
- `get_qqq_sec_primary_flow()` sin retry_call: el provider QQQ SEC
  no hace peticiones de red (lee de disco o cache ya escrita); no
  necesita reintentos.
- `except (...)` con `pd.errors.EmptyDataError` solo en bloque 2:
  coherente porque bloque 2 hace `pd.read_csv` (historico); los
  providers no.

## 5. Patron observado

`flows_primary.py` es simetrico de `flows_secondary.py` (ya
auditado). Ambos orquestan providers externos con try/except por
bloque, devuelven dict de DataFrames-or-None, y comparten el patron
reader-after-writer. `flows_secondary.py` dio 1 BAJA corregida
(FS-1, mtime UTC); `flows_primary.py` da 0 hallazgos materiales.
La diferencia: `flows_secondary` maneja timestamps de fichero
(`st_mtime`); `flows_primary` no.

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 4 INFO,
todos WONT FIX razonado.

`flows_primary.py` queda auditado. Sin acciones pendientes.

## 7. Proximo

Continuar cierre de `src/pipeline/` (9 auditados + este = 10 de 18):
- `finalize.py` (128), `leaders.py` (108), `market_data.py` (101),
  `diagnostics.py` (91), `indices_intl.py` (69).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
