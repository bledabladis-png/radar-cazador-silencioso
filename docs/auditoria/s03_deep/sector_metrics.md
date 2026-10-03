# S-03-deep - src/pipeline/sector_metrics.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 1836fe2
- **Fichero:** `src/pipeline/sector_metrics.py` (154 LOC, 9 KB)
- **mtime:** 2026-09-29
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

154 LOC. Fase 6b del pipeline (refactor C2-7b). Orquestador de 5
metricas sectoriales derivadas: divergencia sector-lideres,
distribucion Wyckoff, RS interno, concentracion, representatividad
del lider.

Funcion publica: `compute_sector_metrics`. Devuelve dict con 5 keys.

Consumidor: `run.py:133`, con `effective_meta=df_stocks_effective_meta`
(FU-C16).

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SM-1 | INFO | L42 | `_compute_wyckoff` no recibe `temporal_meta` (el resto si). Downstream no lo pide. | WONT FIX. |
| SM-2 | INFO | L146 | `_ref_date=None` propaga a downstream, que hace `raise ValueError`. Fail-loud FU-008 + C4-code. Capturado por el except del wrapper. | WONT FIX. |
| SM-3 | INFO | L112 | Dedup extra en `_compute_concentration` (dropna + drop_duplicates + filtro str vacio). Documentado FU-008-b. | WONT FIX. |
| SM-5 | INFO | L146 | `_ref_date` sin normalizar tz. No hay `datetime.now()`. | WONT FIX. |
| SM-7 | INFO | L18 | `_compute_divergencia` requiere `leader_df`; `_compute_concentration` no (FU-008-b). Documentado. | WONT FIX. |
| SM-8 | BAJA | L128 | `_compute_representativeness` lee ruta hardcoded `outputs/history/sector_concentration.csv`; `_compute_concentration` acepta `sc_path` custom. Asimetria reader/writer. | WONT FIX condicional: nadie usa `sc_path` custom; param existe para tests. |

## 3. Contraste con tests

- `tests/test_pipeline_sector_metrics.py`: contrato de las 5
  sub-funciones, persistencia CSV, degradacion (None/df vacio),
  FU-008-b (leader_df=None no bloquea concentration).
- `tests/test_fu_c16_sector_metrics_meta.py`: FU-C16, propagacion de
  effective_meta a `_compute_concentration` y `_compute_representativeness`;
  fallback a `df_stocks.index[-1]`; firma acepta `effective_meta`;
  `run.py` lo propaga.
- `tests/test_rs_internal_dedup.py`: O2, subset `[date,sector,ticker]`
  en `append_dedup` de rs_internal (evita colapso de tickers).

No cubren:
- SM-8: `sc_path` custom en `_compute_concentration` mientras
  `_compute_representativeness` lee del default.

## 4. Falsos positivos descartados

- El raise ValueError de `compute_sector_concentration` cuando
  `reference_date=None`: es contrato fail-loud (FU-008), no bug. El
  except del wrapper lo captura.
- `_ref_date` desde `effective_meta.get('date')` o
  `df_stocks.index[-1]`: cumple R3 (FU-C16). Cubierto por test.
- Persistencia atomica via `.tmp` + `.replace()`: patron familia 3
  (32 sitios). Cubierto por tests de atomicidad en otros ficheros.

## 5. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. 1 BAJA (SM-8) WONT
FIX condicional; 5 INFO, todos WONT FIX razonado.

`sector_metrics.py` queda auditado. Sin acciones pendientes.

## 6. Proximo

Continuar cierre de `src/pipeline/` (8 auditados + este = 9 de 18):
- `flows_primary.py` (141), `finalize.py` (128), `leaders.py` (108),
  `market_data.py` (101), `diagnostics.py` (91).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
