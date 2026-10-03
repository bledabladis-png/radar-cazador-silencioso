# S-03-deep - src/pipeline/sectors_base.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 93e92c1
- **Fichero:** `src/pipeline/sectors_base.py` (155 LOC, 7.8 KB)
- **mtime:** 2026-09-29
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (SS-1 + SS-2). Cero ALTA, cero MEDIA.

## 1. Alcance

155 LOC. Fase 3 del pipeline (refactor C2-5). Orquestador de
rankings sectoriales, rotacion historica, dispersion, correlacion
y cross-asset.

Funcion publica unica: `compute_sectors_base(df_market, temporal_meta=None)`.
Devuelve dict con 10 keys (docstring previo declaraba 11).

Consumidor: `run.py:88`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SS-1 | INFO | docstring | Docstring declaraba `cross_asset_detail_df` como key; return real tiene 10 keys, no 11. La variable se persiste a disco pero no se devuelve. Test `test_sectors_base_contract_keys` verifica 10 keys. | **FIX APLICADO** (commit 2af7e49). |
| SS-2 | BAJA | breadth_values | 5 ocurrencias de `* 11` hardcoded (EMA20/EMA50/EMA200/NH/NL count). `EXPECTED_SECTOR_COUNT = 11` en `config/settings.py:63`. | **FIX APLICADO** (commit 2af7e49). Precedente A5-11. |
| SS-3 | INFO | L~24 | `compute_sector_scores(df_market)` sin `temporal_meta` (el resto de sub-calculos si lo reciben). | Diferido a `regimes/sector_regime.py`. |
| SS-4 | BAJA | L~113-127 | Asimetria cross-asset: si `detail` no vacio pero `summary` vacio, el else pone `detail=None` despues de haberlo persistido a disco. Disco y dict divergen. | WONT FIX condicional. |
| SS-5 | INFO | L~28 | `sector_results['ranking']` sin guard. | WONT FIX: contrato de `compute_sector_scores`. |

## 3. Contraste con tests

`tests/test_pipeline_sectors_leaders.py`:
- `test_sectors_base_contract_keys`: contrato de 10 keys.
- `test_sectors_base_breadth_values_derivados`: `* 11` ->
  EMA20/EMA50/EMA200 count == 6 con b20=0.5.
- `test_sectors_base_degrada_sub_bloque_dispersion`: excepcion en
  dispersion -> `sector_dispersion_df=None`.
- 4 tests de atomicidad to_csv (familia 3): fallo no pierde filas
  en sector_dispersion.csv, sector_correlation_summary.csv,
  cross_asset_correlation.csv, cross_asset_context.csv.

No cubren:
- SS-1: no verifica docstring vs return.
- SS-3: no verifica propagacion de temporal_meta a
  compute_sector_scores.
- SS-4: no simula detail no-vacio + summary vacio.

## 4. Falsos positivos descartados

- `if sector_results:` truthy-check antes de `['ranking']`: cubierto
  por contrato upstream y tests.
- Patron try/except por sub-bloque: coherente con el resto del
  pipeline (patron A5.5). No es bug.
- Guard `if not sector_dispersion_df.empty:` + atomicidad `.tmp` +
  `.replace()`: correcto. Tests de familia 3 lo verifican.

## 5. Patron observado

`sectors_base.py` es el primer fichero auditado de S-03-deep con
mtime >= 1 semana (2026-09-29, cinco dias). Los 7 anteriores eran
mtime 2026-10-02/03. Coincide con mayor densidad de hallazgos
corregibles: 1 INFO + 1 BAJA con fix real (SS-1, SS-2), frente a
los 7 previos que dieron 0-2 BAJA sin fix.

Refuerza la hipotesis de que las zonas no tocadas por el refactor
C2 y los fixes D-04/D-05/D23 concentran el resto del valor de la
auditoria.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 2 corregidos (SS-1, SS-2)
en commit `2af7e49`; 1 INFO diferido (SS-3) a `sector_regime.py`;
2 WONT FIX razonado (SS-4 condicional, SS-5).

`sectors_base.py` queda auditado.

## 7. Proximo

Continuar cierre de `src/pipeline/` (18 ficheros totales, 7
auditados + este = 8):
- `sector_metrics.py` (154), `flows_primary.py` (141),
  `finalize.py` (128), `leaders.py` (108).
Y abrir `regimes/`: `sector_regime.py` (210) para cerrar SS-3.
