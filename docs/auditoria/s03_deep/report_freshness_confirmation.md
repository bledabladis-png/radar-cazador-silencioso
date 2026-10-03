# S-03-deep - bloque src/report/ (freshness, confirmation, volatility_mte, sectorial)

- **Auditoria:** 2026-10-03
- **HEAD inicial:** b441f36
- **Ficheros:**
  - `src/report/freshness.py` (101 LOC, 4.7 KB, mtime 2026-09-30)
  - `src/report/confirmation.py` (95 LOC, 5.4 KB, mtime 2026-10-02)
  - `src/report/volatility_mte.py` (89 LOC, 5.5 KB, mtime 2026-09-28)
  - `src/report/sectorial.py` (85 LOC, 7.4 KB, mtime 2026-09-28)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. freshness.py

101 LOC. `render_data_freshness(pcr_data, darkpool_data, sector_results,
reference_date=None)`. 4 fuentes (CBOE, FINRA, FRED, Yahoo) + llamada
a `_generate_coverage_table` de helpers.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FR-1 | INFO | L27 | `datetime.now()` fallback si `reference_date=None`. Mismo patron F-03/HP-1. | WONT FIX: coherente. |
| FR-2 | INFO | L~70, L~87 | Imports locales (`json`, `Path as _Path`, `Path as _P`). | WONT FIX: cosmetico. |
| FR-3 | INFO | L~70 vs L~87 | Dos alias distintos de `Path` en la misma funcion (`_Path`, `_P`). | WONT FIX: cosmetico. |

FU-007 (`_last_market_session`) aplicado en FRED y Yahoo. Correcto.
D36 (try/except en pd.Timestamp) heredado del helper de cobertura.

## 2. confirmation.py

95 LOC. `render_confirmation(confirmation_data)`. Nivel 2: 10Y-3M,
Realized Vol (21/60d), VRP, FLS, A/D, RECESSION CAPITULATION,
Cross-Asset Ratios (12 pares).

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| CF-1 | INFO | L~70 | `ad` usado fuera del `if ad:` en la rama RECESSION con `.get('nh_nl', 0)`. Dict vacio -> 0. Correcto. | WONT FIX. |
| CF-2 | INFO | L~55 | `fls.get('fls_normalized', 0)*100` sin guard NaN. Contrato upstream. | WONT FIX. |
| CF-3 | INFO | ratio_names | 12 ratios hardcoded. Estables. | WONT FIX. |

## 3. volatility_mte.py

89 LOC. 3 funciones:
- `render_estructura_volatilidad`: VIX, percentiles, term structure,
  PCR z. Guard `_fmt` para NaN (A4-b 2026-09-28).
- `render_calidad_datos`: calidad por fuente con groupby idxmax.
- `render_mte`: MTE v1.0 (scenario, MSI, IPI, SRS, SHS, CLS, IPS).

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| VM-1 | INFO | estructura vol | `_fmt` local con guard NaN (A4-b). Correcto. | WONT FIX. |
| VM-2 | INFO | render_mte | `mte_result.get('cls', 0):.2f` sin guard NaN. Contrato upstream. | WONT FIX. |
| VM-3 | INFO | calidad datos | `groupby('source')['date'].idxmax()`. Correcto. | WONT FIX. |

A4-b (2026-09-28): el `term_structure_ratio=NaN` que se renderizaba
como 'nan' literal quedo resuelto con `_fmt`.

## 4. sectorial.py

85 LOC. 3 funciones:
- `render_sector_breadth`: tabla completa con [BAJA] por cobertura.
- `render_sector_concentration`: Top1/3/5 + medianas + lider.
- `render_sector_dispersion`: percentiles RS y Mom.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SE-1 | INFO | render_sector_breadth | 3 notas al pie (F6-01, F6-04, columna A/D). Intencional. | WONT FIX. |
| SE-2 | INFO | render_sector_breadth | `_fmt_ad_net` aplicado (politica B3). Correcto. | WONT FIX. |
| SE-3 | INFO | 3 funciones | Bloque "pd.to_datetime(...).max() + filtro" repetido 3 veces (mas 4 en sector_context.py). 7 sitios en total. | WONT FIX: DRY bajo, ROI < 1. |
| SE-4 | INFO | render_sector_dispersion | `row['rs_p25']:.4f` sin `_fmt_num`. Contrato upstream: no NaN. | WONT FIX. |

F6-01 (stale_reason con 3 causas: MARKET_CLOSED/DATA_PENDING/ERROR)
documentado y testado. F6-04 (cobertura cercana al 100% esperada)
documentado.

## 5. Contraste con tests

- `tests/test_render_freshness_d36.py` (D36): 15 tests. Cubre
  render_data_freshness con 4 fuentes (CBOE, FINRA, FRED, Yahoo),
  reference_date, parquet vacio, formatos.
- `tests/test_f6_2_stale_reason.py` (F6-2): stale_reason con 3 causas.
- `tests/test_c2f_breadth_fallback.py`: is_stale=True con fallback.
- `tests/test_c3_ad_rendering.py`: `_fmt_ad_net` politica B3.
- `tests/test_render_aclaraciones_d2c_d3_e1_a1.py`: aclaraciones
  D2c, D3, E1, A1.
- `tests/test_render_modules_low_coverage_d29.py` (D29): ramas con
  datos para `render_sector_concentration` y `render_sector_dispersion`.
- `tests/test_report_sections_undercovered.py`: 12 tests para secciones
  con poca cobertura (incluye volatility_mte, confirmation).

No cubren:
- FR-3: los alias de Path no se testean.
- SE-3: bloque de filtro por fecha no se testea con historicos largos.

## 6. Patron: render puro de src/report/

Con este bloque, `src/report/` acumula 12/20 ficheros auditados:
- 1 CON FIX: `helpers.py` (HP-2).
- 1 CON FIX: `leaders.py` + `rankings.py` (LD-1, RK-1, mismo commit).
- 1 CON FIX: `flows_international.py` (FI-2 WONT FIX).
- 9 SIN CAMBIOS.

Confirma el patron: los renders de `src/report/` estan limpios. Los
hallazgos (HP-2, LD-1, RK-1) son siempre el mismo patron "11
hardcoded".

## 7. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 13 INFO
WONT FIX razonado.

Los 4 ficheros quedan auditados.

## 8. Proximo

Continuar `src/report/` (12/20 auditados): `header.py` (71),
`alerts.py` (69), `darkpool.py` (62), `etf_flows.py` (61),
`sentiment.py` (51), `breadth.py` (30), `__init__.py` (2).
