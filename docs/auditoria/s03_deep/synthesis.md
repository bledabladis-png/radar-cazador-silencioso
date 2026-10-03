# S-03-deep - src/report/synthesis.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 7cafe22
- **Fichero:** `src/report/synthesis.py` (149 LOC, 9.1 KB)
- **mtime:** 2026-10-02
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. Alcance

149 LOC. 3 funciones de render puro (devuelven list[str]):
- `render_indices_internacionales(index_phases, index_leaders)`.
- `render_sintesis_senales(macro_regime, slpm_v12_data, liquidity_regime,
  sector_dispersion_data, breadth_values, price_flow_divergences)`.
- `render_matriz_evidencia(evidence_matrix_data)`.

Consumidor: `src/report_generator.py:293, 298, 306`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| DS-1 | INFO | _fmt_evidence | `int(v)` trunca floats. Evidence son enteros (-1/0/+1) por contrato upstream. Tests pasan enteros. | WONT FIX. |
| DS-2 | INFO | render_sintesis_senales cierre | Dos disclaimers casi identicos consecutivos. | WONT FIX: cosmetico. |
| DS-3 | INFO | divergencias | Seccion de divergencias solo aparece si `len(resumen) < 3`. | WONT FIX: diseno. |
| DS-5 | INFO | divergencias[:2] | Maximo 2 divergencias mostradas. | WONT FIX. |
| DS-6 | INFO | _fmt_evidence con pd.isna(v) | Sin guarda de tipo. Tests cubren NaN -> NA. | WONT FIX. |
| DS-7 | INFO | render_indices_internacionales | Acceso a row['rs'], ['rs_mom'], ['flow_proxy_z'], ['wls'], ['wyckoff_phase'] sin guard. | WONT FIX: contrato upstream. |

Nota: se verifico el encoding del fichero. `Índices`, `Señales`,
`Régimen` estan en UTF-8 correcto. El `Ãndices` observado inicialmente
era artefacto de visualizacion de consola (CP1252).

## 3. Contraste con tests

- `tests/test_render_modules_low_coverage_d29.py` (D29): bloque
  synthesis con 5 tests:
  - `test_indices_internacionales_con_phases_y_leaders`.
  - `test_indices_internacionales_sin_leaders`.
  - `test_indices_internacionales_leaders_vacio` (DataFrame vacio ->
    skip sin crash).
  - `test_sintesis_senales_recesion`.
  - `test_sintesis_senales_mixed_con_dispersion`.
  - `test_sintesis_senales_mixed_sin_dispersion`.
  - `test_sintesis_senales_lider_slpm_confirmado`.
  - `test_sintesis_senales_lider_slpm_unresolved`.
  - `test_sintesis_senales_liquidez_alta`.
  - `test_sintesis_senales_divergencias_price_flow`.
- `tests/test_fu003a_ampliacion.py`: 2 tests de
  `render_matriz_evidencia` (cero sin signo, NaN -> NA).

No cubren:
- DS-3: el corte por `len(resumen) < 3` no se testea con resumen
  lleno + divergencias presentes.
- DS-7: acceso a columnas especificas del leader DataFrame.

## 4. Falsos positivos descartados

- `_fmt_evidence` con `int(v)` truncando: los tests confirman que
  el contrato es entero (no hay caso con float).
- Encoding: verificado, fichero en UTF-8.
- `if divergencias and len(resumen) < 3`: decision de diseno
  (mostrar divergencias solo si el resumen no esta saturado).
- D29 documenta el fichero: 66% de cobertura previa. Los tests
  cubren ramas con datos y sin datos.

## 5. Patron: render modules con cobertura post-D29

`synthesis.py` es el segundo fichero de `src/report/` auditado
(tras `flows_international.py`) con 0 hallazgos materiales.
Coincide con ser fichero de `src/report/`: codigo puro, listas
markdown, sin efectos secundarios. Los refactors C1 extrajeron
estos modulos de `report_generator.py` sin bugs.

D29 (2026-09-30) subio la cobertura de los render modules de
`src/report/`. Los 3 de D29 (`sector_context`, `sectorial`,
`synthesis`) tienen tests de ramas con datos.

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 6 INFO
WONT FIX razonado.

`synthesis.py` queda auditado.

## 7. Proximo

Continuar `src/report/` (3/20 auditados: flows_international,
helpers, synthesis):
- `slpm.py` (137), `leaders.py` (128), `rankings.py` (116),
  `iae.py` (106), `market_context.py` (105), `freshness.py` (101),
  `sector_context.py` (97), `confirmation.py` (95),
  `volatility_mte.py` (89), `sectorial.py` (85), `header.py` (71),
  `alerts.py` (69), `darkpool.py` (62), `etf_flows.py` (61),
  `sentiment.py` (51), `breadth.py` (30), `__init__.py` (2).
