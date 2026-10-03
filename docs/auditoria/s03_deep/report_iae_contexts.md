# S-03-deep - bloque src/report/ (iae, market_context, sector_context)

- **Auditoria:** 2026-10-03
- **HEAD inicial:** f340580
- **Ficheros:**
  - `src/report/iae.py` (106 LOC, 4.7 KB, mtime 2026-09-24)
  - `src/report/market_context.py` (105 LOC, 7.5 KB, mtime 2026-09-28)
  - `src/report/sector_context.py` (97 LOC, 6.9 KB, mtime 2026-10-02)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA.

## 1. iae.py

106 LOC. `render_iae_section(iae_section)`. 4 ramas por status:
- `None` -> [].
- `STALE` con `official_list_pending` -> aviso SEC PDF/TXT.
- `STALE` con `insufficient_quarters` -> minimo informativo.
- `ERROR` -> aviso con `error`.
- `OK` -> seccion completa (NIPC, cobertura, observabilidad).

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| IA-1 | INFO | _fmt_int | `int(v)` trunca floats. Contrato entero. | WONT FIX. |
| IA-2 | INFO | _fmt_int | Separador de miles custom (`.` en vez de `,`). Estilo europeo elegido. | WONT FIX. |
| IA-3 | INFO | status desconocido | Rama `status != "OK"` con `**Estado desconocido: {status}**`. | WONT FIX: fail-safe. |

## 2. market_context.py

105 LOC. 5 funciones render puro:
- `render_liderazgo_interno`: top-5 tickers por sector.
- `render_rotacion_reciente`: rank deltas (F6-18, F6-19).
- `render_dispersion_sectores`.
- `render_correlacion_sectores`: solo la ultima fecha.
- `render_contexto_cross_asset`: pivote sector/window/asset_class.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| MC-1 | INFO | render_liderazgo_interno | `pd.to_datetime(...)` sin manejo de errores. Contrato upstream: date valida. | WONT FIX. |
| MC-2 | INFO | render_contexto_cross_asset | `grouped.iterrows()` con `row.get(...)` + `pd.notna` en _fmt local. Correcto. | WONT FIX. |

F6-18 y F6-19 documentados en el codigo (deltas negativos en rank=1;
umbrales de la etiqueta Lectura).

## 3. sector_context.py

97 LOC. 5 funciones render puro:
- `render_matriz_regimen`.
- `render_representatividad_lider`: filtro ultima fecha (C2 fix 2026-09-24).
- `render_wyckoff_sectorial`: filtro ultima fecha.
- `render_divergencia_sector_lideres`: filtro ultima fecha (C3 fix 2026-09-24).
- `render_momentum_amplitud`: filtro ultima fecha.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SC-1 | INFO | 4 funciones | Bloque "filtrar a la ultima fecha" repetido. Candidato a helper `_filter_latest_date(df)`. | WONT FIX: 4 repeticiones, DRY bajo, ROI < 1. |
| SC-2 | INFO | render_matriz_regimen | Sin filtro por fecha (las otras 4 si). Contrato: el DataFrame llega ya filtrado. | WONT FIX: asimetria documentada. |
| SC-3 | INFO | render_matriz_regimen | `row['positive_conditions']:.0f` asume numerico. Contrato upstream. | WONT FIX. |

F6-12 (filtro de sectores en representatividad y divergencia)
documentado en las notas.

## 4. Contraste con tests

- `tests/test_report_iae.py` (11+ tests): STALE (2 razones), OK
  (NIPC, cobertura, observabilidad), ERROR.
- `tests/test_report_market_context_flows_intl.py` (bloque
  market_context): 5 funciones con datos vacios y con datos.
- `tests/test_fu003a_ampliacion.py`: `render_rotacion_reciente` con
  ceros sin signo.
- `tests/test_render_modules_low_coverage_d29.py` (D29): ramas con
  datos para `render_matriz_regimen`, `render_wyckoff_sectorial`,
  `render_momentum_amplitud`, `render_sector_concentration`,
  `render_sector_dispersion`, `render_indices_internacionales`,
  `render_sintesis_senales`.
- `tests/test_render_representatividad_lider.py`.
- `tests/test_render_divergencia_sector_lideres.py`.
- `tests/test_render_aclaraciones_d2c_d3_e1_a1.py`.

No cubren:
- IA-1/SC-3: no se testea con floats. Contrato entero.

## 5. Nota sobre encoding

Los renders tienen tildes en los headers (`RÃ©gimen`, `DispersiÃ³n`,
`CorrelaciÃ³n`, etc.). El examen de otros ficheros del corpus
(helpers.py, synthesis.py) confirma que la convencion del repo es
UTF-8 sin BOM. Las apariencias `Ã©`/`Ã³` en consola son el mismo
artefacto CP1252 ya conocido (documentado en 01_METODO seccion 6).

## 6. Patron: render puro de src/report/ sin hallazgos

Los 3 ficheros de este bloque + synthesis.py + flows_international.py
(5 de 6 auditados de src/report/) siguen el mismo patron: 0 hallazgos
materiales. Coincide con la observacion repetida en S-03-deep:

- Modulos post-refactor C1 (src/report/) y C2 (src/pipeline/):
  limpios.
- Modulos con cobertura post-D29: sin hallazgos.
- Los fixes reales aparecen en modulos con mtime >= 1 semana y
  cobertura desigual (sectors_base, flows_secondary, market_data,
  indices_intl, leaders, data_load, helpers, sector_regime).

## 7. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA, cero BAJA. 8 INFO
WONT FIX razonado.

Los 3 ficheros quedan auditados.

## 8. Proximo

Continuar `src/report/` (8/20 auditados): `freshness.py` (101),
`confirmation.py` (95), `volatility_mte.py` (89), `sectorial.py`
(85), `header.py` (71), `alerts.py` (69), `darkpool.py` (62),
`etf_flows.py` (61), `sentiment.py` (51), `breadth.py` (30),
`__init__.py` (2). Y `market_context` ya auditado en este bloque.
