# sector_regime_matrix schema v3 - contrato

**Fecha:** 2026-11-03.
**Estado:** APROBADO 2026-11-03 (sesion 27 parte 4).
**Motivado por:** columna "Fase dominante" = RANGE en los 11 sectores
del reporte. Auditoria: no era bug de calculo, era problema de
informativa.
**Documentos relacionados:**
- `Consolidacion_Documentos/04_HISTORICO.md` (entrada 2026-11-03 "Fase dominante excluye RANGE").
- `indicators/sector_regime_matrix.py` (implementacion).
- `src/report/sector_context.py` (render).
- `src/pipeline/output_registry.py` (registro del output).

---

## 1. Motivo

La columna "Fase dominante" de la Matriz de Regimen Sectorial era RANGE
en los 11 sectores. No era bug de calculo: la moda bruta de fases de
holdings da RANGE porque la mayoria de holdings no completan cascada
Wyckoff en una sesion dada. Pero la columna era **decorativa**: siempre
RANGE. La senal real (la 2a fase mas frecuente) quedaba oculta.

## 2. Cambio

Aditivo. `wyckoff_phase` original intacta (historico no se rompe). 3
columnas nuevas:

| Columna | Tipo | Descripcion |
|---|---|---|
| `wyckoff_signal_phase` | str | Moda de las 6 fases con senal. Fallback `RANGE`. |
| `wyckoff_signal_share` | float 0-100 | % del sector en la fase ganadora. |
| `wyckoff_signal_n` | int | Count absoluto de holdings en esa fase. |

## 3. Semantica

**Pool de fases con senal** (6):
`ACCUMULATION`, `ACCUMULATION_FORMING`, `MARKUP`, `DISTRIBUTION`,
`DISTRIBUTION_FORMING`, `MARKDOWN`.

**Excluidas del pool:**
- `RANGE`: ausencia de senal, no una fase Wyckoff. Si ningun holding
  tiene senal, se publica `RANGE` con `share=0.0`, `n=0`.
- `CONFLICT`: ambiguedad, no senal direccional.
- `INSUFFICIENT_DATA`: no evaluable.

**Desempate determinista** (si dos fases empatan en %):
`MARKDOWN` > `DISTRIBUTION` > `DISTRIBUTION_FORMING` >
`ACCUMULATION_FORMING` > `ACCUMULATION` > `MARKUP`.

Orden conservador: en empates, gana la fase de mayor deterioro. No
implica prediccion (todas son descriptivas).

## 4. Ejemplo con datos reales (2026-10-30)

| Sector | wyckoff_phase (bruta) | wyckoff_signal_phase | share | n |
|---|---|---|---|---|
| XLK | RANGE | DISTRIBUTION_FORMING | 20.0 | 5 |
| XLF | RANGE | MARKDOWN | 8.82 | 3 |
| XLV | RANGE | MARKUP | 11.54 | 3 |
| XLE | RANGE | DISTRIBUTION | 10.00 | 2 |
| XLY | RANGE | DISTRIBUTION_FORMING | 9.52 | 2 |
| XLP | RANGE | DISTRIBUTION | 5.00 | 1 |
| XLI | RANGE | MARKDOWN | 11.36 | 5 |
| XLB | RANGE | DISTRIBUTION | 10.00 | 2 |
| XLU | RANGE | MARKDOWN | 4.76 | 1 |
| XLRE | RANGE | DISTRIBUTION | 5.00 | 1 |
| XLC | RANGE | MARKDOWN | 10.53 | 2 |

`wyckoff_phase` sigue siendo RANGE en los 11 (correcto: moda bruta).
`wyckoff_signal_phase` muestra la senal real.

## 5. Render en el reporte

Formato de la columna "Fase dominante" del `.md`:
- Senal detectada: `DIST_FORMING (20%, N=5)`.
- Sin senal (fallback): `RANGE (sin senal)`.
- N/D (dato ausente): `N/D`.

La nota del render declara el contrato (por que RANGE no es fase, que
`wyckoff_phase` sigue en el CSV para auditoria). Texto en
`src/report/sector_context.py::render_matriz_regimen`.

## 6. Lo que NO cambia

- `wyckoff_phase` (moda bruta con RANGE): sin cambios. Auditoria
  historica de la serie previa.
- `structure_ratio` y `n_estructuras`: sin cambios.
- Calculo de holdings de cada fase (via `classify_phase`): sin cambios.
- Los 6 CSVs descargables de Wyckoff: sin cambios.
- Semantica de "no clasificar el ETF como entidad": sin cambios
  (retirado en sesion 3).

## 7. Invariantes

- I1: `wyckoff_signal_phase` pertenece al conjunto
  {6 fases con senal, RANGE, N/D}. Nunca CONFLICT ni
  INSUFFICIENT_DATA.
- I2: `wyckoff_signal_share == 0.0` si y solo si `wyckoff_signal_phase
  == "RANGE"` o `"N/D"`.
- I3: `wyckoff_signal_n == 0` si y solo si `wyckoff_signal_phase
  == "RANGE"` o `"N/D"`.
- I4: toda fila de la tabla del `.md` cumple `wyckoff_signal_share =
  pct_*_share` del CSV (misma escala 0-100, sin doble transformacion).
- I5: desempate determinista por el orden documentado en seccion 3.
  Dos ejecuciones sobre el mismo input producen el mismo output.

## 8. Rollback

Revertir `a89eb1f5` (fix en codigo) + `1335cb72` (tests adaptados).
`wyckoff_phase` intacta siempre; las columnas `wyckoff_signal_*` dejan
de aparecer. `schema_version` vuelve a 2.

## 9. Tests

- `tests/test_sector_regime_matrix.py::test_signal_phase_excluye_range`
- `tests/test_sector_regime_matrix.py::test_signal_phase_fallback_range_sin_senal`
- `tests/test_sector_regime_matrix.py::test_signal_phase_desempate_determinista`
- `tests/test_sector_regime_matrix.py::test_signal_phase_forming_compone_pool`
- `tests/test_report_fase_dominante_n_valid.py::test_matriz_nota_fase_dominante_aclaratoria`
- `tests/test_output_writer_schema.py` (T1-T6, adaptados a v3).

## 10. Firma

Diseno e implementacion: sesion 27 (2026-11-03).
Verificacion en produccion: pendiente del proximo run scheduled real
(item `VERIFICAR-WYCKOFF-PHASE-PROD` OPEN).
