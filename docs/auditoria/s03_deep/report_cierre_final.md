# S-03-deep - cierre src/report/ (7 ficheros menores)

Cierre de `src/report/` al 100% (20/20). Este documento cubre los
7 ficheros restantes.

- **Auditoria:** 2026-10-03
- **HEAD inicial:** aeb7191
- **Ficheros:**
  - `src/report/header.py` (71 LOC, 3.7 KB, mtime 2026-09-28)
  - `src/report/alerts.py` (69 LOC, 3.3 KB, mtime 2026-09-30)
  - `src/report/darkpool.py` (62 LOC, 3.6 KB, mtime 2026-09-30)
  - `src/report/etf_flows.py` (61 LOC, 5.2 KB, mtime 2026-10-02)
  - `src/report/sentiment.py` (51 LOC, 3.0 KB, mtime 2026-09-30)
  - `src/report/breadth.py` (30 LOC, 1.6 KB, mtime 2026-09-11)
  - `src/report/__init__.py` (2 LOC)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (BR-1). Cero ALTA, cero MEDIA.

## 1. header.py

71 LOC. `render_regimenes(...)`. Resumen de regimenes (Macro, Cond.
Financieras, Liquidez Real con Liquidity Delta, Volatilidad, Sectores).

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| HD-1 | INFO | render_regimenes | try/except robusto en `macro_score.iloc[-1]`. Coherente con patron del modulo. | WONT FIX. |
| HD-2 | INFO | vol_conf | Logica "Senal neutra (sin desviacion significativa)" cuando `vol_conf < 0.05 and abs(vol_z) < 0.1`. Documentado. | WONT FIX. |

## 2. alerts.py

69 LOC. `render_alerts` + `render_cross_module`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| AL-1 | INFO | render_alerts | Fix F (2026-09-30) aplicado: `div.get('message') or '(sin detalle)'` en lugar de literal fijo. Test test_alerts_price_flow_message.py lo cubre. | WONT FIX. |
| AL-2 | INFO | render_cross_module | Iconos OK/WARN/INFO por conflict_level. Correcto. | WONT FIX. |

## 3. darkpool.py

62 LOC. `render_darkpool`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| DP-1 | INFO | L~15 | `datetime.now()` fallback. Patron F-03/HP-1/FR-1. | WONT FIX. |
| DP-2 | INFO | z_score | `pd.notna(darkpool_data.get('z_score'))` sin default. Cae a `else` si falta. Correcto. | WONT FIX. |
| DP-3 | INFO | week | Bloque week procesado 2 veces (freshness + display). Cosmetico. | WONT FIX. |

## 4. etf_flows.py

61 LOC. `render_flujo_spdr`, `render_flujo_caracteristicas`,
`render_divergencia_precio_flujo`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| EF-1 | INFO | render_flujo_spdr | O1 (2026-09-18) + F6-29b (2026-09-28) aplicados. Test test_i1_o1_notas.py. | WONT FIX. |

## 5. sentiment.py

51 LOC. `render_sentimiento_opciones`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SN-1 | INFO | L~17 | `datetime.now()` fallback. Patron F-03/HP-1/FR-1/DP-1. | WONT FIX. |
| SN-2 | INFO | timestamp | `pcr_data.get('timestamp', 'N/A')` sin formateo. | WONT FIX. |

## 6. breadth.py

30 LOC. `render_breadth_market`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| BR-1 | BAJA | 11 sitios | `11` hardcoded: titulo, 5 defaults `* 11`, 5 f-strings `/11`. | **FIX APLICADO** (commit del fix). |

## 7. __init__.py

2 LOC. Solo docstring. Nada que auditar.

## 8. Contraste con tests

- `tests/test_alerts_price_flow_message.py`: Fix F aplicado.
- `tests/test_report_alerts_header.py`: render_regimenes + render_alerts
  (12 tests).
- `tests/test_report_sections_undercovered.py`: 12+ tests para
  header, alerts, darkpool, sentiment, breadth.
- `tests/test_i1_o1_notas.py`: O1 (fecha efectiva en SPDR).
- `tests/test_fu003a_render.py`: helpers de formato en IWM y DAXEX.
- `tests/test_render_aclaraciones_d2c_d3_e1_a1.py`: aclaraciones
  en estos renders.
- `tests/test_report_market_context_flows_intl.py`: darkpool y
  sentiment.

No cubren:
- BR-1: el valor 11 en cada uno de los 11 sitios (contrato singular).
- DP-3: doble procesado del bloque week.

## 9. Estado del cierre de src/report/

20/20 ficheros auditados. Distribucion:
- SIN CAMBIOS: 16.
- CON FIX: 4 (helpers HP-2; leaders+rankings LD-1/RK-1; breadth BR-1;
  flows_international FI-2 WONT FIX condicional).
- 0 ALTA, 0 MEDIA. 5 BAJA corregidas (HP-2, LD-1, RK-1, BR-1).
- 1 BAJA WONT FIX (FI-2).

## 10. Estado global de S-03-deep

3 rutas cerradas:
- `src/pipeline/`: 18/18.
- `regimes/`: 8/8.
- `src/report/`: 20/20.

Total: 46/46 ficheros auditados. Cero ALTA, cero MEDIA.
- 11 BAJA corregidas (FS-1, SS-1, SS-2, MD-2, LD-1 report/pipeline,
  II-1, II-2, DL-1, SC-2, HP-2, LD-1 report/report, RK-1, BR-1).
- 6 BAJA WONT FIX.
- 90+ INFO WONT FIX.

## 11. Patrones confirmados en S-03-deep

1. **"11 hardcoded"**: 7 ficheros corregidos. Recurrente en codigo
   que itera sobre los 11 sectores. Candidato a regla de estilo:
   "usar EXPECTED_SECTOR_COUNT siempre".
2. **Docstring desalineado tras refactor**: 3 casos (LD-1, II-1,
   DL-1). Candidato a barrido final.
3. **datetime.now() fallback**: 6 casos WONT FIX (F-03, HP-1, FR-1,
   DP-1, SN-1, + validation_gate). Coherente con contrato.
4. **Atomicidad CSV incompleta**: 1 caso corregido (II-2).
5. **except sin ImportError en import tardio**: 1 caso corregido
   (MD-2).
6. **Ficheros post-refactor C1/C2 + cobertura D29**: limpios.

## 12. Proximo

S-03-deep cerrado. Candidatos siguientes:
- Barrido de docstrings desalineados en `src/` (patron 2).
- Barrido de "11 hardcoded" en todo el repo (patron 1).
- Caracterizacion de `macro_regime.py` (12 ramas sin test directo).
- `src/temporal_contracts/` (11 ficheros, A1 CERRADO).
- `src/institutional_accumulation/` (36 ficheros, B/6 CERRADOS).
