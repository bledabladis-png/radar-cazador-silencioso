# S-03-deep - src/report/flows_international.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** c2f5b7d
- **Fichero:** `src/report/flows_international.py` (191 LOC, 12.5 KB)
- **mtime:** 2026-10-02
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

191 LOC. 10 funciones `render_*`, todas con la misma estructura:
reciben un DataFrame o None, devuelven `list[str]` markdown. Sin
efectos secundarios. Fase C1-6d4a (extraccion desde
`src/report_generator.py`).

Funciones:
- `render_flujo_daxex`, `render_flujo_isf`, `render_flujo_lyxi`,
  `render_flujo_iwm`: flujos primarios BlackRock/Amundi.
- `render_flujo_qqq_sec`: QQQ SEC (N-30B-2 / N-CSRS).
- `render_posicionamiento_cftc`: CFTC TFF semanal.
- `render_flujo_posicional_nport`: N-PORT trimestral.
- `render_rendimiento_qqq`: rendimientos Yahoo.
- `render_qqq_nport_flow`: NPORT-P Item B.6.
- `render_flujo_sintesis`: sintesis descriptiva multi-capa.

Consumidor: `src/report_generator.py:44`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FI-1 | INFO | L33 (ISF) | Etiqueta "GBP" con columna `estimated_flow_eur`. El valor es correcto (la columna se produce sin conversion de divisa en `_blackrock_base.py:227`, asi que ISF en GBP y DAXEX en EUR), pero el nombre de columna es enganoso. | WONT FIX: renombrar columna toca provider + tests + render. ROI < 1. Mismo patron que 313/313. |
| FI-2 | BAJA | L69, L97, L113 | `.get(k, 0)` con columna presente y valor NaN -> `f"{NaN:,.0f}"` lanza ValueError. `.get` solo aplica default si la clave no existe. Latente: los proveedores (SEC/BlackRock) garantizan int en esas columnas. | WONT FIX condicional: reabrir si aparece NaN en produccion. |
| FI-3 | BAJA | L129 | `int(row['month'])` sin guard. Mismo tipo que FI-2. | WONT FIX condicional. |
| FI-4 | INFO | L100 | Comentario huerfano `# FLUJO DE PARTICIPACIONES QQQ (NPORT-P)` sin codigo debajo. | WONT FIX: cosmetico. |
| FI-5 | INFO | varias | Guard `hasattr(row['date'], 'strftime')` es False para NaT -> imprime "NaT". | WONT FIX: cosmetico. |
| FI-6 | INFO | L48 | `row['estimated_flow_eur']` sin `.get` dentro de `if pd.notna(shares_change)`. | WONT FIX: contrato upstream. |
| FI-7 | INFO | varias | Nomenclatura IWM (`primary_flow_*`) vs DAXEX/ISF (`estimated_flow_*`). Esquemas upstream distintos. | WONT FIX. |

## 3. Contraste con tests

- `tests/test_report_market_context_flows_intl.py`: cubre None y df
  vacio para todas las funciones, basico por funcion, LYXI con
  shares_change NaN (placeholder N/D), fallback effectiveDate.
- `tests/test_fu003a_render.py`: FU-003a, % AUM a 2 decimales, cero
  sin signo.
- `tests/test_i3_effective_date_render.py`: I3, effectiveDate con
  prioridad sobre as_of_date; fallback a as_of_date truncado.

No cubren:
- FI-1: la etiqueta "GBP" no se verifica como GBP real.
- FI-2: NaN en columna presente (solo claves ausentes).
- FI-3: NaN en `month`.

## 4. Falsos positivos descartados

- El patron `if df is not None and not df.empty: row = df.iloc[-1]`
  es correcto (data vacio -> lista vacia).
- `render_flujo_sintesis({})` -> lista vacia (test lo cubre).
- El default `flow_synthesis.get('european_flow_sign', 0)` es correcto
  (test `test_flujo_sintesis_european_default_cero` lo cubre).
- El try/except de `render_rendimiento_qqq` y `render_qqq_nport_flow`
  captura las excepciones esperadas (AttributeError, TypeError,
  ValueError, KeyError / ValueError, TypeError, AttributeError).
  Correcto por diseno.

## 5. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. 2 BAJA (`.get(k, 0)` con
NaN, `int(month)` sin guard) marcadas WONT FIX condicional (reabrir
si aparece NaN en produccion) y 5 INFO, WONT FIX razonado.

`flows_international.py` queda auditado. Sin acciones pendientes.

## 6. Proximo

Continuar S-03-deep por LOC descendente en `src/report/`:
`helpers.py` (157), `synthesis.py` (149), `slpm.py` (137).
Y `regimes/`: `sector_regime.py` (210), `macro_regime.py` (205).
