# S-03-deep - src/report/helpers.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 840c355
- **Fichero:** `src/report/helpers.py` (157 LOC, 6.5 KB)
- **mtime:** 2026-10-02
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (HP-2). Cero ALTA, cero MEDIA.

## 1. Alcance

157 LOC. 6 helpers de formateo y clasificacion consumidos por 13
modulos de `src/report/` (confirmation, darkpool, etf_flows,
flows_international, freshness, header, leaders, market_context,
rankings, sectorial, sector_context, sentiment, slpm, synthesis).

Funciones publicas (por prefijo _):
- `_fmt_num`, `_fmt_signed`, `_fmt_ad_net`.
- `_classify_freshness`, `_classify_finra_freshness`,
  `_classify_fred_freshness`.
- `_generate_coverage_table`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| HP-1 | INFO | L120 | `datetime.now()` fallback si `reference_date=None`. Mismo patron que F-03 de validation_gate. | WONT FIX: coherente. |
| HP-2 | BAJA | L~123 | `sectores_total = 11` hardcoded. `EXPECTED_SECTOR_COUNT` en config.settings. | **FIX APLICADO** (commit 76b66c5). Precedente A5-11/SS-2. |
| HP-3 | INFO | L~125 | Filtro `s[1] is not None` usa nombre (siempre presente en config), no score. Equivalente en produccion. | WONT FIX: cambio rompe tests sin ganancia. |
| HP-4 | INFO | _fmt_num / _fmt_signed | `except Exception` con `return str(v)`. Fail-safe. | WONT FIX. |
| HP-5 | INFO | 3 _classify_* | Triplicado salvo fuente de umbrales (FRESHNESS_DEFAULT / FINRA / FRED). | WONT FIX: claridad por fuente. |
| HP-6 | INFO | L~121, L~142, L~160 | 3 `import pandas as pd` locales. | WONT FIX: cosmetico. |

## 3. Contraste con tests

- `test_report_generator_helpers.py`: 56 tests. Cubre `_fmt_num`
  (9), `_classify_freshness` (9), `_classify_finra_freshness` (7),
  `_classify_fred_freshness` (6), `_generate_coverage_table` (9),
  y helpers de otros modulos (D6b: reference_date).
- `test_fu003a_ceros.py` (FU-003a): `_fmt_signed` con ceros, NaN,
  porcentajes, separadores de miles, ruido flotante (1e-9, 0.004).
- `test_f6_1_finra_freshness.py` (F6-1): umbrales FINRA (20, 28,
  45), 26 dias ya no es CURRENT.
- `test_c3_ad_rendering.py`: `_fmt_ad_net` con politica B3.

No cubren:
- HP-3: el filtro `s[1] is not None` con nombre ausente no se testea
  fuera del caso `(None, None, None, None)`.

## 4. Falsos positivos descartados

- `_fmt_num` con `all(c in '0.,%' for c in rest)`: cubre los formatos
  usados en el pipeline (punto, coma, porcentaje). Correcto.
- `_fmt_signed` con `digits_only.strip('0')`: chequea visualmente si
  el formato sin signo muestra todos ceros. Correcto.
- `_fmt_ad_net` con politica B3 (ad_net=0 con advances/declines>0
  -> '+0'): documentado, con tests, distinto de N/D. Correcto.
- `_generate_coverage_table` con `except FileNotFoundError` (FU-005)
  vs `except (OSError, ValueError, KeyError, ParserError)`: diferencia
  entre "no existe CSV" (estado esperado) y "CSV corrupto" (WARN).
  Correcto.
- D36 (try/except en `pd.Timestamp(last_date)`): protege strings no
  parseables. Correcto.
- Fix C22b (fallback 0 no 110): documentado. Correcto.

## 5. Patron: helpers de formato con cobertura alta

`helpers.py` tiene ~56 tests directos + 3 ficheros mas de tests
tematica (FU-003a, F6-1, C3). Es el fichero con mayor cobertura
proporcional de S-03-deep hasta ahora. Explica 0 hallazgos
materiales fuera de HP-2: la densidad de tests previene regresiones.

Contraste con `macro_regime.py` (0 tests de las 12 ramas):
`helpers.py` es el extremo opuesto del espectro.

## 6. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 1 BAJA corregida
(HP-2, commit 76b66c5). 5 INFO WONT FIX razonado.

`helpers.py` queda auditado.

## 7. Proximo

Continuar `src/report/` (2/20 auditados: flows_international, helpers):
- `synthesis.py` (149), `slpm.py` (137), `leaders.py` (128),
  `rankings.py` (116), `iae.py` (106), `market_context.py` (105),
  `freshness.py` (101), `sector_context.py` (97), `confirmation.py`
  (95), `volatility_mte.py` (89), `sectorial.py` (85),
  `header.py` (71), `alerts.py` (69), `darkpool.py` (62),
  `etf_flows.py` (61), `sentiment.py` (51), `breadth.py` (30),
  `__init__.py` (2).
