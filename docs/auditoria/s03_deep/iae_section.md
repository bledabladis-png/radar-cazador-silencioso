# S-03-deep - src/pipeline/iae_section.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** e869446
- **Fichero:** `src/pipeline/iae_section.py` (216 LOC, 9.2 KB)
- **mtime:** 2026-10-02
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

216 LOC. Fase 13 del pipeline. Orquestador del NIPC contractual
sobre el ultimo par de trimestres 13F disponibles.

Funciones:
- `_list_available_quarters`: lista `data/sec_13f/processed/20XXQN/`
  con `INFOTABLE.parquet`.
- `_get_catalog_total`: lee `catalog_manifest.json`, snapshot vigente.
- `compute_iae_section`: orquestador, 4 ramas (STALE-insuf, STALE-official,
  ERROR, OK).

Consumidor: `run.py:229` (seccion adicional del reporte).
Renderer: `src/report/iae.py`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| IS-1 | INFO | firma | reference_date/run_id no se usan en el cuerpo (docstring: para logs). Consumidor run.py:229 los pasa. | WONT FIX: contrato de API. |
| IS-2 | BAJA | L~215 | _get_catalog_total() or 242 - fallback obsoleto (catalogo hoy 255). | WONT FIX: activo solo si manifest falla. |
| IS-3 | INFO | L~178 | except Exception - contrato deliberado (docstring: no aborta run.py). Tests cubren. | WONT FIX. |
| IS-4 | INFO | L~215 | max(q4, q1) - sobreestima si q1 > q4. En practica iguales. | WONT FIX. |
| IS-5 | INFO | L~172 | import json local en _get_catalog_total. | WONT FIX: cosmetico. |
| IS-6 | BAJA | L~168 vs L~196 | Formato error inconsistente: STALE-official guarda msg; ERROR guarda type+str. | WONT FIX: ningun consumidor parsea error. |
| IS-7 | INFO | L~245 | coverage_status sin default. | WONT FIX: contrato downstream. |

## 3. Contraste con tests

`tests/test_iae_section.py` (330 lineas, 21 tests):
- _list_available_quarters: dir no existe, vacio, sin INFOTABLE,
  ordenacion, carpetas invalidas.
- compute_iae_section STALE: sin quarters, un solo quarter.
- compute_iae_section OK: basico, cobertura, manifest vigente, helpers.
- compute_iae_section ERROR: RuntimeError("boom").
- Estructura: 23 keys presentes en STALE y en OK.
- stale_reason: insufficient_quarters, official_list_pending, none.
- catalog_coverage_warning: True/False/None en STALE/ERROR.

No cubren:
- IS-1: no verifica que reference_date/run_id se usen.
- IS-2: no hay test del fallback 242 con manifest ausente.
- IS-6: no verifica formato consistente de error entre ramas.

## 4. Falsos positivos descartados

- `_QUARTER_RE` + `_QUARTER_END` correctos. `2024Q9` filtrado.
- Orden lexicografico `year+Q` funciona para Q1-Q4 (una cifra).
- `_get_catalog_total` con `live.sort(key=valid_from, reverse=True)`:
  snapshot mas reciente con valid_to=null. Correcto.
- `catalog_warn` con umbral 0.90 (COVERAGE_WARNING_THRESHOLD).
  Coherente con tests.

## 5. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. 2 BAJA (fallback 242,
formato error inconsistente) y 5 INFO, todos WONT FIX razonado.
`iae_section.py` queda auditado. Sin acciones pendientes.

## 6. Proximo

Continuar S-03-deep por LOC descendente en src/pipeline/:
sectors_base.py (155), sector_metrics.py (154), flows_primary.py (141).
Y src/report/: flows_international.py (191), helpers.py (157),
synthesis.py (149).
