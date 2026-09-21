# A.6.4 - Integracion B1 + P61 + P38

**Origen:** dictamen #73 (integracion autorizada) + #74 (H-73.1 confirmado).
**Objetivo:** demostrar que el adapter B1 <-> P38 funciona sobre records reales
(no como evidencia cuantitativa de cobertura contractual).

---

## Alcance REAL de este probe (corregido 2026-09-21, auditoria externa)

**Este probe NO demuestra cobertura contractual.** Demuestra que:

1. La cadena P61 seccion 5.5 -> B1 `build_target` -> `period_state` ->
   `catalog_p38_adapter` -> `compute_contractual_coverage` se ejecuta sin
   excepciones sobre datos reales Q1 2026.
2. El adapter H-73.1 corregido produce `PositionRecord` con
   `operational_mapping_status="VERIFIED"`.
3. `compute_contractual_coverage` acepta esos records.

**Este probe NO demuestra:**

- Cobertura pairwise real entre Q4 2025 y Q1 2026 (los states Q4 y Q1 son
  identicos por construccion).
- Ponderacion por masa (SSHPRNAMT) real (todos los `PositionRecord.weight`
  son 1.0 hardcoded en el adapter).
- Fail-closed real de Q4 (Q4 = Q1 por mock, no vacio).
- Flujo PIT completo (target_catalog_as_of no se invoca).

## Por que el resultado siempre es 1.0

Tres problemas estructurales (ver ESTADO_DECLARADO.md seccion 8, H-10.1):

1. **Mock Q4=Q1 en el probe.** La llamada
   `catalog_to_p38_targets(universe_sub, universe_sub, state_q4=st_q,
   state_q1=st_q, ...)` pasa el mismo objeto para ambos periodos.
   `coverage_previous` es matematicamente identico a `coverage_current`.

2. **Peso hardcoded.** `catalog_p38_adapter._records` asigna `weight=1.0`
   a todos los records. `paired_weighted_share_coverage` es igual a
   `paired_security_coverage`, no pondera.

3. **Adapter marca VERIFIED incondicionalmente.** `_records` declara
   `operational_mapping_status="VERIFIED"` sin verificar el estado
   operacional real. Combinado con
   `coverage_previous = len(res_q4) / len(all_q4)` (denominador es
   observed con FIGI, no TARGET), el resultado es 1.0 incluso si hubiera
   cobertura real baja.

## Resultado (honesto)

| Etapa | Q1 2026 | Q4 2025 |
|---|---:|---:|
| seccion 5.5 filas | 28.371 | 0 |
| seccion 5.5 tickers equity unicos | 22 | 0 |
| Cruce seccion 5.5 interseccion snapshot | 20 | 0 |
| Keys con FIGI | 18 | mock = Q1 |
| Records VERIFIED | 18 | mock = Q1 |
| coverage_previous | 1.0 | 1.0 (mock, no real) |
| coverage_current | 1.0 | 1.0 (mock) |
| paired_security_coverage | 1.0 | 1.0 (mock) |
| paired_weighted_share_coverage | 1.0 | 1.0 (peso hardcoded) |
| coverage_status | VALID | VALID |

**Lectura:** el probe se ejecuta, pero las cifras son **triviales por
construccion**, no evidencia de cobertura.

## Hallazgo H-73.1 - CORREGIDO (sigue vigente)

`catalog_p38_adapter._records` mapeaba `period_state.weight_status`
directamente a `PositionRecord.operational_mapping_status` (1:1).
Son conceptos ortogonales:

| Campo B1 (`period_state`) | Campo P38 (`coverage.PositionRecord`) |
|---|---|
| weight_status: RESOLVED_OBSERVED / ZERO_REPORTED / NOT_PRESENT | operational_mapping_status: VERIFIED / TEMPORAL_UNVERIFIED / UNRESOLVED / CONFLICT |

`compute_contractual_coverage` filtra por `operational_mapping_status == "VERIFIED"`.
El adapter producia `RESOLVED_OBSERVED` -> cobertura = 0.

**Corregido en dictamen #74.** El adapter ahora:

- Declara `resolution_status="CANONICAL"` y `operational_mapping_status="VERIFIED"`.
- Excluye keys con `identity_status != "RESOLVED"` (defensa por si el caller
  salta PASO 8).

**Test de regresion:** `tests/test_h731_adapter_p38_compat.py` (5 tests).
Verifica no solo estructura del `PositionRecord`, sino su COMPATIBILIDAD
con `compute_contractual_coverage`.

## Hallazgos abiertos (auditoria externa 2026-09-21)

- **H-05:** Q4 declarado vacio pero `coverage_previous=1.0`. Causa: mock
  Q4=Q1 en el probe. Fix propuesto: separar run Q1 positivo del run Q4
  fail-closed.
- **H-06:** no se publican cardinalidades reales
  (`len(target_q4)`, `len(target_q1)`, `len(TARGET_PAIRWISE)`,
  `len(RESOLVED_Q4)`, `len(RESOLVED_Q1)`, `len(PAIRED)`). Fix propuesto:
  instrumentar el probe.
- **H-07:** `paired_weighted_share_coverage=1.0` con todos los pesos = 1.0.
  Fix propuesto: propagar SSHPRNAMT real desde 13F hasta `PositionRecord.weight`.
- **H-10.1 (nuevo, 2026-09-21):** `catalog_p38_adapter._records` marca
  `operational_mapping_status="VERIFIED"` incondicionalmente. `coverage.py`
  divide por observed, no por target. Combinado con peso = 1.0, el resultado
  es estructuralmente 1.0.

Los fixes H-05 a H-10.1 tocan codigo productivo (`coverage.py`,
`catalog_p38_adapter.py`, `period_state.py`). Requieren dictamen externo
antes de implementarse (regla 4 del prompt: prohibicion de modificar
`coverage.py` sin dictamen).

## Uso

    py probe_integration_b1_p61_p38.py

Deterministico. Sin red. Sin datetime.now().

## Referencias

- A.6.4-v2: `evidence/nipc_gate0_top2000_v2/`.
- Dictamenes: #72, #73, #74.
- Estado vigente: `ESTADO_DECLARADO.md` seccion 8 (H-10.1).

## Provenance de los inputs (H-07, resuelto 2026-09-21)

Hashes SHA-256 de los inputs que produjeron el diagnostico:

**Snapshot B2-PIT + mappings:**

| Fichero | SHA-256 (prefijo) |
|---|---|
| snapshot_20260921_01.csv | `e5d8f9c8...` |
| radar_target_catalog.csv | `e5d8f9c8...` (mismo contenido) |
| cusip_ticker_exceptions.csv | `6f545880...` |
| cusip_equivalence.csv | `21a88790...` |
| catalog_manifest.json | `2fa5eee5...` |
| catalog_membership.csv | `ad59cc91...` |
| catalog_assignments.csv | `287061f6...` |

**13F parquets Q4 2025 / Q1 2026:** hashes completos en `HASHES.txt`
y en `data/manifests/sec_13f_*.json`.

La cadena input -> codigo -> resultado es reconstruible a partir de
los hashes registrados. Ver `HASHES.txt`.
