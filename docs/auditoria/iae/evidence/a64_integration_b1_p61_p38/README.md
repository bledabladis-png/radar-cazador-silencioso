# A.6.4 - Integracion B1 + P61 + P38

**Origen:** dictamen #73 (integracion autorizada) + #74 (H-73.1
confirmado + correccion del adapter).
**Objetivo:** demostrar la cadena end-to-end sobre el subconjunto que
atraviesa realmente las tres capas, SIN bypass manual.

## Alcance

**Q1 2026:**
1. P61 resuelve identidades (`resolve_batch_identities`).
2. §5.1-§5.5 (`build_operational_universe`) → 28.371 filas, 22 tickers.
3. Cruce §5.5 ∩ snapshot B2-PIT (242 tickers) → 20 tickers.
4. `build_target` sobre el snapshot → TargetUniverse (242 keys).
5. Sub-universo B1 filtrado a los 20 tickers.
6. Filtro adicional: solo keys con `share_class_figi` real.
   BRK-B y MOG-A no tienen FIGI en el snapshot → 18 keys.
7. `catalog_to_p38_targets` (adapter corregido H-73.1) → records con
   `operational_mapping_status=VERIFIED`.
8. `compute_contractual_coverage` → VALID.

**Q4 2025:** P61 devuelve §5.5 = 0. Sub-conjunto vacio -> fail-closed.
No se fabrica TARGET historico.

## Resultado

| Etapa | Q1 2026 | Q4 2025 |
|---|---:|---:|
| §5.5 filas | 28.371 | 0 |
| §5.5 tickers equity unicos | 22 | 0 |
| Cruce §5.5 ∩ snapshot | 20 | 0 |
| Keys con FIGI | 18 | — |
| Records VERIFIED | 18 | — |
| coverage_previous | 1.0 | fail-closed |
| coverage_current | 1.0 | fail-closed |
| paired_security_coverage | 1.0 | fail-closed |
| paired_weighted_share_coverage | 1.0 | fail-closed |
| coverage_status | **VALID** | fail-closed |

## Hallazgo H-73.1 - CORREGIDO

`catalog_p38_adapter._records` mapeaba `period_state.weight_status`
directamente a `PositionRecord.operational_mapping_status` (1:1).
Son conceptos ortogonales:

| Campo B1 (`period_state`) | Campo P38 (`coverage.PositionRecord`) |
|---|---|
| weight_status: RESOLVED_OBSERVED / ZERO_REPORTED / NOT_PRESENT | operational_mapping_status: VERIFIED / TEMPORAL_UNVERIFIED / UNRESOLVED / CONFLICT |

`compute_contractual_coverage` filtra por `operational_mapping_status == "VERIFIED"`.
El adapter producia `RESOLVED_OBSERVED` → cobertura = 0.

**Corregido en dictamen #74.** El adapter ahora:
- Declara `resolution_status="CANONICAL"` y `operational_mapping_status="VERIFIED"`.
- Excluye keys con `identity_status != "RESOLVED"` (defensa por si el caller salta PASO 8).

**Test de regresion:** `tests/test_h731_adapter_p38_compat.py` (5 tests).
Verifica no solo estructura del PositionRecord, sino su COMPATIBILIDAD
con `compute_contractual_coverage` (el test que faltaba).

**Este probe ya no usa bypass.** Los records vienen del adapter real.

## Limitaciones

- Solo 18 keys (subconjunto real Q1 con FIGI).
- No se ejecuta `target_catalog_as_of` (PIT). El snapshot B2-PIT tiene
  `valid_from=2026-09-21`; no cubre Q1/Q4 por PIT.
- BRK-B y MOG-A excluidos (sin FIGI en snapshot).

## Uso

    py probe_integration_b1_p61_p38.py

Deterministico. Sin red. Sin datetime.now().

## Referencias

- A.6.4-v2: `evidence/nipc_gate0_top2000_v2/`.
- Dictamenes: #72, #73, #74.
- Adapter corregido: `src/institutional_accumulation/aggregation/catalog_p38_adapter.py`.
- Test regresion: `tests/test_h731_adapter_p38_compat.py`.