# A.6.4 - Integracion B1 + P61 + P38

**Origen:** dictamen #73 (integracion autorizada tras A.6.4-v2).
**Objetivo:** demostrar la cadena end-to-end sobre el subconjunto que
atraviesa realmente las tres capas.

## Alcance

**Q1 2026:**
1. P61 resuelve identidades (`resolve_batch_identities`).
2. §5.1-§5.5 (`build_operational_universe`) → 28.371 filas, 22 tickers.
3. Cruce §5.5 ∩ snapshot B2-PIT (242 tickers) → 20 tickers.
4. `build_target` sobre el snapshot → TargetUniverse (242 keys).
5. Sub-universo B1 filtrado a los 20 tickers del §5.5 real.
6. `period_state` sobre el sub-universo → 18 FIGIs unicos.
7. `PositionRecord` con FIGI + `operational_mapping_status=VERIFIED`.
8. `compute_contractual_coverage` → VALID.

**Q4 2025:** P61 devuelve §5.5 = 0 (vigencia de `cusip_ticker_exceptions`
es 2026-03-31, no cubre 2025-12-31). Sub-conjunto vacio -> fail-closed.
**No se fabrica TARGET historico.**

## Resultado

| Etapa | Q1 2026 | Q4 2025 |
|---|---:|---:|
| §5.5 filas | 28.371 | 0 |
| §5.5 tickers equity unicos | 22 | 0 |
| Snapshot B2-PIT tickers | 242 | 242 |
| Cruce §5.5 ∩ snapshot | 20 | 0 |
| Sub-universo FIGIs unicos | 18 | — |
| P38 coverage_status | **VALID** | fail-closed |

## Lectura

**Cadena end-to-end funciona.** El subconjunto de 20 tickers que
atraviesa P61 (§5.5) + B1 (snapshot B2-PIT) + P38 produce un resultado
contractual VALID.

## Hallazgo material H-73.1 — bug adapter B1↔P38

`catalog_p38_adapter._records` mapea `period_state.weight_status`
directamente a `PositionRecord.operational_mapping_status` (1:1). Pero
son conceptos ortogonales:

| Campo B1 (`period_state`) | Campo P38 (`coverage.PositionRecord`) |
|---|---|
| `weight_status`: RESOLVED_OBSERVED / ZERO_REPORTED / NOT_PRESENT | `operational_mapping_status`: VERIFIED / TEMPORAL_UNVERIFIED / UNRESOLVED / CONFLICT |

`compute_contractual_coverage` filtra por
`operational_mapping_status == "VERIFIED"`. El adapter produce
`weight_status = "RESOLVED_OBSERVED"`, que no es VERIFIED → cobertura = 0.

**Impacto:** cuando el adapter se invoque en producción (ninguna ruta
productiva lo hace hoy), P38 devolvera UNAVAILABLE.

**No se modifica el adapter en este probe** (cambio no autorizado por
#73). Documentado como hallazgo.

**Mitigacion en este probe:** `build_records_for_subset` construye los
records con `operational_mapping_status="VERIFIED"` explicitamente
(la cadena §5.5 ya garantiza esa condicion por construccion).

## Limitaciones

- Solo 20 tickers (subconjunto real Q1).
- No se ejecuta `target_catalog_as_of` (PIT) porque el snapshot B2-PIT
  tiene `valid_from=2026-09-21`; no cubre Q1 ni Q4 por PIT. El probe
  invoca `build_target` directamente.
- El adapter B1↔P38 no se modifica; se esquiva el bug con records
  explicitos.

## Uso

    py probe_integration_b1_p61_p38.py

Deterministico. Sin red. Sin datetime.now().

## Referencias

- A.6.4-v2: `evidence/nipc_gate0_top2000_v2/`.
- Dictamenes: #72, #73.
- Adapter B1↔P38: `src/institutional_accumulation/aggregation/catalog_p38_adapter.py`.