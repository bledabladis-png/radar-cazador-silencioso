# IAE - ESTADO DEL SISTEMA

Hechos verificables del sistema. Generado por script.

**Generado en:** 2026-09-23 00:35:16 UTC
**Snapshot tomado sobre:** commit HEAD de la fecha indicada, al momento de generar
**NO editar a mano.** Regenerar con:

    py scripts/generate_estado_sistema.py

Este fichero es complementario a `ESTADO_DECLARADO.md` (declaraciones
subjetivas de fase, prohibiciones, deuda). Los demas documentos del
proyecto describen su tema; NO declaran el estado del sistema.

---

## 1. Git

- **HEAD:** `041c9d2`
- **HEAD completo:** `041c9d209132865a7acd7886f6ff01e6bcbec04b`
- **Fecha commit HEAD:** 2026-09-23 02:35:03 +0200
- **Ahead:** 380
- **Behind:** 0
- **origin/main:** `5f1d9ab`

## 2. Tests (pytest real)

- **Resumen:** 3 failed, 1434 passed, 2 skipped in 27.48s
- **Exit code:** 1

**Tests fallidos (nombre completo):**
- `FAILED tests/test_freshness.py::test_market_data_fresh - AssertionError: mark...`
- `FAILED tests/test_freshness.py::test_stock_prices_fresh - AssertionError: sto...`
- `FAILED tests/test_freshness.py::test_european_tickers_recent - AssertionError...`

## 3. Integridad del codigo

- **compileall:** OK (exit 0)
- **pyflakes:** LIMPIO

## 4. Modulos IAE

| Fichero | LOC |
|---|---:|
| `src/institutional_accumulation/__init__.py` | 28 |
| `src/institutional_accumulation/absence.py` | 77 |
| `src/institutional_accumulation/aggregation/__init__.py` | 97 |
| `src/institutional_accumulation/aggregation/catalog_p38_adapter.py` | 170 |
| `src/institutional_accumulation/aggregation/catalog_validator.py` | 94 |
| `src/institutional_accumulation/aggregation/coverage.py` | 258 |
| `src/institutional_accumulation/aggregation/delta_shares.py` | 380 |
| `src/institutional_accumulation/aggregation/nipc.py` | 395 |
| `src/institutional_accumulation/aggregation/reporting_dedup.py` | 880 |
| `src/institutional_accumulation/catalog_pit.py` | 232 |
| `src/institutional_accumulation/identity/__init__.py` | 13 |
| `src/institutional_accumulation/identity/catalog_key.py` | 375 |
| `src/institutional_accumulation/identity/openfigi_client.py` | 180 |
| `src/institutional_accumulation/identity/period_state.py` | 212 |
| `src/institutional_accumulation/identity/radar_target_catalog.py` | 145 |
| `src/institutional_accumulation/identity/target_builder.py` | 181 |
| `src/institutional_accumulation/identity/target_universe.py` | 171 |
| `src/institutional_accumulation/operational_universe.py` | 194 |
| `src/institutional_accumulation/sec_13f/__init__.py` | 22 |
| `src/institutional_accumulation/sec_13f/downloader.py` | 176 |
| `src/institutional_accumulation/sec_13f/identity/__init__.py` | 161 |
| `src/institutional_accumulation/sec_13f/identity/amendments.py` | 444 |
| `src/institutional_accumulation/sec_13f/identity/cusip_resolver.py` | 149 |
| `src/institutional_accumulation/sec_13f/identity/relationships.py` | 367 |
| `src/institutional_accumulation/sec_13f/identity/sec13f_list.py` | 255 |
| `src/institutional_accumulation/sec_13f/identity/security_identity.py` | 654 |
| `src/institutional_accumulation/sec_13f/identity/temporal_filter.py` | 100 |
| `src/institutional_accumulation/sec_13f/ingest.py` | 131 |
| `src/institutional_accumulation/sec_13f/manifest.py` | 150 |
| `src/institutional_accumulation/sec_13f/parser.py` | 142 |
| `src/institutional_accumulation/sec_13f/schema.py` | 185 |
| `src/institutional_accumulation/sec_13f/storage.py` | 89 |
| `src/institutional_accumulation/security_type.py` | 397 |
| `src/institutional_accumulation/temporal_validity.py` | 92 |
| `src/institutional_accumulation/timestamps.py` | 153 |
| **TOTAL** | **35 ficheros, 7749 LOC** |

## 5. Datos IAE

- `data/mappings/radar_target_catalog.csv`
  - sha256: `4a78e2695aca64ec5601c246d176199ce98e9e10e73e53e4b8f11385fe00ef38`
  - bytes: 29986
  - lineas: 243
- `data/mappings/cusip_radar_crosswalk.csv`
  - sha256: `032ef307e9810fd2b3a8b3361462a03bf04ecc43c5c25313346bcef8c38e858b`
  - bytes: 30203
  - lineas: 247
- `data/mappings/cusip_equivalence.csv`
  - sha256: `21a88790aad50e8b44fbfdaa87c394ef3a782319adaf4208b9cb244dac39b322`
  - bytes: 103
  - lineas: 1

## 6. Scripts IAE

- `scripts/iae_pipeline.py`: 156 LOC
- `scripts/build_catalog_csvs.py`: 245 LOC

## 7. Evidencia empirica (directorios)

- `docs/auditoria/iae/evidence/nipc_gate0_baseline`: 5 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_target_identity_top2000`: 10 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_top2000_v2`: 6 ficheros
- `docs/auditoria/iae/evidence/a64_integration_b1_p61_p38`: 13 ficheros
- `docs/auditoria/iae/evidence/b2_pit_cierre`: 2 ficheros
- `docs/auditoria/iae/evidence/nipc_p70_probe`: NO EXISTE

---

Fin del estado generado. Fuente de verdad: `ESTADO_DECLARADO.md` (declaraciones)
+ `ESTADO_SISTEMA.md` (este fichero, hechos).
