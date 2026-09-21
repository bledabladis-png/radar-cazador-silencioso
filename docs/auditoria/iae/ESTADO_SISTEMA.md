# IAE - ESTADO DEL SISTEMA

Hechos verificables del sistema. Generado por script.

**Generado en:** 2026-09-21 23:13:33 UTC
**Snapshot tomado sobre:** commit HEAD de la fecha indicada, al momento de generar
**NO editar a mano.** Regenerar con:

    py scripts/generate_estado_sistema.py

Este fichero es complementario a `ESTADO_DECLARADO.md` (declaraciones
subjetivas de fase, prohibiciones, deuda). Los demas documentos del
proyecto describen su tema; NO declaran el estado del sistema.

---

## 1. Git

- **HEAD:** `b9a2107`
- **HEAD completo:** `b9a2107139a4b2555bb96d7a8b6f8c05c33cdfcd`
- **Fecha commit HEAD:** 2026-09-22 01:13:13 +0200
- **Ahead:** 311
- **Behind:** 0
- **origin/main:** `5f1d9ab`

## 2. Tests (pytest real)

- **Resumen:** 3 failed, 1415 passed, 2 skipped in 28.61s
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
| `src/institutional_accumulation/aggregation/coverage.py` | 180 |
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
| **TOTAL** | **35 ficheros, 7671 LOC** |

## 5. Datos IAE

- `data/mappings/radar_target_catalog.csv`
  - sha256: `e5d8f9c86b07c4d3981b2fd3c3a804f75d3559fa33fbb7731bfc76308a60b292`
  - bytes: 29849
  - lineas: 243
- `data/mappings/cusip_ticker_exceptions.csv`
  - sha256: `073c68e26774b1bfbc55429d5d31453b28e25be1c4346b84bee9b2ab0f196001`
  - bytes: 3096
  - lineas: 25
- `data/mappings/cusip_equivalence.csv`
  - sha256: `21a88790aad50e8b44fbfdaa87c394ef3a782319adaf4208b9cb244dac39b322`
  - bytes: 103
  - lineas: 1

## 6. Scripts IAE

- `scripts/iae_pipeline.py`: 156 LOC
- `scripts/build_catalog_csvs.py`: 161 LOC

## 7. Evidencia empirica (directorios)

- `docs/auditoria/iae/evidence/nipc_gate0_baseline`: 5 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_target_identity_top2000`: 10 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_top2000_v2`: 6 ficheros
- `docs/auditoria/iae/evidence/a64_integration_b1_p61_p38`: 10 ficheros
- `docs/auditoria/iae/evidence/b2_pit_cierre`: 2 ficheros
- `docs/auditoria/iae/evidence/nipc_p70_probe`: 5 ficheros

---

Fin del estado generado. Fuente de verdad: `ESTADO_DECLARADO.md` (declaraciones)
+ `ESTADO_SISTEMA.md` (este fichero, hechos).
