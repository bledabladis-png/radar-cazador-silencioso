# IAE - ESTADO DEL SISTEMA

Hechos verificables del sistema. Generado por script.

**Generado en:** 2026-09-29 02:55:21 UTC
**Snapshot tomado sobre:** commit HEAD de la fecha indicada, al momento de generar
**NO editar a mano.** Regenerar con:

    py scripts/generate_estado_sistema.py

Snapshot regenerable del sistema. Los documentos del corpus
consolidado (Consolidacion_Documentos/00-05) describen su tema;
NO declaran el estado. Este fichero es la fuente autoritativa de
HEAD, tests e integridad del codigo.

---

## 1. Git

- **HEAD:** `43b0537`
- **HEAD completo:** `43b05376b253a84b77e731d9a9451b6bf6df5f21`
- **Fecha commit HEAD:** 2026-09-29 04:52:59 +0200
- **Ahead:** 0
- **Behind:** 0
- **origin/main:** `43b0537`

## 2. Tests (pytest real)

- **Resumen:** 2294 passed, 2 skipped in 82.31s (0:01:22)
- **Exit code:** 0

## 3. Integridad del codigo

- **compileall:** OK (exit 0)
- **pyflakes:** LIMPIO

## 4. Modulos IAE

| Fichero | LOC |
|---|---:|
| `src/institutional_accumulation/__init__.py` | 28 |
| `src/institutional_accumulation/absence.py` | 78 |
| `src/institutional_accumulation/aggregation/__init__.py` | 97 |
| `src/institutional_accumulation/aggregation/catalog_p38_adapter.py` | 170 |
| `src/institutional_accumulation/aggregation/catalog_validator.py` | 94 |
| `src/institutional_accumulation/aggregation/coverage.py` | 258 |
| `src/institutional_accumulation/aggregation/delta_shares.py` | 429 |
| `src/institutional_accumulation/aggregation/nipc.py` | 393 |
| `src/institutional_accumulation/aggregation/reporting_dedup.py` | 890 |
| `src/institutional_accumulation/catalog_pit.py` | 232 |
| `src/institutional_accumulation/identity/__init__.py` | 13 |
| `src/institutional_accumulation/identity/catalog_key.py` | 375 |
| `src/institutional_accumulation/identity/openfigi_client.py` | 180 |
| `src/institutional_accumulation/identity/period_state.py` | 212 |
| `src/institutional_accumulation/identity/radar_target_catalog.py` | 154 |
| `src/institutional_accumulation/identity/target_builder.py` | 203 |
| `src/institutional_accumulation/identity/target_universe.py` | 171 |
| `src/institutional_accumulation/operational_universe.py` | 251 |
| `src/institutional_accumulation/pipeline_contractual.py` | 233 |
| `src/institutional_accumulation/sec_13f/__init__.py` | 22 |
| `src/institutional_accumulation/sec_13f/downloader.py` | 224 |
| `src/institutional_accumulation/sec_13f/identity/__init__.py` | 161 |
| `src/institutional_accumulation/sec_13f/identity/amendments.py` | 444 |
| `src/institutional_accumulation/sec_13f/identity/cusip_resolver.py` | 149 |
| `src/institutional_accumulation/sec_13f/identity/relationships.py` | 367 |
| `src/institutional_accumulation/sec_13f/identity/sec13f_list.py` | 315 |
| `src/institutional_accumulation/sec_13f/identity/security_identity.py` | 687 |
| `src/institutional_accumulation/sec_13f/identity/temporal_filter.py` | 100 |
| `src/institutional_accumulation/sec_13f/ingest.py` | 131 |
| `src/institutional_accumulation/sec_13f/manifest.py` | 150 |
| `src/institutional_accumulation/sec_13f/parser.py` | 142 |
| `src/institutional_accumulation/sec_13f/schema.py` | 185 |
| `src/institutional_accumulation/sec_13f/storage.py` | 89 |
| `src/institutional_accumulation/security_type.py` | 407 |
| `src/institutional_accumulation/temporal_validity.py` | 92 |
| `src/institutional_accumulation/timestamps.py` | 154 |
| **TOTAL** | **36 ficheros, 8280 LOC** |

## 5. Datos IAE

- `data/mappings/radar_target_catalog.csv`
  - sha256: `4a78e2695aca64ec5601c246d176199ce98e9e10e73e53e4b8f11385fe00ef38`
  - bytes: 29986
  - lineas: 243
- `data/mappings/cusip_radar_crosswalk.csv`
  - sha256: `c344bee55fa5149853ed029e59ee2acbccd94d39990f1b493048ea0cb06a60cc`
  - bytes: 22491
  - lineas: 246
- `data/mappings/cusip_equivalence.csv`
  - sha256: `21a88790aad50e8b44fbfdaa87c394ef3a782319adaf4208b9cb244dac39b322`
  - bytes: 103
  - lineas: 1

## 6. Scripts IAE

- `scripts/iae_pipeline.py`: 159 LOC
- `scripts/build_catalog_csvs.py`: 245 LOC
- `scripts/iae_reconciliation_b1.py`: 193 LOC
- `scripts/iae_validate_crosswalk_openfigi.py`: 179 LOC
- `scripts/iae_test_census.py`: 161 LOC
- `scripts/iae_contractual_coverage.py`: 207 LOC
- `scripts/iae_coverage.py`: 55 LOC
- `scripts/iae_identity_uniqueness_audit.py`: 270 LOC
- `scripts/iae_contractual_nipc_e2e.py`: 142 LOC
- `scripts/regenerate_radar_catalog.py`: 238 LOC
- `scripts/regenerate_cusip_crosswalk.py`: 225 LOC
- `scripts/update_sec_13f.py`: 286 LOC
- `scripts/download_official_list_13f.py`: 147 LOC

## 7. Evidencia empirica (directorios)

- `docs/auditoria/iae/evidence/nipc_gate0_baseline`: 5 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_target_identity_top2000`: 10 ficheros
- `docs/auditoria/iae/evidence/nipc_gate0_top2000_v2`: 6 ficheros
- `docs/auditoria/iae/evidence/a64_integration_b1_p61_p38`: 13 ficheros
- `docs/auditoria/iae/evidence/b2_pit_cierre`: 2 ficheros
- `docs/auditoria/iae/evidence/nipc_p70_probe`: NO EXISTE

---

Fin del estado generado. Fuente autoritativa: este fichero.
