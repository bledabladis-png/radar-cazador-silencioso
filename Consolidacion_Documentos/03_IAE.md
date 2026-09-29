# 03 - IAE

**Referencia on-demand. Subsistema Institutional Accumulation Engine.**
**No se pega al arrancar. El arranque es 00_ARRANQUE.md.**

---

## 1. QUE ES EL IAE

El Institutional Accumulation Engine analiza posiciones institucionales a partir de filings 13F de la SEC. Responde:

> Dado un universo de tickers del radar, que instituciones han aumentado o reducido su posicion entre dos trimestres consecutivos, y con que cobertura de datos se puede afirmar.

Opera integrado en el pipeline productivo del radar con dependencia unidireccional (`run.py -> IAE`). Prohibido lo inverso.

Metrica central: **NIPC** (Net Institutional Position Change).

Pipeline contractual: **§10.1 -> §10.7** (catalogo -> identidad -> delta shares -> NIPC -> coverage).

---

## 2. ARQUITECTURA DEL MODULO

36 ficheros Python, 8280 LOC (metrica `splitlines`, coincide con `ESTADO_SISTEMA.md`).

### 2.1. Estructura de subpaquetes

| Subpaquete | Ficheros | Proposito |
|---|---|---|
| `src/institutional_accumulation/` (raiz) | 8 | Orquestacion + contratos transversales |
| `aggregation/` | 7 | Delta shares, NIPC, coverage, reporting dedup |
| `identity/` | 7 | Identidad de catalogo, targets, periodos |
| `sec_13f/` | 7 | Ingesta SEC 13F (downloader, parser, storage) |
| `sec_13f/identity/` | 7 | Identidad de filings (CUSIP, amendments, relationships) |

### 2.2. Ficheros raiz

- `absence.py` (77 LOC) - P63: ausencia observada vs causa inferida.
- `catalog_pit.py` (232) - B2-PIT: infraestructura temporal del catalogo.
- `operational_universe.py` (251) - §5.5 operational_universe (spec v1.4).
- `pipeline_contractual.py` (223) - Orquestador de la cadena §10.1 -> §10.7.
- `security_type.py` (397) - P38: clasificacion de security_type desde TITLEOFCLASS.
- `temporal_validity.py` (92) - P61: validacion temporal por fuente.
- `timestamps.py` (153) - P63/B3: derivacion de los 3 timestamps de una observacion 13F.
- `__init__.py` (28).

### 2.3. Subpaquete `aggregation/`

- `catalog_p38_adapter.py` (170) - Adaptador catalogo -> dominio P38.
- `catalog_validator.py` (94) - Validadores del flujo normativo (v4 secciones 4-7).
- `coverage.py` (258) - P38: Coverage contractual.
- `delta_shares.py` (420) - DeltaShares: agregacion de shares por reported_position_unit.
- `nipc.py` (387) - NIPC: Net Institutional Position Change.
- `reporting_dedup.py` (880) - P65: Manager Duplication Contract.
- `__init__.py` (97).

### 2.4. Subpaquete `identity/`

- `catalog_key.py` (375) - B1: identidad administrativa del catalogo.
- `openfigi_client.py` (180) - Cliente productivo OpenFIGI.
- `period_state.py` (212) - B1: estados por periodo.
- `radar_target_catalog.py` (154) - Identidad estable de los ~242 tickers del radar.
- `target_builder.py` (181) - Materializacion del TargetUniverse.
- `target_universe.py` (171) - Resolver CUSIP_13F -> target membership.
- `__init__.py` (13).

### 2.5. Subpaquete `sec_13f/`

- `downloader.py` (224) - Descarga y extraccion del dataset SEC Form 13F.
- `ingest.py` (131) - Orquestador de ingestion.
- `manifest.py` (150) - Manifest de lineage SEC.
- `parser.py` (142) - Parser del dataset SEC 13F Data Set.
- `schema.py` (185) - Contrato declarativo del dataset.
- `storage.py` (89) - Persistencia de parquets en filesystem del radar.
- `__init__.py` (22).

### 2.6. Subpaquete `sec_13f/identity/`

- `amendments.py` (444) - FA-2.4: amendments + canonical snapshot.
- `cusip_resolver.py` (149) - Resolver CUSIP -> ticker con vigencia temporal.
- `relationships.py` (367) - Reporting relationships (framework 3 niveles).
- `sec13f_list.py` (315) - Parser SEC Official List of Section 13(f) Securities.
- `security_identity.py` (654) - Resolucion de identidad canonica de security (C1).
- `temporal_filter.py` (100) - Filtro temporal canonico por SUBMISSION.PERIODOFREPORT.
- `__init__.py` (161).
---

## 3. CONTRATOS DEL MODULO IAE

Cinco contratos semanticos (P38, NIPC, P61/P63, B1, B2-PIT) verificados en el pipeline productivo. El contrato P65/P66 (Manager Duplication + L3 cruzada) vive en `06_IAE_P65_P66.md`, extraido del corpus antiguo tras `cdf47ad`.

### 3.1. P38 - Coverage contractual

**Definicion:** el coverage se calcula sobre el universo elegible (target_q4 / target_q1) y la base de datos de posiciones de cada periodo.

**Ficheros:** `aggregation/coverage.py`, `aggregation/catalog_p38_adapter.py`, `aggregation/catalog_validator.py`.

**Regla:** `coverage_available` (medible) + `coverage_quality` (UNAVAILABLE | PARTIAL | COMPLETE con umbral 0.95 sobre dos dimensiones). `coverage_status="VALID"` requiere `denom > 0` y `coverage_previous/current > 0` (fail-closed).

### 3.2. NIPC - Net Institutional Position Change

**Definicion:** delta de shares institucionales entre Q_{k-1} y Q_k, normalizado por clase accionaria.

**Componentes:**
- `nipc_total` = SOLE + DFND + OTR.
- `nipc_sole` = variacion de posiciones declaradas SOLE.
- `nipc_dfnd` = variacion de posiciones DFND.
- `nipc_otr` = variacion de posiciones OTR (residual).

**Regla:** el NIPC contractual (no PROXY) se calcula sobre los `reported_position_units`. Match por `canonical_security` (formato `equity:TICKER` o `figi:FIGI`). Split oficial por `canonical_security`, no por CUSIP (los `observed_security_key=cusip:XXX` son UNRESOLVED_IDENTITY).

**Ficheros:** `aggregation/nipc.py`, `aggregation/delta_shares.py`.

### 3.3. P65 - Reporting dedup (Manager Duplication Contract)

**Definicion:** un filing 13F puede declarar posiciones propias (SOLE) y posiciones de otros managers (DFND/OTR). El Manager Duplication Contract clasifica la evidencia y deduplica.

**Reglas R1-R5 + L3:** framework de 3 niveles para resolver la cadena de reporte.

**Ficheros:** `aggregation/reporting_dedup.py` (880 LOC, 19 funciones AST, 64 casos pytest).

**Evidencia:** `ReportingEvidence` enum; `build_effective_reporting_snapshot` (sin caller productivo). **Estado verificado 2026-09-29:** capa de diagnostico por decision de diseno, NO etapa del pipeline contractual. Coexiste con el desglose SOLE/DFND/OTR en `nipc.py` (via activa). `_apply_intra_period_dedup` v1 no emite DROP_DUP: todas las decisiones son KEEP (overlap no resoluble con 13F aislado). Ver `06_IAE_P65_P66.md` seccion 14.

### 3.4. P61 / P63 - Validez temporal + timestamps

**P61:** validacion temporal por fuente (`temporal_validity.py`). Cada fuente declara `valid_from` / `valid_to` / `period`. Agregable por status.

**P63:** derivacion de 3 timestamps (`timestamps.py`):
- `source_timestamp` = fecha del filing original.
- `knowledge_timestamp` = cuando el radar tuvo acceso.
- `observation_timestamp` = fecha de observacion del instrumento.

Regla: nunca mezclar los 3. La fecha de observacion del dataset viene del periodo del filing, no de `datetime.now()`.

### 3.5. B1 - Identidad administrativa del catalogo

**Definicion:** el catalogo (radar_target_catalog.csv) tiene identidad administrativa: clave (catalog_key), entity_id, sha256 del snapshot.

**Ficheros:** `identity/catalog_key.py`, `identity/period_state.py`, `identity/target_builder.py`.

**Regla:** `TargetUniverse` es la materializacion del catalogo en un periodo. `build_target(snapshot_df, membership_df, assignments_df)` construye el universo.

### 3.6. B2-PIT - Catalogo point-in-time

**Definicion:** el catalogo debe poder consultarse "as of" un period_end. Snapshots versionados con manifest + sha256.

**Ficheros:** `catalog_pit.py`.

**Funciones publicas:** `target_catalog_as_of(period_end)`, `list_snapshots()`, `verify_snapshot_integrity(version_id)`, `load_manifest(catalog_root)`.

**Regla:** no hay ambiguedad (CatalogAmbiguous) ni snapshot corrupto (SnapshotIntegrityError). El estado se declara explicitamente.

### 3.7. Oficial 13(f) - SEC Official List

**Definicion:** universo elegible segun la lista oficial trimestral de la SEC (13flist{year}q{quarter}.txt).

**Ficheros:** `sec_13f/identity/sec13f_list.py`.

**Parser:** clasifica instrument (EQUITY / OPTION / UNRESOLVED) desde el `description` de la lista. La regla CALL/PUT es word-boundary (`\b(?:CALL|PUT|OPTION|OPT)\b`).

**Fallback:** URL_EXCEPTIONS + reintento con sufijo `-txt` si la URL canonica 404 (patron observado en 2025Q3 y 2026Q2).

### 3.8. Amendments + canonical snapshot

**Definicion:** un filing 13F puede tener amendments. El canonical snapshot resuelve la cadena.

**Ficheros:** `sec_13f/identity/amendments.py`.

**Reglas:** `order_filings` -> `classify_strategy` -> `detect_base_filing` -> `apply_amendments`. Estados: `compute_strategy_counts` + `compute_status_counts`.

### 3.9. Relationships - Framework 3 niveles

**Definicion:** cuando un filing declara posiciones de otros managers (OTHERMANAGER / OTHERMANAGER2), se construye un grafo de relaciones.

**Ficheros:** `sec_13f/identity/relationships.py`.

**Regla:** `classify_token` -> `explode_othermanager_edges` -> `build_canonical_relationship`. Un filing puede tener multiples edges.
---

## 4. PIPELINE CONTRACTUAL (§10.1 -> §10.7)

Orquestador: `pipeline_contractual.py::run_contractual_nipc(folders, periods)`.

Cadena:

1. **§10.1** - Cargar snapshots de 2 trimestres consecutivos. `load_canonical(folder, iso)`.
2. **§10.2** - Construir identidades. `build_identities(snap, iso)`.
3. **§10.3** - Delta shares por `reported_position_unit`. `compute_reported_position_units(infotable, submission, coverpage)` + `compute_delta_shares(units_current, units_previous)`.
4. **§10.4** - NIPC contractual. `compute_nipc_contractual(delta_df, *, target_q4, target_q1, records_q4, records_q1, threshold_1=None, threshold_2=None)`.
5. **§10.5** - Coverage contractual. `compute_contractual_coverage(target_q4, target_q1, records_q4, records_q1)`.
6. **§10.6** - Estats observables.
7. **§10.7** - Evidence class = CONTRACTUAL.

**Unidades:** `SSHPRNAMT` se interpreta como miles de acciones. NO se aplica factor x1000.

**Match key:** `canonical_security` = `equity:<TICKER>` o `figi:<FIGI>`. Dualidad teorica del resolver, no observada en E2E con `figi_lookup=None`.

**Filtros SH / PUTCALL:** solo se consideran filas con `SH` (shares) no `PUTCALL`. Filtro 5.1.

---

## 5. SCRIPTS REPRODUCIBLES

`scripts/` (13 activos IAE):

| Script | LOC | Proposito |
|---|---:|---|
| `build_catalog_csvs.py` | 245 | Construccion de catalogos CSV (setup inicial) |
| `download_official_list_13f.py` | 147 | Descarga Official List 13(f) |
| `iae_contractual_coverage.py` | 207 | Cadena contractual completa (target -> adapter -> coverage) |
| `iae_contractual_nipc_e2e.py` | 111 | E2E NIPC contractual vs golden |
| `iae_coverage.py` | 55 | Cobertura reproducible (44 ficheros) |
| `iae_identity_uniqueness_audit.py` | 270 | Auditoria unicidad por shareClassFIGI |
| `iae_pipeline.py` | 159 | Orquestador ingestion + identity |
| `iae_reconciliation_b1.py` | 193 | Reconciliacion radar + complemento = full |
| `iae_test_census.py` | 161 | Censo AST de tests del modulo |
| `iae_validate_crosswalk_openfigi.py` | 179 | Validacion externa OpenFIGI |
| `regenerate_cusip_crosswalk.py` | 225 | Regenera crosswalk CUSIP (trimestral) |
| `regenerate_radar_catalog.py` | 238 | Regenera catalogo radar (daily) |
| `update_sec_13f.py` | 286 | Ingesta trimestral SEC 13F |

**Clasificacion (criterio del auditor externo H3):**
- **Normativo:** `regenerate_radar_catalog.py`, `regenerate_cusip_crosswalk.py`, `update_sec_13f.py`, `download_official_list_13f.py`, `generate_estado_sistema.py`.
- **Auditoria:** `iae_reconciliation_b1.py`, `iae_validate_crosswalk_openfigi.py`, `iae_test_census.py`, `iae_contractual_coverage.py`, `iae_coverage.py`, `iae_identity_uniqueness_audit.py`, `iae_contractual_nipc_e2e.py`.
- **Diagnostico:** `iae_pipeline.py`, `build_catalog_csvs.py`.

---

## 6. EVIDENCIA EMPIRICA

`docs/auditoria/iae/evidence/` (8 directorios):

| Directorio | Ficheros | Contenido |
|---|---:|---|
| `a64_integration_b1_p61_p38` | 13 | Integracion B1 + P61 + P38 |
| `b2_pit_cierre` | 2 | Cierre B2-PIT |
| `nipc_gate0_baseline` | 5 | Baseline NIPC |
| `nipc_gate0_openfigi` | 9 | Validacion OpenFIGI |
| `nipc_gate0_probe` | 11 | Probes NIPC |
| `nipc_gate0_target_identity` | 11 | Identidad target |
| `nipc_gate0_target_identity_top2000` | 10 | Top2000 |
| `nipc_gate0_top2000_v2` | 6 | Top2000 v2 |

Los ficheros de evidencia son reproducibles con los scripts de la Seccion 5.

---

## 7. MAPPINGS

`data/mappings/`:

| Fichero | Bytes | Proposito |
|---|---:|---|
| `catalog_assignments.csv` | 24507 | Asignaciones de identidad |
| `catalog_manifest.json` | 754 | Manifest del catalogo |
| `catalog_membership.csv` | 28330 | Membership de identidades |
| `cusip_equivalence.csv` | 103 | Equivalencia CUSIP (manual) |
| `cusip_radar_crosswalk.csv` | 22491 | Crosswalk CUSIP <-> radar (~246 filas, 242 tickers) |
| `cusip_to_radar_figi.csv` | 11958 | Mapa CUSIP -> FIGI |
| `cusip_to_radar_figi_crosswalk.csv` | 8827 | Crosswalk CUSIP -> FIGI |
| `radar_target_catalog.csv` | 29986 | Catalogo radar (~243 filas) |

Snapshots PIT: `data/mappings/catalog_snapshots/snapshot_*.csv` + `.sha256`.

Regeneracion automatica:
- `radar_target_catalog.csv` se regenera en `daily_run.yml` (step "Regenerar catalogo radar (IAE)").
- `cusip_radar_crosswalk.csv` se regenera en `update_sec_13f.yml` (step "Regenerar crosswalk CUSIP (IAE)").

---

## 8. GOLDEN

`docs/auditoria/iae/golden/`:

| Fichero | Bytes | Proposito |
|---|---:|---|
| `12_5_historic.json` | 2397 | Snapshot historico (FROZEN) - seccion 12.5 del E2E |
| `current.json` | 3159 | Baseline local (ACTIVE_WITH_OPEN_DISCREPANCY) |
| `README.md` | 1802 | Explicacion del directorio |

**Baseline actual:**
- `local_observed_nipc_total` = **-4.264.449.012** (reproducible, commit 72fa824).
- `external_auditor_reference` = **-4.264.449.932** (declarado, sin cadena de custodia).
- `reconciliation_delta` = **920**.
- `reconciliation_status` = **OPEN**.
---

## 9. VERIFICACION EMPIRICA

**Suite IAE:** 777 funciones test_ (criterio AST, 45 ficheros). `885` tests collected por pytest (incluye parametrize). Reproducible con `scripts/iae_test_census.py`.

**Cobertura IAE:** 90% lineas (reproducible con `scripts/iae_coverage.py`).

**E2E contractual:** `scripts/iae_contractual_nipc_e2e.py` reconcilia `compute_nipc_contractual` contra el baseline vigente (`golden/current.json`, default) o contra el historico (`golden/12_5_historic.json`, `--historic`). **Correccion 2026-09-29:** el golden `12_5_historic` (commit 7caa86b, 2026-09-22) NO es reproducible con HEAD por cambios posteriores de codigo (H1-B v2.3 ce32c77 + O1 72fa824). Restaurar solo el crosswalk ya no basta. Estado: `current` reproducible, `12_5_historic` conservado como referencia historica.

**Coherencia verificada:**
- Pesos, MATCH_KEY, DELTA_COLUMNS, filtros SH/PUTCALL, reglas BOTH/NEW/EXIT/UNRESOLVED, fail-closed de TARGET_PAIRWISE vacio: coinciden literalmente con los contratos semanticos consolidados (ver `06_IAE_P65_P66.md` para P65/P66).
- Identidad dual `equity:<TICKER>` / `figi:<FIGI>`: capacidad del resolver, no observada en E2E con `figi_lookup=None`. Auditoria empirica A.1 sobre 4.780.572 units: 0 shareClassFIGI bajo dos canonical_security, 0 uso de rama `figi:*`, delta NIPC = 0 al normalizar por shareClassFIGI.
- GATE 1 nomenclatura catalog / TARGET. GATE 2 identidad dual (verificada empiricamente).
- Validacion externa OpenFIGI: 227/227 match sobre 243 CUSIPs resolubles, 16 sin hit (prefijos no-USA).

**Hallazgos del dictamen externo (2026-09-27):**

| ID | Estado |
|---|---|
| H1-A (drift historico E2E) | CERRADO (drift explicado). `12_5_historic` NO reproducible con HEAD tras H1-B v2.3 + O1; baseline vigente: `current.json` |
| H1-B (clasificacion CALL/PUT) | CERRADO con C2 abierto no bloqueante (920) |
| H2 (contadores documentales) | CERRADO |
| H3 (clasificacion scripts) | CERRADO |
| H4 (golden versionado) | CERRADO |
| H5.1 (criterio seleccion trimestre) | CERRADO |
| H5.2 (fallo ingesta sin alerta) | CERRADO |
| H5.3 (ingesta manual Q2 2026) | TRAZABILIDAD IMPLEMENTADA, pendiente cron nov 2026 |
| H5.4 (validacion inputs.quarter) | CERRADO |
| H5.5 (cache ante republicaciones) | CERRADO |
| H5.6 (Official List Q2 fail) | CERRADO (sufijo -txt) |
| H5.7 (commit mixto c5e3ee0) | HISTORICO |
| O1 (ausencia de imputacion) | CERRADO |

---

## 10. DEUDA ACTIVA

**H5.3 - Trazabilidad cron trimestral:** implementada. Pendiente verificacion en cron real de noviembre 2026.

**C2 (920) - Discrepancia H1-B:** `-4.264.449.012` (local) vs `-4.264.449.932` (auditor). Estado OPEN. Dos salidas: (a) si el auditor aporta comando + HEAD, se reproduce y cierra por atribucion; (b) si no, se mantiene OPEN sin convertir el 920 en "tolerancia aceptada".

**`build_effective_reporting_snapshot`** sin caller productivo (reporting_dedup.py). Funcion testeada (99 tests P65/P66 pasan) pero no invocada en el pipeline. **No es deuda:** es decision de diseno — capa auxiliar de diagnostico. No altera el NIPC (`_apply_intra_period_dedup` v1 solo emite KEEP). Conectar P65 no cambiaria el `nipc_total` actual. Verificado 2026-09-29.

**`scripts/iae_contractual_coverage.py`** reproduce §12.3 pero no esta integrado al flujo continuo de validacion.

**`validate_membership` / `validate_assignment` (catalog_key.py) sin consumidor productivo.** Verificado 2026-09-29: solo invocados en tests. Ademas, `validate_membership` espera un `membership_df` acumulativo (todas las versiones historicas) para resolver `predecessor_row_uid` contra versiones previas. El CSV real (`catalog_membership.csv`) contiene solo la version vigente, producido asi por `build_catalog_csvs.py`. Resultado: la funcion no es usable sobre los datos reales tal como estan. Deuda: o se cambia el generador para acumular versiones, o se cambia el validador para aceptar un `membership_df` por version + el historico como parametro aparte. Sin fix hasta decidir cual de las dos.

**SPCX en catalogo radar** (resuelto 2026-09-29): FIGI `BBG000NQF3Z5` presente en catalogo + crosswalk + snapshot. La deuda previa (sin fuente de identidad) ya no aplica.

**Cobertura FIGI del 12.2% en INFOTABLE.** Un fallback CUSIP -> FIGI via OpenFIGI podria ampliar la cobertura, pero no se ha demostrado que resuelva exhaustivamente los CUSIPs restantes.

**Deuda semantica del nombre NIPC** si se expone en reporte.

**Cobertura unitaria de `run_contractual_nipc`:** 76 stmts no cubiertos por tests con mock.

**Identidad canonica dual (`equity:*` / `figi:*`):** reabrible si se activa `figi_lookup` en algun punto del pipeline, si aparece un caso `figi:*` en units o delta, o si se modifica el modelo de resolucion de identidad.

---

## 11. REGLAS DE OPERACION

- No integracion a produccion sin validacion funcional previa.
- No push a `origin/main` hasta integracion verificada.
- No OpenFIGI masivo. Solo consultas dirigidas.
- No modificar codigo sin ciclo previo.
- Toda escritura pasa por `write_artifact_with_manifest` (parquets) o `append_dedup` (CSVs historicos).
- Todo bloque que haga `git commit` se condiciona a tests verdes (`if ($LASTEXITCODE -eq 0)`).
- **No congelar como golden contractual** ningun valor que no sea reproducible por el sistema local.
- **No presentar cifras declaradas por el auditor como hechos verificados.** Marcarlas siempre como "declaradas, sin cadena de custodia".
- **No mezclar breakdowns de estados distintos** en la misma tabla (el error que produjo el "943.371 fantasma").
- **No inventar explicaciones** para discrepancias no reproducidas. Declararlas y preguntar.
- **No convertir ausencia de evidencia en evidencia negativa.**

---

## 12. LIMITACIONES Y ALCANCE

**Alcance del NIPC:** solo posiciones declaradas en filings 13F sobre el universo del radar. `§10.1` filtro explicito. No es NIPC del mercado completo.

**Cobertura del catalogo:** declarada 99.17% sobre 242 tickers. Cobertura TARGET construido: 100% por construccion (endogeneidad por diseño: el TARGET se construye desde el catalogo, asi que la cobertura TARGET es 100% por definicion).

**Latencia regulatoria:** N-PORT ~60 dias, 13F trimestral (~45d post cierre trimestre).

**Ambiguity de identidad:** el resolver puede en teoria tener dos canonical_security para una misma security (dualidad equity/figi). No observado en E2E. Se reabre si se activa figi_lookup o aparece un caso `figi:*`.

**PIT (Point In Time):** el catalogo tiene snapshots versionados. La identidad historica vs pertenencia historica al radar es un riesgo declarado (B4).

**Unidades:** SSHPRNAMT en miles de acciones. Declarado en codigo, no x1000.

**Oficial 13(f):** el parser puede dar UNRESOLVED para CUSIPs que no aparecen en la lista. No se imputa; se clasifican como UNRESOLVED y se mantienen como tales.