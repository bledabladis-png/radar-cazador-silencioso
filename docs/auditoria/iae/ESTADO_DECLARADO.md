# IAE - ESTADO DECLARADO

Declaraciones subjetivas del proyecto IAE. Los hechos verificables
(HEAD, tests, LOC, hashes) viven en `ESTADO_SISTEMA.md` y se regeneran
por script. Este fichero se edita cuando cambia el estado de una fase,
no cuando cambia el codigo.

**Actualizado:** 2026-09-22
**Responsable de actualizar:** Ingeniero Supervisor.
**Origen de verdad:** este fichero + `ESTADO_SISTEMA.md`. Los demas
documentos describen su tema; NO declaran el estado del sistema.

---

## 1. Fases del modulo IAE

| Fase | Estado | Referencia |
|---|---|---|
| FA-1 (ingestion SEC 13F) | CERRADO / PUSHED | - |
| FA-2 (identity: temporal_filter, cusip_resolver, relationships, amendments) | CERRADO / PUSHED | - |
| B2-PIT (infraestructura temporal catalogo) | CERRADO | #53 |
| B1 (TARGET + adaptador P38, 5 modulos) | CERRADO | #68 |
| B3 (semantica temporal 13F: timestamps + absence) | IMPLEMENTADO SPEC-SIDE (sin integracion) | - |
| Gap spec->codigo seccion 5.1-5.5 | CERRADO | #61-#67 |
| A.6.0 (Gate 0 bloqueantes F2.4) | CERRADO | #43 |
| A.6.1 (dictamen F2.4) | CERRADO | 2026-09-20 |
| A.6.2 (fixes quirurgicos P60/P61/P38) | CERRADO | - |
| A.6.2-bis-B2-PIT | CERRADO | #53 |
| A.6.2-bis-B1 | CERRADO | #68 |
| A.6.2-bis-B3 | IMPLEMENTADO SPEC-SIDE | - |
| A.6.3 (test Q12 Modelo A) | CERRADO | #72 |
| A.6.4-v2 (P38 aislado sobre TOP 2000) | CERRADO | #73 |
| A.6.4 (integracion B1+P61+P38; pre-fix A2) | RECLASIFICADO post-A2: evidencia contractual parcial (rama Q1 + fail-closed); NO pairwise completo | #75 + #76 |
| A.6.5 (actualizar contratos in-place) | CERRADA | dccf70d + 911e95b + 0f0583d + a21b0db |
| A.6.4-fix-A2 (H-05/H-06/H-07/H-10.1) | CERRADO (A2 = GO, #76) | 2f98926..157211e 2026-09-21 |
| A.6.6-saneamiento (post #76) | CERRADO (este commit) | B-01/B-05/B-06/B-07 + H-08 |
| A.6.6 (F2.4-CLOSE) | NO-GO (#76). Bloqueantes: B-02 (P62/PIT), B-03 (TARGET), B-04 (pairwise). | #76 |

## 2. Hallazgos cerrados

| ID | Descripcion | Dictamen |
|---|---|---|
| H-69.1 | source id exceptions -> cusip_ticker_exceptions | #70 |
| H-69.2 | colision BRK-B vs BRK.B; regla R-69.2 | #72 |
| H-73.1 | adapter B1<->P38 (weight_status mapeado a operational_mapping_status) | #75 |
| GHISALLO | +765% caracterizado como artefacto del probe (no economico) | #60 |
| H-05 | probe sin mock Q4=Q1; Q4 vacio -> coverage_previous=None | A2 c5 |
| H-06 | probe publica cardinalidades reales | A2 c5 |
| H-07 | adapter propaga weight desde state.sshprnamt | A2 c3 |
| H-10.1 | adapter propaga operational_mapping_status desde state | A2 c3 |

## 3. Contratos y policies vigentes

| Documento | Estado |
|---|---|
| `NIPC_CONTRATOS_SEMANTICOS_v1.md` | VIGENTE. P60/P61 GO, P38 GO CONDICIONADO |
| `NIPC_COVERAGE_POLICY.md` v1.0 | VIGENTE (normativa) |
| `INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md` v1.4 | Referencia, no normativa |
| `IAE_MAESTRO.md` | Referencia unica del modulo IAE. Estado, arquitectura, evidencia, consulta. |
| `evidence/p66_e2e_probe/` | Evidencia P66 e2e (10.676 filings Q4, 0 R4 no clasificables). Complementa a `p66_gate*`. |

## 4. Prohibiciones vigentes

- NO push a main de codigo (local-first IAE)
- NO activar `compute_nipc_contractual` sin thresholds
- NO ejecutar OpenFIGI masivo
- NO activar `DROP_DUP`
- NO modificar contratos normativos sin dictamen
- NO reescribir snapshots publicados
- NO modificar `coverage.py`, `nipc.py`, `delta_shares.py`
- NO modificar `security_identity.py` ni `temporal_validity.py`

## 5. Deuda activa (no bloqueante)

- Integracion de modulos IAE al pipeline productivo (`daily_run.yml` NO los invoca)
- `compute_nipc_contractual` SIN CALLERS PRODUCTIVOS. Caller de smoke
  anadido al probe A.6.4 (commit `6126c0d`, seccion 7): computa sin
  error con delta vacio. No activa el modulo productivamente
  (thresholds UNDEFINED, dictamen #76).
- `build_effective_reporting_snapshot` SIN CALLERS PRODUCTIVOS (verificado: solo tests)
- `DROP_DUP` NO ACTIVADO (capacidad diferida v2)
- Marcadores seccion 5.4 VERIFICADOS (2026-09-22): no son deuda.
  `security_type.py` esta alineado con dictamenes #61-#63:
  ADR/SPONSORED ADR/SPONSORED ADS excluidos por diseno (L162-164);
  SPON ADS es frase EQUITY autorizada, no marcador residual;
  FUND_EXCLUDED_PHRASES es lista de exclusion legitima;
  SH BEN INT y ACT no existen como marcadores en el modulo.
  Sin cambios de codigo.
- Snapshots historicos Q4 2025 / Q1 2026: **correccion 2026-09-22**.
  Universo contractual real = 242 keys (no 35.649 CUSIPs 13F, que son
  OUT_OF_TARGET). Hueco real: 2 MISS (BRK-B, MOG-A) -> 2 re-queries
  OpenFIGI, NO masivo. Ver `IAE_MAESTRO.md` seccion 4.
- Curacion crosswalk residual CERRADA (2026-09-22): ONB (CUSIP 680033107)
  y PTGX (CUSIP 74366E102) anadidos a `cusip_ticker_exceptions.csv` tras
  verificacion en INFOTABLE Q1 (98 y 617 hits). SPCX NO procede:
  SpaceX es empresa privada; los hits del patron son el ETF ARK Space
  Exploration (CUSIP 00214Q807). La entrada de SPCX en el catalogo
  radar queda marcada como observacion pendiente de dictamen.
- Nueva sub-deuda detectada: BRK-B y MOG-A con `status=MISS` en el
  catalogo radar (sin share_class_figi). No resoluble via crosswalk
  (ambos ya estan en `cusip_ticker_exceptions.csv`). Requiere OpenFIGI.
- Gate-NIPC.2 BLOQUEADO por THRESHOLD_1/2 UNDEFINED
- **Hallazgos #77 (MEDIA, no bloqueantes):**
  - H-19: bundle con multiples HEAD ambiguos. Requiere etiquetar
    `HEAD_DEL_EXPEDIENTE` vs `HEAD_DEL_CODIGO_AUDITADO` vs
    `HEAD_DE_LA_EVIDENCIA` antes del proximo bundle.
  - H-20: afirmacion CI 0 failed sin SHA/run. Requiere aportar
    CI run + commit SHA + resultado en el proximo expediente.
- **B-03 CERRADO (2026-09-22, dictamen #77):** BRK-B (scf BBG001S90346)
  y MOG-A (scf BBG001S5T922) materializados via OpenFIGI re-query.
  Catalogo 240/242 -> 242/242 keys con share_class_figi. Snapshot B2-PIT
  `20260922_01` creado (P62: no se borra el anterior). Probe A.6.4
  actualizado para leer snapshot vigente del manifest: `catalog_keys Q1`
  pasa de 20 a 22, `records_q1_verified: 22`, `coverage_current: 1.0`.
  Membership migrada preservando catalog_key (inmutable) y marcando
  `predecessor_row_uid` para las 2 filas con UID cambiado.
- **B-02 CERRADO (2026-09-22, dictamen #77):** el probe A.6.4 emite
  `pit_status` explicito. Modo `strict-pit` (default): invoca
  `catalog_pit.target_catalog_as_of("2026-03-31")`, que lanza
  `CatalogNotAvailable` porque el snapshot vigente (valid_from
  2026-09-22) no cubre Q1. El probe publica `PIT_UNAVAILABLE`,
  `coverage_current=None`, `coverage_contractual=False`. El numero
  tecnico pre-PIT se preserva bajo `coverage_current_technical_pre_pit`
  para trazabilidad. Modo `--no-pit`: bypass documentado con
  `PIT_BYPASS_DOCUMENTED`. Prohibido retro-fechado (dictamen #77).
- **B-04 CERRADO (2026-09-22, dictamen #77):** rama pairwise ejercitada
  con fixture sintetico explicitamente marcado
  `NON-PRODUCTION / CONTRACT_TEST_FIXTURE`. Ficheros:
  `tests/test_p38_pairwise_fixture.py` (6 tests) +
  `evidence/a64_integration_b1_p61_p38/probe_pairwise_fixture.py`.
  Evidencia: `paired_security_coverage=0.5`, `paired_weighted_share_coverage`
  `= 6/7`, `coverage_status=VALID` sobre TARGET_PAIRWISE={A,D} con PAIRED={A}.
  NO se han fabricado exceptions ni tocado mappings productivos
  (prohibicion explicita #77). Ver `result_pairwise_fixture.json`.
  Pendiente: TARGET_PAIRWISE real > 0 sobre datos 13F reales, que
  requiere snapshot historico PIT (B-02.2, no autorizado).
- Sub-deuda documental (2026-09-22): `NIPC_CONTRATOS_SEMANTICOS_v1.md`
  L332 cita '18 FIGI -> 18 records VERIFIED' en la trazabilidad A.6.4,
  pero tras la curacion ONB/PTGX el valor real es 20. NO se ha tocado
  el contrato (prohibicion vigente: NO modificar contratos normativos
  sin dictamen). Requiere dictamen para actualizar in-place.
- A.6.6 NO-GO (#76). Saneamiento post-dictamen CERRADO.
- Cobertura de tests: 9 funciones publicas IAE sin test directo
  cubiertas por `tests/test_iae_gap_coverage.py` (commit `12e1e3f`).
  Cobertura directa actual: 110/110 funciones publicas mencionadas.
- **Determinismo verificado (2026-09-22):** ejecucion repetida del probe
  A.6.4 (result.json) y del probe P70 (result.json + summary) produce
  hashes SHA-256 identicos bit a bit. Sin `datetime.now()`, sin
  dicts no ordenados. Propiedad verificada empiricamente, no solo
  declarada en docstring.
- **CI-clean verificado (2026-09-22):** los 3 failed locales son
  artefacto de tener parquets stale. En CI (clone sin parquets):
  los tests de integracion usan `@pytest.mark.skipif(not exists)` y
  se saltan; los 7 tests IAE (b06_e2e, iae_gap_coverage,
  iae_gap_semantic, iae_pipeline_report, p38_contract,
  h731_adapter, sec_13f_reporting_dedup) no leen parquets -> pasan.
  Suite en CI: 0 failed. B-07 cerrado con evidencia directa.
- **3 failed preexistentes (`test_freshness.py`) - aclaracion 2026-09-22:**
  no son regresion IAE. Los 3 tests leen `data/market_data.parquet` y
  `data/stock_prices.parquet` con `pytest.mark.skipif(not exists)`.
  En **CI** (clone fresco sin parquets) se **saltan** -> suite global
  verde. En **local** con parquets presentes y stale (last_date muy
  anterior a la fecha de ejecucion), fallan por antiguedad.
  Ambos parquets tienen manifest con status VALID/VALID_WITH_MISSING
  y last_date == expected_session; el problema es solo la distancia
  temporal hasta 'hoy'. No hay causa en codigo IAE.

## 6. Proxima accion autorizada

Saneamiento post-#76 CERRADO (commit actual). Bloqueos residuales NO
autorizados: B-02 (P62/PIT), B-03 (TARGET completo), B-04 (pairwise
real). Requieren dictamen especifico antes de tocarse. Push: NO.

## 7. Saneamiento documental (sub-deuda en curso)

Sesion 2026-09-21: detectadas 82 referencias fantasma en `docs/auditoria/`.

Cerrados:
- `DICTAMENES.md` (23 fantasmas, commit `10f65bb`)
- `INFORME.md` (17 fantasmas, commit `8e04d22`)
- Contradiccion A.6.5 en contrato NIPC (commits `0f0583d` + `a21b0db`)

Cerrado en este commit (2026-09-22):
- `FOLLOWUPS.md`: 12 referencias a ficheros consolidados, cubiertas
  con nota de consolidacion (§3.7). Los 5 hits restantes son
  placeholders pedagogicos del §3.7 o un path vivo no versionado
  (`outputs/report/reporte_diario.md`).
- `PROMPT_MAESTRO.md`: 0 fantasmas reales (verificado por script sobre
  disco, 2026-09-22). Los 5 hits detectados son placeholders
  pedagogicos del propio §3.7. La afirmacion previa '8 fantasmas
  materiales' queda refutada.
- `iae/*.md`: 5 orphans verificados como expedientes legitimos del
  ciclo A.6.x. No se borran.

## 8. Hallazgos del auditor pendientes (2026-09-21)

Hallazgos detectados en la auditoria externa del bundle del 2026-09-21,
tras revisar la evidencia A.6.4. **Requieren dictamen antes de tocar codigo
productivo** (regla: NO modificar `coverage.py`, `nipc.py`, `delta_shares.py`
sin dictamen).

| ID | Severidad | Descripcion | Ficheros afectados |
|---|---|---|---|
| H-08 | MEDIA | Test H-73.1 no cubre estados ortogonales (`identity=RESOLVED` pero `weight=NOT_PRESENT`). | `tests/test_h731_adapter_p38_compat.py` |

**Cerrados 2026-09-21 (fix A2, commits 1-5):**

- H-05 (ALTA): eliminado mock Q4=Q1 en probe (commit 5). Fail-closed cuando Q4 vacio.
- H-06 (ALTA): probe publica cardinalidades reales (commit 5).
- H-07 (ALTA): adapter propaga weight desde state.sshprnamt (commit 3).
- H-10.1 (CRITICA): adapter propaga operational_mapping_status desde state (commit 3).
- H-11 (ALTA): hashes de provenance (cerrado antes del fix A2).
- H-12 (MEDIA): hash de catalogo etiquetado (cerrado antes del fix A2).

**Consecuencia (actualizada post #76):** A.6.4 se reclasifica como
**evidencia contractual parcial**: valida rama Q1 (20/20 sobre TARGET_Q1
materializado) y rama fail-closed (Q4 vacio -> coverage_previous=None),
pero NO valida pairwise real (TARGET_PAIRWISE=0). Ver
`evidence/a64_integration_b1_p61_p38/README.md` y `IAE_MAESTRO.md` seccion 4.

**Pendientes de dictamen:** B-02 (P62/PIT), B-03 (TARGET completo,
requiere OpenFIGI masivo), B-04 (pairwise real). Ver seccion 9.


## 9. Hallazgos del dictamen #76 (2026-09-21)

Dictamen A.6.6: **NO-GO**. A2 = GO (H-05/H-06/H-07/H-10.1 aceptados).

| ID | Sev | Descripcion | Estado |
|---|---|---|---|
| B-01 | CRITICO | Saneamiento documental post-A2 | CERRADO (este commit) |
| B-02 | CRITICO | P62/PIT: snapshot valid_from=2026-09-21 para Q1 2026 (look-ahead) | NO AUTORIZADO |
| B-03 | CRITICO | TARGET contractual completo sin materializar (OpenFIGI masivo) | NO AUTORIZADO |
| B-04 | CRITICO | Pairwise real no ejercitado (TARGET_PAIRWISE=0) | NO AUTORIZADO |
| B-05 | ALTA | Precision: coverage_current=1.0 = 'sobre TARGET materializado' | CERRADO (este commit) |
| B-06 | ALTA | Riesgo doble conteo SSHPRNAMT por FIGI | CERRADO (este commit) |
| B-07 | MEDIA | 3 test_freshness failed sin exclusion formal | CERRADO (este commit, docstring) |
| H-08 | MEDIA | Test ortogonal identity=RESOLVED + weight=NOT_PRESENT | CERRADO (este commit) |

Push: NO AUTORIZADO (#76).
