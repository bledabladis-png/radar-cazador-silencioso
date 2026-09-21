# HISTORICO IAE - Registro de ciclos, fixes y cierres

**Objeto:** registro historico del proyecto IAE y del radar.
**NO normativo.** Para reglas vigentes, ver `PROMPT_MAESTRO.md`.
Para estado actual, ver `ESTADO_DECLARADO.md` + `ESTADO_SISTEMA.md`.

Este fichero recoge lo que antes vivia dentro del prompt (§11.17-§11.32
y SECCION 15), extraido durante la reestructuracion v7 (2026-09-21).

---

### 11.17. K-INDEX-RANGE-01 — build_ticker_df en index_phase y sector_regime (2026-09-18)

**Bug:** `wyckoff_structure_core` degenera al fallback silencioso `RANGE`
(`indicators/wyckoff.py:337-340`) cuando recibe un df con NaN internos. Afectaba a:

- `indicators/index_phase.py:19` — 4/8 indices (`^DJI`, `^IBEX`, `^GDAXI`, `^FTSE`).
- `indicators/index_phase.py:33` — mismo patron en el fallback (router directo).
- `regimes/sector_regime.py:114` — 7/11 ETFs (`XLF`, `XLV`, `XLE`, `XLY`, `XLI`, `XLRE`, `XLC`).

**Confirmado empiricamente** (Gate 0 v22+v23): patron A (`df` directo) vs patron B
(`build_ticker_df`) produce fases distintas para 4/8 indices y 7/11 ETFs.

**Impacto funcional:**

- Tabla *Indices Internacionales — Fases Wyckoff* mostraba 8/8 RANGE.
- Tabla *Indices Internacionales — Oportunidades de Acumulacion y Markup* siempre vacia.
- **SLPM bloqueado en `n=0`** (LIS=0, Eff Breadth=0). El fallback RANGE impedía que
  XLC entrara en ACCUMULATION, y sin ACCUMULATION el SLPM no encontraba lideres.
  Con el fix: LIS=+0.20, Eff Breadth=0.53, 20 tickers en *Acciones Seleccionadas*.
- Etiqueta `Fase Wyckoff` en *Rankings Sectoriales* corregida (no altera score).

**Fix:** `build_ticker_df(df, ticker)` antes de clasificar en los 3 puntos afectados.

**Commit:** `562dc41`. Tests: `tests/test_index_phase_range.py` (4).

### 11.18. PCR Indices N/D — index_pcr ausente del return dict (2026-09-18)

**Bug:** `indicators/options.py::compute_pcr_signals()` construia internamente
`data["index_pcr"]` (usado en L65 para `institutional_hedge_ratio` y guardado en el
CSV historico via `base_cols` L97), pero NO lo incluia en el dict de retorno.
`src/report/sentiment.py:34` leia `pcr_data.get("index_pcr")` y caia al default
`np.nan`, mostrando `PCR Indices: N/D` en el reporte.

**Fix:** 1 linea en el return dict de `indicators/options.py`.

**Commit:** `849f981`. Tests: `tests/test_pcr_indices.py` (2).

### 11.19. K-RENDER-LEADER-TABLE-01 — separator Markdown desalineado (2026-09-18)

**Bug:** el separator Markdown de las tablas de lideres sectoriales
(`indicators/stock_leader.py:180`) tenia 8 columnas mientras el header (L179)
tenia 11. En GFM, un separator mas corto que el header hace que las columnas
sobrantes (`Pers 20d`, `Spring`, `SOS`) se ignoren al renderizar.

**Detectado en el reporte del 2026-09-18** tras desbloquear el SLPM (11.17):
las 4 tablas `## Sector: X (FASE)` con 5 lideres cada una mostraban Spring y
SOS cortados.

**Fix:** separator con 11 grupos (1 linea).

**Commit:** `44cd544` (rebase final `29f393a`). Tests: `tests/test_leader_table_markdown.py` (1).

### 11.20. IAE FA-1 - Ingestion SEC 13F cerrada y pusheada (2026-09-19)

**Ciclo:** 7 commits pusheados a main (HEAD 626b39d). Fase A del modulo
Institutional Accumulation Evidence (IAE), Gate FA-1 superado.

**Alcance FA-1 (cerrado):**
- `src/institutional_accumulation/sec_13f/` (7 ficheros): schema, downloader,
  parser, storage, manifest, ingest + `__init__`.
- Descarga ZIP trimestral SEC con cache + retry 5xx + SHA-256.
- Extraccion validada por CRC (`zipfile.testzip()`) + presencia de los 7 TSVs.
- Parsing con dtypes nullable + `errors="coerce"` (nunca falla por dato sucio).
- Escritura atomica de parquets snappy (.tmp + os.replace).
- Manifest de lineage a 3 niveles: `source.sha256` + `tsv_files[name].sha256`
  + `parquet_files[name].sha256`.
- 72 tests IAE (downloader 15, schema 13, parser 15, storage 10, manifest 11, ingest 8).

**Probe real (FA-1.5, 2026-09-19):** ingest_13f contra 3.3M filas del ZIP
Q1 2026 (SHA-256 `05f4da8f526cd471...`). 7/7 row counts coinciden con Gate 0.
Manifest v1 con 3 niveles verificable. 3 flags validation a True.

**Fuera de alcance FA-1 (va a FA-2):**
- Resolucion de identidad (Q2, framework 3 niveles).
- Filtro `PERIODOFREPORT=31-MAR-2026` (el parser de FA-1 procesa el dataset
  completo por diseno).
- CUSIP->ticker + tabla de excepciones con vigencia temporal.
- Tratamiento DFND, mapeo SH/PRN/Put/Call, resolucion de enmiendas.
- DeltaShares, NIPC, Breadth, New/Exit, clasificacion.

**NIPC permanece BLOQUEADO hasta Gate FA-2.**

**Documentos:**
- Informe Gate FA-1: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA1_INFORME.md`.
- Addendum contrato v1.1 (D1-D5): `docs/auditoria/INSTITUTIONAL_ACCUMULATION_CONTRATO_ADDENDUM.md`.

### 11.21. IAE FA-2 - SEC 13F core cerrado y pusheado (2026-09-19)

**Ciclo:** Gate FA-2 PASS. Commit `85ccf7d` informe + `9543de5` dictamen.

**Pipeline FA-2 completo (4 modulos):**
- FA-2.1 `temporal_filter.py`: filtro por `SUBMISSION.PERIODOFREPORT`.
- FA-2.2 `cusip_resolver.py`: CUSIP -> ticker con vigencia temporal (multi-fila).
  Contrato verificado por tests. Crosswalk sin curar (deuda posterior).
- FA-2.3 `relationships.py`: Column 7 -> `OTHERMANAGER2.SEQUENCENUMBER`.
  6 estados de token + 5 anomalias catalogadas. Multi-edge sin division economica.
- FA-2.4 `amendments.py`: canonical snapshot composicional.
  RESTATEMENT -> REPLACE, NEW HOLDINGS -> ADD. Lineage auditable.

**Metricas probe integrado (Q1 2026):**
- Filtro: 10,776 filings, 3,321,967 filas INFOTABLE.
- Snapshot: 8,762 accessions aplicados, 3,239,273 source lines canonicas.
- Estrategias (10,648 grupos): SINGLE_HR 8,618; SINGLE_NOTICE 1,906;
  HR_PLUS_RESTATEMENT 100; HR_PLUS_NEW_HOLDINGS 19;
  HR_CHAIN_RESTATEMENT 2; HR_COMPOSITE 2; NOTICE_AMENDED 1.
- Edges: 3,509,475 totales; 1,351,172 resueltos; resolution_rate 0.9883;
  duplicate_canonical_edges 0; invalid_source_line_edges 0.
- Anomalias: 2 (NT_AUGMENTED, HR_WITH_AMENDMENT_FLAGS).

**Deuda posterior (fuera de FA-2):**
- Curacion manual crosswalk CUSIP -> ticker (tabla vacia en FA-2).
- NIPC: desbloqueado de FA-2; pendiente su propio Gate.

**Documentos FA-2:**
- Gate FA-2 informe: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_INFORME.md`.
- Gate FA-2 dictamen: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_GATE_FA2_DICTAMEN.md`.

### 11.22. IAE NIPC C1 + C4-revisadas (2026-09-19)

**Ciclo:** dictamen Gate-NIPC.1 v1.1 PASS CONDICIONADO + dictamen probe end-to-end.

**C1-revisada:** `canonical_security` NO es ticker. Dos capas de estado:
  - `security_resolution_status` (5): CANONICAL | OBSERVED_ONLY | UNRESOLVED | AMBIGUOUS | CONFLICT.
  - `canonical_security_kind` (4): CANONICAL_FIGI | CANONICAL_EQUIVALENCE | OBSERVED_CUSIP_ONLY | UNRESOLVED.
  `observed_security_key = cusip:<CUSIP>`. `canonical_security = NULL` salvo `status == CANONICAL`.
  Tabla `data/mappings/cusip_equivalence.csv` (esquema: CUSIP_A, canonical_security, valid_from, valid_to, source, reason, verified_by, source_document). 0 filas por diseno.

**C4-revisada:** identity temporal validity != metadata temporal validity.
  - `shareClassFIGI` estable frente a corporate actions.
  - `TEMPORAL_UNVERIFIED` aplica SOLO a metadata (ticker), no a identidad FIGI.

**Spec NIPC v1.4 activa:** `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION.md`. Versiones v1.0-v1.3 preservadas como `_v1.X.md`.

### 11.23. IAE filer continuity CERRADO como caracterizacion (2026-09-19)

**Dictamen TOP 50 PASS.** Ciclo de caracterizacion CERRADO.

**Hallazgo Vanguard Q4 2025 -> Q1 2026:**
  - Padre VANGUARD GROUP INC (CIK 0000102909): 13F-HR -> 13F-NT.
  - Q1: 2 filiales nuevas con HR (VANGUARD CAPITAL MGMT 35.329B, VANGUARD PORTFOLIO MGMT 21.030B) + FIDUCIARY TRUST (4.124B).
  - Modelo Q4: padre HR + filiales NT. Modelo Q1: padre NT + filiales HR.
  - Match key C2 produce EXIT padre + NEW filiales. No es bug: implementa C2 fielmente.
  - Efecto: nipc_sole ~ -41.16B, nipc_dfnd ~ +41.28B, cancelacion.

**Mini-probes CERRADOS:**
  - TOP 20: 3/24 FILER_DISCONTINUITY (12.5%), todas Vanguard.
  - TOP 50: 4/54 FILER_DISCONTINUITY (7.4%), todas Vanguard.
  - 0 nuevos patrones fuera de Vanguard.

**Decisiones del auditor:**
  - Filer continuity = CONTROL DE INTEGRIDAD / DIAGNOSTICO, NO threshold.
  - `filer_status` por presencia documental (NO por SSHPRNAMT > 0). Nueva dimension `position_mass_status` (HAS_SHARES | ZERO_SHARES | NO_CANONICAL_HOLDINGS).
  - NT -> OTHERMANAGER -> HR = EVIDENCIA DOCUMENTAL. NO inferencia economica.
  - Reconciliacion NT-HR NO AUTORIZADA.

### 11.24. Regla canonical_snapshot CONGELADA (2026-09-19)

**Regla metodologica obligatoria:**

  Todo agregado de SSHPRNAMT debe calcularse sobre el `canonical_snapshot` producido por `apply_amendments`. PROHIBIDO sumar SSHPRNAMT desde `INFOTABLE.parquet` filtrado por periodo cuando existan amendments RESTATEMENT.

**Motivo:** bug detectado en informe mini-probe. Sumar raw duplica el HR original cuando existe RESTATEMENT posterior (ratio 1.988 en CIK 0002100119). El motor NIPC siempre opero sobre canonical_snapshot; solo el informe fue incorrecto.

**Excepcion:** analisis de filings individuales (lineage, deteccion de supersedings). Ahi si se recorre raw, pero nunca se presenta como agregado.

---

### 11.25. IAE Q-CUR - Curacion CUSIP -> ticker Q1 2026 (2026-09-19)

**Ciclo:** Gate 0 + dictamen Q-CUR-1 / Q-CUR-2. Alcance minimo aprobado: 3 filas COM.

**Tabla operativa:** `data/mappings/cusip_ticker_exceptions.csv`.

CUSIPs poblados (equity Q1 2026):

  - `26614N102` -> DD
  - `438516106` -> HON (sustituido tras reverse split 2026-06-29)
  - `30231G102` -> XOM

**Excluidos:**

  - CALL/PUT (`26614N902`, `26614N952`, `438516906`, `438516956`, `30231G902`, `30231G952`): derivados, modelo separado no implementado en este ciclo.
  - HONA / FDXF: FUTURE_CORPORATE_ACTION / POST_Q1_ENTITY (spin-off posterior a PERIODOFREPORT=2026-03-31).

**Regla congelada:** no poblar `valid_from=fecha_spin_off` sin observacion 13F que lo requiera.

**Distincion obligatoria:**

  - `cusip_ticker_exceptions.csv` = tabla operativa (poblada, 3 filas).
  - `cusip_equivalence.csv` = tabla de equivalencias (C1-revisada, vacia por diseno).

**Commits en origin/main:** e4a6576, e8d7e53, 9d4a81e.

---

### 11.26. IAE NIPC coverage baseline Fase A (2026-09-19)

**Ciclo:** Q-T50-5 (post dictamen TOP 50). Sin OpenFIGI, sin thresholds, sin modificacion del motor.

**Metodologia congelada antes del run:** `README.md` del directorio de evidencia (`docs/auditoria/iae/evidence/nipc_gate0_baseline/`).

**4 universos anidados (nomenclatura corregida por el auditor):**

    RAW_13F_SH_NULL      SH + PUTCALL NULL
       v
    ELIGIBLE_SEC         + SEC Official List {ACTIVE, ADDED}
       v
    TECHNICAL_RADAR      + ticker in radar_equities  (cierre del tecnico)
       v
    OPERATIONAL_EQUITY   + get_instrument_class(ticker) == EQUITY

**Metricas pairwise Q4 2025 -> Q1 2026:**
| metrica | RAW_13F | ELIGIBLE_SEC | TECH_RADAR | OP_EQUITY |
|---|---:|---:|---:|---:|
| coverage_previous | 0.0206 | 0.0424 | 1.0000 | 1.0000 |
| coverage_current | 0.0204 | 0.0386 | 1.0000 | 1.0000 |
| paired_security_coverage | 0.0181 | 0.0368 | 0.9909 | 0.9909 |
| paired_weighted_share_coverage | 0.2271 | 0.2772 | 0.9864 | 0.9864 |
| NIPC total (observable) | +23,651,586 | +52,570,648 | -109,460,838 | -109,460,838 |
| STATUS | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT | INSUFFICIENT |

**Hallazgos:**

  H1. `TECHNICAL_RADAR == OPERATIONAL_EQUITY` en units (identidad exacta Q4=474,395 / Q1=490,732). El filtro `get_instrument_class == EQUITY` es no-op sobre el crosswalk actual. El cuarto nivel es verificacion, no filtro efectivo.

  H2. En el universo operativo `coverage_previous == coverage_current == 1.0` por construccion (el pre-filtro a `ticker in radar` ya exige CANONICAL). Los thresholds solo pueden descansar sobre las 2 metricas pairwise.

  H3. NIPC cambia signo entre RAW (+23.65M) y OP_EQUITY (-109.46M). Propiedad del universo operativo (20% de las filas, 100% del radar).

**Fallout ELIGIBLE_SEC -> TECHNICAL_RADAR (Q1 2026):**

| categoria | %units | %peso |
|---|---:|---:|
| IN_RADAR | 20.47% | 28.75% |
| CANONICAL_TICKER_OUTSIDE_RADAR | 13.40% | 10.67% |
| FIGI_CANONICAL_NO_TICKER | 0.00% | 0.00% |
| OBSERVED_ONLY_NO_CANONICAL | 66.13% | 60.58% |
| UNRESOLVED / AMBIGUOUS / CONFLICT | 0.00% | 0.00% |
| OTHER | 0.00% | 0.00% |
**Incidente resuelto:** `baseline_output.txt` se genero con CRLF via `Start-Process -RedirectStandardOutput`. `.gitattributes` (`*.txt eol=lf`) normalizaria a LF en el siguiente clone, invalidando el hash registrado. Correccion: normalizar a LF + recalcular hash + actualizar `HASHES.txt` + actualizar informe + `git commit --amend`. Verificado: los 3 hashes de disco coinciden con los registrados.

**Documentos:**

  - Informe: `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md`.
  - Evidencia: `docs/auditoria/iae/evidence/nipc_gate0_baseline/` (README + probe + output + HASHES).

**Regla:** este ciclo NO fija thresholds. Gate-NIPC.2 SIGUE BLOQUEADO. THRESHOLD_1/2 UNDEFINED. Sin OpenFIGI. Sin push.

### 11.27. Micro-gate Coverage Contract Normalization (2026-09-19)

**Origen:** dictamen del auditor sobre coverage baseline (commit 02b077c, Issue ABIERTO: circularidad `operational_universe`).

**Hallazgo:** `NIPC_COVERAGE_POLICY.md` v1.0 define `operational_universe` incorporando `ticker_mapped`, y despues lo usa para medir coverage. Resultado: `coverage = mapped / total` se vuelve 1.0000 por construccion sobre ese universo. Medido en baseline (commit e9fd830): TECHNICAL_RADAR y OPERATIONAL_EQUITY ambos con coverage = 1.0000.

**Cadena de fases autorizada por el auditor:**

  - F2.1   Gate 0 documental: inventario de los 3 conceptos en la policy v1.0.
  - F2.1-bis  Fuentes de identidad: verificar existencia de RADAR_TARGET_REGISTRY independiente del crosswalk evaluado.
  - F2.2   Propuesta v1.1 (borrador).
  - F2.3   Dictamen del auditor sobre la propuesta.
  - F2.4   Aplicacion de v1.1 sobre `NIPC_COVERAGE_POLICY.md` (solo si F2.3 aprueba).

**F2.1 PASS (commit eda67ef) - inventario 3 conceptos:**

  Inventario linea-por-linea de TARGET / RESOLVED / PAIRED en `NIPC_COVERAGE_POLICY.md` v1.0 (hash 57f2d01f...).

  - TARGET: 0 ocurrencias literales. Existe `operational_universe` que ya incluye `ticker_mapped`.
  - RESOLVED: 0 ocurrencias literales. Existe `mapped / total` sin definir ninguno de los dos terminos.
  - PAIRED: 13 ocurrencias. Existe con 2 metricas pero sin universo base ni criterio de emparejamiento explicito.

**F2.1-bis PASS (commit 4659ce1) - Camino B declarado:**

  Ninguna fuente en disco cumple simultaneamente:
    (a) clave CUSIP/FIGI comparable con 13F,
    (b) ticker del radar USA,
    (c) independiente del crosswalk evaluado.

  Fuentes verificadas:

    - `data/mappings/isin_ticker_map.csv`: 49 tickers, 0 sin sufijo EU.
    - `data/etf_holdings.csv`: 220/242 tickers USA, pero ES el crosswalk evaluado.
    - `data/amundi_yahoo_mapping.csv`: 0 tickers USA.
    - `data/index_holdings.csv`: sin columna CUSIP/identifier.
    - `data/mappings/cusip_equivalence.csv`: 0 filas por diseno.
    - `data/mappings/cusip_ticker_exceptions.csv`: 3 filas, parte del crosswalk.

  Hallazgo colateral: el 90.91% (220/242) que la policy v1.0 L115 cita como
  `mapping_coverage (radar USA)` sale exactamente de `etf_holdings.csv`.
  Es decir, la policy cita como "cobertura" el ratio del propio insumo del
  resolver. Confirmacion numerica de la circularidad.

  Declaracion formal: `TARGET_CUSIP_REGISTRY = NOT AVAILABLE (2026-09-19)`.

**F2.2 NO GO CON CAMBIOS OBLIGATORIOS (commit d4a926e + dictamen):**

  La propuesta v1.1 separaba TARGET / RESOLVED / PAIRED conceptualmente bien,
  pero definia:

    TARGET_UNIVERSE = radar_equities ^ section13f_eligible ^ EQUITY

  con claves incompatibles:
    - `radar_equities`: {ticker} (columnas de `stock_prices.parquet`)
    - `section13f_eligible`: {CUSIP} (SEC Official List)
    - `EQUITY`: {ticker} via `get_instrument_class`

  Intersectar estas claves requiere ejecutar el resolver que se pretende
  medir. Reproduce la circularidad en forma ontologica (H-TARGET-1 + H-TARGET-2).

**8 cambios obligatorios para v1.2 (dictamen F2.2):**

  1. RADAR_TARGET_REGISTRY con identidad estable independiente del resolver evaluado.
  2. TARGET_UNIVERSE sobre clave comun a CUSIP/13F.
  3. RESOLVED_UNIVERSE con enums v1.3: CANONICAL_FIGI / CANONICAL_EQUIVALENCE.
  4. PAIRED_UNIVERSE: `canonical_security` comun, sin "or observed_security_key".
  5. PAIRED_WEIGHTED_SHARE_COVERAGE: convencion Q4/Q1 inequivoca.
  6. Resolver contradiccion filer continuity (seccion 10.bis).
  7. Corregir referencia "Prompt Maestro v6.35" en policy L225.
  8. Trazabilidad explicita del cambio semantico v1.0 -> v1.1 -> v1.2.

**Reordenacion propuesta del pipeline (a dictamen del auditor):**

  v1.0 (descartado):    baseline -> OpenFIGI -> thresholds

  v1.1 (F2.2, NO GO):   baseline -> policy v1.1 -> OpenFIGI -> thresholds

  v1.2 (propuesta):     baseline
                        -> RADAR_TARGET_REGISTRY (via OpenFIGI como fuente)
                        -> policy v1.2
                        -> medicion sobre TARGET independiente
                        -> thresholds

  Es decir, OpenFIGI pasa de ser mejora posterior de coverage a ser fuente
  primaria para construir el target registry, antes de fijar thresholds.

**Documentos generados:**

  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_CONTRACT_INVENTARIO.md`
  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_F21BIS_FUENTES_IDENTIDAD.md`
  - `docs/auditoria/NIPC_COVERAGE_POLICY_V11_PROPUESTA.md` (NO GO)
  - `docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_DICTAMEN.md`

**Commits:** eda67ef (F2.1), d4a926e (F2.2), 4659ce1 (F2.1-bis).

**Estado Gate-NIPC.2:** SIGUE BLOQUEADO. THRESHOLD_1/2 UNDEFINED.
OpenFIGI NO AUTORIZADO todavia. `NIPC_COVERAGE_POLICY.md` v1.0 intacta (hash 57f2d01f...).

### 11.28. IAE NIPC F2.2-v2 - Propuesta v1.2 redactada (2026-09-19)

**Origen:** dictamen F2.2 NO GO CON CAMBIOS OBLIGATORIOS. Aplica los 8 cambios
obligatorios del dictamen mas el Camino B declarado por F2.1-bis.

**Documento:** docs/auditoria/NIPC_COVERAGE_POLICY_V12_PROPUESTA.md (commit 0518b96).
Estado: BORRADOR. No vigente. No sustituye a la policy v1.0.

**Contenido clave:**
- RADAR_TARGET_REGISTRY = NOT AVAILABLE (Camino B, 2026-09-19).
- TARGET_UNIVERSE = semantico, no materializable hoy sin registry.
- RESOLVED_UNIVERSE con enums v1.3 (CANONICAL_FIGI | CANONICAL_EQUIVALENCE).
- PAIRED_UNIVERSE por canonical_security comun (sin observed_security_key).
- PAIRED_WEIGHTED_SHARE_COVERAGE con convencion Q4/Q1 = max() por security.
- coverage = NOT_MEASURABLE (por ausencia de denominador, no por valores bajos).
- filer continuity reafirmado como control de integridad, no threshold.
- Correccion de referencia Prompt Maestro v6.35 -> ruta canonica sin version.

**Cadena F2:**
- F2.1 PASS (eda67ef).
- F2.1-bis PASS (4659ce1).
- F2.2 NO GO CON CAMBIOS (d4a926e).
- F2.2-v2 borrador redactado (0518b96).

**No toca:** motor NIPC, C2, thresholds, OpenFIGI, baseline evidencia, spec v1.4, policy v1.0.

**Estado:** F2.3 PENDIENTE EXTERNO. F2.4 NO AUTORIZADA.
Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO.

### 11.29. IAE OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE (2026-09-19)

**Origen:** dictamen F2.3 (NO GO v1.2) y entrega F2.3 -> F2.4
(inspeccion N-PORT). Auditor: GO CONDICIONADO con Q1=GO auxiliar,
Q2=GO OpenFIGI como capa de resolucion CUSIP_13F -> FIGI/ticker,
Q3=NO FIJAR thresholds, Q4=GO fix N-PORT, Q5=CERRADO.

**Piezas implementadas (7 commits):**

1. `scripts/update_sec_nport_data.py` (a130bf1): fix pivot
   ISIN/TICKER por HOLDING_ID + filtro por SERIES_ID verificado.
   IDENTIFIER_TICKER: 0 -> 1104. CUSIP+ticker pairs: 0 -> 1063.
   SERIES_IDs verificados contra FUND_REPORTED_INFO.tsv (11:
   QQQ, IVV, IJH, IJR, IWM, ITOT, VOO, VTI, VO, VB, VXF).

2. `src/institutional_accumulation/identity/openfigi_client.py`
   (3fccc93): cliente OpenFIGI /v3/mapping. ID_CUSIP, TICKER,
   ID_ISIN, ID_EXCH_SYMBOL. Batching 5/100 segun API key.
   Reintentos solo 429/500/503. extract_stable_identity()
   prioriza exchCode=US. 8 tests.

3. Micro-probe identidad radar (7501825). Evidencia en
   `docs/auditoria/iae/evidence/nipc_gate0_target_identity/`.
   242 tickers radar -> OpenFIGI. 240/242 con FIGI +
   shareClassFIGI (99.17%). 2 MISS: BRK-B, MOG-A (nomenclatura
   Yahoo vs OpenFIGI BRK/B, MOG/A).

4. `src/institutional_accumulation/identity/radar_target_catalog.py`
   (a530ee3): builder desde result_radar.json. Columnas:
   radar_ticker, figi, share_class_figi, composite_figi,
   ticker_from_openfigi, name, security_type, market_sector,
   exch_code, source, source_date, status. 8 tests.

5. `data/mappings/radar_target_catalog.csv` (ae6129b): 242 filas
   materializadas. 240 OK + 2 MISS. Hash sha256
   11eabce8f8aaed1be6aa3c3557b5e392ad305757230333f84f37285c401b62b7.

6. `src/institutional_accumulation/identity/target_universe.py`
   (f81fc17): resolver CUSIP_13F -> target_membership via
   OpenFIGI + cruce por shareClassFIGI con el catalogo.
   8 tests.

7. Piloto TARGET_UNIVERSE_Q1 (32baf9d). Top 500 CUSIPs por
   SSHPRNAMT del 13F 2026Q1 (SH + PUTCALL NULL). Resultado:
   142/500 target (28.4%), 309 NOT_IN_RADAR, 47 NO_ID, 2 ERROR.
   Top target: NVDA, AAPL, AMZN, MSFT, BAC, T, GOOGL, PFE, AVGO,
   NFLX, INTC, GOOG, KO, CMCSA, CSCO.

**Resolucion del bloqueo arquitectonico:**

El auditor autorizo OpenFIGI como capa de resolucion, no como
autoridad unica del catalogo. El flujo validado:

    radar 242 -> OpenFIGI TICKER/US -> shareClassFIGI
                                     (catalogo independiente)
    CUSIP 13F -> OpenFIGI ID_CUSIP -> shareClassFIGI
                                     (target membership)

El cruce por shareClassFIGI es estable porque FIGI no cambia
por corporate actions (documentado por OpenFIGI). El catalogo
es INPUT del resolver, no output. No hay circularidad.

**Hallazgo empirico en el probe previo (152 CUSIPs):**

- 113/152 hits OpenFIGI (74%). 39 errores (29 no identifier,
  10 invalid format).
- 113/113 con ticker no vacio. 112/113 con shareClassFIGI.
- 48/112 emparejan con radar por shareClassFIGI (42.9%).

**No toca:** motor NIPC, C2, policy v1.0, spec v1.4,
baseline evidencia, thresholds.

**Estado:** F2.3 PENDIENTE EXTERNO tras esta entrega.
F2.4 NO AUTORIZADA. THRESHOLD_1/2 UNDEFINED.
OpenFIGI masivo (24.838 CUSIPs) NO AUTORIZADO.
Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO.

---

### 11.30. IAE F2.3-bis + TOP 2000 - validacion cuantitativa ponderada (2026-09-19)

**Origen:** dictamen F2.3-bis (PASS CONDICIONADO) + autorizacion TOP 2000.
H1 arquitectonico CERRADO. Full OpenFIGI NO AUTORIZADO. F2.4 NO AUTORIZADA.

**Convencion temporal aprobada:**

    RADAR_SNAPSHOT_DATE = 2026-09-19
    13F_OBSERVATION_PERIOD = 2026-03-31
    RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

Radar actual aplicado retrospectivamente al 13F Q1 2026. NO modela historical membership.

**Piloto top 2000 (2000 CUSIPs por SSHPRNAMT):**

RAW (shareClassFIGI + exchCode=US estricto):

    target_true    210 (10.50% count, 32.8079% weight)
    target_false  1567 (78.35% count, 55.6839% weight)
    no_id          220 (11.00% count,  8.8561% weight)
    error            3 ( 0.15% count,  2.6521% weight)

CORREGIDO (2 canales de recuperacion: cusip_ticker_exceptions + ticker match):

    target_true    212 (10.60% count, 33.3428% weight)
    target_false  1567 (78.35% count, 55.6839% weight)
    no_id          218 (10.90% count,  8.3211% weight)
    error            3 ( 0.15% count,  2.6521% weight)

Delta pct_target_weight = +0.5349 pp.

**Comparacion top 500 vs top 2000:**

    Categoria       top500    top2000   ratio/delta
    target count    28.40%    10.50%    2.70x caida
    target weight   46.76%    32.81%    -13.95 pp
    target_false_w  41.52%    55.68%    +14.16 pp

Confirma hipotesis del auditor: top 500 sobrerrepresenta large caps del radar.

**Hallazgos materiales:**

H1. shareClassFIGI cambia tras reorganizacion corporativa (caso XOM):
    - Catalogo radar: BBG023CY9NL0 (EXXONMOBIL HOLDINGS CORP).
    - OpenFIGI CUSIP 30231G102: BBG001S69V32 (EXXON MOBIL CORP).
    Matiza premisa "FIGI inmune a corporate actions" del dictamen F2.3-bis.

H2. HON no esta en el radar 242. NO_ID legitimo. Su reverse split
    post-Q1 (2026-06-29) lo excluyo del radar actual.

H3. 221 NO_ID genuinos concentrados en prefijos no-USA (G/H/N/F/Y/D).

H4. exchCode=US causa ~0.53 pp de peso en falsos NO_ID recuperables.

**Deudas declaradas (4):**

D1. multiple_openfigi_hits=0 hardcoded. Cliente no expone hits crudos.
D2. target_universe.py NO consulta cusip_ticker_exceptions.csv (Q-CUR).
D3. Cliente no valida formato CUSIP antes de enviar (3 ERROR por formato).
D4. BRK-B / MOG-A sin resolver (catalogo, 2 MISS). Pendiente evidencia.

**Evidencia:** docs/auditoria/iae/evidence/nipc_gate0_target_identity_top2000/
(README + 8 ficheros + HASHES.txt).

**Documentos:**
- Dictamen F2.3-bis: docs/auditoria/INSTITUTIONAL_ACCUMULATION_OPENFIGI_DICTAMEN_F23BIS.md
- Autorizacion: docs/auditoria/INSTITUTIONAL_ACCUMULATION_TOP2000_AUTORIZACION.md
- Informe: docs/auditoria/INSTITUTIONAL_ACCUMULATION_OPENFIGI_TOP2000_INFORME.md

**Commits:** 3ef4d2d (dictamen + autorizacion + informe + evidence), 9c67ac6 (fix EOL evidence historicos), 06c49ad (chore .gitattributes).

**EOL hygiene (2026-09-19):** 5 ficheros de evidence historicos
(cusips.txt, result.json, sample.json, pilot_13f_top500.csv,
pilot_cusips_sample.csv) tenian CRLF en working tree pero LF en index
por .gitattributes. HASHES.txt registrados se calcularon contra bytes
CRLF, invalidables en clone limpio. Corregido en 9c67ac6 (mismo patron
que baseline_output.txt).

### 11.31. P66 - Reformulacion contrato 14.3 L3 (GO CONTRACTUAL 2026-09-21)

**Origen:** ciclo P65 v2 (DROP_DUP efectivo). Gate 0 extendido
revelo que la evidencia cruzada para L3 no estaba en
`OTHERMANAGER2` ni `ADDITIONALINFORMATION`. Investigacion posterior
confirmo que la fuente estructurada existe en `OTHERMANAGER` del
Data Set SEC.

**Ruta completa del ciclo:**

1. Informe inicial P66 (hipotesis XML crudo, luego refutada).
2. 10 gates empiricos (Gates 0.1 a 0.10).
3. 12 dictamenes externos (#28 a #39).
4. GO CONTRACTUAL por declaracion condicional cumplida (#39).
5. Traslado a `NIPC_CONTRATOS_SEMANTICOS_v1.md` §14.3.

**Hallazgos clave:**

- `OTHERMANAGER` (no `OTHERMANAGER2`) es la tabla que documenta
  la relacion "other managers reporting for this manager".
- Cubre 100% de NT filings y de COMBINATION base en el universo
  analizado.
- `OTHERMANAGER2` es "included in this report" (otra relacion).
- Universo R4 = NOTICE (NT + NT/A) + COMBINATION (HR + HR/A).
- Mapping FormNum -> CIK via COVERPAGE + SUBMISSION: 1:1 estricta.
- Normalizacion FormNum: prefijo {28, 028} -> 028; sufijo padding
  flexible sin truncar.
- Caso Gate 0.9: 1 caso de >1 filing base heterogeneo (CIK
  0002016827 Q4 2025) -> R3 = N/D conservador.
- Gate 0.8 detecto 137-144 FormNum huerfanos en OTHERMANAGER(R4),
  evidencia de posible ingestion incompleta -> PASO 0.

**Resultado contractual (§14.3 reformulada):**

- §14.3.1 R3 tri-state (TRUE/FALSE/N/D) + PASO 0 completitud.
- §14.3.2 Cadena de amendments determinista.
- §14.3.3 NEW HOLDINGS determinabilidad estricta.
- §14.3.4 R4 candidate_A formalizado + CONFLICT scoped a A.
- §14.3.5 Mapping FormNum -> CIK.
- §14.3.6 Canonicalizacion IAE de FormNum.
- §14.3.7 Clausula de cierre contractual.

**Commits clave:** `3a233b4` (traslado §14.3), `6e698ca` (GO + cierre).

**Estado:** CERRADO. §14.3 trasladada. `reporting_dedup.py` NO
tocado (segun indicacion del auditor). `DROP_DUP` NO activado
(segun indicacion del auditor).

**Siguiente ciclo autorizado:** tests contractuales sobre el nuevo
§14.3, implementacion en `reporting_dedup.py`, probe e2e,
auditoria de salida, activacion `DROP_DUP` (requiere nuevo dictamen).

**Evidencia:** `iae/evidence/p66_gate0{3,4_reissue,5,6,7,8,9,10}/`.
**Propuesta congelada:** `iae/P66_L3_REFORMULACION_PROPUESTA.md` (v7-ter).
**Dictamenes:** `iae/DICTAMENES.md` #28 a #40.

---

### 11.32. IAE post-P66 - Estado HISTORICO (snapshot 2026-09-21; ver ESTADO_DECLARADO.md para el estado vigente)

**Origen:** ciclo P66 (CERRADO) + A.6.0 (CERRADO) + A.6.2-bis
(partido en subfases por dictamen #52).

**Estado por bloque del modulo IAE (SEC 13F -> NIPC):**

    FA-1 + FA-2 (ingestion + identity)              CERRADO / PUSHED
    NIPC (5 modulos + 91 tests)                     IMPLEMENTADO (usa proxy)
    P60/P61/P38/P63/P64/P65                         CERRADOS documentalmente
    P66 (§14.3 reformulada + reporting_dedup)       CERRADO (codigo implementado)

    A.6.0 Gate 0 de los 3 bloqueantes               CERRADO
    A.6.2-bis-B2-PIT (infraestructura temporal)     CERRADO (dictamen #53)
    A.6.2-bis-B1 (TARGET + adaptador P38)           CERRADO (#68)
    A.6.2-bis-B3 (semantica temporal 13F)           IMPLEMENTADO SPEC-SIDE
    Gap spec->codigo §5.1-§5.5                       CERRADO (#61-#67)

**Lo que FALTA por implementar (inventario real, no documental):**

1. **B1** — 5 modulos a crear (cero lineas de codigo):
   `catalog_key.py`, `target_builder.py`, `period_state.py`,
   `catalog_p38_adapter.py`, `catalog_validator.py`.


3. **Integracion** de piezas existentes no invocadas desde el pipeline:
   - `coverage.py::compute_contractual_coverage` (nadie lo invoca)
   - `reporting_dedup.py` (no invocado por `build_effective_reporting_snapshot`)
   - `temporal_validity.py` (no invocado por `security_identity.py`, div. D1)
   - `catalog_pit.py` (nadie lo consume aun; B1 sera el consumidor)

4. **Datos**: `catalog_assignments.csv`, `catalog_membership.csv`,
   snapshots historicos Q4 2025 / Q1 2026 (requieren OpenFIGI masivo).

5. **Thresholds / gates**: `THRESHOLD_1/2` UNDEFINED, Gate-NIPC.2
   BLOQUEADO, Gate-NIPC.3 NO AUTORIZADO, `DROP_DUP` NO AUTORIZADO.

**Diagnostico metodologico (importante para el siguiente asistente):**

El ciclo A.6.2-bis (dictamenes #44-#56) ha entrado en un bucle
divergente: cada propuesta de B1 cierra los bloqueos explicitos del
dictamen anterior y descubre 3-4 nuevos. Total: 14 dictamenes, 4
versiones (v1-v4), cero codigo B1.

**Regla nueva (aplicar a partir de ahora):**

    Cuando el diseño de una subfase haya cerrado >=3 rondas de
    dictamen y el patron sea "cierro N, aparecen N nuevos", PARAR.
    Congelar el diseno en la version vigente e IMPLEMENTAR.
    Las dudas se resuelven escribiendo tests, no documentos.

    Aplicado con exito en B3 (2026-09-21 v6.51): 4 commits,
    sin abrir dictamenes adicionales. Ver dictamenes #57-#60
    (los tres ultimos corresponden a GHISALLO, no a B3).

**Decision pendiente del usuario (2026-09-21):**

    Opcion A: congelar B1 v4 e implementar (3 commits definidos).
    Opcion B: aparcar B1 y avanzar otras partes del IAE
              (B3, integraciones, o probe GHISALLO).

**Inventario de lo que NO esta bloqueado por B1 (para Opcion B):**

    - Integracion de B3 en pipeline (timestamps + provenance).
      NO AUTORIZADA: ciclo propio. Ver dictamenes #57-#60.
    - Integracion de `reporting_dedup` en el pipeline NIPC.
      NO AUTORIZADA: P66 prohibe activar DROP_DUP sin dictamen.
    - Curacion adicional del crosswalk CUSIP->ticker (residual:
      ONB, SPCX, PTGX sin match en 13F Q1 2026).

**Ficheros clave del ciclo A.6.2-bis:**

    iae/A62BIS_B1_SUBFASE.md          v4 (diseno B1, sin codigo)
    iae/A62BIS_B2_PIT_SUBFASE.md      cerrado
    iae/A62BIS_PROPUESTA.md           v9 (base historica)
    iae/DICTAMENES.md                 #44-#56
    iae/FASE_A6_PLAN.md               plan maestro A.6
    src/institutional_accumulation/catalog_pit.py    UNICO modulo implementado

---


## SECCION 15 (historica) - Snapshot del 2026-09-21 v5

| Metrica | Valor |
|---|---|
| Cobertura | 313/313 (100%) |
| FAILED | 0 |
| Fuentes europeas | 51 (Euronext 13 + Xetra 19 + BME 19) |
| Fuente commodities | OilPriceAPI (BZ=F, CL=F, GC=F, HG=F, NG=F) |
| Fuente term structure | CBOE (^VIX3M) |
| Tests locales | 1301 passed + 2 skipped + 3 failed preexistentes (freshness) |
| Tests CI | ~610 collected con skips (parquet gitignored) |
| Validation Gate | 10/10 |
| pyflakes | 0 warnings |
| compileall | OK |
| Produccion GH Actions | OK (cron `0 4 * * *` verificado 2026-09-17) |
| Arquitectura | Modular: 19 src/report/ + 16 src/pipeline/ + 10 src/temporal_contracts/ + 7 src/institutional_accumulation/sec_13f/ + 5 sec_13f/identity (temporal_filter, cusip_resolver, relationships, amendments, sec13f_list, security_identity) + 2 aggregation/ (delta_shares, nipc) + 3 identity/ nuevos (openfigi_client, radar_target_catalog, target_universe) + 5 indicators/mte/ + 4 indicators/darkpool/ |
| Contratos temporales | 10 (FU-021-5 = 9, FU-021-3C-bis = +1 SPOT_COMMODITY) |
| Modulo IAE (SEC 13F) | FA-1+FA-2 cerrados. NIPC implementado (usa proxy observacional). P60/P61/P38/P63/P64/P65 CERRADOS. P66 CERRADO (§14.3 + reporting_dedup). A.6.0 CERRADO. GHISALLO CERRADO. A.6.2-bis: B2-PIT CERRADO + **B1 CERRADO (#68, 4 commits + 100 tests)** + B3 IMPLEMENTADO SPEC-SIDE. **Gap spec->codigo §5.1-§5.5 CERRADO** (#61-#67): security_type.py (Nivel A+B), operational_universe.py (handoff validado), contrato ticker_mapped, PIT obligatorio. B1 modulos: catalog_key.py, target_builder.py, period_state.py, catalog_validator.py, catalog_p38_adapter.py. Curacion crosswalk ampliada (3->22 filas Q1 2026). Dictamenes hasta #68. Gate-NIPC.2 BLOQUEADO por THRESHOLD. A.6.3 BLOQUEADO (requiere dictamen especifico). Deuda real: integrar B3/§5.5 al pipeline, obtener snapshots historicos. |
| RADAR_TARGET_CATALOG | MATERIALIZADO 2026-09-19 (242 filas, 240 OK, 2 MISS: BRK-B, MOG-A). Hash 11eabce8... Construido desde OpenFIGI TICKER/US -> shareClassFIGI, independiente del crosswalk interno. TARGET_UNIVERSE resolver operativo (8 tests). |
| Coverage baseline NIPC | Fase A cerrada. TOP 2000 (Q1 2026, CURRENT_RETROSPECTIVE): target_true=210 (10.50% count, 32.8079% weight); corregido 212/33.3428%. target_false=1567 (78.35%, 55.68%). no_id=220 (11.0%, 8.86%). error=3 (0.15%, 2.65%). Delta +0.5349 pp por 2 canales adicionales. THRESHOLD_1/2 UNDEFINED |
| .git size | ~13 MB |
| HEAD | fd082a0 (218 commits locales ahead de origin/main) |

### 15.1. Hitos del ciclo FU-021-3C-bis (2026-09-16)

Objetivo: resolver FU-021-3C (FUTURE_SETTLEMENT BLOCKED). Yahoo agotado, CME/ICE bloqueados por IP. Solucion: OilPriceAPI como provider dedicado.

Commits pusheados (extracto, orden cronologico):

303db48 — feat(temporal_contracts): SPOT_COMMODITY.

78f18a8 — feat(temporal_contracts): FUTURE_SETTLEMENT activo.

fabff51 — feat(providers): OilPriceAPI provider para commodities.

e506b3c — feat(commodities): merge de commodities en df_market.

8a80e3c — test(commodities): merge_commodities_into_market.

bfe3041 — docs(report): nota metodologica commodities.

6d106be — fix(utils): close_cols sin inicializar en write_artifact_with_manifest.

b088ff6 — feat(workflows): update commodities en daily_run.

e49beda — test(temporal_contracts): REF derivada del parquet en tests fragiles.

ab3c6f1 — Daily hist/state (CI, 2026-09-16).

Fixes cerrados:

FUTURE_SETTLEMENT: BLOCKED -> activo (BZ=F, CL=F, settlement_semantics=close_proxy).

SPOT_COMMODITY: nuevo (GC=F, HG=F, NG=F, settlement_semantics=spot_reference).

Bug latente writer FU-002 (close_cols UnboundLocalError con df de 1 fila).

3 tests fragiles con REF hardcodeado (migrados a REF derivada del parquet).

BOM en CSVs OilPriceAPI (resuelto en pipeline externo).

Legacy ICE_BRN_curve_*.csv borrado.

Tests: 469 -> 498 (+29).

### 15.2. Pendientes reales

A2.3 colateral (CERRADO 2026-09-17, `02ca9f0`): docstring de `get_market` extendido con nota A2.3. Verificado: cero consumidores afectados; los 5 callers operan sobre universo ya filtrado a equity o sobre tickers con sufijo europeo.

K-FU-021-3C-bis-02 (OBSOLETO 2026-09-17): FSM correcta (lag=1 <= max_lag=1). No es bug.

K-FU-021-3D-03 (MEDIA) -> RESUELTO 2026-09-17 (`eec6b9c`). K-FU-021-3D-04 (MEDIA) -> WONT FIX / MONITORED 2026-09-17 (sin fuente object en produccion).

K-FU-021-3C-bis-04 (OBSOLETO 2026-09-17): REF solo en mocks por diseno (K-04). Tests con df_real usan `_ref_from_df`.

K-FU-021-3C-bis-09 (OBSOLETO / RESUELTO DE HECHO 2026-09-17): refutado por inspeccion directa.

K7 + K-FU-021-5-01/02/03 (OBSOLETO 2026-09-17): docs ya reflejan Q-P.3 (698eac3). Verificado por grep: "df.attrs es espejo auxiliar, nunca autoridad" en los 3 docs del ciclo FU-021-5.

DT4 (BAJA): reorg de `validation/` y `scripts/`.

K-DT3-YF-DIRECTO (WONT FIX / EXCEPCION ACEPTADA) / K-DT3-SIDE-EFFECT (WONT FIX / MONITORED) / K-DT3-RUNTIMEWARN (RESUELTO 2026-09-17, `2656a5e`): bloque DT3 cerrado.

K-FS-CI-PARITY-01 (CERRADO 2026-09-17, NO BUG): hipotesis de paridad CI-local refutada. `FUTURE_SETTLEMENT=INSUFFICIENT` es estado contractual valido (cobertura parcial del proveedor).

Sin residuales BAJA activos.

Detalle completo en la seccion 13 y en FOLLOWUPS.md.

### 15.3. Cierre del ciclo

FU-021-3C-bis cerrado en local y CI. Gate 10/10 en produccion. Los 5 commodities integrados via OilPriceAPI. Presupuesto API dentro de margen (3 req/dia, 90/mes).

El sistema pasa de 9 a 10 contratos temporales. FUTURE_SETTLEMENT deja de ser el unico contrato BLOCKED.

### 15.4. Ciclo FU-021-3D (2026-09-17)

Objetivo: resolver E5 (Yahoo dejo de servir historico de ^VIX3M). Solucion: CBOE como provider dedicado.

Commits (6): 
- ca223c0 - feat(providers): CboeIndexProvider.
- 3c25262 - feat(scripts): update_cboe.py.
- 904947f - feat(merge): merge_cboe_into_market.
- eb071db - test(merge): 11 tests.
- a608605 - feat(data_loader): integracion.
- a560708 - feat(workflows): step Update CBOE.

R6 aprobada: term structure de volatilidad via CBOE exclusivamente. Alcance ^VIX3M. ^VIX9D fuera.

Resultado: VOLATILITY_INDEX pasa de INSUFFICIENT a OK/STALE con coverage=1.0. Tests nuevos: 28 (17 provider + 11 merge).


### 15.5. Ciclo DT2 - Refactor MTE (2026-09-17)

Objetivo: dividir indicators/mte.py (1963 LOC, 21 funciones) en paquete modular sin cambiar comportamiento.

Fases (7 commits):
- 6433bc1 - Fase 0: golden reference + 7 fixtures.
- 9eeaaa8 - Fase 1a: test_mte_engine.py (3 tests).
- 9473897 - Fase 1b: test_mte_state.py (11 tests).
- 3407909 - Fase 1c: test_mte_scoring.py (11 tests).
- c717e56 - Fase 2: modulo -> paquete (alias sys.modules).
- 2b7363a - Fase 3: state.py + paquete real.
- 43d52a3 - Fase 4: scoring.py.
- f427e80 - Fase 5: decision.py.
- 571f165 - Fase 6: engine.py, mte_legacy.py eliminado.

Arquitectura final: indicators/mte/ con __init__.py + engine.py + state.py + scoring.py + decision.py.

Golden reference: tests/fixtures/mte_golden_2026-09-16.json + 7 fixtures con hashes SHA-256 LF-normalized. Tolerancias beta: exact para scenario, 1e-9 para srs/ipi/ips, 1e-3 para cls/shs/msi/confidence (derivada de variacion FRED).

U+FFFD: 28 -> 26. Los 2 desaparecidos pertenecian a codigo eliminado (docstring del modulo legacy + header de seccion huerfano). No fueron corregidos. K-ID de encoding sigue separado.

Verificacion: 25 tests MTE + 551 suite global verde. Gate 10/10 en E2E. Golden sin cambios semanticos. Import limpio desde proceso independiente.

### 15.6. Ciclo K-DATA-LOADER-01 (2026-09-17) - RESUELTO

Causa raiz: `src/data_loader.py::download_market_data` retornaba `_df` crudo en el cache-hit path, saltando post-procesado (clean_oil_prices, _filter_non_eod_equity, merges commodities/CBOE), `write_artifact_with_manifest` y bloque `[FU-021-5]` que setea `temporal_meta`.

Consecuencia: en runs locales con cache warm, `mte_state.json` con `effective_date=None`, `coverage=None`. CI sin impacto (cache cold).

Fix (`9d77099`): extraccion de `_postprocess_market_data(data, reference_date, run_id, *, write_manifest)`. Cache-hit invoca con `write_manifest=False` (no reescribe parquet, si actualiza `temporal_meta`). Cache-miss con `write_manifest=True`. Idempotencia de `clean_oil_prices`, `_filter_non_eod_equity`, `_trim_market_data_to_equity_eod` y `merge_*` verificada antes de escribir patch.

Tests nuevos: 4 (`tests/test_data_loader_cache_postprocess.py`).

Verificacion: E2E local (cache-hit ejercitado), `mte_state.json` con `effective_date=2026-09-16`, `coverage=0.998`. CI exit 0.

### 15.7. Ciclo K-DT2-GOLDEN-EOL (2026-09-17) - RESUELTO

Causa raiz: `tests/fixtures/*.csv` con doble CRLF (`\r\r\n`) en working tree. Golden se calculo via `_sha256_normalized` que colapsa `\r\n -> \n`, lo cual sobre `\r\r\n` daba el hash del blob CRLF, no el hash LF. Local: `\r\r\n` -> `\r\n` -> hash OK. CI: `\r\n` -> `\n` -> hash mismatch.

Diagnostico: `len(working) - len(blob) = 2606` = nº exacto de secuencias `\r\r\n`.

Fix (`ee34305`): renormalizar fixtures a LF puro + regenerar `sha256` y `size` del golden + `git add --renormalize`.

Verificacion: `test_mte_engine.py` 3/3 verde en local y CI.

### 15.8. Ciclo DT1 - Refactor regimes/sector_regime.py (2026-09-17) - CERRADO

Diagnostico revisado tras Gate 0: 827 lineas brutas = 156 de codigo real, 663 vacias, 8 comentarios. La "deuda ALTA por LOC" era un artefacto de medicion.

Bug latente confirmado: `components` se construia FUERA del bucle principal, copiando los valores del ULTIMO sector (XLC) a los 11. Cero consumidores en produccion.

Fix (`a1406f4`): eliminar bloque `components` (extirpar, no reparar). Colapso de whitespace (827 -> 272). Golden de caracterizacion + 5 tests. 11 tests edge cases.

Verificacion: 571 passed + 2 skipped (en el momento del commit). CI exit 0.

### 15.9. Ciclo DT3 - Refactor indicators/darkpool.py (2026-09-17) - CERRADO

Fases (5 commits: `0ccc603`, `06f3b27`, `5aca394`, `5ec1e21`):

- Fase 0: golden de caracterizacion + 12 tests.
- Fase 1: fix normativo `fecha=week_start` (era `datetime.now()`) + bare except sustituido por `(FileNotFoundError, pd.errors.EmptyDataError)`.
- Fase 2: extraer `darkpool_scoring.py` (4 funciones puras).
- Fase 3: extraer `darkpool_io.py` (2 funciones).
- Fase 4: extraer `darkpool_history.py` (`_backfill_history`).
- Fase 5: 18 tests edge cases. Fix robustez `_get_all_tickers` (isinstance str antes de startswith).

Arquitectura final: `indicators/darkpool.py` (147 LOC, orquestador + re-exports) + `darkpool_scoring.py` + `darkpool_io.py` + `darkpool_history.py`.

Verificacion: 601 passed + 2 skipped. CI exit 0.

### 15.10. Ciclo K-CI-CRON-01 (2026-09-17) - CERRADO pasivamente

Observacion: runs manuales a 00:40-01:57 UTC reportaban `EQUITY_EOD=INSUFFICIENT function_lag=2d`. Diagnostico preliminar: FSM correcta, proveedores no habian propagado cierre del dia anterior.

Verificacion con cron real `0 4 * * *` (`35204004152`): `EQUITY_EOD=OK function_lag=1d`. La FSM funciona. Falsa alarma.

### 15.11. Verificacion continua

Todo commit pusheado ha sido verificado con `workflow_dispatch daily_run.yml` exit 0. Cron `0 4 * * *` verificado sano. Gate 10/10 en todos los runs.

### 15.12. Ciclo K-FU-021-3D-03 (2026-09-17) - RESUELTO

Causa raiz: `commodities_spot.parquet` trae Open/High/Low/Volume como object (None emitido por FuturesProvider). El merge asigna object->float64 en `df_market` -> FutureWarning en pandas 2.3.3, fallo potencial en Pandas 3.x.

Fix (`eec6b9c`): cast defensivo a float64 en `src/commodities_merge.py::merge_commodities_into_market` antes de asignar (`pd.to_numeric(errors='coerce')`). Dato no interpretable -> NaN (nunca imputar).

Tests nuevos: 1 (`TestMergeObjectDtype::test_object_dtype_none_no_futurewarning`). Verificacion: 11/11 provider + 604 global + CI exit 0.

### 15.13. Ciclo K-FUTURES-DTYPE-01 (2026-09-17) - RESUELTO

Causa raiz (Capa 2): `data/providers/futures.py::_rows_to_wide` infiere object para columnas con todos los valores None (Open/High/Low/Volume de spot). El artefacto se persiste asi.

Fix (`128de85`): normalizar `FIELDS` a numerico tras `sort_index()`, iterando por columna porque `wide` tiene MultiIndex (field, ticker). `pd.to_numeric(errors='coerce')`.

Intento previo (revertido): `wide[FIELDS].apply(...)` fallo con `ValueError: Columns must be same length as key` porque `wide[FIELDS]` con lista plana no selecciona 5 columnas del MultiIndex. Leccion: verificar semantica exacta del objeto Pandas (MultiIndex, shape, dtype) antes de autorizar el patch.

Tests nuevos: 2 (`test_dtypes_float64_con_nones`, `test_roundtrip_y_merge_sin_futurewarning`). Verificacion: 19/19 provider + 604 global + CI exit 0.

Parquet historico `data/commodities_spot.parquet` NO reescrito (conservacion historica).

### 15.14. Ciclo K-FU-021-3C-bis-09 (2026-09-17) - OBSOLETO

El prompt v6.20 listaba este K-ID como MEDIA: "6 workflows pushean sin `git pull --rebase`". La inspeccion directa (Gate 0) refuto la afirmacion: los 6 workflows ya tienen `git pull --rebase origin main` antes de `git push` (lineas 41, 41, 43, 45, 45, 45 respectivamente).

Dictamen del auditor: cerrar como OBSOLETO / RESUELTO DE HECHO. No patch. No abrir K-ID sustituto para variaciones cosmeticas (identidad `github-actions` vs `github-actions[bot]`, `--staged` vs `--cached`).

K-ID nuevo detectado en este mismo run: `K-FS-CI-PARITY-01` (MEDIA) — ver Seccion 12/13.

### 15.15. Revision sistematica MEDIA (2026-09-17) - 4 K-IDs cerrados por obsolescencia

Motivacion: 2 K-IDs consecutivos (K-FU-021-3C-bis-09, K-FU-021-3C-bis-04) resultaron obsoletos al ser verificados por Gate 0. Se decidio revisar sistematicamente los 4 MEDIA restantes antes de seguir invirtiendo ciclos.

Resultado: 4 de 4 MEDIA restantes eran obsoletos / mal diagnosticados.

- K-FU-021-3C-bis-04: REF sobrevive solo en mocks (K-04, `18601ee`).
- K-FU-021-3C-bis-09: 6 workflows ya tienen rebase (K-01, `03fcb3e`). Cerrado antes.
- K-FU-021-3C-bis-02: `SPOT_COMMODITY` STALE es la respuesta correcta de la FSM (lag=1 <= max_lag=1). Identico a FUTURE_SETTLEMENT.
- K7 + K-FU-021-5-01/02/03: docs ya reflejan Q-P.3 (698eac3). "df.attrs es espejo auxiliar, nunca autoridad".

MEDIA tecnico pendiente tras la revision: K-FS-CI-PARITY-01. Cerrado posteriormente como NO BUG (ver 15.16).

Leccion operativa: la lista MEDIA requiere re-verificacion periodica. Un K-ID cerrado por un ciclo posterior puede quedar como fantasma en el prompt si no se retira explicitamente. Antes de invertir un ciclo MEDIA, Gate 0 con evidencia directa.

### 15.16. Ciclos K-FS-CI-PARITY-01 + K-FUTURES-REFRESH-01 (2026-09-17)

**K-FS-CI-PARITY-01 (NO BUG):** `FUTURE_SETTLEMENT=INSUFFICIENT` en CI vs `STALE` en local. Gate 0 refuto la hipotesis de paridad rota: el `market_data.parquet` local era anterior al ciclo FU-021-3C-bis. Causa real: `commodities_futures.parquet` con fila parcial (BZ=F con Close=NaN el 09-16, CL=F con NaN el 09-15). La FSM funciono correctamente: cobertura 0.5 < min_coverage 1.0. Test de regresion (`65ba34f`): `test_cobertura_parcial_devuelve_insufficient` + `test_cobertura_completa_no_es_insufficient`.

**K-FUTURES-REFRESH-01 (RESUELTO, `61e27ee`):** Gate 0 contra OilPriceAPI real confirmo que un re-fetch recupera el dato faltante (Caso A). Solucion: `scripts/update_futures.py::_inspect_parquet(path, expected, required)` reemplaza `_already_up_to_date`. Skip solo si `last_date == expected` AND `cobertura == 1.0`. Fetch selectivo: `only_futures` filtra tickers de futuros; `skip_spot=True` evita `_fetch_spot` cuando cobertura spot OK. 9 tests nuevos (`tests/test_update_futures_skip.py`).

**Hallazgo de seguimiento — mutabilidad historica del proveedor:** OilPriceAPI **revisa** valores historicos. `CL=F 2026-09-16`: `100.40` (parquet commiteado) -> `97.21` (hoy). Es comportamiento legitimo de fuente autoritativa, no bug. Relevante para auditorias futuras de reproducibilidad de `FUTURE_SETTLEMENT` y manifests FU-002. Sin K-ID por ahora; registrar.

### 15.17. Cierre bloque DT3 (2026-09-17)

Cierre de los 3 K-IDs residuales del ciclo DT3:

- **K-DT3-RUNTIMEWARN -> RESUELTO (`2656a5e`):** `robust_zscore` con early return `pd.Series([], dtype=float)` si serie vacia. Warning numpy eliminado. Test reforzado con `-W error::RuntimeWarning`.
- **K-DT3-SIDE-EFFECT -> WONT FIX / MONITORED:** Gate 0 demostro doble dedup por `week` (L98 + L105 en `darkpool.py`). FU-002 NO aplica a `outputs/history/*.csv` (convencion: parquet de pipeline). Sin defecto funcional. No migrar a `append_dedup`.
- **K-DT3-YF-DIRECTO -> WONT FIX / EXCEPCION ACEPTADA:** `_backfill_history` usa `yf.download(start/end)` directo. Router no soporta rango arbitrario. Excepcion acotada, documentada. Reabrir solo si aparece 2do consumidor, problema de coste, o cambia contrato del router.

Suite final: 615 passed + 2 skipped, 0 warnings.

### 15.18. Ciclo Gate 0 sistematico BAJA (2026-09-17)

Antecedente: 4 de 4 MEDIA resultaron fantasma el mismo dia. Sospecha de patron similar en BAJA.

Resultado: 9 de 9 K-IDs BAJA/MEDIA revisados eran obsoletos/subrogados. Cero fantasmas residuales.

K-IDs cerrados (3 commits pusheados):

- ef64c71: K-FU-021-3C-bis-05 (OBSOLETO, transfer doc nunca commiteado) + K-FU-021-3C-bis-07 (WONT FIX razonado, ciclo ya documentado).
- 37cc872: E1 (OBSOLETO, mutabilidad Yahoo), E2 (OBSOLETO, subrogado por H-3C-7), E3 (OBSOLETO, cutoff codificado), E4 (OBSOLETO, ciclo .attrs vivo), FU-016 (OBSOLETO, P3 documental + cron 04:00 evita condicion).
- 896097b: FU-003 (OBSOLETO, 0 matches en reporte vivo, FU-003b cubre n=0).

Verificacion: 615 passed + 2 skipped, 0 warnings, Gate 10/10. Working tree limpio.

Leccion consolidada: la lista del prompt acumulaba K-IDs fantasma sin revision periodica. Gate 0 con evidencia directa antes de invertir el ciclo. Coste ~3h. Ahorro ~10-12h. ROI ~5x.

### 15.19. Ciclo H1 - Mutabilidad del dataset historico (2026-09-17) - CERRADO WONT FIX

Origen: hallazgo detectado durante K-FS-CI-PARITY-01. Inicialmente catalogado como mutabilidad de proveedor (OilPriceAPI reviso CL=F 09-16). Gate 0 posterior confirmo 3 mecanismos independientes:

- H1-a: proveedor externo revisa valores (OilPriceAPI, Yahoo).
- H1-b: pipeline regenera historicos por cambio de logica (commit `35af4ba`, 1485 valores reescritos).
- H1-c: `append_dedup(keep='last')` sustituye observaciones sin registrar la revision.

Commits (3):
- `1d5c6c2` - docs(auditoria): informe H1 mutabilidad historica (295 lineas).
- `a42d21d` - docs(auditoria): dictamen auditor + cierre H1.
- (incluye) matiz de formulacion del auditor sobre perdida de informacion.

Dictamen del auditor: **CERRADO - WONT FIX / POLITICA ACEPTADA**. Opciones A (snapshot) y B (congelacion) NO-GO hasta requisito adicional. C (mutabilidad documentada) + D (deteccion cross-run futura) como direccion arquitectonica.

No toca codigo de produccion. Condiciones de reapertura documentadas: requisito regulatorio, reconstruccion exacta de inputs, auditoria externa de dataset completo, o necesidad de distinguir revision de proveedor vs regeneracion pipeline.

### 15.20. Ciclo K-HUERFANO - Deteccion de lote parcial (2026-09-17)

Origen: durante verificacion de fechas por fuente (Gate 0), se detecto que 1 ticker (`KHC`) del run manual del 17/09 11:15 ET no tenia Close valido en `expected_session=2026-09-16`, mientras Yahoo fresco si lo devolvia (Volume=13.87M). El resto de los 561 tickers si tenian la observacion.

Causa raiz: `download_market_data` acepta un lote como OK si `data_batch is not None and not data_batch.empty`. No verifica cobertura por ticker individual. Un lote con 4/5 tickers completos pasa sin retry, dejando 1 ticker sin la sesion esperada.

Fix (`3ccae42`, auditado por dictamen B+C):
- Helper `_check_khuerfano(data_batch, batch, expected_session)`.
- Resolucion de `_expected_session` via `last_expected_market_date(reference_date)`.
- Print `[K-HUERFANO]` tras `all_data.append(data_batch)`.
- Sin retry (NO-GO hasta acumular evidencia de recurrencia).
- +6 tests (`tests/test_khuerfano.py`).

Documentacion: §12 incluye formulacion exacta del auditor. No crear K-ID nuevo. Reabrir ciclo A/D si reaparece: mismo ticker repetidamente, multiples tickers, produccion 04:00 UTC, o cobertura materialmente inferior.

### 15.21. Ciclo de auditoria del reporte 2026-09-17 (2026-09-18)

**Origen:** peticion explicita del usuario de auditar el reporte del run del 2026-09-17
buscando errores, inconsistencias y omisiones. Se diseno un sistema por fases
(Gate 0 -> Diagnostico -> Dictamen -> Decision -> Implementacion -> Verificacion ->
Cierre) aplicado a 7 hallazgos iniciales, ampliado despues a 4 mas.

**Total: 9 fixes aplicados + 3 NO BUG + 2 WONT FIX + 1 incidente revertido.**

#### Ciclos cerrados

| ID | Categoria | Fix | Commit |
|---|---|---|---|
| O2 | BUG ALTA (dedup) | `append_dedup` subset incluye ticker | `9ff8daf` |
| K-RS-INTERNAL-COMMIT-01 | BUG MEDIA (workflow) | `git add outputs/history` | `5fbe16e` |
| I2 | BUG ALTA (clasificador) | `build_ticker_df` en 3 callers | `a6f4666` |
| I3 | DISENO/presentacion | `effectiveDate` en vez de `as_of_date` | `1d7bacd` |
| I1 + O1 | DISENO/notas | Notas aclaratorias + fecha efectiva | `b74790c` |
| K-INDEX-RANGE-01 | BUG ALTA (mismo patron I2) | `build_ticker_df` en 3 callers | `562dc41` |
| PCR-Indices | BUG MEDIA (return dict) | `index_pcr` en el dict de retorno | `849f981` |
| K-RENDER-LEADER-TABLE-01 | BUG BAJA (cosmetico) | separator Markdown a 11 cols | `44cd544` |

#### NO BUG (verificados con evidencia directa)

- **I4** — Evidence Matrix con lag=1 en `data_quality.csv` del run 16/09. No
  reproducible en el run del 17/09 (`last_date=2026-09-17, age=0.0`). El lag del
  16/09 era correcto: `evidence_matrix.py:170` hereda la fecha de
  `sector_breadth_df` y si ese dia tiene lag, evidence_matrix hereda el lag.
- **O3** — FEZ con `primary_flow=0.00`. `shares_outstanding` invariante en
  `outputs/history/etf_primary_flow.csv`. Comportamiento correcto.
- **`sector_regime.py:114`** — inicialmente clasificado como NO BUG con test
  unico (XLK). Reabierto en Gate 0 v23: 7/11 ETFs afectados. Reclasificado como
  BUG y absorbido por K-INDEX-RANGE-01.

#### WONT FIX / MONITORED

- `K-RUN-OUT-OF-WINDOW-01` — incidente 2026-09-18, revertido. Ver §12.
- `VIX3M/VIX nan 2026-09-14` — data artifact. Ver §12.
- `O1 SPDR Ultima fecha N/D` — fallback funcional. Ver §12.

#### Hitos del ciclo

1. **SLPM desbloqueado.** K-INDEX-RANGE-01 no solo arreglo las fases Wyckoff:
   desbloqueo el SLPM entero. Sin XLC en ACCUMULATION, el SLPM nunca encontraba
   lideres. Ahora: LIS=+0.20, Eff Breadth=0.53, 20 tickers en *Acciones Seleccionadas*,
   tabla *Representatividad del lider* poblada.
2. **Doble patron A/B confirmado.** El bug "wyckoff_structure_core degenera a
   RANGE con NaN internos" se manifesto en 3 sitios (I2 en sector_wyckoff_distribution
   + index_leaders, K-INDEX-RANGE-01 en index_phase + sector_regime). Leccion: un
   fix destapa el siguiente, aplicado 2x.
3. **Confirmacion empirica sistematica.** Gate 0 v22+v23 midio A vs B en 8 indices
   + 11 ETFs. Sin esa medicion, `sector_regime.py:114` se habria cerrado como NO BUG
   por el caso unico de XLK (que casualmente no reproduce).
4. **Incidente K-RUN-OUT-OF-WINDOW-01 contenido.** Run manual a 00:00 UTC produjo
   262/313 tickers sin Close. El pipeline gestiono correctamente (FU-020 retrocedio
   `effective_date`). El workflow no tenia guarda y commiteo outputs contaminados.
   Revertido con `fc8faed`. Sin impacto en produccion (cron a 04:00 UTC).
5. **Rebase en push revela acoplamiento con CI.** `44cd544` se convirtio en
   `29f393a` tras rebase contra `37c1029 Daily hist/state`. Comportamiento normal.

#### Lecciones metodologicas consolidadas

- **Gate 0 con evidencia directa antes de invertir un ciclo.** Nuevo caso 2026-09-18:
  `sector_regime.py:114` NO BUG por XLK unico se reabrio con 11 ETFs. La muestra
  unica enmascara el bug.
- **Un solo caller afectado NO implica NO BUG.** Aplicar la regla "un fix destapa el
  siguiente" en cada fix, no solo cuando el sintoma se repite a simple vista.
- **El pipeline puede tener exito parcial sin fallar el Gate.** Caso K-RUN-OUT-OF-WINDOW-01
  (resuelto via guard_coverage, ver 15.22): Gate 10/10 con `coverage_pct=0.16`. El Gate
  valida NaN en columnas especificas, no cobertura global. La guarda de coverage cierra
  el hueco antes del commit, no dentro del Gate.
- **Revert rapido de contaminacion.** El revert quirurgico (`git revert <sha>`) es
  preferible a `git reset` cuando ya esta pusheado. Conserva trazabilidad.

### 15.22. Ciclo H.2 - guard_coverage (2026-09-18)

Origen: run manual 35371884701 (17:02 UTC, USA en sesion) commiteo 10e0608 con
stock_prices.parquet coverage_pct_last=0.1342 y last_date=2026-09-18 > expected=2026-09-17.

Causa raiz: workflow_dispatch commitea a origin/main (no solo cron). stock_prices.parquet
persiste fila intradia porque el merge europeo anade velas EOD legitimas de Euronext/Xetra
mientras USA/UK/BME siguen NaN.

Commits (3):

- dd1b7b1: Revert "Daily hist/state" (revert quirurgico del commit 10e0608).
- 7da0d93: fix(tests): pyflakes limpio en test_pcr_indices.py (MagicMock + pandas).
- f3114b4: feat(workflow): scripts/guard_coverage.py + tests + step en daily_run.yml.

Guard (scripts/guard_coverage.py): 3 condiciones independientes por manifest
(coverage_pct_last >= threshold, status != INVALID, last_date <= expected_session).
Universo explicito GUARDED_MANIFESTS = (stock_prices, market_data). Threshold default 0.95.
Commodities fuera (coverage 0.5 legitimo, ver K-FS-CI-PARITY-01).

Tests: 10 nuevos en tests/test_guard_coverage.py (incluye bordes 0.9499/0.9500,
STALE legitimo, JSON invalido, universo explicito).

Verificacion en produccion real: run 35376243662 (workflow_dispatch, 17:45 UTC) bloqueo
commit con dos [FAIL] en stock_prices (coverage 0.1342 y last_date futuro) y [OK] en
market_data. origin/main intacto en f3114b4. Workflow en rojo esperado; se auto-limpia
con el cron 04:00 UTC (sesion cerrada, coverage 1.0).

Leccion: la primera reproduccion de K-RUN-OUT-OF-WINDOW-01 (run 00:00 UTC) parecia
circunscrita a esa ventana horaria. El segundo caso (17:02 UTC) demostro que el vector
es cualquier run fuera de cron, no solo madrugada. Gate 0 con evidencia directa antes
de cerrar como WONT FIX.

### 15.23. Ciclo IAE NIPC (2026-09-19) - implementacion + dictamenes

**Modulos implementados (5):**
  - `identity/sec13f_list.py` + 24 tests (S1).
  - `identity/security_identity.py` + 30 tests (S2).
  - `aggregation/delta_shares.py` + 18 tests (S4).
  - `aggregation/nipc.py` + 19 tests (S5).
  - `data/mappings/cusip_equivalence.csv` (S3) + `aggregation/__init__.py`.

**Specs:**
  - Spec NIPC v1.4 activa (C1+C4-revisadas + filer continuity).
  - Coverage policy `NIPC_COVERAGE_POLICY.md` (THRESHOLD_1/2 UNDEFINED).
  - Versiones v1.0-v1.3 preservadas.

**Dictamenes del auditor (4):**
  - v1.1 (canonical_security + temporalidad).
  - probe end-to-end (GO mini-probe filer continuity).
  - mini-probe TOP 20 (GO condicionado + hallazgo cuantitativo).
  - TOP 50 (filer continuity CERRADO) + addendum cuantitativo PASS.

### 15.24. Ciclo IAE filer continuity (2026-09-19) - CERRADO

**Mini-probes ejecutados:**
  - TOP 20: 3/24 FILER_DISCONTINUITY (12.5%), todas Vanguard.
  - TOP 50: 4/54 FILER_DISCONTINUITY (7.4%), todas Vanguard.
  - 0 nuevos patrones fuera de Vanguard.

**Evidencia NT -> HR:**
  - Padre Vanguard NT Q1 con 10 OTHERMANAGER targets.
  - Fiduciary Trust NT Q4 -> target padre.

**CERRADO como caracterizacion del periodo.** NO es threshold.

### 15.25. Correccion cuantitativa Vanguard (addendum 2026-09-19)

**Bug detectado por auditor:** informe mini-probe sumaba SSHPRNAMT raw (70.236B) en vez de canonical (35.329B). Ratio 1.988. El HR original (001306) fue SUPERSEDED por RESTATEMENT posterior (001311).

**Cifras corregidas:** filiales Q1 = 56.360B (antes 91.267B NO UTILIZABLE).

**Regla congelada:** agregados SSHPRNAMT desde canonical_snapshot.

---

### 15.26. Ciclo IAE Q-CUR - Curacion CUSIP Q1 2026 (2026-09-19)

Origen: dictamen FA-2.2 y Gate 0 sobre crosswalk CUSIP->ticker vacio.

Resultado: 3 filas COM pobladas en `data/mappings/cusip_ticker_exceptions.csv` (DD, HON, XOM). Excluidos CALL/PUT (derivados) y HONA/FDXF (entidades post-Q1, posteriores a PERIODOFREPORT=2026-03-31).

Dictamen: Q-CUR-1 (excluir CALL/PUT) + Q-CUR-2 (excluir HONA/FDXF). Alcance minimo aprobado por el auditor.

Informe: docs/auditoria/INSTITUTIONAL_ACCUMULATION_CUSIP_CURATION_INFORME.md

Commits: e4a6576, e8d7e53, 9d4a81e.

---

### 15.27. Ciclo IAE NIPC coverage baseline Fase A (2026-09-19)

Origen: Q-T50-5 del dictamen TOP 50. Siguiente fase autorizada tras cierre de filer continuity.

Objetivo: medir las 6 metricas pairwise obligatorias sobre los 4 universos anidados, sin OpenFIGI, sin thresholds, sin tocar motor.

Artefactos generados:

  docs/auditoria/INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md
  docs/auditoria/iae/evidence/nipc_gate0_baseline/
    README.md                     (metodologia congelada antes del run)
    probe_coverage_baseline.py    (probe deterministico, sin datetime.now())
    baseline_output.txt           (salida cruda, exit 0, 7982 bytes, LF)
    HASHES.txt                    (SHA-256 de los 3 ficheros)

Incidente resuelto: `baseline_output.txt` se genero con CRLF via `Start-Process -RedirectStandardOutput`. `.gitattributes` (*.txt eol=lf) normalizaria a LF en el siguiente clone, invalidando el hash registrado. Correccion: normalizar a LF + recalcular hash + actualizar `HASHES.txt` + actualizar informe + `git commit --amend`. Verificado: los 3 hashes de disco coinciden con los registrados.

Verificacion mecanica: 4 checks PASS (sumas %units y %peso, IN_RADAR==TECHNICAL_RADAR, OTHER=0). 14 marcadores de seccion unicos.

Estado: Gate-NIPC.2 SIGUE BLOQUEADO por coverage/thresholds. THRESHOLD_1/2 UNDEFINED. Sin OpenFIGI. Sin push.

Commit local: e9fd830.

### 15.28. Ciclo micro-gate Coverage Contract Normalization (2026-09-19)

Origen: dictamen del auditor sobre coverage baseline (Issue ABIERTO: circularidad `operational_universe`).

Ciclo abierto en 4 fases: F2.1 (inventario) -> F2.1-bis (fuentes de identidad) -> F2.2 (propuesta v1.1) -> F2.3 (dictamen) -> F2.4 (aplicacion).

Resultados:

  F2.1 PASS (commit eda67ef). Inventario linea-por-linea de los 3 conceptos en `NIPC_COVERAGE_POLICY.md` v1.0:
    - TARGET: 0 ocurrencias literales.
    - RESOLVED: 0 ocurrencias literales.
    - PAIRED: 13 ocurrencias con definicion vaga (sin universo base ni criterio de emparejamiento).

  F2.1-bis PASS (commit 4659ce1). Inventario de fuentes de identidad:
    - `isin_ticker_map.csv`: 0 tickers USA.
    - `etf_holdings.csv`: 220/242 USA, pero parte del crosswalk evaluado.
    - `amundi_yahoo_mapping.csv`: 0 tickers USA.
    - `index_holdings.csv`: sin columna CUSIP/identifier.
    - Ninguna fuente cumple (CUSIP + radar USA + independencia).
    - Declaracion: `TARGET_CUSIP_REGISTRY = NOT AVAILABLE`.

  F2.2 NO GO CON CAMBIOS OBLIGATORIOS (commit d4a926e + dictamen). La propuesta v1.1 separaba TARGET/RESOLVED/PAIRED conceptualmente bien, pero definia TARGET con claves incompatibles (ticker ^ CUSIP ^ ticker-por-class). Intersectar requiere ejecutar el resolver que se pretende medir. 8 cambios obligatorios para v1.2.

  F2.3 pendiente. F2.4 NO AUTORIZADA.

Hallazgo colateral: el 90.91% (220/242) que `NIPC_COVERAGE_POLICY.md` v1.0 L115 cita como `mapping_coverage` sale exactamente de `etf_holdings.csv`, parte del crosswalk evaluado. La policy cita como cobertura el ratio del propio insumo del resolver.

Reordenacion propuesta del pipeline (a dictamen del auditor):

  baseline
    -> RADAR_TARGET_REGISTRY (via OpenFIGI como fuente)
    -> policy v1.2
    -> medicion sobre TARGET independiente
    -> thresholds

Thresholds THRESHOLD_1/2 siguen UNDEFINED. Gate-NIPC.2 BLOQUEADO.
`NIPC_COVERAGE_POLICY.md` v1.0 intacta (hash 57f2d01f...).

Cadena de commits:

  eda67ef  Gate 0 documental contrato cobertura - inventario 3 conceptos
  d4a926e  F2.2 propuesta v1.1 - circularidad + TARGET/RESOLVED/PAIRED
  4659ce1  F2.1-bis fuentes identidad - Camino B RADAR_TARGET_REGISTRY NOT AVAILABLE

### 15.29. Ciclo IAE F2.2-v2 - Propuesta v1.2 (2026-09-19)

Origen: dictamen F2.2 NO GO CON CAMBIOS OBLIGATORIOS (texto de sesion,
preservado en TRANSFER_SESION_2026-09-19_POST_F22.md, commit d2e03a1).

Entregable: docs/auditoria/NIPC_COVERAGE_POLICY_V12_PROPUESTA.md
(commit 0518b96, 361 lineas, LF puro, sin BOM).

Aplica los 8 cambios obligatorios del dictamen F2.2:
1. RADAR_TARGET_REGISTRY con identidad estable (NOT AVAILABLE, Camino B).
2. TARGET_UNIVERSE sobre clave comun a CUSIP/13F.
3. RESOLVED_UNIVERSE con enums v1.3 (FIGI / EQUIVALENCE).
4. PAIRED_UNIVERSE por canonical_security comun.
5. PAIRED_WEIGHTED_SHARE_COVERAGE con convencion Q4/Q1 inequivoca (max).
6. Contradiccion filer continuity resuelta: no es condicion de READY.
7. Referencia "Prompt Maestro v6.35" -> ruta canonica sin version.
8. Trazabilidad v1.0 -> v1.1 -> v1.2 explicita en seccion 0.

Verificacion: 911 passed + 2 skipped; pyflakes limpio; compileall OK.
Policy v1.0 intacta (hash 57f2d01f...). Sin push.

Estado: F2.3 PENDIENTE EXTERNO. F2.4 NO AUTORIZADA.
THRESHOLD_1/2 UNDEFINED. OpenFIGI NO AUTORIZADO.
Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO.

### 15.30. Ciclo IAE OpenFIGI + TARGET_UNIVERSE (2026-09-19)

Origen: dictamen F2.3 (NO GO v1.2) + entrega F2.3 -> F2.4
(inspeccion N-PORT). Auditor: GO CONDICIONADO con Q1=GO
N-PORT auxiliar, Q2=GO OpenFIGI CUSIP_13F -> FIGI/ticker,
Q3=NO FIJAR, Q4=GO fix N-PORT, Q5=CERRADO.

Secuencia de 7 commits funcionales (no documentales):
1. a130bf1 fix pivot N-PORT (0 -> 1063 pares CUSIP+ticker).
2. 3fccc93 cliente OpenFIGI + extract_stable_identity + 8 tests.
3. 7501825 micro-probe identidad radar (240/242 con FIGI).
4. a530ee3 RADAR_TARGET_CATALOG builder + 8 tests.
5. ae6129b data/mappings/radar_target_catalog.csv (242 filas).
6. f81fc17 TARGET_UNIVERSE resolver + 8 tests.
7. 32baf9d piloto top500 CUSIPs 13F -> 142/500 target (28.4%).

Estado arquitectonico: el bloqueo H1 del auditor (OpenFIGI no
puede ser autoridad unica del catalogo) queda resuelto por
diseno. El catalogo radar se construye desde OpenFIGI TICKER/US
-> shareClassFIGI, y el target membership se resuelve desde
CUSIP_13F -> OpenFIGI ID_CUSIP -> shareClassFIGI. El cruce por
shareClassFIGI es la identidad estable. No hay circularidad.

Verificacion: 935 passed + 2 skipped; pyflakes limpio; compileall OK.
Policy v1.0 intacta (hash 57f2d01f...). Sin push.

Estado: F2.3 PENDIENTE EXTERNO. F2.4 NO AUTORIZADA.
THRESHOLD_1/2 UNDEFINED. OpenFIGI masivo NO AUTORIZADO.
Gate-NIPC.2 BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO.

---

### 15.31. Ciclo IAE F2.3-bis + TOP 2000 (2026-09-19)

**Ciclo completo:** dictamen F2.3-bis + autorizacion + piloto + informe.

**Estado:** H1 CERRADO arquitectonicamente. Gate-NIPC.2 BLOQUEADO por
THRESHOLD_1/2 UNDEFINED. F2.4 NO AUTORIZADA. Full OpenFIGI NO AUTORIZADO.

**Entregables (5 commits):**

    3ef4d2d  docs(iae): F2.3-bis - dictamen + autorizacion + informe TOP 2000 + evidence
    9c67ac6  chore(iae): normalizar EOL en evidence historicos + recalcular HASHES
    06c49ad  chore: normalizar EOL segun .gitattributes (git add --renormalize .)

**Hallazgos para auditor:**
- count vs weight divergen: 28.4% -> 10.5% (count) y 46.76% -> 32.81% (weight).
- XOM: shareClassFIGI cambia tras reorganizacion corporativa.
- HON: no esta en el radar 242.
- 4 deudas declaradas (D1-D4).

**Esperando dictamen F2.4** con input del informe TOP 2000.

**Ahead al cierre:** 74 commits locales.

---
### 15.32. Ciclo post-F2.3-bis: revision estructural IAE + 6 fixes + F2.4 (2026-09-20)

**Origen:** peticion del usuario de revision completa del modulo IAE.

**Secuencia ejecutada:**

1. Revision estructural completa: 22 ficheros `.py`, 17 tests, 53 documentos.
   95 hallazgos clasificados (3 ALTA, 20 MEDIA, 23 BAJA, ~49 INFORMATIVO).
   2 hipotesis cerradas como NO BUG durante la revision (P14, P29).

2. Dictamen del auditor sobre la revision (materializado en
   `docs/auditoria/NIPC_DICTAMEN_REVISION_ESTRUCTURAL_2026-09-20.md`,
   hash `3d004540...`):
   - P38, P60, P61: NO GO / requieren contrato semantico.
   - P70: GO CONDICIONADO (cerrar invariante primero).
   - P51, P32, P14-BIS, P31, P18/P26: GO FIX.
   - P19, P20, P21, P24, P39, P75/P33: deuda diferida / documental.

3. P70 cerrado con probe empirico sobre Q4 2025 + Q1 2026:
   - 195 grupos `HR_PLUS_RESTATEMENT` auditados, 0 con `len > 2`.
   - Argumento estructural: invariante `len == 2` garantizada por
     `_classify_strategy_from_types`, no por el dataset.
   - Dictamen P70 (documento propio): GO CONDICIONADO.
   - Evidencia en `docs/auditoria/iae/evidence/nipc_p70_probe/`.

4. Contrato semantico v1 redactado (`NIPC_CONTRATOS_SEMANTICOS_v1.md`):
   - P60 (identity_type obligatorio, sin nuevos kinds).
   - P61 (operational_mapping_status separado de
     security_resolution_status).
   - P38 (denominador pairwise = `TARGET_Q4 INTERSECT TARGET_Q1`).
   - Estado: PROPUESTO / SOMETIDO A F2.4.

5. Policy coverage v1.3 propuesta (`NIPC_COVERAGE_POLICY_V13_PROPUESTA.md`):
   sustituye funcionalmente a v1.2. Preserva v1.0, v1.1, v1.2 intactas.

6. Los 6 fixes mecanicos aplicados (todos en local, sin push):

   | Fix | Commit | Archivo |
   |---|---|---|
   | P51 | `0f1e5a4` | `sec13f_list.py` (linea corta = anomalia, no ACTIVE) |
   | P32 | `714457e` | `delta_shares.py` (contador de descartes) |
   | P14-BIS | `eb687c4` | `delta_shares.py` (fillna condicionado a `_merge`) |
   | P31 | `e31d82e` | `nipc.py` (`nipc_total_available` + docstring) |
   | P18/P26 | `54478c3` | `manifest.py` (project_root por env var) |
   | P70 guarda | `1575d17` | `amendments.py` (assert `len == 2`) |

   Regresion acumulada: 0. Tests: 941 -> 946 passed.

7. Documentos de entrega al auditor:
   - `NIPC_INFORME_ENTREGA_F24.md` (referencia para D1-D4, hash `0763f16b...`).
   - `NIPC_INFORME_ESTADO_POST_FIXES_F24.md` (complementario, Q1-Q10, hash `49c397b7...`).

8. Reconciliacion final de hashes en la cadena documental (commit
   `3c59d41`). Cero residuos.

**Estado al cierre:** 97 commits locales ahead de `origin/main`. HEAD
`3c59d41`. 946 passed + 2 skipped. Working tree limpio. F2.4 PENDIENTE
EXTERNO.

**Bloqueos vigentes:** THRESHOLD_1/2 UNDEFINED. Gate-NIPC.2 BLOQUEADO.
Gate-NIPC.3 NO AUTORIZADO. OpenFIGI masivo NO AUTORIZADO. Policy v1.3
aplicacion NO AUTORIZADA. Contratos P38/P60/P61 PROPUESTOS.

**Documentos preservados intactos:** `NIPC_COVERAGE_POLICY.md` v1.0
(hash `57f2d01f...`), v1.1, v1.2.

**Siguiente paso (pendiente externo):** dictamen F2.4 del auditor. Si
es GO, autoriza el paso 4 (tests especificos para P38/P60/P61).

**Leccion operativa documentada:** `-replace` de PowerShell con
argumento numerico se interpreta como `-replace "N", ""`. Corrompio un
informe entero. Regla nueva: sustituciones condicionales (con conteo,
posicion o regex complejo) se ejecutan en Python puro con
`.replace(..., 1)`.

### 15.33. Implementacion de los contratos semanticos P60/P61/P38 (2026-09-20)

**Origen:** tras el cierre de los fixes mecanicos y la redaccion del
contrato semantico v1, se autorizo la implementacion directa en codigo
de los 3 contratos (sin esperar F2.4, en rama local, sin push).

**8 commits:**

    030f239  P60: identity_type obligatorio en _normalize_canonical
    6198d60  P61: modulo temporal_validity
    fc12402  P38: denominador TARGET_PAIRWISE + Q5 coverage_status
    2a13a6f  P61: operational_mapping_status integrado en resolver
    0f3ef8d  P61: end-to-end hasta _mapped_mask
    83e60c3  Q6: unmapped_count_* + Q8: identity_type en CSV

**P60 (implementado):**
- `_normalize_canonical(value, identity_type)`. Firma endurecida.
- identity_type obligatorio. Ausente o invalido -> ValueError.
- TICKER -> equity:<t> (CANONICAL_EQUIVALENCE).
- FIGI -> figi:<F> (CANONICAL_FIGI).
- CUSIP -> None (OBSERVED_CUSIP_ONLY).
- ISIN -> None (UNRESOLVED).
- Prefijo explicito en el valor gana sobre identity_type.

**P61 (implementado):**
- Nuevo modulo `src/institutional_accumulation/temporal_validity.py`.
- `resolve_source_status(source, valid_from, valid_to, period)`.
- `aggregate_status(entries)` con regla Q7 completa.
- Enum: VERIFIED | TEMPORAL_UNVERIFIED | UNRESOLVED | CONFLICT.
- `resolve_security_identity` envuelve `_resolve_identity_inner` y
  anade `operational_mapping_status` al return.
- `_mapped_mask` en `nipc.py` requiere CANONICAL + VERIFIED.
- `operational_mapping_status` propagado en `UNITS_COLUMNS`.

**P38 (implementado):**
- Denominador pairwise = `TARGET_Q4 INTERSECT TARGET_Q1` (interseccion,
  no union).
- `paired_weighted_share_coverage: float | None`.
- `coverage_status: VALID | UNAVAILABLE` (Q5).
- Peso `w(s) = max(Q4, Q1)`.

**Q6 + Q8 (implementados):**
- `unmapped_weight_*` renombrado a `unmapped_count_*`.
- `identity_type` columna opcional en `cusip_equivalence.csv`. Default
  TICKER si falta. `_find_active_equivalence` devuelve tuplas
  (canonical, identity_type).

**Tests:** 946 -> 979 passed + 2 skipped. Regresion acumulada: 0.

**Hueco pendiente:** TARGET real. El denominador P38 usa actualmente
`observed_security_key` como proxy. La proyeccion real
`CUSIP -> shareClassFIGI -> RADAR_TARGET_CATALOG` requiere OpenFIGI
masivo (NO AUTORIZADO hasta F2.4).

**Bloqueos vigentes:** sin cambios. THRESHOLD_1/2 UNDEFINED. Gate-NIPC.2
BLOQUEADO. Gate-NIPC.3 NO AUTORIZADO. OpenFIGI masivo NO AUTORIZADO.
Policy v1.3 aplicacion NO AUTORIZADA. F2.4 PENDIENTE EXTERNO.

**Ahead al cierre:** 104 commits locales.

### 15.34. Ciclo P66 - Reformulacion contrato 14.3 L3 (2026-09-21) - CERRADO

**Objetivo:** reformular el contrato 14.3 para materializar el
requisito 4 de L3 desde el Data Set SEC sin XML crudo.

**Origen:** P65 v2 (DROP_DUP efectivo) se cerro WONT FIX porque
la evidencia cruzada no estaba en el Data Set tabular segun el
analisis inicial. Investigacion posterior confirmo que SI estaba,
en la tabla `OTHERMANAGER`.

**Duracion:** multiples iteraciones (~10 gates + 12 dictamenes).

**Gates ejecutados:**

- 0.1  Cobertura OTHERMANAGER por tipo de filing.
- 0.2  Caso Vanguard.
- 0.3  Granularidad de identidad.
- 0.4  Mapping FormNum -> CIK via COVERPAGE.
- 0.4-reissue  Filtro PERIODOFREPORT.
- 0.5  Amendment probe (cross-period + restatement).
- 0.6  Combination probe (universo R4).
- 0.7  NEW HOLDINGS probe.
- 0.8  Cobertura identidad universo R4 completo.
- 0.9  Multiples filings base.
- 0.10 FormNum representation.

**Dictamenes:** #28 a #40.

**GO contractual:** declaracion condicional del auditor en #39.

**Resultado:** §14.3 reformulada con 7 subsecciones. Estructura
§14.1 a §14.14 preservada.

**Ficheros clave:**

    iae/P66_L3_REFORMULACION_PROPUESTA.md         (v7-ter, propuesta)
    iae/DICTAMENES.md                             (#28 a #40)
    iae/INFORME.md                                (evidencia consolidada)
    iae/evidence/p66_gate*/                       (evidencia empirica)
    NIPC_CONTRATOS_SEMANTICOS_v1.md §14.3         (contrato reformulado)

**Prohibiciones vigentes:**

- NO tocar `reporting_dedup.py` sin nuevo dictamen.
- NO activar `DROP_DUP` sin nuevo dictamen.
- NO modificar §14.3 fuera del cauce previsto por §14.3.7.

**Siguiente ciclo autorizado:** tests §14.3 -> implementacion
`reporting_dedup` -> probe e2e -> auditoria de salida -> activacion
`DROP_DUP`.

---

## Ciclo A2 - Fix H-05/H-06/H-07/H-10.1 (2026-09-21)

**Dictamenes:** #76 (A.6.6 = NO-GO, A2 = GO).

**Commits:**

    2f98926  period_state.py: sshprnamt + operational_mapping_status
    8f8ef87  target_builder.py + probe: SSHPRNAMT del canonical_snapshot
    da5fbcb  catalog_p38_adapter.py: propaga operational + weight (H-10.1+H-07)
    f7f7b20  coverage.py: denominador TARGET + Q5 fail-closed
    157211e  probe sin mock Q4=Q1 + Q5 literal + cardinalidades
    473ce06  chore: regenerar ESTADO_SISTEMA.md
    deb9782  docs: TRANSFER v9.1
    957053f  docs: A66_BUNDLE para dictamen #76
    (actual)  saneamiento B-01/B-05/B-06/B-07 + H-08

**Evidencia empirica** (13F Q4 2025 / Q1 2026 reales):

    coverage_previous              = None    (Q4 vacio)
    coverage_current               = 1.0     (18/18 sobre TARGET_Q1 materializado)
    paired_security_coverage       = None    (TARGET_PAIRWISE=0)
    paired_weighted_share_coverage = None

**Hallazgos cerrados:** H-05, H-06, H-07, H-10.1, H-08, B-01, B-05, B-06, B-07.
**Bloqueos residuales (NO-GO A.6.6):** B-02 (P62/PIT), B-03 (TARGET completo),
B-04 (pairwise real).
