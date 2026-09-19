# INFORME GATE FA-2 - IAE SEC 13F

Version: 1.0 (2026-09-19)
Dictamenes habilitantes:
  FA-2.0: INSTITUTIONAL_ACCUMULATION_FA2_GATE0_DICTAMEN.md
  FA-2.3: INSTITUTIONAL_ACCUMULATION_FA23_DICTAMEN_C.md + DICTAMEN_CIERRE.md
  FA-2.4: INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
Estado: Gate FA-2 superado localmente. Pendiente dictamen final.
NIPC: permanece BLOQUEADO hasta aprobacion del Gate FA-2.

---

## 1. Proposito

Cerrar la Fase A (SEC 13F core) demostrando que el pipeline completo
transforma 3,321,967 filas INFOTABLE en identidad canonica sin doble
conteo, con lineage auditable y sin heuristicas prohibidas.

Criterios dictaminados (Gate FA-2, CONTRATO seccion 8):
  coverage
  duplicate elimination
  manager attribution
  DFND treatment
  amendment resolution

## 2. Pipeline FA-2 completo

  1. filter_by_period (FA-2.1)
     Campo canonico: SUBMISSION.PERIODOFREPORT.

  2. apply_amendments (FA-2.4)
     RESTATEMENT -> REPLACE. NEW HOLDINGS -> ADD.
     Orden: original -> AMENDMENTNO -> FILING_DATE -> ACCESSION.

  3. explode_othermanager_edges (FA-2.3)
     FK: INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQUENCENUMBER.
     Multi-edge. Sin division economica.

  4. resolve_cusip (FA-2.2)
     Modelo multi-fila con vigencia temporal.
     Firma: resolve_cusip(cusip, report_period).

## 3. Metricas del probe integrado

### 3.1. Filtro temporal (FA-2.1)

  SUBMISSION filtrado: 10,776
  INFOTABLE filtrado:  3,321,967

### 3.2. Canonical snapshot (FA-2.4)

  Applied accessions: 8,762
  Snapshot SUBMISSION: 8,762
  Snapshot INFOTABLE:  3,239,273

  Distribucion de estrategias (10,648 grupos):

    SINGLE_HR                    8,618
    SINGLE_NOTICE                1,906
    HR_PLUS_RESTATEMENT            100
    HR_PLUS_NEW_HOLDINGS            19
    HR_CHAIN_RESTATEMENT             2
    HR_COMPOSITE                     2
    NOTICE_AMENDED                   1

  Status:

    CANONICAL                    8,720
    NO_HOLDINGS                  1,906
    CANONICAL_COMPOSITE             21
    SOURCE_ANOMALY                   1

  Anomalias registradas: 2
    SOURCE_ANOMALY_NT_AUGMENTED (CIK 0001581079)
    SOURCE_ANOMALY_HR_WITH_AMENDMENT_FLAGS (CIK 0002051980)

### 3.3. Reporting relationships (FA-2.3)

  Edges totales:                 3,509,475
  Resolved:                      1,351,172
  Unmapped:                         16,001
  Invalid non-numeric:              55,964
  Invalid zero:                        292
  Invalid out-of-domain:               292
  No reference:                  2,085,754

  resolution_rate:                  0.9883
  duplicate_canonical_edges:             0
  unique_canonical_edges:        1,367,173
  source_line_count:             3,239,273
  invalid_source_line_edges:             0

### 3.4. CUSIP resolver (FA-2.2)

  Excepciones cargadas: 0 filas (tabla placeholder).
  Con ticker_resolved:  0 filas (esperado sin curacion).

  La tabla de excepciones CUSIP queda como tarea de curacion manual
  posterior (fuera de FA-2). El contrato del resolver esta implementado
  y verificado por tests unitarios (17 tests).

## 4. Cumplimiento de criterios Gate FA-2

### 4.1. Coverage

  resolution_rate = 98.83% (resolved / (resolved + unmapped)).
  Baseline empirico aprobado por dictamen FA-2.3 cierre.

### 4.2. Duplicate elimination

  duplicate_canonical_edges = 0.
  Invariante satisfecho sobre 3,239,273 source_line_id unicos.

  Precision dictaminada: demuestra ausencia de duplicacion TECNICA
  del edge. NO demuestra ausencia de doble conteo ECONOMICO.

### 4.3. Manager attribution

  FK corregida (dictamen caracterizacion C):
    INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQUENCENUMBER.

  Clave canonica B.1 (dictamen FA-2.0):
    (filing_manager_cik, included_manager_cik, discretion_type, period)
    renombrada semanticamente a canonical_reporting_relationship_key.

  NO economic_owner. Terminos prohibidos ausentes del codigo.

### 4.4. DFND treatment

  DFND preservado sin colapso con SOLE/OTR (dictamen Q2 + dictamen
  caracterizacion C, seccion 15). Tests verifican la preservacion.

  Ultima medicion (dataset completo):
    SOLE  63.93%
    DFND  29.05%
    OTR    7.02%

### 4.5. Amendment resolution

  127 HR/A + 1 NT/A resueltos.
  100 RESTATEMENT simples + 19 NEW HOLDINGS simples
  + 2 chain RESTATEMENT + 2 HR_COMPOSITE (RESTATEMENT + NEW HOLDINGS)
  + 1 NOTICE_AMENDED (anomalia).

  Verificado que el ultimo RESTATEMENT sustituye y que NEW HOLDINGS
  complementa, segun semantica SEC.

## 5. Reglas del dictamen verificadas

| Regla | Estado |
|---|---|
| Filtro por SUBMISSION.PERIODOFREPORT | OK |
| Sin heuristica CIK | OK |
| Sin heuristica por nombre | OK |
| Sin fallback de resolucion de anomalias | OK |
| source_line_id preservado | OK |
| Multi-edge sin division economica | OK |
| Separacion holdings vs graph | OK |
| Anomalias preservadas, no silenciadas | OK |
| snapshot composicional con lineage | OK |

## 6. Verificacion agregada

| Verificacion | Resultado |
|---|---|
| compileall | OK (silencio) |
| pyflakes | 0 warnings |
| pytest identity | 78 passed (13 + 17 + 33 + 30) |
| pytest global | 820 passed + 2 skipped |
| FutureWarning | 0 |
| Probe integrado | OK |

## 7. Estado del repo

  HEAD local: pendiente commit informe FA-2.4 + informe Gate FA-2.
  origin/main: e2c3f02.
  Ahead tras commits: 19.
  Working tree: limpio.
  NIPC: BLOQUEADO (hasta dictamen Gate FA-2).
  Push: NO autorizado (regla local-first IAE).

## 8. Comites de cierre pendientes

  1. Dictamen del auditor sobre este informe Gate FA-2.
  2. Si GO: push unico con todos los commits locales.
  3. Actualizacion del prompt a v6.35.
  4. Entrada en FOLLOWUPS.md (K-INSTITUTIONAL-ACCUMULATION-01).

## 9. Fuera de FA-2 (para el dictamen)

  - Curacion manual de la tabla de excepciones CUSIP.
  - Conectar el CUSIP resolver al pipeline con datos reales.
  - NIPC, Breadth, New/Exit, clasificacion (post Gate FA-2).
  - Fase B (N-PORT) y Fase C (cross-validation).

---

Fin del informe. Version 1.0 (2026-09-19).
