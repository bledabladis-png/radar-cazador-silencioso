# INFORME FA-2.4 - Amendments + Canonical Snapshot

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
Especificacion: INSTITUTIONAL_ACCUMULATION_FA24_ESPECIFICACION.md
Estado: CERRADO local. Sin push (regla local-first IAE).

---

## 1. Alcance

Resolver snapshot canonico del trimestre por (CIK, PERIOD) aplicando
semantica SEC de amendments (RESTATEMENT = REPLACE, NEW HOLDINGS = ADD).
Lineage composicional. Separacion holdings vs graph.

## 2. Artefactos

  src/institutional_accumulation/sec_13f/identity/amendments.py
  tests/test_sec_13f_amendments.py (30 tests)

  __init__.py actualizado con exports FA-2.4.

## 3. Contrato implementado

  order_filings(sub_df, cov_df) -> DataFrame
    Orden: original -> AMENDMENTNO -> FILING_DATE -> ACCESSION.

  classify_strategy(filings_group) -> str
    8 valores (7 + REVIEW_REQUIRED).

  detect_base_filing(filings_group) -> (accession|None, status)
    I2: exactamente 1 base o AMBIGUOUS_BASE.

  apply_amendments(dfs, *, period) -> dict
    canonical_snapshot, applied_accessions, lineage, anomalies,
    per_cik_period.

  compute_strategy_counts / compute_status_counts.

## 4. Probe real Q1 2026 (10,776 filings)

### 4.1. Distribucion de estrategias

| Estrategia | N | Spec | Delta |
|---|---|---|---|
| SINGLE_HR | 8,618 | 8,617 | +1 |
| SINGLE_NOTICE | 1,906 | 1,907 | -1 |
| HR_PLUS_RESTATEMENT | 100 | 100 | 0 |
| HR_PLUS_NEW_HOLDINGS | 19 | 19 | 0 |
| HR_CHAIN_RESTATEMENT | 2 | 2 | 0 |
| HR_COMPOSITE | 2 | 2 | 0 |
| NOTICE_AMENDED | 1 | 1 | 0 |
| TOTAL | 10,648 | 10,648 | 0 |

Delta +/-1 en SINGLE_HR/SINGLE_NOTICE por reclasificacion del caso
CIK 0002051980 (13F-HR con ISAMENDMENT=Y + AMENDMENTTYPE=NEW HOLDINGS).
Dictamen P-AM.4: tratar como original -> SINGLE_HR.

### 4.2. Status

| Status | N |
|---|---|
| CANONICAL | 8,720 |
| NO_HOLDINGS | 1,906 |
| CANONICAL_COMPOSITE | 21 |
| SOURCE_ANOMALY | 1 |
| TOTAL | 10,648 |

### 4.3. Anomalias detectadas

| Anomalia | N |
|---|---|
| SOURCE_ANOMALY_NT_AUGMENTED | 1 |
| SOURCE_ANOMALY_HR_WITH_AMENDMENT_FLAGS | 1 |

  - NT_AUGMENTED: CIK 0001581079, accession 0001193125-26-229329.
  - HR_WITH_AMENDMENT_FLAGS: CIK 0002051980, accession 0001376474-26-000287.

### 4.4. Los 4 casos criticos

HR_COMPOSITE (2):
  0002097005 -> aplica [0002097005-26-000002, 0002097005-26-000004]
  0002100119 -> aplica [0002100119-26-001311, 0002100119-26-001313]

HR_CHAIN_RESTATEMENT (2):
  0000949509 -> aplica [0000949509-26-000004] (ultimo restatement)
  0001349434 -> aplica [0001349434-26-000004] (ultimo restatement)

### 4.5. Impacto en el snapshot

| TSV | Filtrado | Snapshot | Delta |
|---|---|---|---|
| SUBMISSION | 10,776 | 8,762 | -2,014 |
| COVERPAGE | 10,776 | 8,762 | -2,014 |
| INFOTABLE | 3,321,967 | 3,239,273 | -82,694 |

Applied accessions: 8,762.
Los 2,014 filings descartados: 1,906 NT + 1 NT/A + ~107 filings
superados por RESTATEMENT (el HR base o el HR/A #1 anterior).

### 4.6. Invariante

  accessions en snapshot NO en applied: 0.

## 5. Verificacion mini-gate

| Verificacion | Resultado |
|---|---|
| compileall | OK |
| pyflakes | 0 warnings |
| pytest FA-2.4 | 30 passed |
| pytest global | 820 passed + 2 skipped |
| FutureWarning | 0 (corregido) |
| Probe real | OK |
| Snapshot invariante | OK |

## 6. Notas

  - El caso HR_WITH_AMENDMENT_FLAGS (0002051980) se trata como original
    y SE INCLUYE en el snapshot como SINGLE_HR. La anomalia se registra
    pero no bloquea el snapshot.

  - El caso NOTICE_AMENDED (0001581079) queda como SOURCE_ANOMALY:
    ningun filing del grupo contribuye al snapshot de holdings.
    Ambos filings quedan en lineage como NO_HOLDINGS.

  - Los 1,906 NT + 1 NT/A excluidos del snapshot son correctos:
    un notice no contiene holdings.

  - Separacion holdings vs graph: el snapshot contiene solo holdings.
    El grafo de relaciones FA-2.3 se ejecutara sobre el snapshot.

## 7. Impacto en FA-2.5

FA-2.5 ejecutara el pipeline completo:
  filter_by_period (FA-2.1)
    -> apply_amendments (FA-2.4)
    -> explode_othermanager_edges (FA-2.3)
    -> resolve_cusip (FA-2.2)
    -> identidad canonica final + metrics

Y demostrara el Gate FA-2: 3.3M filas -> identidad canonica sin doble
conteo.

---

Fin del informe. Version 1.0 (2026-09-19).
