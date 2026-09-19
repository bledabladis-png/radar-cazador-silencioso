# ESPECIFICACION FA-2.4 - Amendments + Canonical Snapshot

Version: 1.0 (2026-09-19)
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_FA24_DICTAMEN_GATE0.md
Estado: correcciones obligatorias del dictamen aplicadas.
Naturaleza: no normativo. Si hay conflicto, gana el Prompt Maestro.

---

## 1. Proposito

Resolver el snapshot canonico del trimestre por (CIK, PERIOD),
aplicando la semantica SEC de amendments (RESTATEMENT vs NEW HOLDINGS).
Producir lineage composicional auditable. Separar holdings de graph.

Aplicar las dos correcciones obligatorias del dictamen:
  1. Tabla de estrategias formal (sin ambiguedad aritmetica).
  2. Orden de aplicacion: AMENDMENTNO primero.

## 2. Corpus empirico Q1 2026

Filings filtrados: 10,776.
Grupos (CIK, PERIOD): 10,648.

| Tamano de grupo | N | Composicion |
|---|---|---|
| 1 filing | 10,524 | 8,617 HR solo + 1,907 NT solo |
| 2 filings | 120 | 119 (HR, HR/A) + 1 (NT, NT/A) |
| 3 filings | 4 | (HR, HR/A, HR/A) |

Grupos con >=1 amendment: 124.
Amendments totales: 128 = 127 HR/A + 1 NT/A.

## 3. Tabla de estrategias (reconciliada)

| Estrategia | N | Composicion |
|---|---|---|
| SINGLE_HR | 8,617 | HR sin amendment |
| SINGLE_NOTICE | 1,907 | NT sin amendment |
| HR_PLUS_RESTATEMENT | 100 | HR + 1 HR/A RESTATEMENT |
| HR_PLUS_NEW_HOLDINGS | 19 | HR + 1 HR/A NEW HOLDINGS |
| HR_CHAIN_RESTATEMENT | 2 | HR + 2 HR/A RESTATEMENT |
| HR_COMPOSITE | 2 | HR + RESTATEMENT + NEW HOLDINGS |
| NOTICE_AMENDED | 1 | NT + NT/A (anomalia) |
| TOTAL | 10,648 | |

Reconciliacion:
  119 grupos x 1 HR/A = 119
    4 grupos x 2 HR/A =   8
    1 grupo  x 1 NT/A =   1
                      -----
                       128 amendments

Grupos con >=1 amendment: 119 + 4 + 1 = 124.

## 4. Orden de aplicacion (corregido)

Orden canonico:
  1. period (PERIODOFREPORT)
  2. original vs amendment (por SUBMISSIONTYPE)
  3. AMENDMENTNO
  4. FILING_DATE
  5. ACCESSION_NUMBER

Razon: SEC define AMENDMENTNO como orden de presentacion en el
trimestre. NO usar FILING_DATE como primario (puede haber mismo dia).

Ejemplo CIK 0001349434: dos HR/A mismo FILING_DATE=15/05. AMENDMENTNO
1 y 2 resuelven el orden sin ambiguedad.

## 5. Reglas de canonicalizacion

  RESTATEMENT  -> reemplaza snapshot vigente (REPLACE).
  NEW HOLDINGS -> anade entradas al snapshot (ADD).
  NT           -> no contribuye a holdings. Solo grafo.
  NT/A         -> no contribuye a holdings (aunque AMENDMENTTYPE=NEW).

Estado por (CIK, PERIOD):
  CANONICAL            un unico filing final determina el snapshot.
  CANONICAL_COMPOSITE  varios filings contribuyen (chain RESTATEMENT
                       o RESTATEMENT + NEW HOLDINGS).
  NO_HOLDINGS          NT o NT/A sin holdings.
  SOURCE_ANOMALY       flags contradictorios o secuencia invalida.
  REVIEW_REQUIRED      secuencia no reconocida.

## 6. Invariantes

  I1. unique (CIK, PERIOD, AMENDMENTNO) entre amendments del mismo filer.
      Violacion -> AMBIGUOUS_AMENDMENT_ORDER.

  I2. Para cada (CIK, PERIOD): exactamente 1 base_holdings_filing o 0
      (si no hay HR). >1 -> AMBIGUOUS_BASE.

  I3. Si strategy = CANONICAL_COMPOSITE: applied_accessions.len() >= 2.

  I4. Si strategy = SOURCE_ANOMALY o REVIEW_REQUIRED: canonical_snapshot
      puede estar vacio. NO publicar como VALID.

## 7. Anomalias catalogadas

  A1. SOURCE_ANOMALY_NT_AUGMENTED
      NT/A con AMENDMENTTYPE=NEW HOLDINGS (1 caso Q1 2026).

  A2. SOURCE_ANOMALY_HR_WITH_AMENDMENT_FLAGS
      HR con AMENDMENTNO/AMENDMENTTYPE != null (1 caso Q1 2026).

  A3. AMBIGUOUS_AMENDMENT_ORDER
      >1 amendment con mismo AMENDMENTNO para (CIK, PERIOD).

  A4. AMBIGUOUS_BASE
      >1 HR candidato a base para (CIK, PERIOD).

  A5. REVIEW_REQUIRED
      Secuencia no contemplada (ej. HR -> NEW -> RESTATEMENT).

## 8. Contrato del modulo

  src/institutional_accumulation/sec_13f/identity/amendments.py

  Funciones publicas:

    order_filings(sub_df, cov_df) -> DataFrame
      Anade columna _order y _is_amendment. Determinista.

    classify_strategy(filings_group) -> str
      Devuelve una de las 7 estrategias de la seccion 3.

    detect_base_filing(filings_group) -> (accession|None, status)
      Aplica I2. Devuelve AMBIGUOUS_BASE si aplica.

    apply_amendments(dfs, *, period) -> dict
      Orquesta. Devuelve:
        canonical_snapshot    dict[str, DataFrame]
        lineage               DataFrame
        anomalies             DataFrame
        per_cik_period        DataFrame

  Estructura de per_cik_period:
    CIK
    report_period
    strategy          (una de las 7)
    status            (uno de los 5 estados)
    base_accession
    applied_accessions  (lista serializada)
    amendment_sequence  (lista de AMENDMENTNO aplicados)
    filing_count
    amendment_count

  Estructura de lineage:
    CIK
    report_period
    accession
    amendment_no
    amendment_type
    operation      REPLACE | ADD | NO_HOLDINGS
    applied        bool
    reason         str

  Estructura de anomalies:
    CIK
    report_period
    accession
    anomaly_code   (A1..A5)
    detail

## 9. Separacion holdings vs graph

  CANONICAL HOLDINGS SNAPSHOT:
    Solo filings con holdings (HR, HR/A con holdings).
    NT y NT/A excluidos.

  REPORTING / NOTICE LINEAGE:
    Todos los filings.
    NT y NT/A incluidos.

Las dos capas se exponen por separado.

## 10. Fuera de FA-2.4

  - NIPC (post Gate FA-2).
  - CONFIRMED_INSTITUTIONAL (post Gate FA-2).
  - Cross-validation con N-PORT (Fase C).
  - Atribucion economica (no inferible).
  - Heuristicas de resolucion de anomalias.

## 11. Secuencia autorizada (dictamen)

  1. Corregir tabla de estrategias (esta spec).
  2. Corregir orden de aplicacion (esta spec).
  3. Implementar amendments.py.
  4. Mini-gate FA-2.4.
  5. FA-2.5 probe real + informe.
  6. Gate FA-2.
  7. Dictamen final + push.

---

Fin de la especificacion. Version 1.0 (2026-09-19).
