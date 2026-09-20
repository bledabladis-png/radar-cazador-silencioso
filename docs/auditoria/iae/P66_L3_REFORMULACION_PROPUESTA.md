# IAE - P66 L3 Reformulacion propuesta (v4)

**Estado:** v4 consolidada. Pendiente de dictamen contractual definitivo
sobre el texto exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.

**Trazabilidad de dictamenes:**

    #28  corregido
    #29  corregido
    #30  corregido
    #31  corregido
    #32  corregido
    #33  corregido
    #34  4 bloqueos + 2 recomendadas aplicados en este documento v4

**Consolidacion:** este documento reemplaza las versiones v1/v2/v3.
Todas las correcciones anteriores estan integradas. No se conservan
referencias a versiones previas dentro del texto.

**HEAD al redactar:** fc0d223 (o posterior).

---

## 0. Contexto y alcance

### 0.1. Objetivo

Reformular el contrato 14.3 de L3 para permitir la materializacion
del requisito 4 desde el Data Set SEC sin depender de XML crudo ni
de parser paralelo.

La evidencia de los Gates 0.1 a 0.10 demuestra que la tabla
`OTHERMANAGER` contiene la relacion "other managers reporting for
this manager" en el universo de filings aplicable a R4.

### 0.2. Alcance

El concepto de "filing efectivo" (o "conjunto documental efectivo")
definido en esta propuesta se aplica **EXCLUSIVAMENTE** a la
determinacion del filing o conjunto documental de **B** necesaria
para R3/R4.

**La seleccion del filing de A requerida por R1/R2 permanece regida
por las reglas contractuales preexistentes, salvo modificacion
expresa posterior del contrato.**

P66 NO redefine la seleccion del filing de A. NO modifica R1 ni R2.
Cuando §2 menciona "existe filing efectivo de A", se refiere a la
aplicacion de las reglas R1/R2 preexistentes, no a una nueva
definicion introducida por P66.

### 0.3. Terminologia

**filing efectivo / conjunto documental efectivo:** resultado de
aplicar el algoritmo de §3 a los filings de B (NOTICE o COMBINATION)
para un mismo `PERIODOFREPORT`. Determinable cuando la cadena base +
amendments es unica.

**candidate_A(r):** predicado sobre una fila `r` de OTHERMANAGER.
Una fila es candidata a A cuando:

    candidate_A(r) :=
        CIK(r) == CIK_A
        OR
        CIK_A ∈ resolved_ciks(FormNum(r))

donde `resolved_ciks(FormNum)` es el conjunto de CIKs a los que
resuelve el FormNum normalizado segun §4 dentro del scope temporal.

**identidad resuelta:** una fila `r` de OTHERMANAGER tiene identidad
resuelta cuando cumple alguna de:

    - CIK(r) poblado; o
    - FormNum(r) normalizable y resuelve a exactamente 1 CIK.

**inequivocamente:** sin matching por nombre, sin heuristica, sin
inferencia. Solo CIK directo o FormNum con resolucion univoca.

### 0.4. Historial de versiones

| Version | Contenido |
|---------|-----------|
| v1 | commit 6b610b3. Propuesta inicial + primeras correcciones. |
| v2 | Correcciones #28 aplicadas (fail-closed, nomenclatura). |
| v3 | Correcciones #29 a #33 aplicadas (R4 completo, mapping, tri-state). |
| **v4** | **Consolidacion final. Correcciones #34 aplicadas + reescritura canonica.** |

---

## 1. Texto vigente del contrato 14.3 (referencia historica)

    L3(A, B, S, period) == True sii:
      1. existe filing de A con linea L sobre security S
      2. Column 7(L) referencia a B (reference_status = RESOLVED)
      3. existe filing de B en el mismo periodo
      4. B declara explicitamente que A reporta por B
         (NT o Combination con A en "Other Managers Reporting for this Manager")
      5. no existe evidencia contradictoria ni evidencia de reporting
         partition/overlap no resuelto

---

## 2. Propuesta reformulada

### 2.1. R3 (tri-state)

    R3(B, period) evalua en tres estados:

        TRUE
            El conjunto documental efectivo de B es determinable
            inequivocamente y pertenece al universo R4:
              - 13F-NT / 13F-NT/A con REPORTTYPE = 13F NOTICE; o
              - 13F-HR / 13F-HR/A con REPORTTYPE = 13F COMBINATION REPORT.

        FALSE
            No existe ningun filing perteneciente al universo R4
            para B y PERIODOFREPORT.

        N/D
            Existen submissions aplicables al universo R4, pero
            el conjunto documental efectivo no puede determinarse
            inequivocamente (caso tipico: >1 filing base
            heterogeneo, Gate 0.9 CIK 0002016827).

La distincion SUBMISSIONTYPE + REPORTTYPE es obligatoria. `13F-HR`
incluye dos report types con mismo SUBMISSIONTYPE (HOLDINGS REPORT
y COMBINATION REPORT); solo el segundo es valido para R4.

### 2.2. R4 (estados)

    MATCH
        Existe una fila OTHERMANAGER que identifica inequivocamente
        a A mediante CIK o Form13FFileNumber.

    NO_MATCH
        El conjunto documental efectivo es determinable, la ingestion
        es integra, todas las filas OTHERMANAGER tienen identidad
        resuelta inequivocamente y A no aparece.

    N/D
        El conjunto documental efectivo no es determinable, o existe
        al menos una fila OTHERMANAGER cuya identidad no puede
        resolverse inequivocamente y no existe evidencia positiva
        inequivoca de A.

    CONFLICT
        La evidencia candidata a identificar A contiene
        identificadores contradictorios o una resolucion no univoca.

### 2.3. Alcance del CONFLICT (candidate_A)

CONFLICT > MATCH aplica a **CUALQUIER fila OTHERMANAGER que
identifique o pretenda identificar a A** (candidate_A(r) == True),
no solo a la fila que la implementacion eventualmente seleccione.

    CONFLICT cuando existe al menos una fila r con
    candidate_A(r) == True tal que:

    - CIK(r) y FormNum(r) apuntan a CIK distintos; o
    - FormNum(r) resuelve a >1 CIK dentro del scope temporal; o
    - FormNum(r) contradice el CIK(r) explicitamente declarado
      para esa misma fila.

    Si ninguna fila con candidate_A(r) == True presenta conflicto
    y al menos una fila identifica inequivocamente a A -> MATCH.

**Regla de no-seleccion selectiva:** la implementacion NO puede
elegir la fila consistente e ignorar la contradictoria respecto
de A. Si existe contradiccion entre dos filas con candidate_A
== True, el resultado es CONFLICT, no MATCH.

Dos filas que ambas identifican coherentemente a A NO constituyen
conflicto.

### 2.4. Orden de evaluacion

    1. ¿Conjunto documental efectivo determinable?
       NO -> N/D
       SI
       ↓
    2. ¿CONFLICT en alguna fila con candidate_A(r) == True?
       SI -> CONFLICT
       NO
       ↓
    3. ¿Existe fila que identifica inequivocamente a A?
       SI -> MATCH
       NO
       ↓
    4. ¿Todas las filas OTHERMANAGER tienen identidad resuelta?
       NO -> N/D
       SI
       ↓
    5. NO_MATCH

### 2.5. Combinacion booleana L3

    L3(A, B, S, period) = True
    sii:
        R1 == True
        AND R2 == True
        AND R3(B, period) == TRUE
        AND R4(A, B) == MATCH
        AND R5 == True

**Reglas de preservacion de estado:**

    R3 = FALSE       -> L3 no puede ser True
    R3 = N/D         -> L3 no puede ser True
    R4 = NO_MATCH    -> L3 no puede ser True
    R4 = N/D         -> L3 no puede ser True
    R4 = CONFLICT    -> L3 no puede ser True

**N/D y CONFLICT NO se convierten internamente en False.** Sus
estados deben preservarse para auditoria. Son condiciones de
"no puedo afirmar L3", no de "L3 es falso".

---

## 3. Tratamiento de amendments (cadena determinista)

### 3.1. Clasificacion base vs amendment

    BASE:
        ISAMENDMENT != Y

    AMENDMENT:
        ISAMENDMENT == Y

**Nunca usar AMENDMENTNO = null como prueba de que un filing es base.**
La SEC distingue explicitamente el hecho de que un filing es
amendment; el numero solo identifica el orden entre amendments.

Si `ISAMENDMENT == Y` y `AMENDMENTNO` no es determinable:
    N/D.

### 3.2. Construccion de la cadena efectiva

    Para filings de B con mismo CIK + PERIODOFREPORT + familia
    documental (NOTICE o COMBINATION):

    1. Filtrar filings por PERIODOFREPORT igual al periodo de
       analisis. Prohibido filtrar por directorio fisico.

    2. Clasificar por ISAMENDMENT.

    3. Contar bases:
         0 bases + 0 amendments
             -> R3 = FALSE (ausencia documental demostrada).
         0 bases + >=1 amendment aplicable
             -> R3 = N/D (ausencia de antecedente necesario;
                cadena no reconstruible).
         1 base
             -> continuar.
         >1 bases heterogeneas
             -> R3 = N/D o CONFLICT segun naturaleza de duplicidad.

    4. Aplicar amendments en orden AMENDMENTNO ascendente.

    5. Si no puede ordenarse inequivocamente (dos amendments
       indistinguibles en el mismo CIK + periodo + familia):
       N/D.

### 3.3. Semantica de amendments

    RESTATEMENT:
        Sustituye la evidencia OTHERMANAGER anterior.
        Evidencia directa: Gate 0.5, caso CIK 0002056909
        (base Prospector Partners -> restatement Gator Capital).

    NEW HOLDINGS:
        Conserva la evidencia OTHERMANAGER cuando es consistente
        con el estado efectivo previo.
        Evidencia: Gate 0.7, 3/3 casos identicos.
        Fail-closed: si un NEW HOLDINGS presenta OTHERMANAGER
        distinto del estado previo -> N/D o CONFLICT.
        NO asumir union. NO asumir sustitucion.

### 3.4. Multiples bases heterogeneas

**Regla contractual IAE (no atribuida a la SEC):** cuando existen
>1 filings base independientes para mismo CIK + PERIODOFREPORT:

    -> R3 = N/D o CONFLICT segun naturaleza de la duplicidad.

PROHIBIDO:
    - primero encontrado
    - ultimo encontrado
    - MAX(ACCESSION_NUMBER)
    - MAX(FILING_DATE)

Evidencia empirica: Gate 0.9 identifico 1 caso en 4,596 grupos
(CIK 0002016827, Q4 2025: NT + HR COMBINATION mismo periodo, mismo
OTHERMANAGER declarado). Se clasifica como N/D.

### 3.5. Directorios cross-periodo

Los directorios del Data Set SEC (`2025Q4/`, `2026Q1/`) estan
organizados por fecha de presentacion, no por periodo objetivo.
Contienen filings con multiples PERIODOFREPORT.

**Regla obligatoria:** filtrar filings por PERIODOFREPORT igual al
periodo de analisis. Prohibido filtrar por directorio fisico.

Evidencia: Gate 0.5A (48 periodos en 2025Q4, 77 en 2026Q1).


---

## 4. Normalizacion de Form13FFileNumber

### 4.1. Representacion externa

Los filers pueden escribir el Form 13F File Number con padding
variable. Evidencia empirica (Gate 0.10):

    COVERPAGE (estructurado SEC):
        100% con prefijo de 3 digitos y sufijo de 5 digitos.

    OTHERMANAGER (declarado por filer):
        99.09% con sufijo de 5 digitos.
        0.61-0.73% con sufijo de 4 digitos (ej. 28-4545).
        0.07-0.10% con sufijo de 3 digitos (ej. 028-694).
        0.03% con sufijo de 6 digitos (ej. 028-064460).
        0.05-0.10% con sufijo de 7 digitos (ej. 028-2813114).

### 4.2. Regla de canonicalizacion

    entrada: <prefijo>-<sufijo>  (ambos con digitos)

    prefijo_canonico = padding(prefijo, 3)   # 28 -> 028
    sufijo_sin_padding = strip_leading_zeros(sufijo)

    si len(sufijo_sin_padding) > 5:
        -> UNRESOLVED (no canonicalizable)

    sufijo_canonico = padding(sufijo_sin_padding, 5)
    formnum_canonico = f"{prefijo_canonico}-{sufijo_canonico}"

Aplicado a los casos reales:

    028-694       -> 028-00694
    28-4545       -> 028-04545
    028-064460    -> 028-64460
    028-2813114   -> UNRESOLVED (7 digitos tras strip, > 5)
    028-26716     -> 028-26716 (sin cambio)

### 4.3. Mapping FormNum -> CIK

    FormNum + period
        -> 0 CIK    = UNRESOLVED
        -> 1 CIK    = IDENTITY_RESOLVED
        -> >1 CIK   = CONFLICT

Construccion:

    COVERPAGE
        JOIN SUBMISSION ON ACCESSION_NUMBER
        Filtro: FORM13FFILENUMBER no-null AND CIK no-null
        Filtro: PERIODOFREPORT == period
        Agrupacion: FormNum normalizado -> CIK

Cardinalidad observada (Gate 0.4-reissue):
- FormNum -> CIK: 1:1 estricta, 0 casos N:1.
- No se exige la direccion inversa CIK -> FormNum.

El contrato exige la propiedad del mapping. La ubicacion fisica
(tabla, parquet, modulo) pertenece a la capa de implementacion.

### 4.4. Anomalias

Un FormNum que no cumple el formato `<digitos>-<digitos>` se
clasifica como UNRESOLVED.

Un FormNum cuyo sufijo tiene >5 digitos tras strip de leading
zeros se clasifica como UNRESOLVED. NO truncar. NO heuristica.

---

## 5. Ausencia de A en OTHERMANAGER

### 5.1. NO_MATCH

    Filing efectivo R4 (13F NOTICE o 13F COMBINATION REPORT)
        + ingestion integra verificada
        + TODAS las filas OTHERMANAGER tienen identidad resuelta
        + A no aparece entre las filas resueltas
            -> NO_MATCH

NO_MATCH requiere completitud probatoria: no basta con que la
ingestion sea integra. Es necesario que no queden filas con
identidad no resuelta.

### 5.2. N/D

    Filing efectivo R4
        + existe al menos una fila OTHERMANAGER con identidad
          no resuelta
        + no existe MATCH positivo para A
            -> N/D

**Nunca N/D -> NO_MATCH.** La ausencia de evidencia positiva de A
en presencia de filas no resueltas no permite concluir ausencia
demostrada.

Evidencia: Gate 0.8 demuestra que en COMBINATION_BASE hay 227 filas
N/D Q4 y 252 Q1 (mayoria concentradas en filer 0001580642-).
Esos filings no pueden producir NO_MATCH para A; producen N/D.

### 5.3. OTHERMANAGER vacio

    Filing efectivo R4
        + ingestion integra verificada
        + OTHERMANAGER ausente o 0 filas
            -> N/D

Un filing clasificado como R4 con la lista de managers vacia no
permite afirmar NO_MATCH. La ausencia de la lista en un filing
que deberia tenerla es una laguna semantica, no evidencia negativa.

Salvo que una regla documental superior establezca expresamente
que el conjunto vacio es valido para ese caso.

---

## 6. DROP_DUP

    DROP_DUP
        =  L3 completo (R1 + R2 + R3 + R4 + R5)
        AND  evidencia consistente (MATCH, no CONFLICT, no N/D)
        AND  sin overlap_unresolved
        AND  sin reporte contradictorio

DROP_DUP es fail-closed: cualquier N/D, CONFLICT o NO_MATCH en
L3 impide la activacion.

Mantiene la filosofia fail-closed del contrato 14.

---

## 7. Diagrama de decision R4

Evaluacion en orden estricto (dictamen #30 seccion 11 + #33):

                        ¿conjunto documental efectivo determinable?
                        │
                        NO ───────────────> N/D
                        │
                        YES
                        ↓
                 ¿CONFLICT en alguna fila
                  con candidate_A(r) == True?
                        │
                  SI ───┴──> CONFLICT
                        │
                        NO
                        ↓
                 ¿existe fila que
                  identifica inequivocamente a A?
                        │
                  SI ───┴──> MATCH
                        │
                        NO
                        ↓
            ¿todas las filas OTHERMANAGER
              tienen identidad resuelta?
                        │
                  NO ───┴──> N/D
                        │
                        YES
                        ↓
                     NO_MATCH

Previo: si OTHERMANAGER ausente/vacio -> N/D.


---

## 8. Evidencia empirica (Gates 0.1-0.10)

### 8.1. Universo R4 (Gate 0.6)

    Q4 2025:
        NOTICE (13F-NT / 13F-NT/A):        1,938
        COMBINATION (13F-HR / 13F-HR/A):     396
        TOTAL R4:                          2,334
    Q1 2026:
        NOTICE:                            1,908
        COMBINATION:                         411
        TOTAL R4:                          2,319

Cobertura OTHERMANAGER en universo R4: 100.00%.

### 8.2. Mapping FormNum -> CIK (Gate 0.4-reissue)

    Q4 2025:
        FormNum unicos:    906
        Resueltos:         888 (98.01%)
        FormNum con >1 CIK:  0
    Q1 2026:
        FormNum unicos:    901
        Resueltos:         887 (98.45%)
        FormNum con >1 CIK:  0

### 8.3. Granularidad de identidad (Gate 0.3 + 0.4)

    Q4 2025 (2,734 filas OTHERMANAGER NT):
        IDENTITY_RESOLVED por CIK:      1,889 (69.09%)
        IDENTITY_RESOLVED por FormNum:    706 (25.82%)
        N/D FormNum no resuelto:           10 (0.37%)
        N/D ni CIK ni FormNum:            129 (4.72%)
        Cobertura:                       94.92%
    Q1 2026 (2,683 filas):
        IDENTITY_RESOLVED por CIK:      1,820 (67.83%)
        IDENTITY_RESOLVED por FormNum:    738 (27.51%)
        N/D FormNum no resuelto:            2 (0.07%)
        N/D ni CIK ni FormNum:            123 (4.58%)
        Cobertura:                       95.34%

### 8.4. Cobertura identidad universo R4 (Gate 0.8)

Agregada por clase (base + amendments):

    Q4 2025 (4,640 filas):
        NOTICE_BASE:            2,696 (94.84%)
        NOTICE_RESTATEMENT:        37 (100%)
        NOTICE_NEW_HOLDINGS:        1 (100%)
        COMBINATION_BASE:       1,799 (86.33%)
        COMBINATION_RESTATEMENT:  105 (100%)
        COMBINATION_NEW_HOLDINGS:   2 (100%)
        TOTAL R4:               4,640 (91.70%)
    Q1 2026 (4,546 filas):
        TOTAL R4:               4,546 (91.31%)

**Nota de precision (dictamen #34 seccion 5):** estos porcentajes
son "cobertura agregada de resolucion de identidad sobre filas
OTHERMANAGER del universo R4 observado". NO son cobertura del
snapshot efectivo despues de aplicar la cadena documental. NO son
cobertura L3 ni match rate.

Drill-down: el N/D de COMBINATION_BASE esta concentrado (82-85%)
en 2 filings del filer 0001580642-. Composicion: NAME 100%,
CRDNUMBER 85-88%, SECFILENUMBER 64-69%, CIK 0%, FORM13FFILENUMBER
~0%. Caso N/D fail-closed ya cubierto.

### 8.5. Amendments NEW HOLDINGS (Gate 0.7)

    Q4 2025: 2 casos -> 2 identicos.
    Q1 2026: 1 caso  -> 1 identico.

Muestra pequena. Regla fail-closed no depende de afirmacion
estadistica universal.

### 8.6. Amendments RESTATEMENT (Gate 0.5)

Caso CIK 0002056909 (Q4 2025):
- Base: OTHERMANAGER declara CIK <NA> / FormNum 028-04685
  (Prospector Partners).
- RESTATEMENT: OTHERMANAGER declara CIK 0001570284 / FormNum
  028-16376 (Gator Capital Management).

El RESTATEMENT sustituye la evidencia OTHERMANAGER anterior.

### 8.7. Multiples filings base (Gate 0.9)

1 caso en 4,596 grupos (CIK 0002016827, Q4 2025). Patron MIXED:
13F-NT (2026-01-02) + 13F-HR COMBINATION (2026-02-20). Mismo
PERIODOFREPORT, mismo OTHERMANAGER declarado.

Resultado: R3 = N/D por regla conservadora IAE.

### 8.8. FormNum representation (Gate 0.10)

    COVERPAGE: 100% sufijo de 5 digitos.
    OTHERMANAGER: 99.09% sufijo de 5 digitos; resto variable
                  (3, 4, 6, 7 digitos).

Regla de canonicalizacion en §4.2.

---

## 9. Preguntas residuales al auditor

Ninguna. Los 4 bloqueos del dictamen #34 estan aplicados y la
evidencia complementaria (Gate 0.10) esta incorporada.

---

## 10. Lo que NO se ha hecho

- No se ha descargado ningun XML de EDGAR.
- No se ha creado `xml_parser/` ni dependencia externa.
- No se ha modificado `reporting_dedup.py`.
- No se ha modificado `NIPC_CONTRATOS_SEMANTICOS_v1.md`.
- No se ha activado `DROP_DUP`.

---

## 11. Referencias

    | Documento                                        | Rol                |
    |--------------------------------------------------|--------------------|
    | iae/DICTAMENES.md #28-#34                        | Dictamenes P66     |
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 14   | Contrato L3        |
    | iae/P64_P65_EXPEDIENTE.md                        | Ciclo P65          |
    | iae/evidence/p66_gate04_reissue_period/          | Mapping FormNum    |
    | iae/evidence/p66_gate06_combination_probe/       | Universo R4        |
    | iae/evidence/p66_gate07_newholdings_probe/       | NEW HOLDINGS       |
    | iae/evidence/p66_gate08_full_r4_coverage/        | Cobertura R4       |
    | iae/evidence/p66_gate09_multiple_base/           | Multiples bases    |
    | iae/evidence/p66_gate10_formnum_representation/  | FormNum longitudes |
    | SEC Form 13F (documentacion oficial)             | Fuente normativa   |

---

Fin de la propuesta v4. Consolidacion final de los dictamenes
#28 a #34. Pendiente de dictamen contractual definitivo sobre el
texto exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.
