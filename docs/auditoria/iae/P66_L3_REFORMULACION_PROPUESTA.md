# IAE - P66 L3 Reformulacion propuesta (v7-bis)

**Estado:** v7-bis ronda final. Aplicadas las correcciones de los
dictamenes #37 y #38 (flujo unico R3, §3.4 como nota, trazabilidad,
nota de evidencia historica) + §0.5 Clausula de cierre contractual.
Pendiente de dictamen GO sobre esta version.

**Trazabilidad de dictamenes:**

    #28  corregido
    #29  corregido
    #30  corregido
    #31  corregido
    #32  corregido
    #33  corregido
    #34  corregido
    #35  corregido
    #36  corregido
    #37  corregido
    #38  §3.4 como nota + 2 correcciones editoriales (v7-bis ronda final)

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
resuelta cuando cumple:

    IDENTITY_RESOLVED:
        CIK(r) poblado valido
        AND
        (si FormNum(r) es normalizable y resoluble,
         resuelve al mismo CIK)
      OR
        CIK(r) ausente
        AND
        FormNum(r) resuelve a exactamente 1 CIK

**INCONSISTENT:** una fila `r` de OTHERMANAGER es INCONSISTENT cuando:

    CIK(r) presente
    AND
    FormNum(r) presente, normalizable, y resuelve a un CIK distinto

INCONSISTENT conduce a CONFLICT en el resultado de R4.

**Jerarquia:** CONFLICT > IDENTITY_RESOLVED > N/D segun corresponda.

Esta definicion corrige la contradiccion interna de la v4 (una fila
con CIK=A y FormNum->C habria sido simultaneamente IDENTITY_RESOLVED
y CONFLICT bajo la definicion anterior).

**inequivocamente:** sin matching por nombre, sin heuristica, sin
inferencia. Solo CIK directo o FormNum con resolucion univoca.

### 0.4. Historial de versiones

| Version | Contenido |
|---------|-----------|
| v1 | commit 6b610b3. Propuesta inicial + primeras correcciones. |
| v2 | Correcciones #28 aplicadas (fail-closed, nomenclatura). |
| v3 | Correcciones #29 a #33 aplicadas (R4 completo, mapping, tri-state). |
| v4 | Consolidacion. Correcciones #34 aplicadas + reescritura canonica. |
| v5 | Consolidacion. Correcciones #35 aplicadas. |
| v6 | Correcciones #36 aplicadas. |
| v7 | Correcciones #37 aplicadas + §0.5 Clausula de cierre. |
| **v7-bis** | **Ronda final. §3.4 como nota (dictamen #38) + trazabilidad + rotulacion evidencia historica.** |

### 0.5. Clausula de cierre contractual

Este documento constituye el texto contractual candidato definitivo
de P66.

Una vez emitido dictamen GO sobre esta version, el contrato se
trasladara a NIPC_CONTRATOS_SEMANTICOS_v1.md sin nueva iteracion
ordinaria.

Cualquier observacion posterior debera clasificarse expresamente
como:

    A) BLOQUEO MATERIAL
       Cuando la correccion pueda cambiar el resultado contractual,
       modificar una condicion necesaria de L3/R3/R4/R5, alterar el
       criterio de evidencia, introducir una nueva semantica o dejar
       una ambiguedad que permita resultados distintos entre
       implementaciones conformes al contrato.

       -> requiere nueva version y nuevo dictamen.

    B) RECOMENDACION DIFERIBLE
       Cuando la observacion sea exclusivamente editorial,
       documental, de legibilidad o de implementacion y no pueda
       cambiar el resultado contractual definido por este documento.

       -> no bloquea el GO.

El mero hecho de que una observacion sea posterior al GO no
determina por si mismo su materialidad. La clasificacion debera
justificarse por su impacto sobre el contrato.

**Condicion de cierre:** el GO contractual de v7 autoriza el
traslado del texto aprobado a NIPC_CONTRATOS_SEMANTICOS_v1.md.
Cualquier observacion posterior tendra que demostrar impacto
material conforme a esta seccion para reabrir P66.

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

**Criterio contractual (dictamen #36, bloqueo 1):** la clasificacion
se basa en `SUBMISSIONTYPE` + `REPORTTYPE`, NO en `ISAMENDMENT`.

    BASE:
        SUBMISSIONTYPE in {13F-NT, 13F-HR}
        AND REPORTTYPE in universo R4 correspondiente

    AMENDMENT:
        SUBMISSIONTYPE in {13F-NT/A, 13F-HR/A}
        AND REPORTTYPE in universo R4 correspondiente

Definiciones del universo R4:

    NOTICE family:
        SUBMISSIONTYPE = 13F-NT / 13F-NT/A
        AND REPORTTYPE = 13F NOTICE

    COMBINATION family:
        SUBMISSIONTYPE = 13F-HR / 13F-HR/A
        AND REPORTTYPE = 13F COMBINATION REPORT

**Regla de coherencia:** si `SUBMISSIONTYPE` / `ISAMENDMENT` se
contradicen o no permiten clasificar inequivocamente el filing:
R3 = N/D.

Consecuencia: `ISAMENDMENT = <NA>` en un `13F-NT` (caso Gate 0.9)
se clasifica correctamente como BASE por el `SUBMISSIONTYPE`, sin
depender del campo nullable.

**Nunca usar `AMENDMENTNO = null` ni `ISAMENDMENT != Y` como prueba
contractual de que un filing es base.** La SEC distingue el
amendment por el tipo de filing (`13F-NT/A`, `13F-HR/A`) y exige
numero y tipo de amendment en esos filings.

Si `SUBMISSIONTYPE == 13F-NT/A` (o `13F-HR/A`) y `AMENDMENTNO` no es
determinable:
    N/D.

### 3.2. Flujo unico de determinacion de R3

**Este es el unico algoritmo contractual de determinacion de R3.**
No existen reglas solapadas entre secciones. Los pasos se ejecutan
en orden estricto.

**PASO 1. Identificar todos los filings R4 del CIK + PERIODOFREPORT.**

    R4_filings(B, period) = filings de B con:
        mismo CIK
        AND mismo PERIODOFREPORT
        AND (NOTICE family o COMBINATION family)

    NOTICE family:
        SUBMISSIONTYPE in {13F-NT, 13F-NT/A}
        AND REPORTTYPE = 13F NOTICE

    COMBINATION family:
        SUBMISSIONTYPE in {13F-HR, 13F-HR/A}
        AND REPORTTYPE = 13F COMBINATION REPORT

**PASO 2. Clasificar cada filing por familia y por rol.**

    BASE:
        SUBMISSIONTYPE in {13F-NT, 13F-HR}

    AMENDMENT:
        SUBMISSIONTYPE in {13F-NT/A, 13F-HR/A}

`ISAMENDMENT` es SOLO control de coherencia, no criterio de
seleccion. Si `ISAMENDMENT` contradice `SUBMISSIONTYPE`, R3 = N/D.

**PASO 3. Detectar pluralidad de familias documentales.**

    Si entre BASE + AMENDMENT hay representacion de MAS DE UNA
    familia documental (NOTICE y COMBINATION simultaneamente):
        R3 = N/D.

    Esta deteccion es GLOBAL, sobre el conjunto R4 completo del
    CIK + PERIODOFREPORT. No se evalua por familia aislada.
    El caso Gate 0.9 (CIK 0002016827: NT + HR COMBINATION mismo
    periodo) se captura aqui.

**PASO 4. Contar bases y amendments dentro de la familia unica.**

    BASE = 0 AND AMENDMENT = 0
        -> R3 = FALSE (ausencia documental demostrada).

    BASE = 0 AND AMENDMENT >= 1
        -> R3 = N/D.
        Un amendment no constituye evidencia de ausencia del
        filing antecedente. No se puede afirmar FALSE sin base.

    BASE = 1
        -> continuar con la validacion de cadena (§3.2-bis).

    BASE > 1
        -> R3 = N/D.
        Prohibido: primero encontrado, ultimo encontrado,
        MAX(ACCESSION), MAX(FILING_DATE).

### 3.2-bis. Validacion de la cadena (familia unica ya establecida)

Esta subseccion aplica SOLO despues de §3.2 PASO 3 (familia unica
confirmada) y §3.2 PASO 4 (BASE = 1). No introduce reglas solapadas
con §3.2.

**Validacion de cada amendment aplicable a la base:**

    AMENDMENT valido:
        SUBMISSIONTYPE in {13F-NT/A, 13F-HR/A}
        AND REPORTTYPE consistente con la familia de la base
        AND AMENDMENTNO entero en 1..99
        AND AMENDMENTTYPE in {RESTATEMENT, NEW_HOLDINGS}

    Si existen amendments aplicables a la base:

        - Numeros unicos.
        - Secuencia sin huecos desde 1 hasta N.
          (Regla conservadora contractual IAE. La SEC exige
          numeracion 1..99 y orden, pero no que la secuencia
          este fisicamente completa. Se conserva la regla por
          prudencia fail-closed.)
        - Tipos reconocibles.

    Si cualquiera falla:
        -> R3 = N/D.

Casos que deben producir N/D:
    - Huecos: base + amendment 1 + amendment 3 (falta el 2).
    - Duplicados: base + amendment 1 + amendment 1.
    - Amendment sin numero determinable.
    - Amendment con tipo desconocido.

### 3.3. Semantica de amendments

    RESTATEMENT:
        Sustituye la evidencia OTHERMANAGER anterior.
        Evidencia directa: Gate 0.5, caso CIK 0002056909
        (base Prospector Partners -> restatement Gator Capital).

    NEW HOLDINGS:
        Conserva la evidencia OTHERMANAGER cuando es consistente
        con el estado efectivo previo. Evidencia: Gate 0.7,
        3/3 casos identicos.

        Definicion operacional de "consistente" (dictamen #35
        bloqueo 5 + dictamen #36 bloqueo 3):

        Preliminar: determinar si ambos estados son construibles.

            OTHERMANAGER_state(filing) =
                conjunto de identidades de managers obtenido tras:
                    - resolver CIK
                    - resolver FormNum
                    - canonicalizar FormNum (seccion 4.2)

        **Regla de determinabilidad (bloqueo 3):**

            Si alguna fila del filing tiene:
                N/D
                o CONFLICT
                    -> OTHERMANAGER_state NO es determinable.
                    -> NEW HOLDINGS = N/D.

        Es decir: la comparacion de conjuntos solo es posible si
        AMBOS estados (previo y amendment) son completamente
        determinables. Cualquier N/D o CONFLICT en cualquiera de
        los dos impide esa comparacion.

        Solamente despues, cuando ambos son determinables:

            NEW HOLDINGS es consistente sii
                conjunto normalizado(amendment)
                ==
                conjunto normalizado(estado efectivo previo)

        La comparacion es de igualdad exacta de conjunto. Ignora
        orden de filas y diferencias puramente representacionales
        (padding, espacios).

        Cambio de conjunto -> N/D.
        NO asumir union. NO asumir sustitucion.

### 3.4. Multiples bases heterogeneas (nota explicativa)

**Nota historica/evidencial. No constituye regla decisional
independiente.** La determinacion de R3 se realiza exclusivamente
por el flujo unico de §3.2.

**Regla absoluta (de §3.2 PASO 4):**

    BASE > 1 para mismo CIK + PERIODOFREPORT
        -> R3 = N/D.

**CONFLICT no puede producirse en R3 por pluralidad documental.**
CONFLICT pertenece exclusivamente a la evaluacion de identidad de
A dentro de R4 (§2.3). Mezclar ambos conceptos introduce
interpretacion subjetiva y se prohibe.

PROHIBIDO como criterio de resolucion:
    - primero encontrado
    - ultimo encontrado
    - MAX(ACCESSION_NUMBER)
    - MAX(FILING_DATE)

Evidencia empirica: Gate 0.9 identifico 1 caso en 4,596 grupos
(CIK 0002016827, Q4 2025: NT + HR COMBINATION mismo periodo, mismo
OTHERMANAGER declarado). La decision R3 = N/D se produce en §3.2
PASO 3 (pluralidad de familias), no en §3.4.

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
variable. La regla de canonicalizacion que sigue es una
**canonicalizacion IAE basada en correspondencia observada con
COVERPAGE**, no una afirmacion sobre un formato SEC obligatorio
de cinco digitos.

Evidencia empirica (Gate 0.10):

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

    Regla de prefijo (dictamen #35, bloqueo 4):

        prefijo in {"28", "028"}   -> prefijo_canonico = "028"
        cualquier otro prefijo      -> UNRESOLVED

        NO usar padding() generico. La evidencia Gate 0.10 solo
        observo prefijos "28" y "028". Otros prefijos (2, 02, 002)
        no estan demostrados y deben rechazarse.

    Regla de sufijo:

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

## 9. Estado de trazabilidad y notas residuales

### 9.1. Trazabilidad de dictamenes

Estado actual del documento v7-bis:

    #28  corregido
    #29  corregido
    #30  corregido
    #31  corregido
    #32  corregido
    #33  corregido
    #34  corregido
    #35  corregido
    #36  corregido
    #37  corregido
    #38  §3.4 reformulada como nota + 2 correcciones editoriales
         + nota de rotulacion de evidencia historica

### 9.2. Nota sobre evidencia historica de Gates

Los README.md de los directorios `iae/evidence/p66_gate0*/` conservan
la metodologia tal como se ejecuto cada Gate en su momento. Algunos
incluyen reglas que han sido posteriormente superadas por el contrato:

- `p66_gate04_reissue_period/`: normalizacion estricta de 5 digitos.
- `p66_gate09_multiple_base/`: uso de `ISAMENDMENT != Y` como
  identificador de base.

**Estos READMEs NO se modifican retroactivamente.** Son evidencia
historica de como se ejecuto cada Gate. Su contenido refleja la
metodologia vigente en ese momento, no la semantica contractual
actual.

La semantica contractual vigente es la de este documento (v7-bis).
En caso de conflicto, prevalece siempre este documento.

Los HASHES.txt de cada directorio de evidencia permanecen validos
porque los READMEs no se han modificado.

### 9.3. Preguntas residuales al auditor

Ninguna. Los 4 bloqueos del dictamen #34 estan aplicados, las
correcciones de #35, #36, #37 estan integradas, y las correcciones
del #38 (bloqueo material §3.4 + 2 editoriales) estan aplicadas.

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

Fin de la propuesta v7-bis. Consolidacion final de los dictamenes
#28 a #34. Pendiente de dictamen contractual definitivo sobre el
texto exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.
