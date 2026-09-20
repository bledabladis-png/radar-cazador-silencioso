# IAE - P66 L3 Reformulacion propuesta (v2)

**Objeto:** propuesta de reformulacion del contrato 14.3 requisito 4,
corregida tras dictamen #27 (P66 GO CONDICIONADO).

**Estado:** PROPUESTA v2. Pendiente de dictamen final sobre el texto
contractual exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.

**Origen:** Gate 0.1 + Gate 0.2 (2026-09-20) tras NO-GO de P66.
Correcciones aplicadas segun dictamen externo #27.

**HEAD al redactar:** 551ea52 (o posterior).
**Dictamen mas reciente:** #28 (P66 v2-bis NO-GO contractual, 2026-09-21).
Estado actual: 6 bloqueos del dictamen #28, 5 resueltos; #4 en
invariante del mapping (seccion 10).

**Historial:** v1 en commit 6b610b3. v2 incorpora las 8 correcciones
obligatorias del dictamen #27.

---

## 0. Contexto

El informe P66 inicial (XML crudo 13F-NT) fue NO-GO. Dos Gates
demostraron que la evidencia esta en la tabla OTHERMANAGER del Data
Set SEC, sin XML.

El dictamen #27 emitio GO CONDICIONADO sobre la propuesta v1. Las
correcciones son de precision contractual, no de arquitectura.

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

## 2. Propuesta reformulada (texto base + correcciones #28)

    L3(A, B, S, period) == True sii:
      1. existe filing efectivo de A con linea L sobre security S;
      2. Column 7(L) referencia a B mediante la resolucion
         estructurada de Other Included Managers (OTHERMANAGER2);
      3. existe filing efectivo de B para el mismo PERIODOFREPORT;
      4. B declara explicitamente que A reporta por B mediante una
         fila de OTHERMANAGER perteneciente al filing efectivo de B,
         identificando inequivocamente a A:

         a) mediante CIK; o

         b) cuando CIK no este disponible, mediante Form 13F File
            Number cuya resolucion a CIK sea inequivoca dentro del
            mismo PERIODOFREPORT.

         No se permite matching por nombre.

         La evidencia R4 es aplicable cuando B presenta:

         - 13F NOTICE; o
         - 13F COMBINATION REPORT.

         En presencia de amendments, la evidencia OTHERMANAGER se
         determina mediante la cadena documental efectiva del
         periodo. Un RESTATEMENT sustituye la evidencia anterior.
         Un NEW HOLDINGS solo conserva la evidencia R4 cuando su
         OTHERMANAGER es consistente con el estado efectivo previo;
         cualquier cambio no resoluble produce N/D o CONFLICT.
      5. no existe evidencia contradictoria ni evidencia de
         reporting partition/overlap no resuelto.

**Nota terminologica (correccion #1 del dictamen):** no se habla de
"evidencia estructurada cruzada entre filings". La evidencia es "estructurada cruzada
entre el filing de A y el filing de B". La SEC no exige simetria
reciproca.

---

## 3. Estados formales

    MATCH     evidencia positiva inequivoca:
              - CIK coincide exactamente, o
              - Form13FFileNumber resuelve inequivocamente a A, o
              - ambos coinciden.

    NO_MATCH  filing efectivo B inspeccionado, ingestion integra
              verificada, A no aparece en OTHERMANAGER.

    N/D       no se puede determinar:
              - filing efectivo B no inspeccionable (ausente,
                corrupto, periodo distinto).
              - fallo de ingestion/extraccion/lineage.
              - identificadores ambiguos sin tabla de resolucion.

    CONFLICT  identificadores contradictorios:
              - CIK presente y Form13FFileNumber presente, ambos
                poblados, apuntan a entidades distintas.

**Reglas duras:**

    N/D != NO_MATCH       (correccion #6)
    CONFLICT != NO_MATCH  (correccion #5)
    N/D y CONFLICT son fail-closed respecto de DROP_DUP.

---

## 4. Tratamiento de identidad (correccion #4)

Prioridad:

    1. CIK_A == OTHERMANAGER.CIK           -> MATCH
       aunque FORM13FFILENUMBER este vacio.

    2. FORM13FFILENUMBER resuelve a CIK_A
       via tabla canonica inequivoca         -> MATCH
       solo si CIK esta vacio.

    3. Ambos presentes y coherentes         -> MATCH.

    4. Ambos presentes y contradictorios    -> CONFLICT.

    5. Ninguno permite identificar A        -> N/D.

**NO se exige la presencia simultanea de ambos campos.** El dictamen
explicita que ambos son nullable en OTHERMANAGER y en los filings
reales aparecen casos con uno u otro.

---

## 5. Tratamiento de amendments (correccion #3, Gates 0.5-0.7)

### 5.1. Hallazgo critico: directorios cross-periodo

Los directorios del Data Set SEC (`2025Q4/`, `2026Q1/`) contienen
filings con multiples `PERIODOFREPORT`. NO son homogeneos.

Evidencia Gate 0.5:
- Directorio 2025Q4: 48 periodos distintos (desde 2013-12-31).
- Directorio 2026Q1: 77 periodos distintos (desde 2008-03-31).
- NT contaminantes en cada directorio: 108 / 138.

**Regla obligatoria:** filtrar filings por `PERIODOFREPORT` igual
al periodo de analisis. Prohibido filtrar por directorio fisico.

### 5.2. Algoritmo de filing efectivo

    R3(B, period) := existe filing efectivo de B
                     para ese PERIODOFREPORT

    "Filing efectivo":
      1. Filtrar filings de CIK=B con PERIODOFREPORT=period
         y (SUBMISSIONTYPE in {13F-NT, 13F-NT/A}
             con REPORTTYPE = 13F NOTICE
             O
             SUBMISSIONTYPE in {13F-HR, 13F-HR/A}
             con REPORTTYPE = 13F COMBINATION REPORT).

      2. Ordenar filings por AMENDMENTNO ascendente
         (base AMENDMENTNO=null tratado como orden 0).

      3. Aplicar semantica SEC (texto afinado por dictamen #29):

         Para R4, la evidencia OTHERMANAGER se considera valida
         unicamente cuando su estado en la cadena de amendments sea
         inequivoco:

         - RESTATEMENT: sustituye la evidencia OTHERMANAGER anterior.
                        Evidencia directa: Gate 0.5, caso CIK 0002056909.
         - NEW HOLDINGS: se acepta sin transformacion cuando la
                         relacion OTHERMANAGER coincide con el estado
                         efectivo anterior. Evidencia: Gate 0.7,
                         3/3 casos identicos.
         - NEW HOLDINGS con cambio de OTHERMANAGER: N/D o CONFLICT.
                         NO asumir union ni sustitucion sin evidencia.

      4. R4 se evalua sobre el snapshot efectivo, NO sobre el base.

    Salvaguardas:
      - Si no existe filing en el periodo: R3 = False.
      - Trazabilidad: registrar cadena de amendments aplicados.

### 5.3. Caso RESTATEMENT que cambia OTHERMANAGER

Evidencia directa (Gate 0.5): CIK `0002056909`.
- Base: OTHERMANAGER declara CIK `<NA>` / FormNum `028-04685`
  (Prospector Partners).
- RESTATEMENT: OTHERMANAGER declara CIK `0001570284` /
  FormNum `028-16376` (Gator Capital Management).

**El RESTATEMENT sustituye el contenido de OTHERMANAGER.** Un
analisis de R4 sobre el base produciria un falso positivo.

### 5.4. Estadistica del universo

Filtrando por PERIODOFREPORT correcto:

| Periodo | NT base | NT/A RESTATEMENT | NT/A NEW HOLDINGS |
|---------|--------:|-----------------:|------------------:|
| 2025-12-31 | 1,900 | 37 | 1 |
| 2026-03-31 | 1,907 | 0 | 1 |

Casos con OTHERMANAGER distinto entre base y amend: 1 (CIK 0002056909).

### 5.5. Consecuencia

La formalizacion anterior (v2 seccion 5) era incompleta: definia
la semantica SEC sin filtrar por PERIODOFREPORT. La version v2-bis
incluye el filtro obligatorio.

---

## 6. Requisito 3 explicito (correccion #2)

R3 NO se absorbe en R4. Se mantiene como control independiente:

    R3 = existe filing efectivo de B
         con el mismo PERIODOFREPORT que A
         y SUBMISSIONTYPE valido.

R4 presupone R3. R4 no puede sustituirlo: la existencia de una fila
en OTHERMANAGER[ACCESSION=B] presupone B, pero no valida por si sola
que B sea el filing efectivo del periodo.

---

## 7. NT sin A en OTHERMANAGER (correccion #6)

    NT efectivo, ingestion integra verificada:
        -> NO_MATCH

    NT efectivo, ingestion NO verificada (fallo de descarga,
       parseo, o lineage):
        -> N/D

    Nunca N/D -> NO_MATCH.

Aplica tambien a NT con 0 filas de OTHERMANAGER (si existieran):
NO_MATCH, no N/D.

---

## 8. DROP_DUP (correccion #7)

DROP_DUP NO se autoriza por R4 aislado.

    DROP_DUP
        =  L3 completo (R1 + R2 + R3 + R4 + R5)
        AND  evidencia consistente (MATCH, no CONFLICT, no N/D)
        AND  sin overlap_unresolved
        AND  sin reporte contradictorio

Mantiene la filosofia fail-closed del contrato 14.

---

## 9. Evidencia empirica (correccion #8: granularidad)

Gate 0.1 midio cobertura al 100% de NT sobre OTHERMANAGER. El
dictamen exige conservar granularidad completa:

### 9.1. Cobertura por tipo (Gate 0.1)

| Trimestre | Tipo | Filings | Con filas OTHERMANAGER | Cobertura |
|-----------|------|--------:|-----------------------:|----------:|
| 2025Q4    | NT   | 2,008   | 2,008                  | 100.0%    |
| 2025Q4    | HR   | 9,364   | 419                    | 4.5%      |
| 2026Q1    | NT   | 2,045   | 2,045                  | 100.0%    |
| 2026Q1    | HR   | 9,716   | 441                    | 4.5%      |

### 9.2. Granularidad de identidad (Gate 0.3, EJECUTADO)

Medicion ejecutada 2026-09-20. Evidencia en
`iae/evidence/p66_gate03_granularidad/`.

| Categoria | Q4 filas | Q4 % | Q1 filas | Q1 % |
|-----------|---------:|-----:|---------:|-----:|
| con CIK poblado      | 1,955 | 69.72% | 1,948 | 68.91% |
| con FormNum poblado  | 2,567 | 91.55% | 2,596 | 91.83% |
| con ambos poblados   | 1,815 | 64.73% | 1,820 | 64.38% |
| solo CIK             |   140 |  4.99% |   128 |  4.53% |
| solo FormNum         |   752 | 26.82% |   776 | 27.45% |
| ninguno              |    97 |  3.46% |   103 |  3.64% |

Consistencia CIK <-> FormNum (sobre filas con ambos):

| Trimestre | CIK con 1 FormNum | CIK con >1 | FormNum con 1 CIK | FormNum con >1 |
|-----------|------------------:|-----------:|------------------:|---------------:|
| 2025Q4    | 716 / 716         | 0          | 716 / 716         | 0              |
| 2026Q1    | 696 / 696         | 0          | 696 / 696         | 0              |

**Cardinalidad (correccion #4 del dictamen #28):** la invariante
contractual es unidireccional:

    FormNum -> 0 CIK    = UNRESOLVED
    FormNum -> 1 CIK    = IDENTITY_RESOLVED
    FormNum -> >1 CIK   = CONFLICT

Siempre dentro del scope temporal del periodo. NO se exige la
invariante inversa (CIK -> 1 FormNum). Un CIK podria, por
circunstancias historicas o administrativas, aparecer asociado a
mas de un FormNum a lo largo del universo temporal.

### 9.2-bis. Hallazgo de normalizacion FormNum

Del 91.55-91.83% con FormNum poblado, el 93.42-93.34% usa prefijo
`028-`. El resto (6.58-6.66%) usa prefijo `28-` (sin cero a la
izquierda). **Todas las filas afectadas tienen CIK=`<NA>`.**

Normalizacion trivial propuesta: `28-XXXXX -> 028-XXXXX` con
zero-padding a 3 digitos.

### 9.3. Caso Vanguard confirmado (Gate 0.2)

Q4 2025: multiples NT de filiales declaran al parent con FormNum 028-06408.
Q1 2026: parent 13F-NT (ACC=0000102909-26-002707) declara 10 managers,
incluyendo las filiales CIK 0002100119 y 0002100121.

Evidencia estructurada cruzada documental directa en OTHERMANAGER.

### 9.4. Column 7 no apunta a OTHERMANAGER

INFOTABLE.OTHERMANAGER (Column 7) matchea OTHERMANAGER2.SEQUENCENUMBER
(90% aprox), no OTHERMANAGER_SK. Requisito 2 y requisito 4 usan
tablas distintas. El dictamen confirma la separacion R2/R4 como
correcta.

---

### 9.5. Tabla canonica Form13FFileNumber -> CIK (Gate 0.4-reissue)

**Fuente corregida (dictamen P66 v2, Bloqueo A):** la tabla canonica
se construye vía `COVERPAGE` + `SUBMISSION`, no vía `OTHERMANAGER`.

    COVERPAGE  (ACCESSION_NUMBER, FORM13FFILENUMBER, ...)
        JOIN
    SUBMISSION (ACCESSION_NUMBER, CIK)
        ON ACCESSION_NUMBER
    Agrupacion: FORM13FFILENUMBER (normalizado) -> CIK

**Normalizacion:** `28-XXXXX -> 028-XXXXX` y `028-XXXXX -> 028-XXXXX`.
El prefijo se normaliza a 3 digitos y el sufijo a 5.

**Cobertura del mapping (Gate 0.4-reissue, filtro PERIODOFREPORT):**

| Trimestre | FormNum unicos | Resueltos | No resueltos | Cobertura |
|-----------|---------------:|----------:|-------------:|----------:|
| 2025Q4    | 906            | 888       | 18           | 98.01%    |
| 2026Q1    | 901            | 887       | 14           | 98.45%    |

**Cardinalidad:** 1:1 en direccion FormNum -> CIK (0 casos N:1).
No se exige 1:1 en direccion inversa (correccion #4 del dictamen #28).

**Impacto en filas (Gate 0.4-reissue):**

| Categoria | Q4 filas | Q4 % | Q1 filas | Q1 % |
|-----------|---------:|-----:|---------:|-----:|
| A. IDENTITY_RESOLVED por CIK | 1,889 | 69.09% | 1,820 | 67.83% |
| B. IDENTITY_RESOLVED por FormNum | 706 | 25.82% | 738 | 27.51% |
| C. N/D FormNum no resuelto | 10 | 0.37% | 2 | 0.07% |
| D. N/D ni CIK ni FormNum | 129 | 4.72% | 123 | 4.58% |
| **Cobertura de resolucion de identidad** | **94.92%** | | **95.34%** | |
| **N/D total** | **5.08%** | | **4.66%** | |

**Nota sobre el delta:** el Gate 0.4 original (sin filtro de periodo)
reportaba 96.01% / 96.18%. El reissue corrige la contaminacion
cross-periodo: delta −1.09 pp (Q4) y −0.84 pp (Q1).

**Los porcentajes son cobertura de resolucion de identidad de las
filas OTHERMANAGER. NO son cobertura L3 ni "match rate".** L3
requiere ademas R1 + R2 + R3 + R5 y la comparacion CIK_resuelto == CIK_A.

**Bloqueo #3 del dictamen #28: RESUELTO.**

**Bloqueo #5 (sin truncar) aplicado:** normalizacion estricta
`28-XXXXX` -> `028-XXXXX` con 5 digitos exactos. El caso
`028-2813114` (7 digitos) queda como N/D automaticamente.

**Bloqueo #6 (nomenclatura) aplicado:** IDENTITY_RESOLVED reemplaza
MATCH en toda esta seccion.

**Anomalia detectada:** el FormNum `028-2813114` (7 digitos) aparece
en N/D por FormNum no resuelto. Volumen despreciable (4+2=6 filas).
Documentado como hallazgo colateral en
`iae/evidence/p66_gate04_mapping/README.md` seccion 2.5.

### 9.7. Cobertura de identidad universo R4 completo (Gate 0.8)

**Origen:** dictamen #29. Gate 0.4-reissue midio solo
`OTHERMANAGER(NT)`. El universo contractual de R4 es mas amplio:
NOTICE + COMBINATION + amendments.

**Cobertura por clase:**

| Clase | Q4 Filings | Q4 Filas | Q4 Cobertura | Q1 Filings | Q1 Filas | Q1 Cobertura |
|-------|----------:|---------:|------------:|----------:|---------:|------------:|
| NOTICE_BASE | 1,900 | 2,696 | 94.84% | 1,907 | 2,682 | 95.34% |
| NOTICE_RESTATEMENT | 37 | 37 | 100% | 0 | 0 | — |
| NOTICE_NEW_HOLDINGS | 1 | 1 | 100% | 1 | 1 | 100% |
| COMBINATION_BASE | 388 | 1,799 | 86.33% | 402 | 1,841 | 85.33% |
| COMBINATION_RESTATEMENT | 7 | 105 | 100% | 9 | 22 | 100% |
| COMBINATION_NEW_HOLDINGS | 1 | 2 | 100% | 0 | 0 | — |
| **TOTAL R4** | **2,334** | **4,640** | **91.70%** | **2,319** | **4,546** | **91.31%** |

**Delta vs NOTICE-solo:** −3.21 pp (Q4), −4.03 pp (Q1).

**Drill-down del N/D en COMBINATION_BASE:**

- 82-85% del N/D_null se concentra en 2 filings del mismo filer
  (`0001580642-`).
- Composicion de las filas N/D_null: NAME 100%, CRDNUMBER 85-88%,
  SECFILENUMBER 64-69%, CIK 0%, FORM13FFILENUMBER ~0%.
- Patron: filer declara sub-managers sin CIK ni FormNum.
- Es exactamente el caso N/D que el contrato prohibe resolver por
  nombre.

**Conclusion:** Gate 0.8 PASS. El delta de −3/−4 pp no introduce
nueva clase de ambiguedad. Es el mismo N/D fail-closed, concentrado
en un filer concreto.

**Cobertura de identidad efectiva del universo contractual R4:**
**91.70% / 91.31%**.

Nota: los porcentajes son cobertura de resolucion de identidad de
las filas OTHERMANAGER del universo R4. NO son cobertura L3 ni
"match rate".

## 10. Invariante del mapping (bloqueo #4)

### 10.1. Direccion contractual

La unica direccion contractual obligatoria es:

    FormNum -> CIK

NO se establece como invariante la direccion inversa:

    CIK -> FormNum

Un CIK podria aparecer asociado a mas de un FormNum a lo largo del
universo temporal. El sistema no debe explotar si esto ocurre: la
resolucion es siempre por FormNum como clave.

### 10.2. Construccion del mapping

    period
    normalized_form13f_filenumber    (clave)
    cik                              (valor)
    source_accessions                (lineage)
    resolution_status                (UNRESOLVED | IDENTITY_RESOLVED | CONFLICT)

Construido por `PERIODOFREPORT`, no por directorio fisico.

### 10.3. Uso

    Dado un FormNum F (normalizado) en una fila OTHERMANAGER:

      - Si F no resuelve en el mapping del periodo: N/D.
      - Si F resuelve a 1 CIK: IDENTITY_RESOLVED (comparar con CIK_A
        para decidir R4).
      - Si F resuelve a >1 CIK: CONFLICT.

    La direccion inversa (dado CIK, buscar FormNum) no se usa
    para R4.

### 10.4. Bloqueos resueltos por este apartado

    #4 Mapping unidireccional: RESUELTO.

Dictamen #29 (2026-09-21): APROBADO explicitamente. La invariante
util para R4 es FormNum -> CIK, no la direccion inversa.

---

## 11. Preguntas residuales al auditor

Una vez aplicadas las 8 correcciones, solo quedan por definir:

    Q1. La formalizacion exacta de amendments (seccion 5) debe
        implementarse antes del GO contractual o puede diferirse
        a la implementacion en reporting_dedup.py?

    Q2. La tabla canonica Form13FFileNumber -> CIK existe en el
        sistema o debe construirse? Donde debe vivir?

    Q3. El enum MATCH/NO_MATCH/N/D/CONFLICT debe anadirse al
        contrato 14.3 como definicion formal, o se mantiene como
        especificacion operativa?

---

## 12. Lo que NO se ha hecho

- NO se ha descargado ningun XML de EDGAR.
- NO se ha creado xml_parser/.
- NO se ha modificado reporting_dedup.py.
- NO se ha reformulado 14.3 efectivamente.
- NO se ha activado DROP_DUP.
- NO se ha tocado NIPC_CONTRATOS_SEMANTICOS_v1.md.

---

## 13. Referencias

    | Documento                                      | Rol                |
    |------------------------------------------------|--------------------|
    | iae/DICTAMENES.md #27                          | Dictamen P66       |
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 14 | Contrato L3        |
    | iae/P64_P65_EXPEDIENTE.md                      | Ciclo P65          |
    | iae/P66_INFORME_HALLAZGO.md                    | Informe NO-GO      |
    | iae/DICTAMENES.md #26                          | Dictamen P65 v3    |
    | SEC Form 13F Data Sets (documentacion)         | Fuente oficial     |

---

Fin de la propuesta v2. Pendiente dictamen final del auditor sobre
el texto contractual exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.
