# IAE - P66 L3 Reformulacion propuesta (v2)

**Objeto:** propuesta de reformulacion del contrato 14.3 requisito 4,
corregida tras dictamen #27 (P66 GO CONDICIONADO).

**Estado:** PROPUESTA v2. Pendiente de dictamen final sobre el texto
contractual exacto antes de tocar NIPC_CONTRATOS_SEMANTICOS_v1.md.

**Origen:** Gate 0.1 + Gate 0.2 (2026-09-20) tras NO-GO de P66.
Correcciones aplicadas segun dictamen externo #27.

**HEAD al redactar:** 3255f77 (o posterior).

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

## 2. Propuesta reformulada (texto exacto del auditor #27)

    L3(A, B, S, period) == True sii:
      1. existe filing efectivo de A con linea L sobre security S;
      2. Column 7(L) referencia a B mediante la resolucion
         estructurada de Other Included Managers (OTHERMANAGER2);
      3. existe filing efectivo de B para el mismo PERIODOFREPORT;
      4. B declara explicitamente que A reporta por B, mediante
         una fila de OTHERMANAGER asociada al ACCESSION_NUMBER
         del filing efectivo de B, identificando inequivocamente
         a A:
         a) mediante CIK, o
         b) cuando CIK no este disponible, mediante Form 13F File
            Number cuya resolucion a CIK sea inequivoca dentro del
            mismo PERIODOFREPORT.

         La evidencia es valida cuando el filing B es:
         - 13F NOTICE (SUBMISSIONTYPE in {13F-NT, 13F-NT/A}
                       y REPORTTYPE = 13F NOTICE), o
         - 13F COMBINATION REPORT (SUBMISSIONTYPE in {13F-HR, 13F-HR/A}
                       y REPORTTYPE = 13F COMBINATION REPORT).

         La ausencia de resolucion inequivoca produce N/D o CONFLICT
         segun corresponda. No se permite matching por nombre.
      5. no existe evidencia contradictoria ni evidencia de
         reporting partition/overlap no resuelto.

**Nota terminologica (correccion #1 del dictamen):** no se habla de
"evidencia bidireccional". La evidencia es "estructurada cruzada
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

## 5. Tratamiento de amendments (correccion #3, Gate 0.5 PASS)

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

      3. Aplicar semantica SEC:
         - Base:          punto de partida.
         - RESTATEMENT:   sustituye el snapshot efectivo
                          (incluye OTHERMANAGER). Evidencia
                          directa: Gate 0.5, caso CIK 0002056909.
         - NEW HOLDINGS:  tratar como identidad sobre OTHERMANAGER.
                          Evidencia: Gate 0.7, 3/3 casos identicos.
                          Fail-closed: si un NEW HOLDINGS presenta
                          OTHERMANAGER distinto del base, el estado
                          es N/D o CONFLICT. NO asumir union.
         - Resultado:     snapshot efectivo.

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

**Cardinalidad 1:1 perfecta.**

### 9.2-bis. Hallazgo de normalizacion FormNum

Del 91.55-91.83% con FormNum poblado, el 93.42-93.34% usa prefijo
`028-`. El resto (6.58-6.66%) usa prefijo `28-` (sin cero a la
izquierda). **Todas las filas afectadas tienen CIK=`<NA>`.**

Normalizacion trivial propuesta: `28-XXXXX -> 028-XXXXX` con
zero-padding a 3 digitos.

### 9.5. Tabla canonica Form13FFileNumber -> CIK (Gate 0.4 PASS)

**Fuente corregida (dictamen P66 v2, Bloqueo A):** la tabla canonica
se construye vía `COVERPAGE` + `SUBMISSION`, no vía `OTHERMANAGER`.

    COVERPAGE  (ACCESSION_NUMBER, FORM13FFILENUMBER, ...)
        JOIN
    SUBMISSION (ACCESSION_NUMBER, CIK)
        ON ACCESSION_NUMBER
    Agrupacion: FORM13FFILENUMBER (normalizado) -> CIK

**Normalizacion:** `28-XXXXX -> 028-XXXXX` y `028-XXXXX -> 028-XXXXX`.
El prefijo se normaliza a 3 digitos y el sufijo a 5.

**Cobertura del mapping (Gate 0.4):**

| Trimestre | FormNum unicos | Resueltos | No resueltos | Cobertura |
|-----------|---------------:|----------:|-------------:|----------:|
| 2025Q4    | 914            | 894       | 20           | 97.81%    |
| 2026Q1    | 908            | 892       | 16           | 98.24%    |

**Cardinalidad:** 1:1 perfecta en ambas direcciones (0 casos N:1).

**Impacto en filas (Gate 0.4b):**

| Categoria | Q4 filas | Q4 % | Q1 filas | Q1 % |
|-----------|---------:|-----:|---------:|-----:|
| A. MATCH por CIK directo | 1,955 | 69.72% | 1,948 | 68.91% |
| B. MATCH por FormNum fallback | 737 | 26.28% | 771 | 27.27% |
| C. N/D FormNum no resuelto | 15 | 0.53% | 5 | 0.18% |
| D. N/D ni CIK ni FormNum | 97 | 3.46% | 103 | 3.64% |
| **Cobertura efectiva MATCH** | **96.01%** | | **96.18%** | |
| **N/D total** | **3.99%** | | **3.82%** | |

**Bloqueo A del dictamen P66 v2: RESUELTO.**

**Anomalia detectada:** el FormNum `028-2813114` (7 digitos) aparece
en N/D por FormNum no resuelto. Volumen despreciable (4+2=6 filas).
Documentado como hallazgo colateral en
`iae/evidence/p66_gate04_mapping/README.md` seccion 2.5.

### 9.6. Bloqueo B pendiente

El dictamen P66 v2 mantiene un segundo bloqueo: formalizar como se
determina el "filing efectivo de B" cuando existen NT/A y multiples
amendments.

Requiere un `amendment probe` especifico que mida el comportamiento
de `OTHERMANAGER` en:

    NT
    NT/A RESTATEMENT
    NT/A ADDS NEW HOLDINGS ENTRIES

Pendiente de ejecucion. Sin el, la semantica de R3 (filing efectivo)
no puede formalizarse contractualmente.

### 9.3. Caso Vanguard confirmado (Gate 0.2)

Q4 2025: multiples NT de filiales declaran al parent con FormNum 028-06408.
Q1 2026: parent 13F-NT (ACC=0000102909-26-002707) declara 10 managers,
incluyendo las filiales CIK 0002100119 y 0002100121.

Bidireccionalidad documental directa en OTHERMANAGER.

### 9.4. Column 7 no apunta a OTHERMANAGER

INFOTABLE.OTHERMANAGER (Column 7) matchea OTHERMANAGER2.SEQUENCENUMBER
(90% aprox), no OTHERMANAGER_SK. Requisito 2 y requisito 4 usan
tablas distintas. El dictamen confirma la separacion R2/R4 como
correcta.

---

## 10. Preguntas residuales al auditor

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

## 11. Lo que NO se ha hecho

- NO se ha descargado ningun XML de EDGAR.
- NO se ha creado xml_parser/.
- NO se ha modificado reporting_dedup.py.
- NO se ha reformulado 14.3 efectivamente.
- NO se ha activado DROP_DUP.
- NO se ha tocado NIPC_CONTRATOS_SEMANTICOS_v1.md.

---

## 12. Referencias

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
