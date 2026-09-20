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
         a A mediante CIK o, cuando el CIK no este disponible,
         mediante Form 13F File Number resoluble inequivocamente
         a A;
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

## 5. Tratamiento de amendments (correccion #3)

Prohibido:

    MAX(ACCESSION_NUMBER)
    MAX(FILING_DATE)
    ultima fila del dataframe

Requerido:

    La seleccion del "filing efectivo de B" debe realizarse
    sobre los campos:
        CIK
        PERIODOFREPORT
        SUBMISSIONTYPE   (13F-HR | 13F-NT | .../A)
        ISAMENDMENT
        AMENDMENTNO
        AMENDMENTTYPE

La semantica aplicada debe ser la de la spec SEC:
- Amendment RESTATEMENT: sustituye el filing base.
- Amendment NEW HOLDINGS: suplementa el filing base.
- Multiples amendments coexisten; el efectivo se resuelve por
  la cadena documental, no por fecha.

Esta formalizacion es OBLIGATORIA antes del GO contractual definitivo.
Sin ella, R3 puede resolver al filing equivocado en presencia de
amendments.

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

### 9.2. Granularidad de identidad (pendiente de Gate 0.3)

Antes del GO contractual, medir:

    numero total de NT
    numero con OTHERMANAGER non-null
    numero total de filas OTHERMANAGER en NT
    numero con CIK poblado
    numero con FORM13FFILENUMBER poblado
    numero con ambos poblados
    numero con ninguno poblado
    numero con CIK <NA> y FormNum <NA>
    numero con CIK poblado y FormNum <NA>
    numero con CIK <NA> y FormNum poblado

Esta medicion es requisito del dictamen #27, condicion del GO.

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
