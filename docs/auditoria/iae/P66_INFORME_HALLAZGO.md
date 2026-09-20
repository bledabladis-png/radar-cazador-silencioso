> **NO-GO — CERRADO 2026-09-20**
>
> Este informe fue redactado sobre una premisa incorrecta: que la
> relacion "other managers reporting for this manager" solo estaba
> disponible en el XML crudo del filing 13F-NT.
>
> Gate 0.1 demostro que `OTHERMANAGER` (no `OTHERMANAGER2`) del Data
> Set SEC contiene esa relacion al 100% de cobertura sobre NT filings.
> Gate 0.2 confirmo el caso Vanguard directamente en esa tabla.
>
> El ciclo P66 (XML crudo + parser propio + 4.000 descargas) queda
> CANCELADO. La reformulacion de L3 (contrato 14.3) se hara desde
> `OTHERMANAGER` sin XML.
>
> Se preserva este informe como registro del error metodologico.
>
> Ver: `iae/FOLLOWUPS.md` (entrada P66) y `iae/INFORME.md` (cierre).

---

# IAE - P66 Informe de hallazgo: fuente externa para L3

**Objeto:** elevar al auditor externo el hallazgo de una fuente oficial SEC
estructurada que puede satisfacer el requisito 4 de L3 (contrato 14.3).

**HEAD al redactar:** 62314ac (o posterior).
**Estado:** BORRADOR. Pendiente de envio al auditor. NO modifica contrato.
NO autoriza descarga. NO toca codigo.

**Autor:** Ingeniero Supervisor.
**Fecha:** 2026-09-20.
**Destinatario:** auditor externo (solicitud de dictamen).

**Origen:** Gate 0 extendido del ciclo P65 v2 (DROP_DUP efectivo),
cerrado como WONT FIX razonado por ausencia de evidencia cruzada en
el Data Set tabular. Investigacion posterior de fuentes externas.

---

## 0. Resumen ejecutivo

El ciclo P65 v2 fue cerrado como WONT FIX razonado por Gate 0 que
demostro que el Data Set tabular 13F de la SEC no expone evidencia
cruzada entre filings. El requisito 4 de L3 (contrato 14.3) exigia
que B declarase que A reporta por B. Esa declaracion no existe en
`ADDITIONALINFORMATION` ni en `OTHERMANAGER2` (NT lo tienen vacio).

La investigacion posterior de fuentes externas confirma que:

**La evidencia SI existe, en el XML crudo del filing 13F-NT.**

El XML de un NT contiene un bloque `<otherManagersInfo>` (schema
antiguo) o `<otherManagers2Info>` (schema nuevo) con la lista
estructurada de managers que reportan en su nombre, con CIK y
Form 13F File Number completos, validados contra schema SEC.

Esto convierte P65 v2 en un problema de **contrato probatorio**, no
de disponibilidad de fuente.

**Solicitud al auditor:** dictamen sobre si el XML del NT satisface
el requisito 4 de 14.3, y en que condiciones autoriza DROP_DUP.

---

## 1. Hallazgo

### 1.1. La fuente

EDGAR publica cada filing 13F como XML firmado digitalmente. El
documento se sirve en:

    https://www.sec.gov/Archives/edgar/data/<CIK>/<ACCESSION>/
        primary_doc.xml

Para un filing 13F-NT, el XML contiene en su `<coverPage>` (o en
`<formData><summaryPage>` segun schema) una lista estructurada de
managers. Cada entrada lleva:

- `<cik>` - CIK del manager representado
- `<form13FFileNumber>` - numero 13F asignado por SEC
- `<name>` - nombre del manager

Esta lista es la declaracion formal, firmada, de que el NT filer
esta reportando en nombre de esos managers.

### 1.2. Caso de referencia - Yorktown

Filing 13F-NT publicado por SEC. Fragmento XML literal:

    <coverPage>
      <reportType>13F NOTICE</reportType>
      <form13FFileNumber>028-17365</form13FFileNumber>
      <otherManagersInfo>
        <otherManager>
          <cik>0001423080</cik>
          <form13FFileNumber>028-17368</form13FFileNumber>
          <name>Yorktown Energy Partners VII, L.P.</name>
        </otherManager>
        <otherManager>
          <cik>0001423201</cik>
          <form13FFileNumber>028-17370</form13FFileNumber>
          <name>Yorktown VII Associates LLC</name>
        </otherManager>
      </otherManagersInfo>
      <provideInfoForInstruction5>Y</provideInfoForInstruction5>
      <additionalInformation>Yorktown VII Company LP is the sole general
        partner of Yorktown Energy Partners VII, L.P. ...</additionalInformation>
    </coverPage>

Interpretacion directa: el NT filer (Yorktown Energy Partners VII,
CIK 0001423080) declara que reporta en nombre de Yorktown VII
Associates LLC (CIK 0001423201) y otros. Es exactamente la relacion
A -> B que L3 exige, en XML estructurado.

### 1.3. Cobertura estimada

Del Gate 0 previo: 2,008 NT en Q4 2025 + 2,045 NT en Q1 2026. Total
~4,053 filings potencialmente portadores del bloque.

De los NT, se estima que una fraccion significativa contiene
`otherManagersInfo` (o `otherManagers2Info`). Medicion exacta
requiere descarga (bloqueada hasta dictamen).

---

## 2. Dos schemas XML

El schema ha evolucionado. Un parser debe cubrir ambos.

### 2.1. Schema antiguo

    <edgarSubmission>
      <coverPage>
        <reportType>13F NOTICE</reportType>
        <form13FFileNumber>...</form13FFileNumber>
        <otherManagersInfo>
          <otherManager>
            <cik>...</cik>
            <form13FFileNumber>...</form13FFileNumber>
            <name>...</name>
          </otherManager>
          ...
        </otherManagersInfo>
        ...
      </coverPage>
    </edgarSubmission>

### 2.2. Schema nuevo

    <edgarSubmission>
      <formData>
        <coverPage>...</coverPage>
        <summaryPage>
          <otherManagers2Info>
            <otherManager2>
              <sequenceNumber>1</sequenceNumber>
              <otherManager>
                <cik>...</cik>
                <form13FFileNumber>...</form13FFileNumber>
                <name>...</name>
              </otherManager>
            </otherManager2>
            ...
          </otherManagers2Info>
        </summaryPage>
      </formData>
    </edgarSubmission>

### 2.3. Equivalencia semantica

Ambos formatos declaran la misma relacion: el NT filer
(B) declara que A reporta por B. La diferencia es estructural
(posicion en el arbol XML y presencia/ausencia de
`sequenceNumber`). Un parser productivo debe aceptar ambos y
normalizar el output.

---

## 3. Por que el Data Set tabular no lo tiene

El PDF oficial de los SEC Data Sets declara:

> "The FORM 13F data sets are extracted from EDGAR FORM 13F XML
> filings in a flat file format to assist users in constructing
> the data for analysis."

La tabla `OTHERMANAGER2` del Data Set se construye desde el XML,
pero **solo desde filings que la contienen en formato tabular
extraible**. Gate 0 midio: 0 filas de NT en `OTHERMANAGER2`
(Q4 2025 y Q1 2026). La extraccion tabular omite el bloque
`otherManagersInfo` de los NT porque no tiene mapping obvio a la
tabla OTHERMANAGER2.

Conclusion: la informacion existe en XML crudo y es la fuente
primaria oficial. El Data Set tabular es una simplificacion
parcial.

---

## 4. Propuesta de reformulacion del contrato 14.3

### 4.1. Texto vigente (14.3)

    L3(A, B, S, period) == True sii:
      1. existe filing de A con linea L sobre security S
      2. Column 7(L) referencia a B (reference_status = RESOLVED)
      3. existe filing de B en el mismo periodo
      4. B declara explicitamente que A reporta por B
         (NT o Combination con A en "Other Managers Reporting for this Manager")
      5. no existe evidencia contradictoria ni evidencia de reporting
         partition/overlap no resuelto

### 4.2. Propuesta de reformulacion (sujeta a dictamen)

    L3(A, B, S, period) == True sii:
      1. existe filing de A con linea L sobre security S
      2. Column 7(L) referencia a B (reference_status = RESOLVED)
      3. existe filing de B en el mismo periodo
      4. B declara explicitamente que A reporta por B, via:
         (4a) XML del NT de B con otherManagersInfo/otherManager
              (schema antiguo) con CIK(A) en <cik> y/o
              Form 13F File Number de A en <form13FFileNumber>, o
         (4b) XML del NT de B con otherManagers2Info/otherManager2
              (schema nuevo) con la misma declaracion, o
         (4c) filing de B tipo Combination con A declarado
              explicitamente en campo estructurado
      5. no existe evidencia contradictoria ni evidencia de reporting
         partition/overlap no resuelto

### 4.3. Estados propuestos del requisito 4

    MATCH     La relacion A -> B existe en el XML del NT de B.
    NO_MATCH  Se descargo el XML de B y NO contiene la relacion.
    N/D       No se pudo descargar, parsear o validar el XML de B.

N/D NO se transforma en NO_MATCH. Fail-closed: N/D no autoriza DROP_DUP.

### 4.4. Requisitos adicionales propuestos

Antes de activar DROP_DUP efectivo, ademas de L3:

    R1  El filing NT de B debe ser del MISMO PERIODOFREPORT que el
        filing de A y que el periodo evaluado.
    R2  El filing NT de B debe ser el filing efectivo del periodo
        (respetar amendements: el NT/A posterior prevalece).
    R3  CIK(A) debe estar normalizado (10 digitos, sin ceros
        significativos perdidos).
    R4  Si el Form 13F File Number se usa como clave, debe
        resolverse consistentemente a CIK.
    R5  Si el XML contiene AMBOS campos (cik y form13FFileNumber)
        con valores incompatibles, estado = CONFLICT.

---

## 5. Preguntas al auditor

### 5.1. Pregunta binaria central

    Autoriza el auditor que un filing 13F-NT de B, en el que el XML
    SEC contiene otherManagersInfo/otherManager o
    otherManagers2Info/otherManager2 indicando explicitamente a A
    mediante CIK y/o Form 13F File Number, constituya evidencia
    bidireccional estructurada suficiente para satisfacer el
    requisito 4 de L3 del contrato 14.3, siempre que A y B
    pertenezcan al mismo periodo de reporte y la relacion pueda
    normalizarse de forma inequivoca?

### 5.2. Pregunta secundaria critica

    Es suficiente esta evidencia para autorizar DROP_DUP por si
    sola, o exige una segunda condicion independiente de
    validacion?

### 5.3. Preguntas operativas (R1-R6)

    R1. Es autorizado el patron de acceso HTTP a EDGAR para descargar
        ~4,053 XML (rate limit SEC 10 req/s, User-Agent obligatorio,
        cache local, reintentos en 429/503)?

    R2. Como se versiona el storage nt_other_managers.parquet si el
        schema XML cambia entre periodos?

    R3. El nuevo campo convive con OTHERMANAGER2.parquet (HR) o se
        mantiene tabla separada?

    R4. La reformulacion propuesta de 14.3 (seccion 4.2) es
        aceptable, o requiere ajuste?

    R5. Los NT sin otherManagersInfo (estimado significativo) deben
        reportar como NO_MATCH o como N/D?

    R6. Que tratamiento se aplica a los casos CONFLICT (seccion 4.4 R5)?

---

## 6. Alcance propuesto del ciclo P66

Sujeto a dictamen del auditor. El orden es
**dictamen -> GO contractual -> implementacion -> probe -> activacion**.

    P66.1  Especificacion
           - Aplicar las respuestas del auditor.
           - Fijar schema normalizado.
           - Fijar criterios de versionado.
           Estado: propuesto, no autorizado.

    P66.2  Implementacion parser XML
           - Modulo nuevo: src/institutional_accumulation/sec_13f/
             xml_parser/ (parser propio con lxml).
           - Storage: data/sec_13f/nt_other_managers.parquet
             (gitignored, coherente con FA-1).
           - Manifest de lineage.
           Estado: propuesto, no autorizado.

    P66.3  Integracion L3
           - Reformular contrato 14.3 (tras GO formal).
           - Conectar con reporting_dedup.classify_evidence_level.
           - Activar DROP_DUP efectivo.
           Estado: propuesto, no autorizado.

    P66.4  Evidencia e2e
           - Probe sobre Q4 2025 + Q1 2026.
           - Reportar match rate L3 bidireccional real.
           - Identificar casos Vanguard y similares.
           Estado: propuesto, no autorizado.

---

## 7. Lo que NO se ha hecho

- NO se ha descargado ningun XML de EDGAR.
- NO se ha creado src/institutional_accumulation/sec_13f/xml_parser/.
- NO se ha modificado reporting_dedup.py.
- NO se ha reformulado 14.3 (solo propuesto).
- NO se ha activado DROP_DUP.
- NO se ha instalado edgartools ni ninguna dependencia nueva.
- NO se ha tocado el contrato NIPC_CONTRATOS_SEMANTICOS_v1.md.

---

## 8. Decisiones de proceso recomendadas

| Decision                              | Estado                              |
|---------------------------------------|-------------------------------------|
| Preparar informe de hallazgo          | GO (hecho)                          |
| Elevar al auditor                     | GO (este documento)                 |
| Descargar masivamente EDGAR           | NO, hasta dictamen                  |
| Crear xml_parser/                     | NO                                  |
| Modificar reporting_dedup.py          | NO                                  |
| Reformular 14.3                       | Propuesta al auditor, no efectiva   |
| Parser productivo                     | lxml propio, sujeto a GO            |
| edgartools                            | Herramienta auxiliar de contraste   |
| Activar DROP_DUP                      | NO hasta autorizacion contractual   |

---

## 9. Referencias

    | Documento                                        | Rol                |
    |--------------------------------------------------|--------------------|
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 14   | Contrato L3        |
    | iae/P64_P65_EXPEDIENTE.md                        | Ciclo P65          |
    | iae/DICTAMENES.md #26                            | Dictamen P65 v3    |
    | iae/DICTAMENES.md #24                            | Dictamen F2.4      |
    | iae/INFORME.md #19                               | P65 v1 cierre      |
    | iae/evidence/nipc_p65_probe/                     | Evidencia v1       |

Fuentes externas consultadas:

    - SEC Form 13F XML schema (documentacion publica EDGAR).
    - SEC Data Sets PDF (declaracion de extraccion parcial).
    - Caso Yorktown Energy Partners VII (13F-NT publico).

---

Fin del informe P66. Pendiente de envio al auditor externo.
