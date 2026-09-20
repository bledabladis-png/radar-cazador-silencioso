# IAE - P66 L3 Reformulacion propuesta

**Objeto:** propuesta de reformulacion del contrato 14.3 requisito 4,
basada en evidencia directa del Data Set SEC (`OTHERMANAGER`).

**Estado:** PROPUESTA. No vigente. No modifica el contrato. Pendiente
de dictamen del auditor externo.

**Origen:** Gate 0.1 + Gate 0.2 ejecutados 2026-09-20 tras NO-GO de P66.

**HEAD al redactar:** 2edb462 (o posterior).

---

## 0. Contexto

El informe P66 (informe de hallazgo, XML crudo 13F-NT) fue marcado
como NO-GO tras dos Gates:

- Gate 0.1: `OTHERMANAGER` del Data Set contiene la relacion "other
  managers reporting for this manager" al 100% de cobertura sobre
  NT filings (2,008/2,008 en Q4; 2,045/2,045 en Q1). `OTHERMANAGER2`
  es una relacion distinta ("included in this report") y no es
  relevante para L3.
- Gate 0.2: el caso Vanguard Q4 2025 -> Q1 2026 esta documentado
  bidireccionalmente en `OTHERMANAGER` sin necesidad de XML.

Conclusion: L3 requisito 4 es materializable desde el Data Set, sin
descargas, sin parser, sin dependencia nueva.

---

## 1. Texto vigente del contrato 14.3

    L3(A, B, S, period) == True sii:
      1. existe filing de A con linea L sobre security S
      2. Column 7(L) referencia a B (reference_status = RESOLVED)
      3. existe filing de B en el mismo periodo
      4. B declara explicitamente que A reporta por B
         (NT o Combination con A en "Other Managers Reporting for this Manager")
      5. no existe evidencia contradictoria ni evidencia de reporting
         partition/overlap no resuelto

El requisito 4 cita la relacion correcta por nombre pero no la ancla
a la tabla del Data Set. La implementacion v1 no la encontro porque
buscaba en `OTHERMANAGER2` (incorrecto) y en `ADDITIONALINFORMATION`
(no aplicable).

---

## 2. Propuesta de reformulacion

    L3(A, B, S, period) == True sii:
      1. existe filing de A con linea L sobre security S
      2. Column 7(L) referencia a B (reference_status = RESOLVED)
      3. existe filing de B en el mismo periodo
      4. B declara explicitamente que A reporta por B, via:
         OTHERMANAGER[ACCESSION_NUMBER = B.ACCESSION_NUMBER,
                      CIK = A.CIK]
         (la tabla OTHERMANAGER del Data Set SEC documenta "list of
         other managers reporting for this manager")
      5. no existe evidencia contradictoria ni evidencia de reporting
         partition/overlap no resuelto

---

## 3. Estados propuestos del requisito 4

    MATCH     Existe fila en OTHERMANAGER[ACCESSION=B, CIK=A].
    NO_MATCH  El filing B fue inspeccionado y NO tiene fila con A.
    N/D       No se pudo inspeccionar el filing B (ausente, corrupto,
              periodo distinto).

N/D NO se transforma en NO_MATCH. Fail-closed: N/D no autoriza DROP_DUP.

---

## 4. Evidencia empirica del Gate 0.1 + 0.2

### 4.1. Cobertura de OTHERMANAGER por tipo de filing

| Trimestre | Tipo | Filings | Con filas OTHERMANAGER | Cobertura |
|-----------|------|--------:|-----------------------:|----------:|
| 2025Q4    | NT   | 2,008   | 2,008                  | 100.0%    |
| 2025Q4    | HR   | 9,364   | 419                    | 4.5%      |
| 2026Q1    | NT   | 2,045   | 2,045                  | 100.0%    |
| 2026Q1    | HR   | 9,716   | 441                    | 4.5%      |

### 4.2. Caso Vanguard confirmado

Q4 2025:
- Parent `0000102909` presenta 13F-HR.
- Multiples NT de filiales declaran al parent con FormNum 028-06408.
- Ejemplo: `ACC=0000933478-26-000002 CIK=0000102909`.

Q1 2026:
- Parent `0000102909` presenta 13F-NT (ACC=0000102909-26-002707).
- Declara 10 other managers, entre ellos:
  - `0002100121` VANGUARD PORTFOLIO MANAGEMENT LLC
  - `0002100119` VANGUARD CAPITAL MANAGEMENT LLC
- Las filiales presentan 13F-HR en Q1.

Bidireccionalidad documental directa en `OTHERMANAGER`, sin XML.

### 4.3. Column 7 no apunta a OTHERMANAGER

Hallazgo adicional: `INFOTABLE.OTHERMANAGER` (Column 7) matchea
`OTHERMANAGER2.SEQUENCENUMBER` (90% aprox), no `OTHERMANAGER_SK`.
La spec SEC es coherente: Column 7 referencia al "included manager"
del propio filing, no al "reporting manager".

Esto significa que L3 requisitos 2 (Column 7 -> B) y 4 (B declara
A) usan **dos tablas distintas**:

    Requisito 2: INFOTABLE.OTHERMANAGER -> OTHERMANAGER2.SEQ
    Requisito 4: OTHERMANAGER[ACCESSION=B, CIK=A]

Aparentemente ambas coexisten sin contradiccion. Requiere validacion
del auditor.

---

## 5. Preguntas al auditor

### 5.1. Pregunta binaria

    Autoriza el auditor que la tabla OTHERMANAGER del Data Set SEC
    (documentada como "list of other managers reporting for this
    manager") constituya evidencia bidireccional estructurada
    suficiente para satisfacer el requisito 4 de L3 del contrato
    14.3, siempre que:
    - el filing B sea del mismo periodo que el filing A,
    - la fila OTHERMANAGER[ACCESSION=B, CIK=A] exista,
    - CIK y FORM13FFILENUMBER esten poblados y sean coherentes?

### 5.2. Pregunta secundaria

    Es suficiente esta evidencia para autorizar DROP_DUP por si
    sola, o exige una segunda condicion independiente?

### 5.3. Preguntas operativas

    Q1. El requisito 2 (Column 7 -> OTHERMANAGER2) y el requisito 4
        (OTHERMANAGER -> CIK_A) usan tablas distintas. Es correcto
        asi, o deben unificarse bajo una unica fuente?

    Q2. Como se tratan los casos en que OTHERMANAGER[ACCESSION=B,
        CIK=A] existe pero FORM13FFILENUMBER esta vacio (<NA>)?

    Q3. Si OTHERMANAGER contiene la fila pero CIK esta vacio y solo
        FORM13FFILENUMBER esta poblado, es MATCH o N/D?

    Q4. La reformulacion propuesta (seccion 2) es aceptable, o
        requiere ajuste?

    Q5. Los NT que declaran CERO other managers (si existen) deben
        reportar como NO_MATCH o como N/D?

    Q6. El requisito 3 original ("existe filing de B en el mismo
        periodo") ya esta cubierto por la propia existencia de la
        fila OTHERMANAGER con ACCESSION=B. Se simplifica o se
        mantiene explicito?

---

## 6. Lo que NO se ha hecho

- NO se ha descargado ningun XML de EDGAR.
- NO se ha creado xml_parser/.
- NO se ha modificado reporting_dedup.py.
- NO se ha reformulado 14.3 efectivamente (solo propuesto).
- NO se ha activado DROP_DUP.
- NO se ha tocado NIPC_CONTRATOS_SEMANTICOS_v1.md.
- NO se ha enviado P66 al auditor.

---

## 7. Referencias

    | Documento                                    | Rol                |
    |----------------------------------------------|--------------------|
    | iae/NIPC_CONTRATOS_SEMANTICOS_v1.md seccion 14 | Contrato L3      |
    | iae/P64_P65_EXPEDIENTE.md                    | Ciclo P65          |
    | iae/P66_INFORME_HALLAZGO.md                  | Informe NO-GO      |
    | iae/DICTAMENES.md #26                        | Dictamen P65 v3    |
    | SEC Form 13F Data Sets (documentacion)       | Fuente oficial     |

---

Fin de la propuesta de reformulacion L3.
