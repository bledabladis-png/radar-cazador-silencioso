# DICTAMEN DEL AUDITOR EXTERNO - GATE 0 FIGI

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / Mapping / FIGI
**Resultado:** **PASS - FIGI DESCARTADO COMO PIVOTE PRIMARIO**
**Siguiente paso:** **GO para Gate 0 OpenFIGI**
**Codigo productivo:** no autorizado todavia

La caracterizacion es suficiente para tomar una decision arquitectonica. Ademas, la propia SEC define `FIGI` como campo nullable de `INFOTABLE` y advierte que sus datasets contienen informacion "as filed", pudiendo incluir inconsistencias, discrepancias y errores de extraccion/compilacion.

---

# 1. Q-FIGI-1 - Se descarta FIGI como fuente primaria?

## **SI - DESCARTADO COMO PIVOTE PRIMARIO**

La evidencia es suficientemente fuerte:

    FIGI presente en filas       12.01%
    SHARES con FIGI valido       10.98%
    CUSIPs con algun FIGI        38.04%

Para NIPC, el dato decisivo es:

    ~89% del SSHPRNAMT
    sin FIGI valido

Por tanto, FIGI no puede sostener la capa primaria:

    CUSIP
     v
    security identity
     v
    ticker

La SEC define `FIGI` simplemente como el Financial Instrument Global Identifier dentro de `INFOTABLE`; no lo presenta como la clave maestra del dataset ni garantiza su completitud. El campo ademas es nullable.

### Pero NO eliminaria FIGI

Debe conservarse como:

    FIGI_RAW
    FIGI_FORMAT_STATUS
    FIGI_ANOMALY

y eventualmente como evidencia auxiliar de validacion.

Esto es importante porque habeis demostrado una propiedad interesante:

    intra-filing consistency > 99.9%

El problema no es que un filer cambie aleatoriamente FIGI dentro del mismo filing; es la **inconsistencia entre filers**.

Por tanto:

    FIGI
        -> auxiliar / diagnostico
        X pivote canonico

### Q-FIGI-1 = **SI**

---

# 2. Q-FIGI-3 - Que hacer con los 1.267 FIGI no-BBG?

## **APROBADO: conservar + clasificar como anomalia**

No recomiendo:

    (c) ignorar el campo FIGI completo

La opcion correcta es una variante de **(a) + (b)**:

    raw FIGI
        v
    normalizacion
        v
    format_status

Por ejemplo:

    VALID_BBG
    NONSTANDARD_FORMAT
    MALFORMED_IDENTIFIER
    SENTINEL

Los ejemplos:

    CA15101Q2071
    US60937P1066
    IE00BK9ZQ967

son especialmente utiles como evidencia de que el valor no debe introducirse directamente en el resolver como FIGI.

La especificacion SEC solo garantiza que el campo es un `VARCHAR2(12)` nullable; no establece que cada valor aportado por un filer sea necesariamente un FIGI valido.

### Importante

No haria:

    FIGI no-BBG
        v
    intentar convertir automaticamente
        v
    CUSIP/ISIN

Eso volveria a introducir heuristica en una capa que estamos intentando mantener determinista.

**Q-FIGI-3 = conservar y etiquetar; no usar para resolver.**

---

# 3. Q-FIGI-4 - Comprobar tambien Q4 2025?

## **NO es necesario para decidir la arquitectura**

Q1 2026 ya proporciona evidencia suficiente para descartar FIGI como pivote primario.

Por tanto:

    Q1 2026
       v
    decision arquitectonica

ya esta justificado.

### Pero si aprobaria un probe Q4 como control no bloqueante

Tiene utilidad para comprobar si:

    10.98% weighted coverage
    66% multi-FIGI/CUSIP

son fenomenos persistentes o especificos de Q1.

Pero:

**no condiciona el inicio del siguiente Gate.**

### Q-FIGI-4 = **OPCIONAL / NO BLOQUEANTE**

No abriria otro ciclo de investigacion antes de OpenFIGI solo para confirmar algo que ya esta suficientemente demostrado.

---

# 4. Q-FIGI-2 - Pasamos a OpenFIGI?

## **SI - GO**

El descarte de FIGI como pivote interno no elimina la utilidad de investigar OpenFIGI como **servicio externo de crosswalk**.

Pero hay que separar dos conceptos:

    FIGI presente en SEC
            !=
    OpenFIGI como servicio

La primera pregunta era:

> Podemos utilizar el FIGI que ya viene en 13F?

Respuesta: **NO**.

La segunda es:

> Puede un servicio externo resolver de forma reproducible CUSIP -> security/ticker?

Todavia no lo sabemos.

Por ello:

**Q-FIGI-2 = GO para Gate 0 OpenFIGI.**

---

# 5. Q-FIGI-5 - alcance del probe OpenFIGI

Aqui **no apruebo exactamente "top 100 CUSIPs del crosswalk por SSHPRNAMT" como unica muestra**.

El problema es sesgo de muestra:

    etf_holdings.csv
           v
    sector ETFs
           v
    large/mega caps sobreponderadas

Si usamos exclusivamente los 100 primeros, estariamos midiendo OpenFIGI precisamente donde probablemente funciona mejor.

Eso no nos permitiria concluir cobertura del universo NIPC.

## Probe aprobado: muestra estratificada

Propongo cinco estratos:

### A. Universo radar USA con mapping conocido

Para medir:

    CUSIP -> ticker

en casos donde conocemos la respuesta esperada.

### B. Los 22 radar USA actualmente unmapped

Son los casos mas utiles para saber si OpenFIGI puede cerrar el gap real del radar.

### C. CUSIPs 13F de mayor `SSHPRNAMT`

Para medir impacto economico.

### D. CUSIPs de `SSHPRNAMT` medio/bajo

Para evitar que el resultado sea unicamente large-cap.

### E. Casos de clases/instrumentos problematicos

Por ejemplo:

    BRK-B
    MOG-A
    ADR
    CL A / CL B

y casos conocidos de anomalia.

### Tamano

Un probe de alrededor de **200-300 CUSIPs distintos** seria mas informativo que 100 unicamente del crosswalk ETF.

No necesitamos resolver los 24.838 para este Gate 0.

---

# 6. Que debe responder OpenFIGI

El probe no debe preguntar unicamente:

    Devuelve ticker?

Debe registrar por CUSIP:

    input CUSIP
    HTTP/result status
    FIGI
    ticker
    name
    security type
    market sector
    share class / class info
    exchange / market

y clasificar:

    EXACT
    NOT_FOUND
    MULTIPLE_CANDIDATES
    CONFLICT
    NON_EQUITY
    ERROR
    RATE_LIMIT

La informacion decisiva sera:

### Coverage

    CUSIPs encontrados / consultados

### Operational coverage

    radar USA resueltos / radar USA

### Weighted coverage

    SSHPRNAMT mapped / SSHPRNAMT sampled

### Exactness

    one unambiguous security / hits

### Class handling

Especialmente:

    BRK-A / BRK-B
    GOOG / GOOGL
    MOOG-A
    ADR

### Temporal validity

Una respuesta actual no demuestra que sea valida para:

    2025-12-31

o:

    2026-03-31

Por eso la temporalidad debe formar parte del Gate.

---

# 7. Criterio fundamental: OpenFIGI no se convertira automaticamente en autoridad

Aunque OpenFIGI consiguiera:

    99% coverage

eso **no implica automaticamente**:

    OpenFIGI = truth

La arquitectura debe conservar:

    source
    mapping_method
    mapping_timestamp
    mapping_version
    input_identifier
    resolved_security
    confidence/status

y, cuando haya conflicto:

    SEC / internal / external
            v
    CONFLICT

No:

    OpenFIGI gana siempre

Esto sera especialmente importante para las clases y corporate actions.

---

# 8. El hallazgo de `TITLEOFCLASS` queda reforzado

El analisis FIGI confirma que FIGI tampoco soluciona el problema de clasificacion.

Los dos casos NIPST siguen:

    SH
    PUTCALL NULL
    FIGI NULL
    TITLEOFCLASS = CONVERTIBLE BOND

Por tanto, queda confirmado:

    FIGI
        X
    security_type

y la especificacion NIPC tendra que mantener una capa explicita de clasificacion de security.

La SEC confirma que `TITLEOFCLASS`, `CUSIP`, `FIGI`, `SSHPRNAMT`, `SSHPRNAMTTYPE` y `PUTCALL` son campos independientes del Information Table.

---

# 9. Un detalle del informe que debe conservarse

Hay una conclusion muy buena y tecnicamente importante:

    INTRA-FILING
    99.91% consistent

    CROSS-FILING
    high ambiguity

Esto significa que el problema no se resuelve con:

    "normalizar FIGI dentro de cada filing"

El problema es de **canonicalizacion cross-filing**.

Eso encaja perfectamente con la arquitectura que estamos disenando:

    13F
     v
    raw identifier
     v
    external/internal security resolver
     v
    canonical security

y no:

    13F FIGI
     v
    usar directamente como security key

---

# 10. Dictamen consolidado

| Pregunta                       | Dictamen                                                |
| ------------------------------ | ------------------------------------------------------- |
| Q-FIGI-1                       | **SI - descartar FIGI como fuente primaria/pivote**     |
| Uso futuro de FIGI             | **Auxiliar / diagnostico**                              |
| Q-FIGI-2                       | **GO - OpenFIGI**                                       |
| Q-FIGI-3                       | **Conservar anomalias; no resolver heuristicamente**    |
| Q-FIGI-4                       | **Q4 opcional, no bloqueante**                          |
| Q-FIGI-5                       | **GO, pero muestra estratificada, no solo top-100 ETF** |
| Codigo NIPC                    | **NO**                                                  |
| Especificacion NIPC definitiva | **todavia NO**                                          |
| Gate 0 OpenFIGI                | **AUTORIZADO**                                          |

---

# DICTAMEN FORMAL

## **GATE 0 FIGI -> PASS**

**FIGI queda descartado como pivote primario de security identity para NIPC.**

La evidencia es suficientemente contundente:

    12.01% filas
    10.98% SHARES
    38.04% CUSIPs con algun FIGI
    66% de esos CUSIPs -> multiples FIGIs
    1.57% FIGIs -> multiples CUSIPs

Ademas, la propia SEC advierte que los datos son `as-filed`, pueden contener inconsistencias y no garantizan exactitud completa.

## **Gate 0 OpenFIGI -> GO**

Pero el probe debe ser **estratificado y orientado al universo operativo del radar**, no un simple top-100 del crosswalk ETF.

La secuencia queda:

    Gate 0 Mapping
           [OK]
           v
    Gate 0 FIGI
           [OK] DESCARTADO
           v
    Gate 0 OpenFIGI
           [GO]
           v
    Gate 0 SEC 13(f)
           v
    Gate-NIPC.1
           v
    Especificacion definitiva

**No escribas todavia `delta_shares.py`, `nipc.py` ni el contrato definitivo.** El siguiente artefacto correcto es el **probe empirico de OpenFIGI**, con evidencia real y muestra estratificada.
