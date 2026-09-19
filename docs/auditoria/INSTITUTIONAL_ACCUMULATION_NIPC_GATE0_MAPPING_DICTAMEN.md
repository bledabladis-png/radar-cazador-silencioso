# DICTAMEN DEL AUDITOR EXTERNO - GATE 0 MAPPING CUSIP -> SECURITY/TICKER

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / Gate 0 Mapping
**HEAD:** `93fda55` local
**Estado:** **PASS CONDICIONADO - GO para Gate-NIPC.1**
**Codigo NIPC:** no autorizado todavia

He contrastado especialmente la semantica de `TITLEOFCLASS`, `SSHPRNAMTTYPE`, `PUTCALL` y la naturaleza de las posiciones 13F con documentacion oficial SEC. La SEC define `SSHPRNAMTTYPE` como `SH` o `PRN`, pero tambien advierte que para opciones muchas columnas describen el **underlying security** y que `PUT/CALL` es lo que identifica que la linea es una opcion. La lista oficial de Section 13(f) es un universo independiente que la SEC actualiza trimestralmente.

---

## 1. Gate 0 de mapping: resultado

### **APROBADO COMO DIAGNOSTICO**

La evidencia responde correctamente a las cuatro metricas solicitadas:

    13F CUSIPs mapped                 2.021%
    Radar USA securities              90.91%
    Radar USA weighted shares         29.995%
    Radar USA weighted value          55.035%

Esto demuestra una cosa muy importante:

> **El crosswalk interno actual no es un crosswalk del universo 13F; es esencialmente un crosswalk derivado de holdings ETF sectoriales.**

Por tanto, los `502` CUSIPs mapeados no pueden utilizarse como prueba de que exista ya una capa de identidad general.

### Conclusion

**Gate 0 Mapping = PASS como caracterizacion.**

No significa:

    NIPC productivo = listo

El propio dato de `~30%` de shares del universo radar impide esa conclusion.

---

# 2. El dato mas importante no es 90.91%, sino 29.995%

La cobertura de securities:

    220 / 242 = 90.91%

es util, pero no suficiente.

La cobertura ponderada:

    29.995%

significa que aproximadamente el **70% del `SSHPRNAMT` del universo radar USA no esta actualmente cubierto** por el crosswalk disponible.

Por tanto, mantengo:

    NIPC status = INSUFFICIENT

para el universo operativo actual.

No aprobaria una regla del tipo:

    coverage securities >= 90%
            v
    NIPC OK

porque ese umbral estaria practicamente construido a partir del valor actual (`90.91%`) y no de un requisito independiente.

## Dictamen sobre el umbral propuesto de 90%

### **NO APROBADO como umbral contractual**

Debe definirse despues de disenar la capa de mapping y sus controles.

El 90.91% actual es un **resultado observado**, no una justificacion para convertir 90% en criterio de suficiencia.

---

# 3. Q-NIPC-8 - `TITLEOFCLASS`

## **RECHAZO la allowlist textual como mecanismo primario**

No apruebo la opcion:

    (b) allowlist de TITLEOFCLASS

como filtro principal del NIPC.

La razon es estructural: `TITLEOFCLASS` es un campo textual del filing y la SEC muestra en filings reales multitud de descripciones validas diferentes, como `COM`, `COMMON STOCK`, `CL A`, `SPONSORED ADS`, `AMERICAN DEPOSITORY`, etc.

Una allowlist:

    COM
    STOCK
    COMMON
    ORD
    ...

introduciria un catalogo heuristico que puede fallar ante una nueva variante legitima.

Y una blocklist tampoco es perfecta: bastaria que apareciera una denominacion no prevista para que volviera a pasar ruido.

---

# 4. Que hacer en su lugar

## **Q-NIPC-8 -> GO condicionado a clasificacion de security**

El filtro contractual base continua siendo:

    SSHPRNAMTTYPE == "SH"
    AND
    PUTCALL IS NULL

pero NIPC no debe considerar que eso, por si solo, demuestra que la linea representa una accion ordinaria del universo radar.

La condicion completa debe ser conceptualmente:

    13F line
      v
    SH + PUTCALL null
      v
    security identity resolved
      v
    security_type == EQUITY
      v
    ticker IN operational radar universe

Eso es muy superior a:

    TITLEOFCLASS =~ allowlist

### Por que

Tu propio hallazgo demuestra que existen lineas:

    SH
    +
    PUTCALL NULL
    +
    TITLEOFCLASS no-equity

La solucion correcta no es ensenar al sistema a reconocer cada posible cadena textual, sino **resolver primero que security es y que tipo de instrumento representa**.

---

# 5. El hallazgo `TITLEOFCLASS` si debe conservarse

No lo eliminaria del pipeline.

Debe funcionar como **campo de validacion/anomalia**, no como autoridad primaria de clasificacion.

Por ejemplo:

    security_type = EQUITY
    title_of_class = "COMMON STOCK"

-> coherente.

Pero:

    security_type = EQUITY
    title_of_class = "CONVERTIBLE BOND"

->

    TITLE_CLASS_CONFLICT

Eso es mucho mas auditable.

La SEC confirma que hay posiciones de bonos convertibles en los datasets 13F y que su naturaleza se refleja ademas mediante `PRN` en ejemplos reales.

---

# 6. Caso NIPST - decision

Los ejemplos que has encontrado son particularmente utiles:

    TITLEOFCLASS = CONVERTIBLE BOND
    SSHPRNAMTTYPE = SH
    PUTCALL = NULL

No deben reinterpretarse automaticamente como acciones.

Y tampoco recomendaria crear una excepcion textual especifica para `NIPST`.

La linea debe seguir existiendo en el dataset tecnico, pero quedar fuera del **universo operativo de equity** salvo que la capa de identidad/security classification demuestre que realmente es una equity security.

Esto mantiene separadas:

    13F raw eligibility

de:

    NIPC equity eligibility

---

# 7. Universo tecnico vs universo operativo

El diseno del Gate anterior queda reforzado:

### Universo tecnico

    13F
    +
    SH
    +
    PUTCALL null

con todas las anomalias conservadas.

### Universo operativo NIPC

    technical universe
    INTERSECT
    resolved security
    INTERSECT
    security_type = EQUITY
    INTERSECT
    radar universe

y, posteriormente:

    mapping coverage sufficient

Solo entonces puede pasar a:

    NIPC status = READY

---

# 8. Sobre `Section 13(f) eligible`

Hay otra mejora importante para Gate-NIPC.1.

La SEC mantiene una **Official List of Section 13(f) Securities**, actualizada trimestralmente. La lista es una fuente normativa de elegibilidad 13(f), distinta del ticker.

Por tanto, la especificacion deberia distinguir:

    section13f_eligible

de:

    security_type = EQUITY

y de:

    ticker_mapped

No son sinonimos.

Una security puede ser:

    13f_eligible = true
    security_type = non-equity
    ticker = null

y eso no seria una contradiccion.

---

# 9. Los 22 tickers sin mapping

No recomiendo abrir una investigacion individual ahora sobre los 22.

El objetivo de este Gate era medir el **sistema de mapping**, no resolver manualmente la cola actual.

La clasificacion adecuada es:

    UNMAPPED_RADAR_SECURITY

y el inventario queda como input para Gate-NIPC.1.

Especialmente:

    BRK-B
    MOG-A

deben tratarse como problemas de **clase/security identity**, no como simples faltantes de ticker.

---

# 10. Fuentes internas: resultado del Gate

La conclusion del inventario esta aprobada:

    cusip_ticker_exceptions
        -> puntual

    etf_holdings
        -> util, pero universo parcial

    isin_ticker_map
        -> no resuelve CUSIP

    index_holdings
        -> sin CUSIP

    amundi_yahoo_mapping
        -> no resuelve universo USA

Por tanto:

## **NO EXISTE TODAVIA FUENTE INTERNA SUFICIENTE PARA UN CROSSWALK MASIVO CUSIP -> TICKER**

Esto no es un defecto de FA-2 ni de NIPC.

Es el **gap estructural principal que Gate-NIPC.1 tiene que disenar**.

---

# 11. Cobertura ponderada: shares vs VALUE

La eleccion de:

    SSHPRNAMT

para la cobertura de NIPC queda aprobada.

`VALUE` debe mantenerse unicamente como control secundario.

La razon es contractual y semantica:

    NIPC = DeltaShares

por lo que una cobertura del 55% en `VALUE` no puede compensar una cobertura de solo 30% en `SHARES`.

La diferencia:

    30% shares
    55% value

es informativa, pero no cambia el estado:

    INSUFFICIENT

---

# 12. Que debe especificar Gate-NIPC.1

Ahora si hay suficiente evidencia para cerrar la estructura del contrato interno.

Debe definir, como minimo:

    security_identity
    section13f_eligibility
    security_type
    ticker_mapping
    source_precedence
    temporal_validity
    one_to_one / ambiguous
    mapping_coverage
    weighted_share_coverage
    unmapped_weight
    operational_universe
    status

Y especialmente:

    mapped
    !=
    equity
    !=
    section13f eligible
    !=
    radar member

Son cuatro propiedades distintas.

---

# 13. Decision sobre fuente externa

### **NO elegir todavia OpenFIGI ni otra fuente**

El Gate 0 ha cumplido su funcion: demuestra que las fuentes internas no son suficientes.

Ahora corresponde a Gate-NIPC.1 comparar alternativas externas **antes** de implementar.

La comparacion debera incluir:

    coverage
    temporal validity
    CUSIP support
    US equities
    class handling
    rate limits
    reproducibility
    licensing/usage
    failure modes

Solo despues se selecciona una fuente o combinacion.

---

# 14. Decisiones consolidadas

| Punto                               | Dictamen                                       |
| ----------------------------------- | ---------------------------------------------- |
| Gate 0 Mapping                      | **PASS**                                       |
| Crosswalk interno masivo            | **NO EXISTE**                                  |
| 90.91% radar securities             | **dato observado, no umbral**                  |
| 29.995% radar shares                | **INSUFFICIENT**                               |
| 55.035% radar value                 | **solo contraste**                             |
| Q-NIPC-8 allowlist `TITLEOFCLASS`   | **NO**                                         |
| Q-NIPC-8 clasificacion por security | **GO**                                         |
| `TITLEOFCLASS`                      | **validacion/anomalia, no autoridad primaria** |
| `SH + null`                         | **se mantiene como filtro base**               |
| Section 13(f) eligibility           | **dimension separada**                         |
| NIPST                               | **no interpretar automaticamente como equity** |
| 22 radar unmapped                   | **UNMAPPED_RADAR_SECURITY**                    |
| Fuente externa                      | **pendiente de especificacion**                |
| NIPC productivo actual              | **INSUFFICIENT**                               |
| Gate-NIPC.1                         | **AUTORIZADO**                                 |
| Codigo                              | **NO autorizado todavia**                      |

---

# DICTAMEN FORMAL

## **GATE 0 MAPPING -> PASS**

La investigacion ha cerrado correctamente la incognita que bloqueaba el diseno:

> **las fuentes internas disponibles no proporcionan un crosswalk masivo suficiente para NIPC.**

El siguiente paso no es seguir curando CUSIPs manualmente uno por uno. Es disenar una **capa formal de identidad de security/mapping**.

Y dejo una regla arquitectonica congelada para el siguiente Gate:

    TITLEOFCLASS
          v
    NO es una allowlist de equity

    security identity
          v
    security_type
          v
    EQUITY / NON_EQUITY
          v
    ticker mapping
          v
    radar intersection

La cobertura actual de `29.995%` de shares del universo radar mantiene el NIPC en:

    INSUFFICIENT

hasta que Gate-NIPC.1 defina y posteriormente implemente una capa de mapping suficiente.

**Siguiente paso autorizado: `Gate-NIPC.1 - Especificacion`, sin codigo.**
