# DICTAMEN DEL AUDITOR EXTERNO - GATE-NIPC.1

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / Especificacion v1.0
**HEAD:** `e6f550a`
**Resultado:** **PASS CONDICIONADO - especificacion conceptualmente aprobada; Gate-NIPC.2 todavia NO autorizado**
**Codigo productivo:** NO

He revisado la especificacion contra las decisiones acumuladas y la documentacion SEC vigente. La SEC mantiene `CUSIP`, `SSHPRNAMT`, `SSHPRNAMTTYPE`, `PUTCALL`, `INVESTMENTDISCRETION` y `OTHERMANAGER` como dimensiones distintas del Information Table; asimismo, `Column 7` identifica managers adicionales vinculados a las posiciones reportadas.

La estructura es buena, pero **hay cuatro cuestiones que deben quedar cerradas antes de convertir esta especificacion en codigo**.

---

## Q-ESP-1 - Estructura del documento

### **APROBADA**

Las 16 secciones son adecuadas.

La separacion:

    identity/
        v
    mapping / eligibility
        v
    aggregation/

es correcta y evita mezclar "quien es la security" con "cuantas shares cambian".

Tambien apruebo que el documento sea: complementario + contractual para NIPC + sin codigo.

**Q-ESP-1 = GO.**

---

# Q-ESP-2 - `reported_position_unit`

## **APROBADA, pero falta congelar la identidad de `canonical_security`**

La definicion:

    (
        report_period,
        filing_manager_cik,
        canonical_security,
        discretion_type
    )

es correcta **como unidad logica por periodo**.

Tambien queda definitivamente aprobada:

    SSHPRNAMT = una vez por source line

y:

    multi-edge OTHERMANAGER != multiplicacion economica

Esto es coherente con la estructura 13F: Column 7 enlaza una linea con otros managers, pero la cantidad de shares sigue siendo la cantidad reportada en esa linea.

## Pero hay un punto critico

`canonical_security` **no puede quedar definido implicitamente como ticker ni como CUSIP crudo**.

Ya hemos observado corporate actions donde el CUSIP cambia temporalmente. Y BRK-A/BRK-B demuestra que ticker e identidad de security tampoco son equivalentes.

Por tanto, debe existir una distincion explicita:

    observed_security_identifier = CUSIP del 13F

    canonical_security = identidad estable de la security

y una capa:

    (report_period, observed_CUSIP)
            v
    canonical_security

Esto es **obligatorio** para poder calcular:

    Q4 2025
       v
    Q1 2026

sin tratar un cambio de CUSIP como un `Exit + New`.

### Decision

**Q-ESP-2 = GO condicionado a definir formalmente `canonical_security`.**

No aceptaria implementar `compute_delta_shares()` mientras esto siga ambiguo.

---

# Q-ESP-3 - universo tecnico vs operativo

## **APROBADO**

La separacion es correcta:

    technical_universe = 13F canonical lines

y:

    operational_universe = radar ^ 13F eligible ^ mapped ^ EQUITY

La Official List de la SEC es una dimension independiente y se publica trimestralmente.

Ademas:

    section13f_eligible != equity != mapped != radar_member

debe permanecer como regla contractual.

**Q-ESP-3 = GO.**

---

# Q-ESP-4 - secuencia de Gates

## **APROBADA**

La secuencia Gate-NIPC.1 -> .2 -> .3 -> .4 es correcta.

Y mantengo la regla: NO PUSH hasta Gate-NIPC.3 PASS.

No hace falta ampliar el programa.

**Q-ESP-4 = GO.**

---

# Q-ESP-5 - interfaces

## **APROBADAS CON UNA MODIFICACION**

Las tres funciones tienen responsabilidades claras:

    compute_reported_position_units()
    compute_delta_shares()
    compute_nipc()

pero `compute_delta_shares()` no deberia aceptar `match_keys=None` como mecanismo ambiguo.

La clave de matching debe quedar **fijada en el contrato**, porque un consumidor podria introducir accidentalmente `(report_period, ...)` o `filing_manager_cik + CUSIP` y producir resultados inconsistentes.

### Clave de matching aprobada

Para cruzar dos periodos:

    (
        filing_manager_cik,
        canonical_security,
        discretion_type
    )

**sin `report_period`**.

`report_period` pertenece a cada snapshot, no a la union entre snapshots.

Por tanto:

    units_2025Q4
        FULL OUTER JOIN
    units_2026Q1
        ON (filing_manager_cik, canonical_security, discretion_type)

Esto genera BOTH / NEW / EXIT y no una falsa dependencia temporal en la clave.

### Otro punto importante

La salida deberia conservar `match_status` con al menos: BOTH / NEW / EXIT, y posiblemente UNRESOLVED_IDENTITY para no confundir ausencia de posicion con incapacidad de matching.

**Q-ESP-5 = GO condicionado a fijar la clave de matching en el contrato.**

---

# Q-ESP-6 - tests

## **APROBADOS, pero falta una familia critica**

Los 14 previstos son buenos, pero anadiria obligatoriamente:

    test_cusip_change_same_canonical_security

Ejemplo conceptual:

    Q4: CUSIP_A -> Security_X -> 100 shares
    Q1: CUSIP_B -> Security_X -> 120 shares

resultado: delta = +20

y **no**: EXIT(A) + NEW(B).

### Anadir tambien

    test_multi_edge_does_not_duplicate_delta

con una linea que tenga varios managers incluidos.

    test_unresolved_mapping_blocks_pair_match

porque una security mapeada en Q1 pero no en Q4 no debe generar un delta artificial.

    test_new_and_exit_zero_baseline_current

para garantizar NEW: previous = 0; EXIT: current = 0.

### Por tanto

No son 14 pruebas conceptuales; recomiendo **18 familias minimas**.

No hace falta que sean exactamente 18 funciones de test, pero esas cuatro propiedades deben estar cubiertas.

**Q-ESP-6 = GO condicionado a ampliar cobertura.**

---

# Q-ESP-7 - umbrales

## **NO apruebo el `>=90%` como umbral contractual**

90.91% es un dato observado. No debe transformarse en threshold = 90% porque el threshold estaria practicamente derivado del propio resultado.

### Si apruebo

Que v1.0 diga: thresholds = UNDEFINED, y que sean fijados **antes de Gate-NIPC.2**.

Pero hay una cuestion adicional:

No basta con un unico `weighted_share_coverage` para un calculo entre dos periodos.

Necesitamos al menos:

    coverage_previous
    coverage_current
    paired_security_coverage
    paired_weighted_share_coverage
    unmapped_weight_previous
    unmapped_weight_current

porque una security puede estar mapeada en Q1 y no en Q4, y entonces su `DeltaShares` no es observable aunque exista un mapping en Q1.

Esto debe quedar en la especificacion.

**Q-ESP-7 = GO condicionado.**

---

# Q-ESP-8 - temporalidad OpenFIGI

## **NO recomiendo aplazar el diseno hasta Gate-NIPC.2**

Este punto afecta directamente a la correccion del NIPC.

El probe OpenFIGI utilizo la base **actual**. Ya sabemos que una respuesta actual no demuestra por si sola que esa metadata sea valida historicamente para 2025-12-31 o 2026-03-31.

La documentacion de OpenFIGI identifica campos como `ticker`, `securityType`, `shareClassFIGI`, etc., pero no convierte automaticamente la metadata actual en un snapshot historico.

Por tanto, Gate-NIPC.1 debe congelar la politica:

    OpenFIGI actual
            v
    NO equivale automaticamente
            v
    mapping historico

### Regla aprobada

Si una resolucion OpenFIGI no puede demostrarse valida para el `report_period`:

    mapping_status = TEMPORAL_UNVERIFIED

y **no entra en el operational_universe**. Podra utilizarse como candidato/diagnostico.

Esto evita que el sistema convierta una respuesta actual en una verdad historica.

### No hace falta resolver todavia como obtener la evidencia historica

Eso si puede investigarse en Gate-NIPC.2.

Pero la **politica de no asumir temporalidad** debe quedar ya en el contrato.

**Q-ESP-8 = NO tal como esta; GO despues de introducir esta regla en la especificacion.**

---

# Q-ESP-9 - parser SEC 13(f)

## **GO - `identity/sec13f_list.py`**

Es la ubicacion correcta. La responsabilidad es:

    identity/
        sec13f_list.py

porque el modulo responde: "Es esta security elegible para Section 13(f) en este periodo?".

No: "Cuanto cambio la posicion?". Por tanto no lo pondria en `aggregation/`.

### Responsabilidades

Debe limitarse a parsear fixed-width 80 (CUSIP, option_indicator, issuer_name, issuer_description, status) y devolver:

    section13f_eligible
    status
    option_indicator
    raw_line

No debe resolver ticker ni security identity.

La SEC define formalmente `01-09 CUSIP`, `10 Option Indicator`, `11-40 Issuer Name`, `41-67 Issuer Description`, `68-70 STATUS` y posicion 80 como `Misc/Unused`.

**Q-ESP-9 = GO.**

---

# Q-ESP-10 - autorizar Gate-NIPC.2?

## **TODAVIA NO**

La arquitectura esta suficientemente madura, pero antes de escribir codigo deben incorporarse **cuatro correcciones obligatorias** a la v1.0:

### 1. Definir `canonical_security`

    CUSIP observado
            v
    canonical_security estable

Debe ser explicito.

### 2. Fijar la clave de matching

    (filing_manager_cik, canonical_security, discretion_type)

sin `report_period`.

### 3. Definir coverage pairwise

No solo `coverage_current`, sino cobertura **de comparabilidad Q4->Q1**.

### 4. Fijar `TEMPORAL_UNVERIFIED`

Una respuesta OpenFIGI actual no puede entrar silenciosamente en el snapshot historico.

---

# 13. Una observacion adicional sobre `security_type`

Este es el quinto punto que recomiendo anadir a la especificacion.

Actualmente pone `security_type == EQUITY` pero todavia no esta suficientemente definido **quien tiene autoridad para afirmar `EQUITY`**.

Hemos descartado `TITLEOFCLASS` allowlist y FIGI del filer como pivote.

OpenFIGI si devuelve `securityType`, pero no queremos convertirlo automaticamente en autoridad unica.

Por ello Gate-NIPC.1 debe introducir `security_type_source` y `security_type_status`, por ejemplo: RESOLVED_EQUITY / RESOLVED_NON_EQUITY / UNRESOLVED / CONFLICT.

Esto sera determinante para casos como SH + PUTCALL NULL + CONVERTIBLE BOND, que ya descubristeis.

---

# 14. Una observacion adicional sobre `section13f_eligible`

La regla `CUSIP in Official List AND STATUS != DELETED` es valida como aproximacion contractual actual.

Pero recomiendo expresarla como:

    section13f_eligible(period, cusip)
    =
    estado oficial del CUSIP
    en la Official List correspondiente al periodo

con: NOT_IN_LIST | DELETED | ADDED | ACTIVE | CONFLICT.

No reducir toda la dimension a un simple booleano internamente. El booleano puede derivarse posteriormente.

La SEC define trimestralmente la Official List y `STATUS` indica incorporaciones/deletions.

---

# 15. Dictamen consolidado

| Pregunta                | Dictamen                                            |
| ----------------------- | --------------------------------------------------- |
| Q-ESP-1                 | **GO**                                              |
| Q-ESP-2                 | **GO condicionado a definir canonical_security**    |
| Q-ESP-3                 | **GO**                                              |
| Q-ESP-4                 | **GO**                                              |
| Q-ESP-5                 | **GO condicionado; fijar match key**                |
| Q-ESP-6                 | **GO condicionado; ampliar tests**                  |
| Q-ESP-7                 | **GO condicionado; no threshold 90%**               |
| Q-ESP-8                 | **NO tal como esta; fijar politica temporal ahora** |
| Q-ESP-9                 | **GO - `identity/sec13f_list.py`**                  |
| Q-ESP-10                | **NO todavia**                                      |
| security_type authority | **debe anadirse**                                   |

---

# DICTAMEN FORMAL

## **Gate-NIPC.1 -> PASS CONDICIONADO**

La especificacion **no necesita ser redisenada**. La arquitectura fundamental queda aprobada:

    13F raw
      v
    canonical snapshot
      v
    security identity
      v
    section13f eligibility
      v
    security type
      v
    mapping
      v
    operational universe
      v
    reported_position_unit
      v
    Q4 <-> Q1 matching
      v
    DeltaShares
      v
    NIPC

Y quedan congeladas las reglas fundamentales:

    1 source line -> 1 SSHPRNAMT
    multi-edge -> provenance != multiple shares
    canonical_reporting_relationship_key -> provenance
    reported_position_unit -> aggregation

### Antes de Gate-NIPC.2

El documento v1.0 debe incorporar: **`canonical_security` estable**, **match key interperiodo**, **coverage pairwise**, **TEMPORAL_UNVERIFIED** y **autoridad/status de `security_type`**.

Una vez corregidos esos puntos: **Gate-NIPC.2 podra recibir GO para implementacion.**

No recomiendo hacer ningun codigo antes de esas modificaciones. La razon principal es que, sin `canonical_security` estable y sin una politica de matching interperiodo explicita, un pipeline perfectamente programado podria producir un NIPC matematicamente correcto pero **semanticamente falso** ante un cambio de CUSIP o una mapping gap entre Q4 y Q1.

**Estado final: v1.0 = PASS CONDICIONADO; revision v1.1 requerida; despues GO a implementacion.**
