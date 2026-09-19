# DICTAMEN DEL AUDITOR EXTERNO - GATE-NIPC.1 v1.1

**Fecha:** 19-09-2026
**Referencia:** IAE / NIPC / Especificacion v1.1
**HEAD:** `52fc53c`
**Resultado:** **PASS CONDICIONADO - GO para cerrar especificacion tras 2 ajustes semanticos; despues Gate-NIPC.2 puede recibir GO**
**Codigo productivo:** todavia NO autorizado

La v1.1 incorpora correctamente las cinco correcciones obligatorias. La arquitectura de `reported_position_unit`, la separacion tecnico/operativo, el tratamiento de `DFND`, la cobertura pairwise y el estado `TEMPORAL_UNVERIFIED` quedan bien encaminados. La SEC mantiene separadas en 13F las dimensiones de CUSIP, numero de shares, tipo de instrumento, PUT/CALL, investment discretion y otros managers; OpenFIGI, por su parte, distingue `FIGI`, `shareClassFIGI`, `ticker` y `securityType`, y establece que el FIGI no cambia cuando se produce una corporate action. ([SEC][1])

Hay, sin embargo, **dos correcciones que considero obligatorias antes de autorizar codigo**.

---

## 1. Q-ESP-V11-1 - `canonical_security`

### **APROBADO CONDICIONADO**

La separacion:

```text
observed_security_identifier
        !=
canonical_security
```

es correcta y necesaria.

Tambien apruebo:

```text
(report_period, observed_CUSIP) -> canonical_security
```

y la tabla temporal:

```text
data/mappings/cusip_equivalence.csv
```

para corporate actions.

### Pero hay un problema en la representacion propuesta

Has definido:

```text
figi:<shareClassFIGI>
ticker:<normalized>
cusip:<CUSIP>
unresolved:<CUSIP>
```

como posibles valores de `canonical_security`.

**No apruebo `ticker:<ticker>` como identidad canonica estable.**

OpenFIGI distingue explicitamente `shareClassFIGI` de `ticker`; ademas, su documentacion de corporate actions establece que el FIGI permanece intacto cuando cambia el ticker. ([OpenFIGI][2])

Por tanto:

```text
ticker
   !=
stable security identity
```

y no debe poder aparecer como `canonical_security` definitivo.

### Tambien hay que corregir `cusip:<CUSIP>`

Un CUSIP puede ser perfectamente valido como **fallback de identidad observada**, pero no necesariamente como identidad estable entre periodos.

Ejemplo:

```text
Q4: CUSIP_A
Q1: CUSIP_B
```

con evidencia de que ambas representan la misma clase:

```text
CUSIP_A --equivalence--> canonical_security_X
CUSIP_B --equivalence--> canonical_security_X
```

Sin esa equivalencia, el sistema debe comportarse conservadoramente como:

```text
Q4 = security_A
Q1 = security_B
```

y no fabricar continuidad.

### Correccion que queda congelada

Sugiero distinguir internamente:

```text
canonical_security
    = identidad estable realmente resuelta

observed_security_key
    = cusip:<CUSIP>
```

Y los estados:

```text
CANONICAL_FIGI
CANONICAL_EQUIVALENCE
OBSERVED_CUSIP_ONLY
UNRESOLVED
```

Asi:

```text
cusip:<CUSIP>
```

puede seguir existiendo tecnicamente, pero **no se presenta como identidad canonica estable si no hay evidencia suficiente**.

### Decision

**Q-ESP-V11-1 = GO condicionado a esta precision.**

---

# 2. Q-ESP-V11-2 - clave de matching

## **APROBADA**

La clave:

```text
(
    filing_manager_cik,
    canonical_security,
    discretion_type
)
```

es correcta.

Y:

```text
FULL OUTER JOIN
```

es el mecanismo correcto para detectar:

```text
BOTH
NEW
EXIT
UNRESOLVED_IDENTITY
```

No debe incluir `report_period` en la clave de union.

**Q-ESP-V11-2 = GO.**

---

# 3. Q-ESP-V11-3 - coverage pairwise

## **APROBADA**

Muy buena correccion.

Las seis metricas:

```text
coverage_previous
coverage_current
paired_security_coverage
paired_weighted_share_coverage
unmapped_weight_previous
unmapped_weight_current
```

son suficientes como contrato minimo.

Y mantengo una regla:

> **NIPC no puede publicar una unica cifra de coverage.**

La cobertura de Q1 no demuestra que la misma poblacion pueda ser emparejada con Q4.

**Q-ESP-V11-3 = GO.**

---

# 4. Q-ESP-V11-4 - `TEMPORAL_UNVERIFIED`

## **APROBADA, con un matiz importante**

La politica:

```text
OpenFIGI actual
!=
mapping historico automatico
```

es correcta.

Pero esta regla:

```text
mapping_timestamp_utc > report_period
-> TEMPORAL_UNVERIFIED
```

es demasiado mecanica.

OpenFIGI establece que **el FIGI no cambia por una corporate action**. Por tanto, una resolucion obtenida hoy puede identificar correctamente una clase que ya existia en Q1 2026; lo que no podemos asumir automaticamente es que el **ticker o metadata temporal asociada** fuera la misma. ([OpenFIGI][3])

Por tanto, el contrato debe distinguir:

```text
identity temporal validity
        !=
metadata temporal validity
```

Ejemplo:

```text
shareClassFIGI = estable
ticker = actual
```

podria ser suficiente para identificar la security historica, pero **no** para afirmar que el ticker actual era el ticker de 2026-03-31.

### Decision

Mantengo:

```text
TEMPORAL_UNVERIFIED
```

pero recomiendo que el contrato diga explicitamente:

> **La fecha de consulta OpenFIGI no invalida por si sola una identidad FIGI estable; si impide asumir automaticamente que los atributos temporales de la respuesta actual eran validos en el `report_period`.**

**Q-ESP-V11-4 = GO condicionado a esta precision.**

---

# 5. Q-ESP-V11-5 - `security_type_source/status`

## **APROBADA**

La separacion:

```text
security_type_source
security_type_status
```

es correcta.

Tambien apruebo:

```text
RESOLVED_EQUITY
RESOLVED_NON_EQUITY
UNRESOLVED
CONFLICT
```

y:

```text
TITLEOFCLASS
```

como evidencia, no como allowlist.

La propia SEC mantiene `TITLEOFCLASS`, `PUTCALL` y `SSHPRNAMTTYPE` como campos separados, por lo que no existe razon para colapsarlos en una clasificacion textual unica. ([SEC][1])

**Q-ESP-V11-5 = GO.**

---

# 6. Q-ESP-V11-6 - observaciones adicionales

## **APROBADO**

### `section13f_eligible` con cinco estados

Correcto.

La Official List es trimestral y distingue adiciones, eliminaciones y estado activo; la SEC confirma oficialmente el campo `STATUS` como la dimension de cambios de elegibilidad. ([SEC][4])

### 18 familias de tests

**Aprobadas.**

### `identity/sec13f_list.py`

**Aprobado.**

La responsabilidad queda limpia:

```text
SEC list parser
    v
eligibility

identity resolver
    v
security identity

aggregation
    v
DeltaShares
```

No mezclar.

---

# 7. Q-ESP-V11-7 - ?autorizar Gate-NIPC.2?

## **Todavia no de forma inmediata**

No porque haya un problema de arquitectura general, sino por los dos ajustes anteriores:

```text
1. canonical_security no puede ser ticker definitivo
2. temporalidad FIGI debe separar identidad de metadata
```

Una vez incorporados esos dos cambios documentales:

## **GO para Gate-NIPC.2**

Y el alcance aprobado es exactamente:

```text
aggregation/
+---- delta_shares.py
+---- nipc.py

identity/sec13f_list.py

OpenFIGI cache
cusip_equivalence.csv
tests
```

No incluir:

```text
breadth.py
new_exit.py
classification.py
report integration
```

---

# 8. Q-ESP-V11-8 - API key OpenFIGI

## **SI - AUTORIZADA**

Se puede solicitar la API key.

OpenFIGI documenta actualmente:

```text
sin key:
25 requests/min
10 jobs/request

con key:
25 requests/6 sec
100 jobs/request
```

([OpenFIGI][2])

Pero quedan congeladas estas reglas:

```text
API key
    v
env var / secret
```

Nunca:

```text
Git
manifest
source code
logs
README
```

Y la clave no debe formar parte del hash de reproducibilidad.

**Q-ESP-V11-8 = GO.**

---

# 9. Q-ESP-V11-9 - coverage thresholds

## **NO fijarlos todavia en este dictamen**

Estoy de acuerdo en que la v1.1 permanezca:

```text
THRESHOLD_1 = UNDEFINED
THRESHOLD_2 = UNDEFINED
```

Pero los thresholds deben fijarse **antes de Gate-NIPC.2**, no durante la implementacion.

Recomiendo un pequeno dictamen/appendix especifico:

```text
NIPC_COVERAGE_POLICY.md
```

o una seccion explicita de la v1.2.

El criterio debe considerar al menos:

```text
paired_security_coverage
paired_weighted_share_coverage
unmapped_weight_previous
unmapped_weight_current
```

No:

```text
number of mapped securities only
```

Y evitaria tambien fijarlos por simple analogia con el actual 90.91%.

**Q-ESP-V11-9 = GO condicionado a dictamen de thresholds previo a codigo.**

---

# 10. Q-ESP-V11-10 - `cusip_equivalence.csv` vacio

## **APROBADO**

Si:

```text
data/mappings/cusip_equivalence.csv
```

puede crearse inicialmente con:

```text
0 filas de datos
+ cabecera
```

Esto es incluso preferible.

No debemos introducir ahora equivalencias historicas "por conocimiento general" que no hayan sido necesitadas y documentadas.

La regla:

```text
sin equivalence entry
    v
no inferir equivalencia
```

es coherente con vuestro estandar de no inventar.

### Pero debe anadirse

Cada equivalencia futura deberia contener:

```text
CUSIP_A
CUSIP_B / canonical_security
valid_from
valid_to
source
reason
verified_by
```

y, cuando sea posible:

```text
source_document
```

**Q-ESP-V11-10 = GO.**

---

# 11. Una cuestion que recomiendo congelar antes de implementar

Hay una pequena incoherencia terminologica entre:

```text
canonical_security
```

y:

```text
CUSIP fallback
```

La solucion no requiere cambiar la arquitectura; solo la nomenclatura.

Propongo:

```text
observed_security_identifier
    v
security_resolution
    +---- CANONICAL
    +---- OBSERVED_ONLY
    +---- UNRESOLVED
    +---- AMBIGUOUS
    +---- CONFLICT
```

Y:

```text
canonical_security
```

solo cuando existe una identidad que puede utilizarse legitimamente en el matching interperiodo.

Esto evitara un bug conceptual extremadamente peligroso:

```text
Q4 CUSIP_A
   v
CUSIP fallback
   v
canonical_security = A

Q1 CUSIP_B
   v
CUSIP fallback
   v
canonical_security = B

NIPC = NEW + EXIT
```

cuando en realidad podria ser:

```text
A = B
NIPC = DeltaShares
```

---

# 12. Resumen de dictamen

| Pregunta     | Dictamen                                            |
| ------------ | --------------------------------------------------- |
| Q-ESP-V11-1  | **GO condicionado**                                 |
| Q-ESP-V11-2  | **GO**                                              |
| Q-ESP-V11-3  | **GO**                                              |
| Q-ESP-V11-4  | **GO condicionado**                                 |
| Q-ESP-V11-5  | **GO**                                              |
| Q-ESP-V11-6  | **GO**                                              |
| Q-ESP-V11-7  | **GO condicionado**                                 |
| Q-ESP-V11-8  | **GO**                                              |
| Q-ESP-V11-9  | **Thresholds obligatorios antes de implementacion** |
| Q-ESP-V11-10 | **GO**                                              |

# DICTAMEN FORMAL

## **GATE-NIPC.1 v1.1 -> PASS CONDICIONADO**

La especificacion esta **arquitectonicamente madura** y no requiere una nueva reformulacion general.

Antes de Gate-NIPC.2 solo deben cerrarse:

```text
C1
canonical_security != ticker
CUSIP fallback != identidad estable automatica

C4
FIGI identity validity
!=
current metadata validity
```

Y despues:

```text
threshold policy
        v
Gate-NIPC.2
        v
implementation
```

### La arquitectura queda congelada asi:

```text
13F source line
      v
observed CUSIP
      v
security resolution
      +---- internal
      +---- OpenFIGI
      +---- equivalence
      +---- unresolved
      v
canonical security
      v
section13f eligibility
      v
security type
      v
radar
      v
reported_position_unit
      v
Q4 <-> Q1
      v
DeltaShares
      v
NIPC
```

La separacion entre `shareClassFIGI` y ticker es especialmente importante: OpenFIGI establece que los FIGI permanecen sin cambios ante corporate actions, mientras el ticker asociado puede cambiar. ([OpenFIGI][3])

**Una vez aplicadas esas dos precisiones a v1.1 y fijada la politica de thresholds, autorizo Gate-NIPC.2.**