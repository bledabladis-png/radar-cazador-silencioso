# DICTAMEN FORMAL DEL AUDITOR

**Objeto:** evaluacion del probe end-to-end NIPC Q4 2025 -> Q1 2026
**HEAD revisado:** `b75c262`
**Naturaleza:** caracterizacion previa a Gate-NIPC.3; no normativo
**Dictamen:** **GO para mini-probe de continuidad de filer**
**Gate-NIPC.2:** **NO CERRADO**
**Gate-NIPC.3:** **NO AUTORIZADO**

---

## 1. Conclusion ejecutiva

El probe descubre una limitacion estructural que debe tratarse **antes de fijar los thresholds de coverage**.

La causa no es un fallo de implementacion del `match_key`. El contrato actual:

```text
(filing_manager_cik,
 canonical_security,
 discretion_type)
```

mide continuidad del **filer CIK**, no continuidad economica del conjunto reportado.

El caso Vanguard demuestra que ambas dimensiones pueden divergir: una posicion puede seguir presente sobre el mismo CUSIP mientras el reporting pasa de un filer a otros filers relacionados mediante la mecanica 13F-NT. La SEC define precisamente el 13F-NT como el aviso en el que las posiciones del reporting manager son reportadas por otro(s) reporting manager(s), y exige la identificacion de esos otros managers en el Cover Page.

Por tanto:

> **El motor esta funcionando conforme al contrato; el contrato no cubre todavia la discontinuidad de filer entre periodos.**

Este hallazgo **precede** a la fijacion de thresholds.

---

# 2. Q-PROBE-1 - Tratamiento del hallazgo

### Decision: **ACEPTAR COMO LIMITACION CONOCIDA + TRIGGER DE DISENO**

No recomiendo describirlo simplemente como:

> "NIPC asume continuidad de filer".

Es demasiado amplio y puede inducir a error.

La formulacion tecnicamente correcta es:

> **"El matching interperiodo actual utiliza `filing_manager_cik` como componente de la unidad de posicion. Por tanto, mide continuidad del reporting manager/filer. Una reorganizacion del reporting entre managers puede producir pares `EXIT` + `NEW` sobre la misma security aunque la exposicion reportada continue dentro de la estructura de filing."**

Esto es exactamente lo que el caso Vanguard pone de manifiesto.

Debe documentarse en:

```text
NIPC_COVERAGE_POLICY.md
INSTITUTIONAL_ACCUMULATION_NIPC_ESPECIFICACION_v1.3.md
```

pero **sin modificar todavia la semantica de C2**.

La C2 actual permanece vigente.

### Estado

**CERRADO COMO HALLAZGO DOCUMENTAL.**

No requiere patch.

---

# 3. Q-PROBE-2 - ?Reconciliacion via OTHERMANAGER2?

### Decision: **AUTORIZADO COMO INVESTIGACION; NO AUTORIZADA TODAVIA COMO RECONCILIACION**

La primera investigacion debe utilizar la evidencia que ya existe en SEC 13F:

```text
13F-NT
   v
List of Other Managers Reporting for this Manager
   v
CIK / relacion de reporting
```

La propia SEC contempla que un 13F-NT identifique a los managers que reportan las posiciones del manager notificante.

Por tanto, **si esta autorizado investigar los `OTHERMANAGER2` ya resueltos por FA-2.3**.

Pero hay una frontera critica:

```text
reporting relationship
        !=
economic ownership
        !=
economic continuity
```

Un `OTHERMANAGER2` puede demostrar:

> "este manager declara que otro manager reporta sus posiciones"

pero **no basta por si solo para afirmar que un `EXIT` en CIK A y un `NEW` en CIK B constituyen la misma exposicion economica entre Q4 y Q1**.

Esto queda fuera de la semantica actual y de las prohibiciones vigentes sobre `economic_owner_cik`.

### Por tanto

No se autoriza todavia:

```text
CIK_Q4 -> parent/child -> CIK_Q1
```

ni:

```text
EXIT(Q4, CIK_A) + NEW(Q1, CIK_B)
-> mismo propietario economico
```

como regla productiva.

Primero debe demostrarse que existe una **relacion de reporting suficientemente determinista y longitudinal** para resolver el caso.

### Fuente adicional

En esta fase **no exigiria todavia una fuente externa adicional**.

Primero debe explotarse toda la evidencia SEC ya ingerida:

```text
COVERPAGE
13F-HR
13F-NT
OTHERMANAGER
OTHERMANAGER2
ACCESSION
PERIODOFREPORT
```

Si esa evidencia resulta insuficiente para construir continuidad longitudinal de forma segura, entonces se abrira una decision separada sobre fuente adicional.

---

# 4. Q-PROBE-3 - Thresholds + continuidad de filer

### Decision: **ACEPTAR**

Los thresholds de coverage **no deben fijarse ignorando este hallazgo**.

Pero tampoco deben modificarse para "compensarlo" arbitrariamente.

La politica correcta debe separar dos dimensiones:

```text
A) Coverage / comparabilidad de securities
B) Continuidad de reporting manager
```

Conceptualmente:

```text
Coverage suficiente
        +
Continuidad suficiente
        v
condiciones necesarias para interpretar NIPC
```

No recomiendo convertir la continuidad de filer en un unico ratio artificial antes de medirla.

Primero hay que observar:

* proporcion de `NEW/EXIT` atribuible a cambios de filer;
* peso `SSHPRNAMT` afectado;
* numero de securities afectadas;
* numero de filers afectados;
* concentracion de ese fenomeno en grandes managers;
* saldo bruto y neto generado por esas discontinuidades.

El caso Vanguard muestra por que esto es necesario: con el `match_key` actual aparecen aproximadamente:

```text
SOLE  ~= -41.16 B
DFND  ~= +41.28 B
```

mientras el neto observable es de solo unas decenas de millones.

Esto es evidencia de una **cancelacion interna muy grande**, pero todavia no prueba por si sola que todo ese bruto sea una reorganizacion economica. Esa conclusion requiere cerrar la relacion SEC del caso.

### Decision sobre thresholds

**NO fijar aun `THRESHOLD_1` ni `THRESHOLD_2`.**

Primero:

```text
Q-PROBE-5
    v
medicion de continuidad
    v
revision de cobertura
    v
propuesta de thresholds
```

---

# 5. Q-PROBE-4 - `paired_security_coverage < 4%`

### Decision: **CONFIRMAR MODO DIAGNOSTICO**

Con el crosswalk interno actual:

```text
ALL       1.81%
ELIGIBLE  3.68%
```

la cobertura de identidad canonica es extremadamente limitada.

Esto no significa que el 98% de los valores del dataset sean "erroneos". Significa algo mas concreto:

> **el sistema actualmente solo puede construir continuidad interperiodo canonica para una fraccion muy pequena del universo bajo la definicion C1 revisada.**

Por tanto:

**NO es publicable como NIPC representativo del universo 13F.**

### ?Autorizar un "canonical subset"?

**Si, pero solamente como scope experimental/diagnostico.**

Puede existir:

```text
scope = CANONICAL_SUBSET
```

para:

* estudiar comportamiento del motor;
* validar DeltaShares;
* estudiar distribucion de NEW/BOTH/EXIT;
* evaluar efectos de C2;
* comprobar reconciliaciones;
* comparar baseline vs OpenFIGI.

Pero **no debe denominarse NIPC publicable del mercado institucional** sin una politica especifica que demuestre representatividad y cobertura suficiente.

Ademas, nunca debera utilizarse para ocultar el problema de cobertura del universo completo.

---

# 6. Q-PROBE-5 - Mini-probe de top 20 filers

### Decision: **GO - AUTORIZADO**

Esta es ahora la **siguiente accion prioritaria**.

El mini-probe debe determinar si Vanguard es:

```text
CASO AISLADO
```

o:

```text
FENOMENO SISTEMICO / RECURRENTE
```

### Universo

Top 20 managers/filers por:

```text
sum(SSHPRNAMT)
```

preferentemente tras aplicar el mismo scope y filtros que usa NIPC.

Debe compararse Q4 2025 contra Q1 2026.

### Para cada filer

Medir como minimo:

```text
CIK_Q4
nombre_Q4
SSHPRNAMT_Q4

CIK_Q1
nombre_Q1
SSHPRNAMT_Q1

presente_Q4
presente_Q1

tipo_filing_Q4
tipo_filing_Q1

NEW/EXIT bruto asociado

n securities afectadas
SSHPRNAMT afectado
```

Y, fundamentalmente:

```text
NT_Q4 -> OTHERMANAGER2
NT_Q1 -> OTHERMANAGER2
```

cuando exista.

### Clasificacion del resultado

El probe no debe inventar una clasificacion de "misma entidad economica".

Debe limitarse a estados observables:

```text
CONTINUOUS_FILER
FILER_DISCONTINUITY
NT_TO_HR_RELATION_OBSERVED
HR_TO_HR_WITHOUT_RELATION
UNRESOLVED
```

Esto mantiene la separacion entre **hecho documental** e **inferencia economica**.

---

# 7. Hallazgo adicional importante sobre Vanguard

El dato mas relevante del probe no es realmente el `+52M`.

Es este:

```text
mismo CUSIP
mismo periodo economico
distintos filing managers
```

junto con:

```text
13F-NT
+
otros managers
+
13F-HR
```

La SEC contempla expresamente estructuras en las que un manager presenta un 13F-NT porque todas sus posiciones son reportadas por otro manager.

Ademas, existen ejemplos SEC en los que el propio 13F-NT declara que posiciones previamente reportadas por separado pasan a estar reportadas por otro manager, demostrando que la discontinuidad del filer puede ser una transicion documental real y no una variacion de posiciones.

Por ello, el hallazgo debe considerarse **estructuralmente plausible y relevante**, pero todavia no debe generalizarse desde un unico caso.

---

# 8. Sobre los resultados del NIPC actual

Los siguientes numeros deben conservarse exactamente como evidencia del probe:

```text
ALL
NIPC observable       ~= +23.65 M

ELIGIBLE
NIPC observable       ~= +52.57 M
```

pero acompanados de una advertencia:

```text
NO INTERPRETAR COMO FLUJO INSTITUCIONAL ECONOMICO.
```

La razon no es unicamente la reorganizacion Vanguard.

Existen **dos limitaciones independientes**:

```text
1. Cobertura canonica muy baja.
2. Discontinuidad potencial del filer.
```

No deben mezclarse en un unico "factor de descuento".

---

# 9. Consecuencia para Gate-NIPC.2

El resultado del probe **refuerza**, en lugar de cerrar, el bloqueo previo.

Gate-NIPC.2 continua:

```text
NO AUTORIZADO
```

porque todavia faltan:

```text
coverage empirical evidence
+
threshold proposal
+
filer-continuity evidence
```

No procede todavia:

```text
THRESHOLD_1 = X
THRESHOLD_2 = Y
```

ni cambiar la C2 productiva.

---

# 10. Consecuencia para Gate-NIPC.3

**NO GO.**

Gate-NIPC.3 no debe iniciarse formalmente hasta resolver metodologicamente:

```text
A. cobertura de identidad
B. continuidad de filer
C. tratamiento de NT -> HR
```

El presente probe ha cumplido precisamente una funcion util: ha descubierto este problema **antes** de fijar thresholds y antes de convertir el resultado en senal de producto.

---

# 11. Orden de trabajo aprobado

Queda establecido el siguiente orden:

```text
[1] CONSERVAR este probe como evidencia
            v
[2] Documentar limitacion de filer
            v
[3] MINI-PROBE Q-PROBE-5
            v
[4] Verificar NT -> OTHERMANAGER2 en casos afectados
            v
[5] Cuantificar continuidad/discontinuidad
            v
[6] Revisar cobertura baseline
            v
[7] Ejecutar OpenFIGI sobre universo pendiente
            v
[8] Proponer THRESHOLD_1 / THRESHOLD_2
            v
[9] Dictamen Gate-NIPC.2
            v
[10] Disenar, si procede, reconciliacion NT <-> HR
```

---

# DICTAMEN FINAL

### **GO**

Q-PROBE-1 - documentar limitacion contractual.

### **GO CONDICIONADO**

Q-PROBE-2 - investigar `OTHERMANAGER2` como evidencia de reporting, **sin convertirlo todavia en reconciliacion economica**.

### **GO**

Q-PROBE-3 - incorporar continuidad de filer como dimension independiente antes de fijar thresholds.

### **GO DIAGNOSTICO**

Q-PROBE-4 - permitir `CANONICAL_SUBSET` solo como scope experimental, no como publicacion representativa.

### **GO**

Q-PROBE-5 - ejecutar mini-probe top 20.

## Estado global

```text
MICRO-GATE COVERAGE        -> ABIERTO
LIMITACION FILER           -> HALLAZGO CONFIRMADO
C2                         -> VIGENTE / SIN PATCH
RECONCILIACION NT-HR       -> NO AUTORIZADA
THRESHOLD_1                -> UNDEFINED
THRESHOLD_2                -> UNDEFINED
Gate-NIPC.2                -> BLOQUEADO
Gate-NIPC.3                -> NO AUTORIZADO
```

**No tocar codigo productivo.**

La proxima pieza de evidencia que debe producir el proyecto es **exclusivamente el mini-probe de continuidad de filer Q4->Q1 sobre los top 20**, con especial atencion a `13F-NT -> OTHERMANAGER2 -> 13F-HR` y a la cantidad/peso de `NEW/EXIT` que queda explicado documentalmente por esas transiciones.

