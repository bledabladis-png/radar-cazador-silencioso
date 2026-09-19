# DICTAMEN FORMAL DEL AUDITOR

**Objeto:** Coverage Baseline NIPC - Q4 2025 -> Q1 2026
**Evidencia:** `docs/auditoria/evidence/nipc_gate0_baseline/`
**Informe revisado:** Coverage baseline Gate 0
**Estado:** **PASS COMO EVIDENCIA DIAGNOSTICA**
**OpenFIGI masivo:** **NO AUTORIZADO TODAVIA**
**Gate-NIPC.2:** **BLOQUEADO**

---

## 1. Dictamen ejecutivo

El probe esta correctamente construido para la finalidad declarada:

* metodologia congelada antes de ejecutar;
* `canonical_snapshot` como unica fuente de agregados;
* funciones productivas reutilizadas;
* cuatro universos anidados;
* seis metricas pairwise;
* checks mecanicos PASS;
* ejecucion determinista;
* sin modificacion del motor;
* sin thresholds.

Por tanto:

**Coverage Baseline = PASS.**

Pero el resultado revela una cuestion mas importante que los propios numeros:

> **La actual definicion contractual de `operational_universe` incorpora `ticker_mapped` antes de evaluar la cobertura.**

Esto provoca que:

```text
TECHNICAL_RADAR
=
ELIGIBLE_SEC
+
mapping exitoso al radar
```

y posteriormente:

```text
coverage_previous = 1.0000
coverage_current  = 1.0000
```

sea practicamente una consecuencia de la construccion del universo, no una medicion independiente de cobertura de mapping.

Por tanto, esos `1.0000` **NO pueden utilizarse como evidencia de cobertura suficiente del universo NIPC**.

---

## 2. Hallazgo critico - circularidad de denominador

La policy actual establece:

```text
operational_universe
=
radar_equities
^ section13f_eligible
^ ticker_mapped
^ EQUITY
```

y simultaneamente establece:

```text
coverage = mapped / total
```

El baseline demuestra el efecto:

```text
ELIGIBLE_SEC
    v
474,395 unidades
    v
TECHNICAL_RADAR
```

y despues:

```text
coverage_previous = 1.0000
coverage_current  = 1.0000
```

Eso es correcto **matematicamente**, pero no responde a la pregunta:

> ¿que proporcion del universo elegible que deberia alimentar NIPC podemos identificar/mapping correctamente?

Responde a otra pregunta:

> ¿que proporcion del subconjunto que ya hemos decidido que esta mapeado permanece emparejable entre Q4 y Q1?

Son metricas diferentes.

---

## 3. Que demuestran realmente los resultados

La lectura correcta del baseline es:

### Nivel 1 - universo 13F bruto

```text
coverage_previous = 2.06%
coverage_current  = 2.04%
paired_security   = 1.81%
```

### Nivel 2 - despues de SEC eligibility

```text
coverage_previous = 4.24%
coverage_current  = 3.86%
paired_security   = 3.68%
paired_weighted  = 27.72%
```

Esto muestra que el crosswalk interno actual resuelve una fraccion pequena del universo total 13F.

### Nivel 3 - subconjunto ya identificado como radar

```text
coverage_previous = 100%
coverage_current  = 100%

paired_security   = 99.09%
paired_weighted  = 98.64%
```

Estas ultimas cifras significan:

> **una vez que nos limitamos a las posiciones que ya han entrado por el mapping al radar, la estabilidad interperiodo es muy alta.**

No significan:

> "el 99.09 % del universo 13F/radar esta cubierto."

Esta distincion debe quedar congelada en la documentacion.

---

## 4. El baseline ha localizado el cuello de botella

Sin convertir esto todavia en un juicio economico, el experimento permite separar dos fenomenos:

```text
MAPPING / RESOLUTION
        v
gran perdida antes de TECHNICAL_RADAR

LONGITUDINAL PAIRING
        v
muy poca perdida una vez dentro de TECHNICAL_RADAR
```

Concretamente:

```text
ELIGIBLE_SEC
Q4 = 2,335,031 units
Q1 = 2,397,186 units

TECHNICAL_RADAR
Q4 =   474,395
Q1 =   490,732
```

Por tanto, aproximadamente cuatro quintas partes de las unidades `ELIGIBLE_SEC` **no llegan a TECHNICAL_RADAR**.

La causa no debe etiquetarse todavia como "CUSIPs faltantes" unicamente, porque el propio fallout distingue:

```text
CANONICAL_TICKER_OUTSIDE_RADAR
+
OBSERVED_ONLY_NO_CANONICAL
```

y esas categorias tienen significados diferentes.

Eso esta correctamente reflejado en el probe.

---

## 5. `OPERATIONAL_EQUITY` = `TECHNICAL_RADAR`

El resultado:

```text
474,395 = 474,395
490,732 = 490,732
```

en ambos periodos es correcto y util como check.

Significa que, bajo la lista concreta de `radar_equities` utilizada, todos los tickers que llegan a `TECHNICAL_RADAR` son clasificados:

```text
get_instrument_class(ticker) == EQUITY
```

Por tanto:

```text
instrument_class
```

no esta introduciendo perdida adicional en este baseline.

Esto **no significa que el filtro sea innecesario en el contrato**; demuestra solamente que no es un cuello de botella en este snapshot.

---

## 6. Contradiccion adicional de la policy: 242 frente a 313

Hay una segunda cuestion que debe corregirse antes de fijar thresholds.

La policy seccion 4.B dice:

```text
universo radar real = 313 tickers
```

mientras el micro-gate USA esta utilizando:

```text
radar_equities = 242
```

La diferencia esta explicada por la investigacion anterior:

```text
313 total
 +-- 51 europeos con provider
 +-- 20 .L sin provider
 +-- 242 sin sufijo
```

Para NIPC/13F, el universo USA de **242** puede ser la definicion metodologica apropiada del micro-gate, pero la policy debe decirlo explicitamente.

No debe coexistir:

```text
313 = "universo radar real"
```

y:

```text
242 = universo efectivo usado por thresholds
```

sin explicar la diferencia.

### Decision

**Corregir documentalmente antes de thresholds.**

No es un bug del probe.

---

## 7. Consecuencia para THRESHOLD_1 y THRESHOLD_2

No autorizo todavia fijarlos.

La razon es ahora mas precisa que antes:

```text
THRESHOLD_1
paired_security_coverage

THRESHOLD_2
paired_weighted_share_coverage
```

estan definidos sobre un universo que ya contiene:

```text
ticker_mapped
```

Por tanto, existe el riesgo de fijar thresholds excelentes sobre una muestra seleccionada por el propio proceso de mapping.

El resultado:

```text
99.09%
98.64%
```

es util para estudiar **comparabilidad de lo ya mapeado**, pero insuficiente para decidir cuanto mapping total necesita el producto para ser publicable.

---

## 8. No recomiendo parchear el motor

Este hallazgo es **contractual/documental**, no una evidencia de bug en:

```text
compute_delta_shares()
compute_nipc_and_coverage()
```

El motor esta haciendo exactamente lo que se le pidio.

No tocar:

```text
C2
delta_shares
nipc.py
security_identity.py
```

en esta fase.

La correccion debe producirse primero en la definicion de los universos y metricas de la policy.

---

## 9. Que debe resolverse antes de OpenFIGI

El siguiente micro-gate debe ser **Coverage Contract Normalization**, no todavia el OpenFIGI masivo.

Hay que congelar dos conceptos diferentes:

```text
A. UNIVERSO OBJETIVO
¿Que queremos cubrir?

B. UNIVERSO RESUELTO
¿Que podemos identificar actualmente?
```

y una tercera dimension:

```text
C. COMPARABILIDAD
¿Que proporcion de lo resuelto puede emparejarse Q4 <-> Q1?
```

La medicion debe poder distinguir:

```text
TARGET
   v
RESOLVED
   v
PAIRED
```

y no:

```text
TARGET ^ RESOLVED
   v
PAIRED
```

sin conservar el tamano del primer conjunto.

---

## 10. OpenFIGI

### Estado: **NO AUTORIZADO TODAVIA**

No porque OpenFIGI este rechazado.

Al contrario: el baseline demuestra que la capa de identidad es el principal elemento que merece investigacion.

Pero ejecutar 24.000-50.000 consultas ahora produciria mas datos sobre una **definicion contractual todavia no normalizada**.

Primero debe quedar congelado que significa:

```text
coverage
operational_universe
mapping coverage
pairwise coverage
```

Despues si:

```text
BASELINE
   vs
BASELINE + OPENFIGI
```

tendra una interpretacion inequivoca.

---

## 11. Lo que si puede conservarse del baseline

Debe permanecer como evidencia historica intacta:

```text
baseline_output.txt
probe_coverage_baseline.py
README.md
HASHES.txt
```

No recomiendo reescribir sus resultados para adaptarlos a una futura policy.

El baseline representa correctamente:

> "como se comporta el sistema bajo la definicion contractual vigente durante la ejecucion".

Una futura v1.1 de la policy debe producir **una nueva medicion**, no reinterpretar retroactivamente el resultado.

---

## 12. Nuevo estado de gates

```text
Coverage Baseline                 PASS
Mechanical integrity              PASS
Canonicalization                  PASS
Filer continuity                  CERRADO
Baseline mapping bottleneck       CONFIRMADO

Operational-universe semantics    ISSUE ABIERTO
242 vs 313 policy wording         ISSUE DOCUMENTAL
Threshold_1                       UNDEFINED
Threshold_2                       UNDEFINED

OpenFIGI masivo                   NO AUTORIZADO
Gate-NIPC.2                       BLOQUEADO
Gate-NIPC.3                       NO AUTORIZADO
```

---

# DICTAMEN FINAL

### **PASS**

El baseline esta correctamente ejecutado y constituye evidencia reproducible valida.

### **HALLAZGO CRITICO**

La policy actual es circular al definir `operational_universe` mediante `ticker_mapped` y despues utilizar ese mismo universo para medir coverage.

### **NO GO**

No fijar todavia `THRESHOLD_1` ni `THRESHOLD_2`.

### **NO GO**

No ejecutar todavia OpenFIGI masivo.

### **GO**

Abrir un micro-gate documental especifico para normalizar la semantica:

```text
TARGET UNIVERSE
        v
MAPPING / RESOLUTION
        v
RESOLVED UNIVERSE
        v
PAIRWISE COMPARABILITY
```

### **GO**

Mantener el baseline actual como evidencia historica sin modificar.

La consecuencia mas importante del experimento es que **el 99,09 % / 98,64 % no demuestra cobertura suficiente del radar; demuestra estabilidad Q4->Q1 del subconjunto que ya consiguio entrar en el radar mediante mapping**.

Hasta que esa distincion quede incorporada formalmente a `NIPC_COVERAGE_POLICY`, cualquier threshold que se fije sobre esos 99 % seria metodologicamente prematuro.

**Mi decision de auditoria, por tanto, es detener aqui la cadena `baseline -> OpenFIGI -> thresholds` y hacer primero la normalizacion contractual de coverage.** El baseline ha hecho exactamente su trabajo: ha encontrado una circularidad que era invisible antes de medir.