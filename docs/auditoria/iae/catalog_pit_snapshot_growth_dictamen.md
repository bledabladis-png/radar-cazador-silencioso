\# Dictamen del auditor externo



\## Conclusión ejecutiva



\*\*El expediente identifica correctamente un defecto de diseño real y de severidad alta.\*\* El `IndexError` actual es, paradójicamente, el comportamiento seguro: impide que el sistema escriba un `catalog\_membership` incorrecto. El riesgo grave es el escenario descrito en §1.4: cuando el número de assignments alcance también 255, el mecanismo posicional puede dejar de fallar y \*\*reasignar silenciosamente keys a tickers distintos\*\*.



Por tanto:



> \*\*No debe parchearse el `IndexError`; debe eliminarse la dependencia posicional como mecanismo de identidad.\*\*



Mi decisión es \*\*aprobar una variante de (A) + membership acumulativo\*\*, pero con una corrección importante respecto de la propuesta del expediente:



> \*\*No aprobaría declarar una biyección global e inmutable `radar\_ticker ↔ catalog\_key`.\*\*

> Aprobaría una relación \*\*uno-a-uno por versión activa\*\*, mientras que la identidad permanente debe seguir siendo `catalog\_key → identidad estable de la entidad/share class`.



El ticker es un atributo de mercado; no debe convertirse en la identidad normativa que A1 precisamente intenta hacer inmutable.



\---



\# 1. Severidad de los hallazgos



| Hallazgo                                        | Estado                        |                     Severidad |

| ----------------------------------------------- | ----------------------------- | ----------------------------: |

| Alineamiento posicional snapshot → assignment   | Confirmado                    |                      \*\*Alta\*\* |

| `IndexError` 255 vs 242                         | Confirmado                    |                          Alta |

| Posible reasignación silenciosa con 255/255     | Confirmado como riesgo lógico | \*\*Crítica si se materializa\*\* |

| Ausencia de contrato ticker ↔ key               | Confirmado                    |                          Alta |

| `membership` per-version vs A1 §3.3 acumulativo | Confirmado                    |                          Alta |

| `validate\_membership` no operativo              | Confirmado                    |                    Media/Alta |

| CI deja `origin/main` rojo                      | Confirmado                    |              Alta operacional |

| Tests anclados a cardinalidad histórica         | Confirmado                    |                         Media |

| Race lógica entre `Run tests` y regeneración    | Confirmado                    |                          Alta |



\### Punto especialmente importante



Actualmente el sistema \*\*falla cerrado por accidente\*\*:



```text

242 assignments

255 snapshot

&#x20;       ↓

IndexError

&#x20;       ↓

no se escribe membership incorrecto

```



Pero una futura generación de 255 assignments puede convertirlo en:



```text

255 snapshot

255 assignments

&#x20;       ↓

alineamiento posicional

&#x20;       ↓

ticker nuevo intercalado

&#x20;       ↓

keys desplazadas

&#x20;       ↓

CSV aparentemente válido

&#x20;       ↓

binding incorrecto

```



Ese segundo escenario es sustancialmente peor que el fallo actual.



\---



\# 2. Dictamen sobre la opción (A)



\## Aprobación: SÍ, con modificación contractual



Apruebo la idea de:



```text

catalog\_assignments.csv

&#x20;   catalog\_key

&#x20;   assigned\_entity\_id

&#x20;   radar\_ticker

&#x20;   valid\_from

&#x20;   valid\_to

&#x20;   source

&#x20;   reason

```



porque elimina una dependencia implícita fundamental y coloca el binding donde conceptualmente corresponde: en el artefacto administrativo de assignments.



Pero no aprobaría este enunciado:



> `radar\_ticker ↔ catalog\_key` es una biyección permanente.



Debe sustituirse por algo más preciso:



\### Invariante temporal



Para cada versión `V` del catálogo:



```text

cada ticker activo en V → exactamente una catalog\_key

cada catalog\_key activa en V → exactamente un ticker

```



Por tanto:



```text

Ticker(T,V) ↔ Key(K,V)

```



es uno-a-uno \*\*en el universo activo de una versión\*\*.



Pero históricamente puede ocurrir:



```text

V1: ABC → K001

V2: ABC → K001

V3: ABC desaparece

V4: ABC reaparece → K257

```



y eso no contradice la estabilidad de `catalog\_key`.



Así se conserva la filosofía de A1:



```text

K001 nunca pasa a otra identidad

K257 nunca había sido utilizada

```



\---



\# 3. Corrección adicional fundamental: el ticker no debe ser la clave de reconciliación definitiva



Este punto debe entrar en el nuevo contrato.



El algoritmo propuesto dice:



> si el ticker existe en assignments, conserva su key.



Eso \*\*no es suficientemente robusto\*\*.



Supongamos:



```text

V1

ticker = FB

share class identity = X

key = K001

```



y después:



```text

V2

ticker = META

share class identity = X

```



Un algoritmo basado solamente en ticker interpretaría:



```text

FB desapareció

META apareció

```



cuando en realidad puede ser:



```text

misma identidad

nuevo ticker

misma key

```



Por ello el orden correcto debe ser:



```text

identidad estable

&#x20;      ↓

catalog\_key

&#x20;      ↓

radar\_ticker como atributo observado

```



No:



```text

radar\_ticker

&#x20;      ↓

catalog\_key

```



si existe disponible una identidad estable mejor.



En vuestro contexto, `assigned\_entity\_id`/identificador estable de share class debe utilizarse para reconciliar cambios; `radar\_ticker` debe ser una representación del instrumento en esa versión.



\### Consecuencia de auditoría



\*\*La columna `radar\_ticker` en `catalog\_assignments.csv` sí la apruebo\*\*, pero como binding/atributo explícito y auditable, no como identidad criptográficamente inmutable de la key.



\---



\# 4. Política de bajas: NO aprobaría la regla automática propuesta



El expediente propone:



```text

ticker desaparece

&#x20;     ↓

valid\_to = fecha del snapshot

```



\*\*No lo aprobaría así.\*\*



Existe un riesgo que considero más importante que el que aborda la propuesta: una ausencia en un snapshot puede ser una anomalía de extracción y no una baja real.



Por ejemplo:



```text

V1: ticker presente

V2: proveedor omite ticker

V3: ticker presente

```



Si V2 provoca inmediatamente:



```text

valid\_to = V2

```



el sistema ha fabricado una baja histórica.



Además, si V3 vuelve a crear la key:



```text

K001 → cerrada

K257 → nueva

```



habremos generado artificialmente rotación de identidad.



\## Política que sí apruebo



Una key solo se cierra cuando existe una \*\*baja de universo confirmada\*\*, no simplemente ausencia puntual.



El contrato debe distinguir:



```text

ABSENCE

≠

RETIREMENT

```



La baja debe requerir al menos:



```text

snapshot válido y completo

\+

ausencia confirmada

\+

regla de retiro explícita

```



La regla concreta puede ser:



```text

desaparición en snapshot de universo autoritativo

```



si ese snapshot tiene garantías suficientes de completitud.



Si no las tiene, debe exigirse confirmación adicional.



\### Reaparición



Una vez que una key ha sido realmente retirada:



```text

K001 --valid\_to--> retirada

```



y la identidad reaparece posteriormente en el universo:



```text

K257

```



\*\*Sí aprobaría no reactivar K001.\*\*



Esto preserva el principio:



> una key cerrada no vuelve a vida operativa.



Pero esta regla debe aplicarse a una \*\*baja real\*\*, no a una ausencia accidental.



\---



\# 5. Hay una precisión temporal que A1 v5 debe cerrar



El contrato debería definir formalmente si:



```text

valid\_from

valid\_to

```



representan intervalos:



```text

\[valid\_from, valid\_to)

```



o extremos inclusivos.



Yo recomiendo:



```text

\[valid\_from, valid\_to)

```



porque evita ambigüedades en cambios de versión:



```text

K001

valid\_from = 2026-01-01

valid\_to   = 2026-10-01

```



y la nueva relación empieza:



```text

K257

valid\_from = 2026-10-01

```



Sin solapamiento.



Esto debería formar parte del amendment, no quedar implícito en código.



\---



\# 6. La reconstrucción histórica: apruebo el objetivo, no el método único propuesto



Aquí haría una corrección importante.



El expediente propone reconstruir los 242 bindings mediante:



```python

sorted(snapshot\["radar\_ticker"])

```



y posición.



Esto puede ser una \*\*hipótesis histórica\*\*, pero no debería ser la prueba final de identidad.



¿Por qué?



Porque precisamente estamos auditando un sistema cuyo defecto es depender de posiciones.



No deberíamos utilizar el mismo supuesto que queremos retirar como única evidencia de migración.



\## Método que sí aprobaría



Usaría dos reconstrucciones independientes:



\### Camino A — reconstrucción histórica del generador



```text

sorted(snapshot\_20260922)

→ posición

→ key

```



\### Camino B — evidencia materializada existente



```text

catalog\_membership 20260922

&#x20;       ↓

catalog\_key

&#x20;       ↓

snapshot\_row\_uid

&#x20;       ↓

snapshot\_20260922

&#x20;       ↓

radar\_ticker

```



Entonces:



```text

mapping A == mapping B

```



para las 242 keys.



Solo en ese caso aprobaría la migración automática.



Esto convierte la fórmula posicional en \*\*evidencia de segunda fuente\*\*, no en autoridad única.



\---



\# 7. Criterio de aceptación de la migración histórica



Exigiría exactamente:



```text

242/242 keys existentes

242/242 row\_uids resolubles

242/242 tickers resolubles

242/242 coincidencias entre método A y método B

0 duplicados

0 keys reasignadas

0 row\_uids huérfanos

0 ticker ambiguos

```



Y además:



```text

snapshot pre

→ migration

→ snapshot post

```



debe demostrar que las 242 relaciones preexistentes permanecen idénticas.



Los 13 nuevos deben ser los únicos elementos nuevos:



```text

K existentes: sin cambio

K nuevas: 13

```



\---



\# 8. Membership acumulativo: APROBADO



Aquí el expediente tiene razón.



Existe una contradicción material:



```text

A1 §3.3

membership histórico acumulativo

```



frente a:



```text

implementación

membership = solo versión vigente

```



No mantendría ambas semánticas.



\## Dictamen



\*\*Migrar a membership acumulativo.\*\*



Después:



```text

20260921: 242

20260922: 242

20261001: 255

```



el fichero contendría:



```text

739 filas

```



si se conservan esas tres versiones, no 497, porque 242 + 255 solo contempla las dos versiones que el expediente cita en la migración.



Esto es un detalle menor, pero conviene que el plan de migración lo contabilice exactamente.



\---



\# 9. La unidad correcta de unicidad de membership



Aprobaría:



```text

(version\_id, catalog\_key)

```



como único histórico.



Y además:



```text

(version\_id, snapshot\_row\_uid)

```



también único.



Por tanto:



```text

(version\_id, catalog\_key)        UNIQUE

(version\_id, snapshot\_row\_uid)   UNIQUE

```



y debe existir:



```text

snapshot\_row\_uid → snapshot real de esa versión

```



Esto impide duplicaciones silenciosas.



\---



\# 10. `predecessor\_row\_uid`: hay que precisar la semántica antes de activar el validator



El expediente propone:



> `predecessor\_row\_uid` no vacío si y solo si el `snapshot\_row\_uid` difiere de la versión anterior.



Eso es correcto \*\*solo si ésa es exactamente la semántica que definió A1 §3.3\*\*.



Hay dos posibles contratos:



\### Modelo 1 — predecessor solo ante cambio



```text

V1 row A

V2 row A idéntico

&#x20;   predecessor = NULL



V3 row A cambiado

&#x20;   predecessor = UID(V2)

```



\### Modelo 2 — cadena histórica completa



```text

V2

predecessor = UID(V1)



V3

predecessor = UID(V2)

```



Son contratos distintos.



Antes de activar `validate\_membership`, A1 v5 debe fijar uno de los dos.



\*\*No activaría el validator sobre una interpretación ambigua.\*\*



\---



\# 11. `validate\_membership`: APROBADO, pero no como simple “descomentar”



Mi decisión:



> \*\*Debe pasar de código muerto a validador productivo.\*\*



Pero después de hacer dos cosas:



```text

1\. fijar la semántica normativa de membership

2\. migrar el CSV a ese formato

```



No recomiendo:



```text

activar validator

```



antes de haber hecho ambas.



Lo incorporaría como validación obligatoria del flujo:



```text

load snapshots

↓

load assignments

↓

load cumulative membership

↓

validate\_membership

↓

build\_target

```



No tiene tanta importancia que el documento lo llame literalmente "PASO 0"; la propiedad importante es:



> \*\*ningún `build\_target` puede ejecutarse sobre membership inválido.\*\*



\---



\# 12. Los tests actuales deben cambiar de naturaleza



Aquí coincido totalmente con el diagnóstico.



Esto:



```python

assert len(df) == 242

```



es un test de estado histórico, no un test del contrato.



Debe sustituirse por invariantes.



Los cinco tests afectados —el expediente habla de cuatro anclados y además está `test\_idempotencia`— deberían quedar claramente clasificados.



\## Tests obligatorios



\### Invariantes de estructura



```text

cada key una sola vez

cada ticker activo una sola vez

assignments ↔ membership coherentes

```



\### Invariante de estabilidad



```text

snapshot 242

→ snapshot reordenado

→ mismo binding

```



Este test es \*\*obligatorio\*\*.



\### Invariante de crecimiento



```text

242 → 255



las 242 keys antiguas permanecen idénticas

13 keys nuevas

```



\### Invariante de repetición



```text

main()

main()



segunda ejecución:

mismo output byte-for-byte

```



o, donde existan timestamps legítimos, igualdad semántica.



\### Invariante de baja



```text

retirement confirmado

→ valid\_to

→ key jamás reutilizada

```



\### Invariante de reaparición



```text

retirada

→ reaparece

→ nueva key

```



\### Invariante de rename



Este test falta y lo considero importante:



```text

same stable identity

ticker A → ticker B

&#x20;       ↓

same catalog\_key

```



si esa es la semántica finalmente aprobada.



\---



\# 13. Opción B: descartada



Coincido con el expediente.



Añadir:



```text

catalog\_ticker\_binding.csv

```



crearía:



```text

assignments

\+

binding

```



para resolver una relación que naturalmente pertenece al dominio de assignments.



Eso introduce otro punto de sincronización y otro contrato.



\*\*No la aprobaría.\*\*



\---



\# 14. Opción C: descartada



También coincido.



Hash del ticker:



```text

radar\_<hash>

```



soluciona un problema que ya se resuelve de forma mucho más limpia manteniendo keys administrativas históricas.



Además:



```text

ticker

```



no es la identidad adecuada para representar entidades a perpetuidad.



\*\*No migraría 242 keys existentes para esto.\*\*



\---



\# 15. Contención inmediata: NO dejaría `main` rojo



Aquí mi dictamen es diferente de la opción conservadora del expediente.



\*\*No considero aceptable mantener indefinidamente `origin/main` rojo cuando el fallo ya está identificado y afecta a la siguiente ejecución programada.\*\*



Pero tampoco haría una eliminación destructiva del snapshot 255.



\## Contención aprobada



```text

revert de la promoción del snapshot 255

\+

conservar snapshot 255 como artefacto histórico/quarantined

\+

pausar temporalmente la regeneración que volvería a introducir 255

```



Es importante la segunda parte.



Solo revertir:



```text

255 → 242

```



sin impedir la regeneración significa:



```text

Run tests

&#x20; ↓

verde

&#x20; ↓

regenerate

&#x20; ↓

255

&#x20; ↓

main vuelve a quedar incompatible

```



Por tanto, la contención debe ser:



```text

estado contractual conocido = 242

regeneración automática = pausada

```



hasta que exista el fix.



\### No recomiendo



```text

borrar definitivamente snapshot\_2026-10-01\_01.csv

```



Debe conservarse como evidencia del incidente.



Y tampoco recomiendo reescribir la historia Git. Mejor un commit de reversión trazable.



\---



\# 16. Alcance del fix



\## Mi decisión: cerrar ambos problemas, pero en dos commits auditables



No limitaría el trabajo únicamente a:



```text

242 → 255

```



porque quedaría:



```text

binding corregido

pero

membership sigue contradiciendo A1

validator sigue muerto

```



Eso no sería un A1 v5 realmente cerrado.



Pero tampoco mezclaría todo en un único cambio imposible de auditar.



\### Commit/Change A



```text

A1 amendment

\+

binding explícito

\+

eliminación de positional mapping

\+

migración 242→255

\+

tests de estabilidad/crecimiento

```



\### Commit/Change B



```text

membership acumulativo

\+

predecessor formalizado

\+

validate\_membership productivo

\+

tests históricos

```



Después:



```text

E2E completo

→ Gate 10/10

→ auditoría

```



Así cada transición puede verificarse independientemente.



\---



\# 17. Una cuestión de nomenclatura que recomiendo corregir



No utilizaría en el contrato algo ambiguo como:



```text

radar\_ticker = identidad

```



Preferiría:



```text

catalog\_key = identidad administrativa

radar\_ticker = identificador de mercado observado en versión V

```



Y, si el esquema lo permite:



```text

stable\_identity / share\_class\_figi

```



como ancla para reconciliar cambios de ticker.



Esto evita que dentro de seis meses aparezca otro expediente:



> “Cambio de ticker rompe catalog\_key”.



\---



\# 18. Contrato A1 v5 que considero aprobable



Conceptualmente debería quedar así:



```text

1\. catalog\_key es identidad administrativa inmutable.

2\. Una key jamás se reasigna a otra identidad.

3\. En cada versión V, cada instrumento activo del universo tiene

&#x20;  exactamente una catalog\_key.

4\. En cada versión V, cada catalog\_key activa corresponde exactamente

&#x20;  a un instrumento.

5\. radar\_ticker es atributo versionado, no identidad permanente.

6\. El binding entre versiones se resuelve mediante identidad estable,

&#x20;  nunca mediante posición.

7\. La ausencia de un ticker no implica automáticamente retirement.

8\. Una key retirada no se reactiva; una reaparición confirmada obtiene

&#x20;  una nueva key.

9\. catalog\_membership es acumulativo.

10\. Cada fila histórica identifica inequívocamente:

&#x20;      version\_id

&#x20;      catalog\_key

&#x20;      snapshot\_row\_uid

&#x20;      predecessor\_row\_uid cuando corresponda.

11\. Ningún productor ni consumidor puede depender del orden de filas.

12\. Toda transformación debe ser idempotente.

```



Este sería, a mi juicio, un contrato mucho más resistente que A1 v4.



\---



\# 19. Resultado de las siete preguntas



| Pregunta                  | Dictamen                                                                                                                                  |

| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |

| \*\*1. Extensión A1\*\*       | \*\*APROBADA con modificación:\*\* `radar\_ticker` explícito sí; biyección solo por versión activa, no global/inmutable.                       |

| \*\*2. Bajas\*\*              | \*\*APROBADA con condición:\*\* solo tras baja confirmada; ausencia puntual no basta. Reaparición tras retirement → nueva key.                |

| \*\*3. Membership\*\*         | \*\*APROBADO:\*\* acumulativo. `validate\_membership` debe pasar a productivo tras formalizar su semántica.                                    |

| \*\*4. Reconstrucción 242\*\* | \*\*APROBADA como migración\*\*, pero no mediante posición como única evidencia; debe cruzarse con `snapshot\_row\_uid` + membership histórico. |

| \*\*5. Contención CI\*\*      | \*\*APROBADA:\*\* restaurar último estado coherente y pausar regeneración; conservar snapshot 255 como evidencia. No dejar `main` rojo.       |

| \*\*6. Alcance\*\*            | \*\*Aprobado en dos cambios auditables:\*\* binding/growth primero; membership/validator después.                                             |

| \*\*7. Trazabilidad\*\*       | \*\*Ambos:\*\* expediente específico como evidencia normativa; `04\_HISTORICO.md` como índice de decisión, no como depósito de detalle.        |



\---



\# 20. Dictamen final



\### Estado del sistema



\*\*NO APTO para aceptar crecimiento adicional del snapshot bajo el algoritmo actual.\*\*



\### Causa



\*\*Dependencia posicional no contractual entre snapshot y assignments.\*\*



\### Riesgo



Actualmente:



```text

FAIL-CLOSED

```



pero con posibilidad inmediata de evolucionar a:



```text

SILENT DATA CORRUPTION

```



cuando las cardinalidades coincidan.



\### Solución aprobada



```text

A1 v5/amendment

&#x20;       +

binding explícito

&#x20;       +

reconciliación por identidad estable

&#x20;       +

sin índices posicionales

&#x20;       +

membership acumulativo

&#x20;       +

validator productivo

&#x20;       +

tests basados en invariantes

```



\### Solución que NO aprobaría



```text

if len(snapshot) != len(assignments):

&#x20;   fabricar/ajustar posiciones

```



ni:



```text

ticker → key

```



como identidad permanente, ni tampoco crear un segundo CSV de binding como parche.



La decisión de arquitectura que considero más importante es ésta:



> \*\*`catalog\_key` debe identificar; `radar\_ticker` debe describir. Nunca al revés.\*\*



Con esa separación, el crecimiento 242 → 255 deja de ser una migración excepcional y pasa a ser un caso normal de evolución del catálogo.



