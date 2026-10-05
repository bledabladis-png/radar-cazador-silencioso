# 57b - Dictamen del auditor sobre frente Wyckoff (57)

Fecha: 2026-10-05
Ramas auditadas: gold-standard-v3 HEAD 6c4912f + sow-v4-validation HEAD a505b87
Copia literal del dictamen recibido. Normalizado a ASCII sin alterar contenido.

---

## Dictamen ejecutivo

Decision global: APROBADO CON REFORMULACION CONTROLADA.

La separacion de los tres claims es metodologicamente correcta. Los
resultados de v3/v5/v8/v9 no permiten declarar efecto, ausencia de
efecto ni validez productiva de SOW/DISTRIBUTION. El problema principal
de Capas 1+2 es actualmente de identificabilidad/potencia muestral bajo
las restricciones fijadas, no una demostracion de que el fenomeno no
exista.

Por tanto:

| Decision | Dictamen                               |
| -------- | -------------------------------------- |
| D1       | SI                                     |
| D2       | REFORMULAR                             |
| D3       | SI, CONDICIONADO                       |
| D4       | SI, CONDICIONADO A NUEVO PROTOCOLO     |
| D5       | SI                                     |

---

# D1. Separar Capa 1, Capa 2 y Capa 3

DICTAMEN: SI

Esta separacion debe quedar formalmente incorporada al contrato del
frente.

Son tres afirmaciones distintas:

Capa 1 - Semantica
    D=1 <=> debilidad estructural Wyckoff.
    Pregunta: el detector identifica correctamente el estado que
    pretende representar?

Capa 2 - Fase
    SOW desplaza RANGE -> DISTRIBUTION.
    Pregunta: la senal SOW aporta informacion suficiente para
    reclasificar estructuralmente la fase?

Capa 3 - Predictiva
    D=1 anticipa deterioro H20.
    Pregunta: la senal posee capacidad predictiva sobre un resultado
    futuro?

No es metodologicamente valido utilizar evidencia de Capa 3 para
"rescatar" Capa 1, ni utilizar una buena concordancia de fases para
declarar capacidad predictiva.

Consecuencia auditora: cada capa puede tener metodo de validacion
distinto, hipotesis distinta, muestra distinta, estimador distinto,
gate distinto, estado distinto. Un PASS en una capa no transmite
automaticamente PASS a las demas.

D1 queda APROBADO.
---

# D2. Archivar Capas 1+2 bajo el techo actual

DICTAMEN: REFORMULAR

No recomiendo aprobar literalmente: "Capas 1+2 como no validables".
Es demasiado fuerte desde el punto de vista epistemologico.

Lo que si esta demostrado es: Capas 1+2 no son validables con potencia
suficiente bajo el diseno actualmente autorizado, el techo de 800
unidades por anotador y la rareza observada, salvo que exista un
mecanismo de enriquecimiento independiente y metodologicamente
admisible.

Esta diferencia es importante.

Lo que los experimentos si establecen:

Con q_D = 2559/243435 = 1.0512%, P(FN|D=0) = 584/240876 ~ 0.242%,
techo n <= 800 por anotador, el muestreo aleatorio de D=0 produce una
expectativa de FN extraordinariamente pequena.

    400 * 0.00242 ~ 0.97

Es decir, aproximadamente 1 FN esperado en 400 unidades D=0. Frente a
una necesidad del orden de 164 FN para alcanzar el objetivo de error
indicado, el salto requerido es enorme. La conclusion practica es
inequivoca: el diseno actual esta limitado por capacidad estadistica,
no por falta de sofisticacion del estimador.

Ademas, v8 ya muestra que sustituir el estimador o modificar la
formulacion sin alterar la estructura informacional del problema no
elimina el muro: 12/12 escenarios fallan en e(Se).

Por tanto, la formulacion correcta de D2 seria: archivar temporalmente
Capas 1+2 como NO VALIDADAS y NO VALIDABLES CON EL DISENO ACTUAL,
manteniendo abierta una reapertura condicionada a un nuevo diseno que
demuestre ex ante capacidad de identificacion/potencia sin violar las
prohibiciones vigentes.

Esto evita convertir "no hemos podido identificarlas con este diseno"
en "las capas son imposibles de validar".

Importante: el archivo debe etiquetarse como

    UNVALIDATED / INCONCLUSIVE DUE TO SAMPLE-CAPACITY CONSTRAINT

y NO como

    NEGATIVE / ABSENCE OF EFFECT

D2 se aprueba unicamente con esta reformulacion.

---

# D3. Migracion a wyckoff_v1.py v1.8-core

DICTAMEN: SI, CONDICIONADO

La migracion propuesta es metodologicamente razonable porque no depende
de SOW ni de DISTRIBUTION.

El nucleo (MARKUP, ACCUMULATION, RANGE, MARKDOWN) es una clasificacion
diferente del bloque SOW/DISTRIBUTION y, por tanto, puede aislarse.

Ademas, el hecho de que legacy este produciendo 20/20 sectores = RANGE
es una evidencia operacional importante de que el clasificador legacy
tiene poca capacidad discriminante en el estado actual.

Eso justifica estudiar la migracion del core, pero no constituye por si
mismo una validacion de que v1.8 sea "correcto" desde el punto de vista
semantico.

Condicion critica: la autorizacion es migrar la implementacion core,
no declarar validado el significado economico/Wyckoff de sus cuatro
fases.

La 5c comparativa legacy vs v1 debe cerrarse antes de migrar
consumidores, tal como exige el propio expediente.

Yo anadiria una condicion auditora explicita - Gate 5c minimo. Debe
quedar documentado:

    1. Que entradas recibe exactamente legacy y v1.
    2. Que diferencias de clasificacion aparecen.
    3. Que diferencias son esperadas por diseno.
    4. Que no existe dependencia residual de SOW/DISTRIBUTION.
    5. Que los consumidores no reciben campos semanticamente
       incompatibles.
    6. Que el cambio no introduce look-ahead ni fuga temporal.
    7. Que existe regresion reproducible del comportamiento del core.

Y, especialmente: una mejora de discriminacion frente a legacy no
constituye evidencia de verdad de la clasificacion.

Por tanto: D3 = SI, condicionado al cierre formal de 5c. No debe
realizarse merge a main hasta existir el dictamen favorable
correspondiente.
---

# D4. Reapertura de Capa 1 con expertos o eventos externos

DICTAMEN: SI, CONDICIONADO A PROTOCOLO NUEVO

Esta es, en mi opinion, la unica via razonable actualmente abierta para
atacar el problema semantico sin forzar el diseno estadistico actual.

Pero hay una diferencia fundamental entre "pedir a expertos que
confirmen el detector" y "construir una referencia independiente de
verdad (gold standard)". Solo la segunda es admisible.

Si se utiliza panel experto, el protocolo deberia exigir como minimo:

    - anotacion independiente;
    - etiquetado ciego respecto al detector;
    - criterios escritos antes de observar resultados;
    - desacuerdo entre anotadores registrado;
    - tasa de acuerdo / inter-rater reliability;
    - adjudicacion separada;
    - prohibicion de retroajustar la definicion despues de conocer D;
    - separacion entre expertos que definen el criterio y quienes
      validan la muestra, cuando sea viable.

Si se utilizan eventos externos, deben ser:

    - documentados;
    - temporalmente anteriores al etiquetado;
    - definidos por una regla ex ante;
    - independientes de la variable D;
    - suficientemente precisos para poder reproducir la etiqueta;
    - protegidos contra seleccion retrospectiva.

Un "evento externo" descubierto despues de observar que senales aciertan
seria esencialmente otra forma de label leakage.

Regla fundamental para D4: no se podra utilizar el nuevo mecanismo para
modificar q_D, pi_Y, MDE = 5 pp, criterios de exito, horizonte, grid, B,
R_MC, despues de observar resultados favorables o desfavorables. Todo
eso debe quedar congelado antes de la nueva ejecucion.

Por tanto: D4 = SI, pero exclusivamente como proyecto nuevo, con
protocolo independiente y nueva firma. No reabre automaticamente v8 ni
invalida sus resultados.

---

# D5. Mantener las prohibiciones

DICTAMEN: SI

Las prohibiciones son coherentes con el estado del frente y deben
permanecer vigentes. Especialmente importantes son estas cuatro:

1. No reducir/eliminar e(Se). Correcto. Dado que el fallo fundamental
   identificado por v8 esta precisamente en la estimacion de
   sensibilidad, eliminar el componente problematico del gate seria
   cambiar la pregunta para conseguir PASS. No admisible.

2. No modificar q_D, pi_Y ni Y tras observar resultados. Correcto. Son
   elementos estructurales del diseno estadistico. Alterarlos post hoc
   seria una clara contaminacion de la validacion.

3. No cambiar B / R_MC para conseguir PASS. Correcto. Los parametros
   computacionales no pueden utilizarse como perilla de optimizacion de
   resultado.

4. No declarar PASS por Sp/PPV/NPV sin Se. Correcto. Una validacion del
   detector SOW sin sensibilidad correctamente identificada estaria
   incompleta para el claim que se pretende probar.

---

# Estado final recomendado del frente

    Componente                        Estado
    --------------------------------  ---------------------------------
    Capa 1 - Semantica SOW            NO VALIDADA
    Capa 2 - RANGE -> DISTRIBUTION    NO VALIDADA
    Capa 3 - Predictiva H20           NO VALIDADA / NO EJECUTADA
                                      CONFIRMATORIAMENTE
    SOW v1.9                          FROZEN FOR VALIDATION / NOT PROD
    SOW params                        FAIL-CLOSED / NO ACTIVAR
    DISTRIBUTION                      FUERA DE PRODUCCION
    v1.8-core 4 fases                 MIGRACION AUTORIZABLE TRAS 5c
    Legacy                            Permanece en produccion hasta
                                      completar 5c + migracion
    Reapertura Capa 1                 Permitida solo mediante nuevo
                                      protocolo firmado
    Merge a main                      NO autorizado todavia

---

# Veredicto cerrado D1-D5

D1 - SI.
Las tres capas son claims independientes y deben tener validaciones
independientes.

D2 - REFORMULAR.
No declarar "imposibles de validar" en terminos absolutos. Declarar
NO VALIDADAS Y NO VALIDABLES CON EL DISENO ACTUAL, salvo nuevo
mecanismo de identificacion/enriquecimiento admisible.

D3 - SI, CONDICIONADO.
Autorizar v1.8-core de cuatro fases una vez cerrada 5c, sin activar SOW
ni introducir DISTRIBUTION.

D4 - SI, CONDICIONADO.
Reapertura mediante panel experto y/o eventos externos unicamente bajo
nuevo protocolo pre-registrado, independiente, ciego y firmado.

D5 - SI.
Todas las prohibiciones de la seccion 7 del documento 57 permanecen
vigentes.

# Dictamen global

NO EXISTE BASE PARA ACTIVAR SOW, VALIDAR DISTRIBUTION NI DECLARAR
CONFIRMADO NINGUNO DE LOS CLAIMS.

SI EXISTE BASE PARA CERRAR EL BLOQUE SOW/DISTRIBUTION EN SU ESTADO
ACTUAL, preservar integramente los resultados de v8 y trasladar el
desarrollo operativo unicamente al core v1.8, sujeto a 5c.

La unica reapertura cientificamente justificable de Capa 1 debe producir
nueva evidencia independiente; no debe reinterpretar ni rescatar
retrospectivamente los cinco intentos anteriores.

---

Fin del dictamen 57b.