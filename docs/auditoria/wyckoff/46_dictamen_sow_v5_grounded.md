Este hallazgo **bloquea la vía grounded actual**, pero también permite cerrar definitivamente una cuestión que estaba mal planteada desde v5.



## Dictamen ejecutivo



**No acepto `MUESTRA INSUFICIENTE` como diagnóstico científico de SOW.**



El resultado `N_eval=2` es el resultado de intentar estimar un modelo demasiado exigente **dentro de ventanas OUTER TEST de 1–2 años**, y además mediante ajuste y evaluación sobre la misma ventana. Eso produce exactamente los problemas observados:



* `R_stress` constante → `SOW × R_stress` no identificable.

* muy pocos episodios normales → separación/inestabilidad.

* colinealidad entre `D1/D2/D3`.

* y, fundamentalmente, ajuste en el propio OUTER TEST → **ya no es una evaluación externa de generalización**.



Tengo que corregir expresamente una cosa de mi dictamen anterior: **fui demasiado permisivo al aceptar el “grounded” estimado directamente dentro del OUTER TEST**. Que pueda ser un estimador descriptivo no significa que pueda utilizarse como validación externa de un detector. La razón de ser del nested CV es precisamente mantener la selección/ajuste separado de la evaluación exterior. ([Scikit-learn][1])



Por tanto:



> **v5 grounded no debe repararse cambiando el tamaño de ventana, quitando la interacción o relajando el modelo fold por fold. Debe abandonarse como estimando primario.**



\---



# 1. Los números de fold 1 demuestran además que el problema no era solo convergencia



Esto es muy revelador:



```text

Fold 1

RD_stress = +0.1135



bootstrap:

IC95 B20 ≈ [-0.1016, +0.3419]

```



La anchura es aproximadamente:



$$

0.4435

$$



Eso cambia completamente la interpretación.



La versión anterior producía:



```text

RD ≈ +0.11

IC ≈ ±0.001

```



porque estaba midiendo la estabilidad de las **predicciones del modelo congelado**.



La grounded produce:



```text

RD ≈ +0.11

IC ≈ ±0.22

```



porque ahora intenta cuantificar incertidumbre de un contraste estimado en una ventana muy pequeña y heterogénea.



**Este resultado no dice que SOW sea malo. Dice que ese estimando es demasiado inestable para esas ventanas exteriores.**



\---



# 2. No acepto la afirmación de que `LowerCI95 > 0.05` carece de justificación



Aquí hay que hacer otra distinción.



Si tuvierais un estimador válido de:



$$

RD_r

$$



y un IC95 con cobertura adecuada, entonces:



$$

LowerCI95(RD_r)>0.05

$$



es un criterio perfectamente legítimo de evidencia de que el contraste supera el mínimo práctico.



El problema **no es el umbral de 5 pp**.



El problema es que actualmente:



```text

estimador

\+

ventana

\+

modelo

```



no son adecuados para proporcionar ese IC como evidencia externa.



En otras palabras:



> **No hay que tirar `δ_min=5 pp`; hay que cambiar el estimador que debe sostenerlo.**



\---



# 3. Rechazo las cuatro soluciones A–D tal como están



| Alternativa                 | Dictamen                                         |

| --------------------------- | ------------------------------------------------ |

| **A. Aceptar IC actuales**  | ❌                                                |

| **B. Reajustar en TEST**    | ❌ como parche                                    |

| **C. Bootstrap TRAIN+TEST** | ⚠️ útil, pero no arregla por sí solo el problema |

| **D. RD crudo**             | ❌ como estimando principal                       |



Y recomiendo una **E corregida**.



\---



# 4. La solución que recomiendo para v6



## Volver a evaluación verdaderamente OUT-OF-SAMPLE



El candidato continúa siendo seleccionado exclusivamente en INNER:



```text

INNER TRAIN

&#x20;    ↓

240 combinaciones

&#x20;    ↓

1 candidato

&#x20;    ↓

OUTER TRAIN

&#x20;    ↓

ajuste del estimador

&#x20;    ↓

OUTER TEST

&#x20;    ↓

Y observado

```



La diferencia fundamental es:



> **OUTER TEST nunca se utiliza para seleccionar ni ajustar el modelo que produce la predicción individual del episodio.**



Esto recupera la propiedad esencial de nested validation. ([Scikit-learn][1])



\---



# 5. Pero necesitamos que Y del OUTER participe realmente



Aquí está la solución que considero más fuerte.



Mantendría `δ_min=5 pp` y utilizaría un **contraste de riesgo ajustado y observado mediante un estimador de un paso / AIPW**, con todos los nuisance models estimados exclusivamente en OUTER TRAIN.



Formalmente, para cada episodio:



$$

m_a(X,R)=P(Y=1\\mid SOW=a,X,R)

$$



y:



$$

e(X,R)=P(SOW=1\\mid X,R)

$$



ambos estimados en OUTER TRAIN.



Después, en OUTER TEST:



$$

\\widehat{RD}

=

\\frac1n\\sum_i

\\left[

m_1(X_i,R_i)-m_0(X_i,R_i)

\+

\\frac{SOW_i}{e_i}(Y_i-m_1)

\-

\\frac{1-SOW_i}{1-e_i}(Y_i-m_0)

\\right]

$$



y lo mismo estratificado por `stress` y `normal`.



Esto tiene tres ventajas:



**Primera:** `Y_TEST` participa directamente.



**Segunda:** conserva el ajuste por `X0` y régimen.



**Tercera:** mantiene el contraste en **puntos porcentuales**, por lo que `δ_min=5 pp` sigue teniendo sentido.



La g-computation clásica obtiene el contraste estandarizado a partir de los modelos de outcome y las predicciones bajo las dos condiciones; los estimadores aumentados añaden una corrección basada en los outcomes observados. ([PubMed Central (PMC)][2])



### Importante



No lo llamaría efecto causal.



Lo denominaría:



> **contraste de riesgo ajustado asociado a SOW**



porque SOW no fue aleatorizado.



\---



# 6. Hay que añadir propensity/overlap



La alternativa E introduce una variable nueva:



$$

e(X,R)

$$



Por tanto, debemos impedir pesos extremos.



Con frecuencia SOW ≈ 6%, esto es particularmente importante.



Debe existir un soporte ex ante como:



```text id="8u6b0x"

ε <= e(X,R) <= 1-ε

```



con `ε` preespecificado.



Si aparecen demasiadas observaciones fuera de ese soporte, el régimen/fold debe declararse no estimable.



**No recomendaría decidir ε después de mirar los resultados.**



\---



# 7. ¿Y el bootstrap?



Aquí sí tenemos una solución limpia.



En cada OUTER TEST:



```text id="z33v2t"

OUTER TRAIN

&#x20;   ↓

fit m(X,SOW,R)

fit e(X,R)

&#x20;   ↓

congelar ambos

&#x20;   ↓

OUTER TEST

&#x20;   ↓

MBB B20/B40/B50

&#x20;   ↓

para cada réplica:

&#x20;   recalcular AIPW usando Y observado

&#x20;   NO reajustar nuisance models

```



Esto mantiene la filosofía de evaluación externa y hace que el bootstrap incorpore:



* variabilidad de `Y`;

* variabilidad temporal;

* dependencia cross-sectional;

* composición de los episodios.



No incorpora la variabilidad del ajuste de `m` y `e`, porque esos modelos se consideran parte del predictor entrenado en el periodo previo.



Eso es aceptable para una evaluación del rendimiento de un procedimiento ya entrenado.



Para estimadores de g-computation, volver a estimar el modelo en cada bootstrap es una vía habitual cuando el objetivo es inferencia sobre el procedimiento de estimación completo; pero aquí queremos preservar la separación entre entrenamiento y evaluación exterior. ([PubMed Central (PMC)][3])



\---



# 8. Existe una alternativa todavía más sencilla



Yo añadiría como **métrica predictiva primaria secundaria**:



### Modelo completo



```text

SOW + R + X0

```



frente a:



### Modelo baseline



```text

R + X0

```



ambos ajustados exclusivamente en OUTER TRAIN.



Y después medir en OUTER TEST:



$$

\\Delta Brier =

Brier_{baseline}-Brier_{SOW}

$$



y:



$$

\\Delta LogLoss =

LogLoss_{baseline}-LogLoss_{SOW}

$$



usando los `Y` realmente observados.



El Brier score es una regla estrictamente propia para probabilidades y compara directamente probabilidades predichas con outcomes reales; log loss tiene la misma propiedad para probabilidades. ([Scikit-learn][4])



Esto responde exactamente:



> **¿SOW añade capacidad predictiva fuera de muestra?**



Y además evita que una RD model-implied enorme con mala calibración pueda confundirse con una buena predicción.



\---



# 9. De hecho, recomiendo dos estimandos en v6



Esto permite resolver todo el conflicto que hemos ido descubriendo.



### Estimando primario de activación



$$

RD^{AIPW}_{stress}

$$



$$

RD^{AIPW}_{normal}

$$



con:



$$

\\delta_{min}=5pp

$$



Este conserva la lógica económica original.



### Estimando secundario de robustez predictiva



$$

\\Delta Brier

$$



y:



$$

\\Delta LogLoss

$$



Esto verifica que el detector realmente mejora las predicciones sobre datos exteriores.



Así:



```text id="o3dph7"

RD AIPW

&#x20;  +

predictive skill

&#x20;  +

placebos

&#x20;  +

temporal robustness

```



constituirían un paquete probatorio mucho más fuerte que cualquiera de los enfoques aislados.



\---



# 10. ¿Qué hacemos con la interacción `SOW × R`?



No recomiendo eliminarla simplemente para evitar `LinAlgError`.



Pero tampoco recomiendo obligar a que cada fold TEST estime esa interacción.



La interacción debe pertenecer al **modelo entrenado en OUTER TRAIN**.



Y el régimen específico se obtiene posteriormente mediante:



$$

RD^{AIPW}_{stress}

$$



y:



$$

RD^{AIPW}_{normal}

$$



en la ventana exterior.



Si un OUTER TRAIN tiene soporte suficiente, la interacción queda identificada.



Si un OUTER TEST tiene pocos normales, eso afecta a la **precisión** de `RD_normal`, no obliga a cambiar el modelo.



\---



# 11. Esto también resuelve los folds 4 y 5



La situación:



```text

2020: normal = 0

2021: normal = 0

```



ya no provoca:



```text

matrix singular

```



porque el modelo no se estima dentro de esos TEST.



El modelo se estima usando el histórico anterior:



```text

Fold 4:

TRAIN 2015-2019

TEST 2020

```



```text

Fold 5:

TRAIN 2015-2020

TEST 2021

```



Si TRAIN contiene ambos regímenes, la interacción puede estimarse.



Después, en 2020, simplemente:



```text

RD_stress = estimable

RD_normal = no estimable

```



Eso es exactamente lo que debería significar la ausencia de observaciones normales.



No se fuerza una regresión imposible.



\---



# 12. Pero hay una consecuencia sobre la regla de `N_eval`



Aquí introduciría un cambio importante.



Actualmente:



```text

N_eval >= 6

```



es una condición global.



Pero ya hemos comprobado que la evaluabilidad por régimen puede ser muy distinta.



Yo definiría:



```text id="le3n2e"

N_eval_stress

N_eval_normal

```



y:



```text

N_eval_total

```



por separado.



No permitiría que:



```text

N_eval_total = 7

```



oculte:



```text

N_eval_normal = 2

```



cuando queremos declarar universal.



\---



# 13. El resultado actual de v5 no debe considerarse un FAIL de SOW



La salida:



```text id="hjqy2d"

MUESTRA INSUFICIENTE

N_eval=2

```



debe archivarse como:



> **“No estimable bajo el estimando grounded v5”**



no:



> “SOW tiene muestra insuficiente”.



Eso es una distinción crítica.



El dataset puede tener cientos o miles de episodios por fold y aun así el **modelo específico** ser no identificable.



Es exactamente lo que ha ocurrido aquí.



\---



# 14. S-01: arreglar inmediatamente



Esto no necesita debate.



```text

manifest.protocol_version = "v4.1"

```



cuando el protocolo/ejecución es v5 es un fallo de trazabilidad.



### Dictamen



**CORREGIR antes de cualquier ejecución posterior.**



Y el manifiesto debería contener:



```text id="stn4dw"

protocol_version

protocol_sha256

code_base_commit

prereg_commit

data_sha256

```



No es deuda aceptable para un experimento confirmatorio.



\---



# 15. S-02: la asimetría también debe desaparecer



Correcto el hallazgo.



Actualmente:



```text id="u9b4w8"

placebo

→ paradigma predictivo



bootstrap

→ paradigma grounded

```



Eso hace que `p_rand` y el IC primario pertenezcan a objetos estadísticos distintos.



En v6:



```text id="w2pgd5"

REAL

PLACEBO A

PLACEBO B

```



deben pasar por **el mismo estimador final**, cambiando solamente la asignación placebo de SOW.



Esto es importante.



\---



# 16. Por tanto, mi decisión sobre las cuatro preguntas



| Pregunta                              | Dictamen                                                                                                                              |

| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |

| **1. ¿Aceptar MUESTRA INSUFICIENTE?** | **No como conclusión científica sobre SOW.** Sí como diagnóstico de que el estimando grounded v5 no es estimable.                     |

| **2. ¿Cambiar criterio LowerCI?**     | **No todavía.** `δ_min=5pp` sigue siendo defendible si el nuevo RD está correctamente estimado.                                       |

| **3. ¿Redefinir grounded?**           | **Sí, pero no con ventanas TEST más largas ni eliminando variables ad hoc. Abrir v6 con estimador OUT-OF-SAMPLE que utilice Y_TEST.** |

| **4. S-01/S-02**                      | **Corregir ambos obligatoriamente antes de la siguiente ejecución.**                                                                  |



\---



# 17. Qué NO haría



No haría ninguna de estas cuatro cosas:



```text id="4p6s0a"

Fold 4 → quitar interacción porque no converge

Fold 5 → quitar normal

Fold 6 → quitar D2_prev

Fold 7 → quitar D3_prev

```



Eso sería **model surgery después de observar qué folds fallan**.



Tampoco:



```text id="9pmrgm"

2020 + 2021 + 2022 → combinar porque así hay más normal

```



porque destruiría la estructura temporal del outer.



Y tampoco:



```text id="nbx7sp"

usar Y_TEST para recalibrar el modelo y luego declarar ese mismo

TEST como evidencia exterior

```



porque volveríamos al problema de v5.



\---



# 18. Mi recomendación de v6 queda así



```text id="7gl47p"

&#x20;               SOW v6

&#x20;                 │

&#x20;      ┌──────────┴──────────┐

&#x20;      ▼                     ▼

&#x20;   INNER                  OUTER

selección 240            candidato fijo

&#x20;      │                     │

&#x20;      └──────────┬──────────┘

&#x20;                 ▼

&#x20;         modelos fit TRAIN

&#x20;                 │

&#x20;       ┌─────────┴─────────┐

&#x20;       ▼                   ▼

&#x20;     m(Y|.)             e(SOW|.)

&#x20;       │                   │

&#x20;       └─────────┬─────────┘

&#x20;                 ▼

&#x20;             OUTER TEST

&#x20;                 │

&#x20;       ┌─────────┴─────────┐

&#x20;       ▼                   ▼

&#x20;     Y real             SOW real

&#x20;       │                   │

&#x20;       └─────────┬─────────┘

&#x20;                 ▼

&#x20;            RD AIPW

&#x20;         stress / normal

&#x20;                 │

&#x20;                 ▼

&#x20;            MBB B20/B40

&#x20;                 │

&#x20;                 ▼

&#x20;        placebo A / placebo B

&#x20;                 │

&#x20;                 ▼

&#x20;           Brier / LogLoss

&#x20;                 │

&#x20;                 ▼

&#x20;         decisión universal

&#x20;         / condicional / no

```



Esto sí responde a las dos cosas que queremos demostrar:



> **¿SOW añade información?**



y:



> **¿esa información funciona sobre outcomes que no participaron en el ajuste?**



\---



# 19. Dictamen formal



# DICTAMEN EXTERNO — SOW v5 GROUNDED



**Fecha:** 2026-10-04

**Referencia:** `sow-v4-validation`, HEAD `80a0e08`

**Objeto:** evaluación del estimando grounded y del resultado `MUESTRA INSUFICIENTE`

**Estado:** no aprobado para activación



## Dictamen



**S-03 se confirma como bloqueante, pero la conclusión correcta no es que el dataset SOW sea insuficiente.**



El estimando grounded implementado ajusta el modelo en el propio OUTER TEST y posteriormente evalúa el mismo conjunto. Esto transforma la ventana exterior en una muestra de estimación in-sample y deja de constituir una evaluación externa de generalización.



La nested validation requiere que selección y evaluación permanezcan separadas; utilizar la misma información exterior para ajustar y evaluar produce optimismo por reutilización de datos. ([Scikit-learn][1])



La no convergencia observada en cinco de siete folds demuestra además que la especificación `SOW × R + D1_prev + D2_prev + D3_prev` no resulta identificable de forma estable en varias ventanas TEST.



El fold 1 proporciona evidencia adicional: con el procedimiento grounded correctamente reajustado, el IC95 tiene anchura aproximada 0,44, mostrando que la incertidumbre real de este estimando es muy superior a la de los intervalos del paradigma predictivo anterior.



## Decisión sobre el resultado



La salida automática:



`MUESTRA INSUFICIENTE (N_eval=2)`



se acepta únicamente como **diagnóstico del estimando grounded v5**, no como conclusión científica de que el dataset carece de muestra suficiente para SOW.



No se autoriza utilizar ese resultado para declarar SOW rechazado ni para modificar retrospectivamente el protocolo.



## Alternativa recomendada



Se recomienda abandonar el grounded in-sample como estimando primario y desarrollar **SOW v6** con una evaluación auténticamente out-of-sample.



La candidata continuará siendo seleccionada exclusivamente en INNER.



Los modelos de resultado y propensity necesarios para el estimador ajustado se estimarán exclusivamente en OUTER TRAIN.



El OUTER TEST aportará los outcomes observados y se utilizará para estimar un contraste de riesgo ajustado mediante un estimador de un paso/AIPW, manteniendo `delta_min = 5 pp`.



La interpretación seguirá siendo predictiva/asociativa y no causal.



## Robustez predictiva adicional



Se recomienda incorporar como análisis secundario la diferencia de Brier score y log loss entre:



* modelo baseline `R + X0`;

* modelo `R + X0 + SOW`.



Ambos deberán entrenarse exclusivamente en OUTER TRAIN y evaluarse contra `Y` observado en OUTER TEST. El Brier score y log loss son reglas estrictamente propias para evaluar predicciones probabilísticas frente a outcomes reales. ([Scikit-learn][4])



## Bootstrap



El bootstrap OUTER deberá conservar el carácter externo.



No se reajustarán modelos con datos del OUTER TEST para producir el predictor.



El MBB de evaluación utilizará `Y` observado y recalculará el estimador ajustado en cada muestra bootstrap, de acuerdo con el estimando v6 que sea pre-registrado.



## Placebos



Los placebos A y B deberán utilizar exactamente el mismo estimador final que el efecto real, modificando únicamente la asignación de SOW conforme a las reglas de cada negative control.



No se mezclarán paradigmas predictivo y grounded entre efecto real y placebo.



## Trazabilidad



S-01 debe corregirse obligatoriamente antes de la siguiente ejecución.



`manifest.protocol_version = v4.1` cuando la ejecución corresponde a v5 es un defecto de trazabilidad.



También deberá corregirse S-02 antes de cualquier nueva ejecución.



## Decisión final



**v5 grounded: NO APROBADO COMO VALIDACIÓN CONFIRMATORIA.**



**MUESTRA INSUFICIENTE de v5: no interpretar como insuficiencia del dataset SOW.**



**No se autoriza activar SOW basándose en los resultados grounded actuales.**



Se autoriza el desarrollo de **SOW Protocol v6**, cuyo estimando primario deberá utilizar outcomes observados del OUTER TEST sin reutilizar dichos datos para seleccionar la candidata ni entrenar el predictor que se evalúa.



El `delta_min = 5 pp` puede mantenerse como margen práctico hasta que el nuevo estimando sea especificado y auditado.



**DICTAMEN: v5 RECHAZADO COMO ESTIMANDO CONFIRMATORIO; v6 REQUERIDO.**



### La conclusión importante



Hemos llegado a un punto en el que **no conviene seguir parcheando v4/v5**.



El experimento ha descubierto una distinción esencial:



```text

predicción model-implied

&#x20;       ≠

evidencia observada out-of-sample

```



La primera dio IC artificialmente estrechos. La segunda, mediante grounded in-sample, dio IC enormes y problemas de identificabilidad.



La solución no está en buscar “el tamaño de bootstrap correcto”. Está en construir un **estimador externo que conserve `Y_TEST`, conserve los 5 pp como magnitud económica y no reutilice el TEST para entrenarse**.



Mi recomendación firme es **abrir v6**, corregir S-01/S-02 y hacer que el núcleo sea **OUTER-TRAIN → predictor/nuisance fijo → OUTER-TEST con Y observado → contraste ajustado + Brier/LogLoss → MBB → placebos**. Eso es defendible; seguir intentando rescatar el grounded de v5 no lo es.



[1]: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html?utm_source=chatgpt.com "Nested versus non-nested cross-validation — scikit-learn 1.9.1 documentation"

[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8012235/?utm_source=chatgpt.com "Machine learning for causal inference: on the use of cross-fit estimators - PMC"

[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5223318/?utm_source=chatgpt.com "G-computation of average treatment effects on the treated and the untreated - PMC"

[4]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration — scikit-learn 1.9.1 documentation"



