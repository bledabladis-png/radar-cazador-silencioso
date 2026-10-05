# 55b - Dictamen del auditor sobre protocolo v8 revision 4

Fecha: 2026-10-05
Decision: GO METODOLOGICO, con condiciones de congelacion menores.

---

## 1. Q14 - R_MC = 200 / 500

APROBADO.

    SCREENING:      R_MC = 200, B = 500
    CONFIRMACION:   R_MC = 500, B = 2000

Separacion correcta entre variabilidad entre muestras (R_MC) y
estimacion bootstrap de incertidumbre dentro de cada muestra (B).
Evita el error de r3 de usar B como sustituto de simulacion de
potencia.

---

## 2. Q15 - Generacion determinista de Y

APROBADO con una correccion de redaccion.

§17.2 contenia contradiccion: "conteos exactos" seguido de round()
por cada celda. Debe cambiarse a redondeo global + largest remainder
para TP y FP.

    Para D=1:
        total_TP = redondeo determinista del total D=1.
        Distribuir TP entre C1 y C2 mediante largest remainder,
        respetando tamanos.

    Para D=0:
        total_FP = redondeo determinista del total D=0.
        Distribuir FP entre C3 y C4 mediante largest remainder,
        respetando tamanos.

La palabra "exactos" debe sustituirse por "conteos enteros
deterministas, con redondeo global y asignacion largest remainder".

El implementador no puede elegir posteriormente entre Bernoulli,
round() por celda, floor, ceiling, etc. La regla queda congelada.

---

## 3. Q16 - Estratos censales

APROBADO.

    si n_h = N_h:
        peso fijo = 1
        no calcular lambda_h
        no remuestrear el estrato
        contribucion fija en todas las replicas

No debe aparecer 0/0 ni aproximacion artificial para lambda_h.

---

## 4. §17.5 - Regla para aumentar n

Exigencia: definir como se aumenta n sin dejar decision al
implementador.

Regla congelada:

    Si 800 PASS: diseno aprobado = 800.

    Si 800 FAIL: calcular n_min requerido <= 800.
        Si n_min <= 800:
            redimensionar protocolo antes de anotacion.
            repetir confirmacion.
        Si n_min > 800:
            SUSPENSION.

No se modifica la muestra sobre la marcha.

---

## 5. PASS mide precision, no cobertura

No bloqueante, pero debe quedar documentado.

    PASS = criterio de precision de intervalos.
    NO implica validacion de cobertura nominal 95%.

La palabra "IC95%" se refiere al procedimiento nominal; el gate
verifica anchura/precision, no cobertura.
---

## 6. §15.3 - Cuadricula

Mantengo la aprobacion:

    pi_Y en {0.005, 0.008, 0.010, 0.012}
    Se en {0.60, 0.70, 0.80}

12 escenarios, todos compatibles con q_D = 2559/243435.

---

## 7. §15.4 - Stress test

APROBADO.

Correccion r1 = P(ctx=1 | D=1), r0 = P(ctx=1 | D=0) adecuada.

Conserva la estructura D x ctx. No impone independencia artificial.

Separado del gate como STRESS_TEST_MODEL_BASED.

---

## 8. §17.3 - invalid_rate <= 1%

APROBADO.

    invalid_rate <= 0.01
    B=2000 -> maximo 20 replicas invalidas

NO imputacion, NO sustitucion por cero, NO eliminacion silenciosa.

El resultado debe informar numero exacto de validas e invalidas por
estimador. Una replica invalida para Se no contamina PPV si PPV esta
definida.

---

## 9. §16.3 - Streams de seeds

SeedSequence(master_seed) como origen de dos streams independientes:
stream_MC, stream_bootstrap. No reutilizacion accidental.

Semilla congelada y registrada para MC, bootstrap, SRSWOR.

---

## 10. Q17 - Correcciones adicionales

Cuatro ajustes de texto obligatorios antes de "firmado":

    A. §17.2: redondeo global + largest remainder para TP y FP.
    B. §17.5: si 800 FAIL, buscar menor n_total <= 800 que pase;
       repetir confirmacion; no aumentar durante anotacion.
    C. Terminologia: PASS = criterio de precision, NO cobertura
       nominal 95%.
    D. Semillas: streams separados MC y bootstrap.

---

## DICTAMEN FINAL

v8 revision 4: GO METODOLOGICO.

Autorizada materializacion de gold-standard-v3, siempre que las
cuatro correcciones de cierre anteriores se incorporen antes de
escribir el codigo y queden congeladas en el protocolo.

Firmas del auditor:

| Item                                            | Estado           |
| ----------------------------------------------- | ---------------- |
| Arquitectura v8                                 | APROBADA         |
| ctx = AND                                       | APROBADO         |
| Una muestra unica                               | APROBADO         |
| 4 celdas x sector x periodo                     | APROBADO         |
| SRSWOR                                          | APROBADO         |
| Hamilton                                        | APROBADO         |
| n_h >= 5 / censo                                | APROBADO         |
| RWY + FPC                                       | APROBADO         |
| 12 escenarios gate                              | APROBADO         |
| 45 escenarios stress                            | APROBADO         |
| R_MC = 200/500                                  | APROBADO         |
| B = 500/2000                                    | APROBADO         |
| invalid_rate <= 1%                              | APROBADO         |
| Ceguera total                                   | APROBADO         |
| 800 unidades por anotador                       | APROBADO         |
| gold-standard-v3                                | APROBADO         |
| Materializacion de codigo                       | GO               |
| Merge a main                                    | NO AUTORIZADO    |
| Modificacion de v1/v2/v3/legacy/config/detector | NO AUTORIZADA    |

Conclusion: GO.

La revision 4 deja de tener defecto metodologico bloqueante. Las
cuatro correcciones de cierre son de especificacion determinista, no
de rediseno. Una vez incorporadas literalmente, no se necesita otra
revision conceptual antes de materializar v8.

---

Fin del dictamen.