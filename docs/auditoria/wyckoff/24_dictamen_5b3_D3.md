# DICTAMEN EXTERNO - Expediente 5b.3 / D3

**Documento normativo. Fuente autoritativa de la decision sobre 5b.3.**
**Fecha:** 2026-10-02.
**Emitido en respuesta a:** 23_expediente_5b3_D3.md.
**Resultado:** 5b.3 declarada FAIL. Se abre 5b.4.

---

## 0. Veredicto

**No se aprueba la congelacion de ningun conjunto de parametros con los
resultados actuales de 5b.3.**

El motivo no es solo el incumplimiento de D3. El dictamen detecta un
problema metodologico anterior que afecta a la interpretacion del
incremental_lift_H20: **el grupo confirmado y el grupo baseline no
estan anclados al mismo momento temporal (inmortal time bias).**

Eso obliga a tratar el resultado `lift > 0` como **evidencia
exploratoria prometedora, no como validacion de que SOW aporta
informacion incremental**.

---

## 1. Hallazgo critico adicional: ancla temporal asimetrica

El protocolo ejecutado en 5b.3 hace:

    CONFIRMED:
    candidate_t0
        -> SOW en t_sow
        -> outcome = struct[t_sow + 20] < struct[t_sow]

    BASELINE:
    candidate_t0
        -> outcome = struct[t0 + 20] < struct[t0]

No estan comparando el mismo reloj.

Para un candidato confirmado:

    t0 ------- SOW --------- +20
          espera 0..30

Para el baseline:

    t0 -------------- +20

La diferencia es material: los candidatos confirmados pueden necesitar
hasta M=30 sesiones para conseguir el SOW. El episodio confirmado
tiene un periodo entre candidate y SOW que no existe en el mismo
sentido para el baseline.

Esto es inmortal time bias: analizar una exposicion variable en el
tiempo como si el grupo estuviera definido desde el origen puede
introducir sesgos temporales.

### Consecuencia

No se acepta `240/240 lift > 0` como demostracion definitiva de que
SOW es confirmatorio. Puede ser senal real. Pero el diseno actual no
permite aislar cuanto del lift procede de SOW y cuanto del diferente
momento de anclaje.

---

## 2. Este es ahora el problema prioritario

El expediente decia: "SOW aporta informacion mas alla de candidate".
Para demostrarlo:

    candidate + SOW
        vs
    candidate sin SOW

Ambos evaluados desde un **mismo landmark temporal**. Esa es la
condicion que falta.

---

## 3. Solucion recomendada: landmark por M

No se recomienda matching inicial. La solucion limpia es convertir M
en un landmark predefinido:

    L = t0 + M

Hasta L:

    aparece un SOW valido?

Entonces:

    confirmed = existe SOW en [t0, L]
    baseline  = no existe SOW en [t0, L]

**Ambos grupos empiezan el outcome en L.**

    candidate
        |
        +-- ventana M ----+
        |                 |
        |             LANDMARK
        |                 |
        +- SOW -> CONF    |
        +- no SOW -> BASE |
                          |
                      + H20

Esto es mas limpio. El analisis landmark fija un momento comun y
determina la pertenencia a grupos usando la informacion disponible
hasta ese momento, evaluando despues el outcome.

---

## 4. La solucion corrige tambien el problema de M=30

Con el diseno actual:

    M=30
    outcome H20 desde SOW

SOW puede ocurrir en t0+30. En la nueva arquitectura:

    M=30
    landmark = t0+30
    outcome = landmark +20

No hay utilizacion del futuro del outcome para determinar el grupo.
Por tanto `M <= H20` deja de ser necesario.

---

## 5. Reinterpretacion obligatoria del resultado actual

La candidata:

    N=60, M=30, X=0.50, Y=1.20

muestra actualmente:

    cal lift = +0.192
    hold lift = +0.226

Bajo el nuevo diseno hay que recalcular `lift_landmark`. No se sabe si
seran +0.20, +0.05, o incluso <= 0. Por tanto:

> No se puede usar el +0.192/+0.226 actual como argumento para congelar A.

---

## 6. D3=50% no debe relajarse ahora

No se aprueba 0.50 -> 0.40. Aunque 40% sea alcanzable, seria
modificacion motivada por el resultado. El protocolo congelado decia
D3 >= 50%, y 0/240 lo cumple. El resultado formal de 5b.3 es:

> **D3 FALLA. 5b.3 no selecciona parametros.**

Eso se conserva como resultado historico.

---

## 7. Tampoco sustituir D3 inmediatamente por lift > 0

Hay que evitar otro ajuste post-hoc:

    D3 falla
    -> "entonces usamos lift > 0"

No es aceptable como cierre de 5b.3. La conclusion correcta es:

    5b.3 -> FAIL del protocolo original

y despues:

    5b.4 -> nuevo protocolo -> nuevo diseno temporal -> nuevos
    criterios -> nueva ejecucion

---

## 8. Lo que si cambia en 5b.4

El lift debe pasar a ser el **outcome principal**. El objetivo real no
es "que al menos el 50% de los confirmados sigan deteriorandose", es:

> "que la confirmacion SOW discrimine mejor el deterioro futuro que un
> candidato equivalente sin confirmacion."

Eso se mide con `lift_H20`.

---

## 9. lift > 0 es demasiado debil como criterio

Un criterio `lift > 0` puede aceptar +0.3 pp con muestra enorme aunque
economicamente sea irrelevante. Necesitamos distinguir significancia
estadistica de magnitud descriptiva.

### Recomendacion

Para cada configuracion:

    lift_H20 + intervalo de confianza

Debido a multiples episodios por ticker, la inferencia debe respetar
la agrupacion temporal. No tratar los 2.415 episodios como
observaciones independientes. Opcion razonable:

    cluster/bootstrap por ticker

Exigir para validacion final:

    lower_CI(lift_H20) > 0

---

## 10. El criterio absoluto de 50% queda como diagnostico

No se borra de la historia. Se convierte en:

    D3_diagnostic: pct_conf_struct_H20

pero ya no como puerta binaria de seleccion. El informe puede mostrar:

    confirmed outcome = 44.6%
    baseline          = 25.4%
    lift              = +19.2 pp

Eso es mas informativo que decir "44.6% < 50% -> FAIL" cuando 50% es
simplemente un nivel absoluto.

---

## 11. Seleccion de parametros tras corregir el protocolo

No se usa top holdout para seleccionar. Regla:

    CALIBRACION -> seleccion -> HOLDOUT -> confirmacion

---

## 12. El holdout actual ya no es "ciego"

El equipo ha visto top-10 holdout (X=1.00) y ha usado esa divergencia
en el analisis. Aunque no se haya usado formalmente para seleccionar A,
el holdout ya ha sido inspeccionado.

> No se reutiliza como holdout final de una nueva 5b.4.

Puede usarse como diagnostico historico, no como validacion final
independiente del nuevo protocolo.

---

## 13. Consecuencia importante para el proyecto

Si no existe actualmente un bloque temporal no inspeccionado:

    5b.4 puede hacer desarrollo/robustez
    NO puede producir validacion final completamente ciega

La validacion final tendra que usar:
- un bloque historico reservado no usado en el nuevo proceso, si
  existe; o
- una nueva ventana temporal futura una vez disponible.

No es correcto inventar independencia estadistica reutilizando el
mismo holdout ya examinado.

---

## 14. Divergencia X=0.50 vs X=1.00

No se elige ninguna por ese resultado. El hecho:

    cal:  X=0.50 domina
    hold: X=1.00 domina

puede proceder de regimen + muestra + ruido + seleccion. No sabemos
cual. Hay una senal interesante: N=60, M=30 aparece con frecuencia
entre configuraciones fuertes. No es suficiente para congelar.

---

## 15. Analisis adicional aprobado

Una vez corregido el landmark: sensibilidad temporal por bloques.

    bloque A / bloque B / bloque C / bloque D

Para cada combinacion: lift_H20_A, lift_H20_B, lift_H20_C, lift_H20_D.
Medir mediana de lifts, minimo lift, proporcion de bloques positivos.

Pregunta util: la senal funciona consistente o depende de un unico
regimen?

No elegir "mejor media" si tiene picos aislados. Preferir estabilidad.

---

## 16. Nota semantica

`struct[t+20] < struct[t]` NO significa deterioro continuo. Significa
struct menor en t+20 que en t. Puede haber oscilado durante las 20
sesiones. El nombre `struct_deterioration_H20` es aceptable **si se
documenta como comparacion punto-a-punto**. No introducir metrica de
continuidad ahora.

---

## 17. Decision sobre la Candidata A

**NO CONGELAR.**

    N=60, M=30, X=0.50, Y=1.20

Es candidata diagnostica fuerte, no parametro aprobado. Tiene buen
resultado puntual, muestra suficiente, vecindario estable, lift
positivo. Pero tiene anclaje incorrecto, holdout ya observado, D3
original incumplido.

---

## 18. Protocolo 5b.4 recomendado

No ejecutar hasta congelar estas reglas:

- **Unidad:** candidate episode.
- **Landmark:** L = candidate_start + M.
- **Confirmacion:** SOW valido en [candidate_start, L].
- **Baseline:** candidate sin SOW en [candidate_start, L].
- **Outcome:** ambos desde L, no desde fechas distintas.
- **Primario:** incremental_lift_H20.
- **Secundarios:** H10, H40, price weakness, below support, lower low.
- **Inferencia:** cluster/bootstrap por ticker.
- **Criterio de seleccion:** definir antes de mirar resultados.
- **Holdout:** verdaderamente independiente del proceso de seleccion;
  el actual queda como diagnostico historico, no como nuevo holdout
  ciego.

---

## 19. Respuesta a P1-P3

| Pregunta | Dictamen |
|---|---|
| **P1 (D3)** | No relajar 50% ahora. 5b.3 falla D3. 5b.4 debe rediseñar D3 como diagnostico, no puerta binaria. Lift_H20 como outcome principal + robustez. |
| **P2 (X=0.50 vs X=1.00)** | No elegir ninguna usando el holdout observado. Resolver tras rediseno temporal con estabilidad entre bloques de desarrollo. |
| **P3 (Candidata A)** | No aprobada para congelacion. Compite de nuevo bajo protocolo landmark. |

---

## 20. Estado de auditoria

    5b.3 original
        -> PROTOCOL FAIL (D3 = 0/240)
        -> resultado lift positivo
        -> PROMETEDOR, NO VALIDATORIO
        -> se descubre desalineacion temporal
        -> 5b.4 requerida

Parametros:

    K=0.25   CONGELADO
    N        PROPUESTO (no calibrado)
    M        PROPUESTO (no calibrado)
    X_ATR    PROPUESTO (no calibrado)
    Y_VOL    PROPUESTO (no calibrado)

5c.4 y 5d: BLOQUEADAS.

---

## 21. Dictamen final

No se aprueba congelar N=60, M=30, X=0.50, Y=1.20. Tampoco convertir
retrospectivamente D3=50% en 40% ni sustituirlo inmediatamente por
lift>0.

La siguiente version de la metodologia debe corregir el problema
fundamental:

    candidate
        -> ventana M
        -> LANDMARK comun
           +- SOW
           +- no SOW
        -> mismo reloj H20

Entonces se podra determinar si el hallazgo `240/240 lift positivo` es
propiedad robusta de SOW o en parte consecuencia del diseno de anclaje
actual.

Aprobacion condicionada a reformular 5b.4 con:
- landmark temporal,
- outcome primario basado en lift,
- inferencia agrupada por ticker,
- holdout verdaderamente no inspeccionado para validacion final.

---

**Fin del dictamen 24.**