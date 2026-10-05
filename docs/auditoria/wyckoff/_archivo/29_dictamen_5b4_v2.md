# DICTAMEN EXTERNO - Segunda ronda 5b.4

**Documento normativo. Fuente autoritativa de la decision sobre 5b.4.**
**Fecha:** 2026-10-02.
**Documento auditado:** 27_calibracion_5b4_resultados.md +
28_expediente_5b4_heterogeneidad.md + protocolo v2 + commit 9fb6bcf.
**Resultado:** 5b.4 FAIL en seleccion, PASS en diagnostico
metodologico. No se congela ningun parametro. Se abre 5b.4-bis.

---

## 0. Veredicto general

5b.4 se ejecuto correctamente respecto del anclaje temporal. NO ha
producido una configuracion seleccionable.

No se modifica D3 para hacer pasar ninguna de las 17.

Conclusiones oficiales:

    5b.4 FAIL como proceso de seleccion bajo protocolo v2.
    La evidencia es valida como diagnostico/desarrollo, NO como
    autorizacion para congelar parametros.

El landmark ha corregido el defecto temporal de 5b.3. NO esta
demostrado que haya eliminado todos los sesgos posibles.

**Correccion linguistica obligatoria:** no escribir
`0.1920 - 0.0637 = 0.1283` como "el sesgo". Es la diferencia entre dos
estimaciones bajo disenos diferentes. La formulacion correcta es:

> "El +0.192 de 5b.3 no es una estimacion valida bajo el diseno
> temporal corregido; al aplicar el landmark, el lift de la misma
> configuracion se reduce a +0.0637."

---

## 1. Evaluacion de resultados

| Hallazgo | Dictamen |
|---|---|
| Landmark reduce sustancialmente el lift de 5b.3 | Consistente con el problema temporal detectado |
| 17/240 con IC95% inferior > 0 | Hallazgo exploratorio |
| Las 17 con solo 2 bloques positivos | Evidencia de heterogeneidad |
| Ninguna cumple D3 | Fallo conforme al protocolo |

La diferencia 0.1920 -> 0.0637 confirma que el resultado de 5b.3 era
muy sensible al anclaje temporal. No autoriza a cuantificar "el sesgo".

---

## 2. P1 - D3 inalcanzable

**Dictamen: NO modificar D3 para cerrar 5b.4.**

Bloque A maximo: 12 confirmed. D3 exige >= 20. D3 es incompatible con
la muestra disponible en A bajo el diseno actual.

No aplicar retroactivamente "3 de 3 evaluables" ni "mitad de
evaluables". Seria cambiar la regla tras ver que falla.

**Decision:** 5b.4 v2 permanece FAIL. Se abre 5b.4-bis / protocolo v3.

No seleccionar ningun parametro de 5b.4.

---

## 3. P2 - Heterogeneidad temporal

El hallazgo es mas importante que D3. Para la candidata A:

    A   -0.31
    B   -0.18
    C   +0.11
    D   +0.11

Bloque B tiene n_confirmed=90 y muestra lift negativo. **La hipotesis
C ("desbalance de muestra") no explica por si sola el patron.**

Pero tampoco esta demostrada la hipotesis A ("SOW solo funciona en
determinados regimenes").

**Dictamen: D - tratar la heterogeneidad como hallazgo que debe
probarse formalmente antes de seleccionar parametros.**

La siguiente fase debe separar: efecto global -> efecto por bloque ->
heterogeneidad -> posible interaccion SOW x regimen temporal.

**Muy importante:** los 4 bloques NO son automaticamente 4 regimenes
de mercado. La coincidencia temporal A-B negativos / C-D positivos
permite hipotesis de regimen, no la demuestra. Pueden ser: cambios de
universo, frecuencia de candidate, cambios estructurales del indicador,
distribucion de episodios, volatilidad, dependencia temporal,
concentracion por tickers.

---

## 4. P3 - Agregacion

**Dictamen: B modificada - ALL descriptivo + analisis por bloque +
heterogeneidad. ALL no gobierna seleccion.**

ALL = +0.0637 es matematicamente valido como diferencia agregada. No
es valido interpretarlo como "SOW genera +6.4 pp de forma estable".

Componentes: A negativo, B negativo, C positivo, D positivo.

Publicar: ALL, A, B, C, D. Medida formal de heterogeneidad con el
mismo esquema de remuestreo por ticker (no chi-cuadrado elemental:
hay dependencia).

Regla fundamental: **no hace falta demostrar heterogeneidad
significativa para dejar de confiar en el agregado.** El patron de
signos opuestos ya es suficiente.

---

## 5. P4 - M=5 vs M=30

**Dictamen: C - neutral.**

M=5 responde a "SOW proximo al candidate anade informacion".
M=30 responde a "SOW en ventana amplia anade informacion".

Son hipotesis distintas, no "pico vs robustez". M queda dentro del
grid, sin premio ni castigo por tamano.

---

## 6. P5 - Las 17 candidatas

**Dictamen: NO congelar ninguna.**

Las 17 fueron descubiertas con el mismo dataset utilizado para
evaluarlas. Todas tienen D3=FAIL y n_bloques_lift_pos=2. No hay
candidata que haya superado la totalidad del protocolo.

Tampoco llevar las 17 a 5b.X: seria 17 apuestas simultaneas y
permitiria seleccionar posteriormente la ganadora.

Arquitectura correcta:

    240 configs -> DESARROLLO -> regla congelada -> UNA config
    -> VALIDACION FUTURA

No: 240 -> 17 -> validar 17 -> escoger.

---

## 7. P6 - Multiple testing

**Dictamen: D2 = filtro exploratorio/desarrollo. NO prueba
confirmatoria.**

No aplicar Bonferroni ni FDR como solucion para cerrar 5b.4. La
arquitectura correcta es: usar historicos para desarrollar una
configuracion y despues validarla en datos temporalmente nuevos.

D2 filtra candidatas, no declara 17 descubrimientos confirmados.

---

## 8. P7 - Bloque A

**Dictamen: no eliminar A post-hoc.**

Tampoco mantenerlo como equivalente a C/D.

**Rediseñar bloques antes de la siguiente ejecucion, sin usar el
lift observado para escoger fronteras.**

El objetivo: construir unidades temporalmente comparables y con
capacidad muestral suficiente. No buscar fronteras donde el SOW
funcione.

Preferencia del auditor: pasar de 4 bloques a **3 bloques
temporales predefinidos**, con fronteras basadas solo en calendario.

No construir fronteras mirando donde desaparece el signo negativo.

---

## 9. Debilidad conceptual de D3

D3 combina dos cosas:

    A. suficiencia muestral
    B. direccion del efecto

Cuando ambas estan en una misma puerta, no sabemos si el fallo
significa "SOW no funciona" o "no hay muestra para saberlo".

**Separar en el proximo protocolo:**

    Gate de suficiencia: n_confirmed >= X AND n_baseline >= X
    Evaluacion de estabilidad: lift > 0

Permite decir `Bloque A = INSUFFICIENT_SAMPLE` en vez de `FAIL`.

---

## 10. Arquitectura recomendada

POR BLOQUE:

    1. Hay muestra suficiente? NO -> INEVALUABLE
                              SI
    2. lift + IC
    3. signo/precision

GLOBAL:

    configuracion -> suficiencia minima -> estabilidad temporal
                  -> efecto agregado -> seleccion

NO:

    configuracion -> obligacion de que 4 bloques tengan
                     simultaneamente suficiente muestra

---

## 11. Correccion sobre IC

B=2000, IC95 = percentiles [2.5, 97.5], lower_CI > 0. Esto es IC
bilateral del 95% usando su limite inferior.

**No llamarlo "test unilateral al 5%".** Es mas conservador que un
unilateral. Mantener IC bilateral como estadistico descriptivo.
Congelar que lower_CI > 0 es **puerta de desarrollo**, no prueba
confirmatoria unilateral.

---

## 12. Dependencia

Bootstrap ticker-cluster es correcto para dependencia intraticker.
NO afirmar que "elimina la dependencia". Los tickers comparten shocks.

Formulacion correcta:

> "La dependencia intraticker se conserva mediante bootstrap por
> cluster; permanece una sensibilidad potencial a dependencia
> temporal/common-shock."

Sensibilidad especificada antes de la siguiente ejecucion.

---

## 13. Heterogeneidad: no congelar SOW por regimen

Los datos sugieren 2021-2024 negativo, 2024-2026 positivo. NO
demuestra "SOW solo funciona en regimen alcista". La frontera puede
capturar simultaneamente varias cosas.

Proxima fase debe comprobar como minimo: lift por bloque + IC por
bloque + diferencia C-D vs A-B + frecuencia de candidate + frecuencia
de confirmed + medida de estabilidad que no convierta la fecha en
explicacion causal.

---

## 14. Conclusion retirada oficialmente

> "El SOW tiene lift positivo universal."

**Con el landmark, esa afirmacion ya no es sostenible.**

Lift_ALL > 0, pero Lift_A < 0, Lift_B < 0, Lift_C > 0, Lift_D > 0.

El signo positivo agregado **no puede considerarse evidencia de
estabilidad temporal**. Es el hallazgo mas importante de toda la
secuencia 5b.3 -> 5b.4.

---

## 15. Que NO debe hacerse

    NO bajar el umbral de D3
    NO pasar de 4 a 3 bloques usando el resultado observado
    NO eliminar A porque perjudica el resultado
    NO congelar la primera de las 17
    NO elegir M=30 porque tiene IC mas estrecho
    NO elegir M=5 porque tiene mayor lift
    NO convertir C-D en "regimen valido" sin prueba adicional
    NO aplicar Bonferroni solo para encontrar una configuracion
    NO usar D como validacion historica

Especialmente: **ninguna modificacion cuyo objetivo implicito sea
conseguir que alguna configuracion pase.**

---

## 16. Conclusiones demostradas

    C1. Correccion temporal: landmark es metodologicamente superior
        al anclaje asimetrico de 5b.3.
    C2. El efecto de 5b.3 era extremadamente sensible al diseno
        temporal: +19.2 pp -> +6.37 pp.
    C3. No existe evidencia de lift universal estable.
    C4. D3 no puede evaluar el SOW bajo las fronteras actuales.
    C5. Hay 17 configuraciones exploratorias interesantes, pero
        ninguna ha satisfecho la especificacion completa.
    C6. No hay base normativa para congelar parametros.
        WYCKOFF_X_ATR = None, WYCKOFF_Y_VOL = None deben permanecer.

---

## 17. Decision P1-P7

| Pregunta | Dictamen |
|---|---|
| P1 D3 | D - no modificar dentro de 5b.4; abrir nuevo protocolo |
| P2 heterogeneidad | D - tratarla como hallazgo a probar formalmente |
| P3 agregacion | B modificada - ALL descriptivo + bloques + heterogeneidad |
| P4 M | C - neutral |
| P5 17 candidatas | C - no congelar; resolver diseno/heterogeneidad primero |
| P6 multiplicidad | C con precision - D2 exploratorio; no confirmatorio |
| P7 bloque A | D - rediseñar bloques ex-ante; no eliminar post-hoc |

---

## 18. Estado de 5b.4

    Ejecucion tecnica:  APROBADA
    Resultados:         VALIDOS como diagnostico
    Seleccion:          FAIL (D1 240/240, D2 17/240, D3 0/240)
    Parametros:         NO CONGELAR
    SOW:                HIPOTESIS ABIERTA (ya no "confirmador universal")
    5b.X:               BLOQUEADA
    5c.4 / 5d:          BLOQUEADAS

---

## 19. Ruta recomendada

5b.4-bis / protocolo v3 con:

    1. 3 bloques temporales predefinidos
    2. fronteras fijadas antes de la nueva ejecucion
    3. suficiencia de muestra separada de calidad
    4. lift ALL como descriptivo
    5. lift por bloque como componente principal
    6. medida formal de heterogeneidad
    7. M tratado neutralmente
    8. D2 como filtro de desarrollo, no significacion confirmatoria
    9. UNA configuracion congelada tras desarrollo
    10. validacion final solo sobre datos futuros no inspeccionados

Bloque D actual: **NO llamarlo "holdout" en ningun documento futuro.**
Ya esta contaminado por inspeccion en 5b.3.

---

## 20. Dictamen final

    NO APROBAR EL CIERRE DE 5b.4
    NO CONGELAR NINGUNA DE LAS 17
    NO RELAJAR D3
    NO ELIMINAR A POST-HOC
    ABRIR NUEVO PROTOCOLO v3/5b.4-bis

5b.4: FAIL en seleccion, PASS en diagnostico metodologico.

La evidencia ha eliminado la falsa apariencia de "lift universal" de
5b.3. El resultado corregido es mas debil y temporalmente heterogeneo.
La siguiente fase debe investigar esa heterogeneidad, no esconderla
mediante una nueva regla de seleccion.

---

**Fin del dictamen 29.**