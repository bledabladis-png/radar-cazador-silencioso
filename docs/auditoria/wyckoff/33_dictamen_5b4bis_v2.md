# DICTAMEN EXTERNO - 5b.4-bis / Expediente 32

**Documento normativo. Fuente autoritativa de la decision sobre el
freeze de la candidata.**
**Fecha:** 2026-10-02.
**Documento auditado:** 32_expediente_5b4bis_freeze.md + 31 + protocolo
v3 + commit 848ee23.
**Resultado:** NO se autoriza freeze productivo. SI se autoriza freeze
especificativo para validacion futura, con condiciones.

---

## 0. Veredicto

NO AUTORIZO EL FREEZE PRODUCTIVO.

SI AUTORIZO UN FREEZE ESPECIFICATIVO PARA VALIDACION FUTURA, bajo
condiciones estrictas.

Tres problemas impiden considerar demostrada la robustez de la
candidata:

1. La candidata ha sido seleccionada tras buscar entre 240
   configuraciones sobre el mismo historico.
2. La supuesta homogeneidad +++ se basa solo en el signo del lift; no
   se aportan IC por bloque.
3. La candidata queda congelada por una seleccion optimizada sobre
   datos historicos, pero todavia no existe validacion independiente
   posterior al freeze.

El landmark sigue siendo una correccion valida para el problema
temporal de 5b.3. La literatura advierte que los resultados de
landmark dependen de su momento y no es una solucion universal a
todos los sesgos temporales.

---

## 1. Lo aprobado y lo no aprobado

### Aprobado

    Landmark t0 -> M -> L -> L+H
    Exposicion [t0+1, L]
    Mismo L para confirmed/baseline
    Bootstrap por ticker
    B=2000 + seed fija
    Grid de 240
    Rediseno a P1/P2/P3
    D1 / D2 / D3 v3 (ejecutados conforme al protocolo)
    17 candidatas como resultado valido de desarrollo
    Candidata #1 como candidata seleccionada por regla definida

### No aprobado

    Activar parametros en settings.py
    Retirar fail-closed
    Considerar 5b.4-bis validacion final
    Considerar demostrado un efecto estable
    Pasar directamente a produccion
    Considerar +++ prueba suficiente de homogeneidad
    Considerar +0.074 un efecto confirmatorio

---

## 2. Interpretacion correcta del resultado

La candidata N=60, M=30, X=0.25, Y=1.10 presenta:

    Lift_ALL = +7.40 pp
    IC95% = [+2.34, +12.42] pp
    P1 = +9.3 pp
    P2 = +5.0 pp
    P3 = +10.6 pp

Evidencia historica favorable al criterio definido. NO equivale a
"queda probado que SOW es confirmador robusto".

La candidata fue seleccionada tras explorar 240 configuraciones. La
inferencia sobre la ganadora esta condicionada por ese proceso de
seleccion (selective inference).

Esto no invalida el desarrollo. Si impide usar el IC95% historico
como prueba confirmatoria independiente.

---

## 3. Sobre el rediseno P1/P2/P3

El cambio A/B/C/D -> P1/P2/P3 ha resuelto el problema muestral del
bloque A (max 12 confirmed -> P1=32). Aparecen configuraciones con 3
bloques evaluables y 3 positivos.

NO es una relajacion automatica de D3. Es un cambio del diseno de
estratificacion.

### Precision sobre "ex-ante"

El texto de 30_protocolo_fase_5b4bis_v3.md dice "fronteras fijadas
ex-ante". No es completamente exacto: las fronteras se eligieron
despues de observar en 5b.4 que el primer bloque tenia insuficiencia
muestral y usando la distribucion de candidate starts.

No es outcome leakage (no se usaron los lifts para elegir fronteras).
Pero SI es una decision informada por resultados del experimento
anterior.

**Formulacion correcta que debe usarse:**

> "Fronteras fijadas antes de la ejecucion de 5b.4-bis, informadas
> por diagnosticos de disponibilidad muestral de la fase anterior y
> no por el signo del outcome."

---

## 4. D3 v3 aprobado como puerta de desarrollo

La regla ceil(2/3) bloques evaluables con lift > 0 es razonable para
desarrollo. Es alcanzable (69/240).

Limitacion: para la candidata #1, P1 n=32, P2 n=304, P3 n=293.
El +++ solo dice "lift > 0 en los tres". No dice que los tres efectos
esten estimados con similar precision.

**+++ demuestra consistencia de signo, no homogeneidad estadistica.**

---

## 5. Falta principal: IC por bloque

El expediente afirma delta_hetero=5.56 pp -> heterogeneidad moderada.
Eso no esta demostrado sin IC por bloque.

**Decision obligatoria:** antes del freeze documental definitivo debe
existir:

    P1: lift + lower_CI + upper_CI
    P2: lift + lower_CI + upper_CI
    P3: lift + lower_CI + upper_CI

con el mismo principio de bootstrap por ticker.

---

## 6. delta_hetero no es test de heterogeneidad

max-min es descriptivo, no test. La diferencia entre P2 y P3 puede
ser o no material respecto a la incertidumbre.

Reportar: delta_hetero + IC individuales + bootstrap de una
estadistica de heterogeneidad.

No confundir "bajo rango observado" con "ausencia de heterogeneidad
estadisticamente relevante".

---

## 7. "Sin ambiguedad" debe matizarse

Procedimentalmente #1 esta correctamente determinado por la regla.
Pero "gana sin ambiguedad" es demasiado fuerte: existe un conjunto
cercano (#1, #4, #5, #7, #8) con estructura similar.

Hay un ganador determinista del ranking, pero no se ha demostrado
separacion estadistica clara entre la candidata y sus vecinas.

---

## 8. Proximidad de candidatas: buena senal

No vemos una candidata aislada +7.4% y el resto 0%. Vemos un grupo
~4.7% -> ~7.4% con estructuras +++. Eso indica que el resultado no
depende de una configuracion aislada.

Documentar como "robustez local del conjunto de parametros", NO como
"robustez fuera de muestra".

---

## 9. Candidata en los limites del grid

N=60 (max), M=30 (max), X_ATR=0.25 (min), Y_VOL=1.10 (min).

No es defecto, pero no conocemos comportamiento fuera del grid.

NO afirmar "0.25 y 1.10 son los valores optimos". Solo: "0.25/1.10
son los valores seleccionados dentro del dominio previamente
autorizado".

No ampliar el grid ahora porque los limites hayan ganado: seria otra
busqueda condicionada por resultados.

---

## 10. D2 17/240: no es evidencia confirmatoria

7.08% es exploratorio. No aplicar Bonferroni retrospectivamente.

Tampoco afirmar "por tanto no hay data snooping".

La solucion correcta:

    desarrollo historico -> UNA configuracion congelada -> datos
    futuros no vistos

---

## 11. Bootstrap ticker-cluster

Apropiado para dependencia intraticker. NO escribir "bootstrap
elimina la dependencia". Formulacion correcta:

> "controla la dependencia entre episodios del mismo ticker."

---

## 12. El SOW no es "confirmador universal"

Existe evidencia historica consistente con capacidad incremental del
SOW bajo el diseno landmark. No: "SOW es confirmador universal".

La palabra universal debe desaparecer hasta que exista evidencia
fuera de la muestra de desarrollo.

---

## 13. price_weakness: correccion narrativa

El expediente dice "consistente con la definicion (SOW es senal
estructural)". No aprobado asi.

En P1:

    struct_deterioration: +9.3 pp
    price_weakness:       -19.7 pp

Eso no demuestra que SOW sea "estructural". Lo que demuestra es:
bajo esa definicion concreta de outcome, el SOW presenta lift
positivo para struct_deterioration y signo contrario para
price_weakness.

Debe quedarse como dato. No usar la definicion de la senal para
convertir un resultado contradictorio en coherencia semantica.

---

## 14. No abrir 5b.5

Hay senal suficientemente consistente para justificar validacion
independiente. Cambiar el estado:

    5b.4-bis = PASS desarrollo / NO confirmado

---

## 15. P1 - Freeze

SI, pero unicamente freeze de validacion; NO freeze productivo.

No autorizar:

    WYCKOFF_SOW_WINDOW_N = 60
    WYCKOFF_SOW_MAX_AGE_M = 30
    WYCKOFF_SOW_X_ATR = 0.25
    WYCKOFF_SOW_Y_VOL = 1.10

en settings.py productivo todavia.

SI autorizar: congelar N=60, M=30, X_ATR=0.25, Y_VOL=1.10 como
especificacion normativa de validacion futura.

    FREEZE DE VALIDACION != ACTIVACION PRODUCTIVA

---

## 16. P2 - Alcance

Variante de (b) con condicion:

    Parametros congelados documentalmente
    settings.py sigue None
    detect_sow permanece fail-closed

No quitar fail-closed. Mientras no pase 5b.X, no debe ser posible
que una llamada productiva descubra accidentalmente los valores
congelados.

---

## 17. P3 - v1.9

SI, crear v1.9. NO como contrato productivo final. Titulo:

> v1.9 - SOW candidate frozen for out-of-sample validation

STATUS = FROZEN_FOR_VALIDATION
NOT_PRODUCTION

Incluir explicitamente:

    validacion 5b.X pendiente
    produccion no activada
    fail-closed mantenido

---

## 18. P4 - 5b.X

Sin regla de "6-12 meses". Condicion por disponibilidad de sesiones,
no por tiempo elegido.

Cutoff del freeze: 2026-10-01 (ultima fecha del historico actual).
Todo dato posterior pertenece potencialmente a la validacion.

NO volver a seleccionar N/M/X/Y cuando lleguen los nuevos datos.

---

## 19. P5 - Analisis adicional antes del freeze

SI, pero no otra calibracion. Un unico expediente de QA con:

### A. IC por bloque

Obligatorio: P1 lower/upper, P2 lower/upper, P3 lower/upper.

### B. Distribucion de episodios por ticker

    n_tickers_confirmed
    n_tickers_baseline
    episodios/ticker
    max episodios por ticker

para asegurar que +7.4% no esta excesivamente concentrado.

### C. Robustez local

Comprobar descriptivamente las vecinas ya existentes, sin ampliar
grid ni seleccionar.

### D. Integridad de artefactos

Verificar grid, bootstrap, hetero, summary, log y reproducibilidad
desde commit 848ee23.

---

## 20. 5b.X debe ser UNA sola configuracion

    commit 848ee23 -> candidata #1 congelada -> cutoff 2026-10-01
    -> N=60/M=30/X=0.25/Y=1.10 -> NUEVOS DATOS -> resultado unico

No: nuevos datos -> 17 configuraciones -> elegir la mejor.

---

## 21. Landmark: documentar limitaciones

La literatura reciente enfatiza limitaciones del landmark (atenuacion
del efecto, dependencia del momento). No rehacer 5b.4. Documentar:

> "El diseno landmark corrige el defecto de anclaje asimetrico
> identificado en 5b.3 y evita el immortal-time problem concreto de
> esa implementacion; sus restantes limitaciones se mantienen fuera
> del alcance de esta fase."

---

## 22. Decision final P1-P5

| Pregunta | Decision |
|---|---|
| P1 Freeze | SI, freeze de validacion; no activacion productiva |
| P2 Alcance | Documental; settings=None; fail-closed intacto |
| P3 v1.9 | SI, FROZEN_FOR_VALIDATION, no produccion |
| P4 5b.X | Ventana futura posterior al cutoff; sin regla de 6-12 meses |
| P5 QA previo | SI: IC por bloque + composicion por ticker + QA |

---

## 23. Estado normativo recomendado

    v1.8  CONTRATO PRODUCTIVO VIGENTE
    v1.9  CANDIDATA SOW CONGELADA PARA VALIDACION
          N=60 M=30 X=.25 Y=1.10
          NO produccion / settings=None / fail-closed activo /
          pendiente 5b.X

    5b.3        FAIL
    5b.4        FAIL seleccion / diagnostico valido
    5b.4-bis    PASS desarrollo
    5b.X        BLOQUEADA hasta datos futuros
    5c.4        BLOQUEADA
    5d          BLOQUEADA
    Legacy      INTACTO

---

## 24. Dictamen definitivo

NO AUTORIZO:
- activar SOW en settings.py
- retirar fail-closed
- migrar consumidores
- abrir 5c.4
- considerar demostrado el efecto
- considerar la candidata validada

SI AUTORIZO:
- reconocer 5b.4-bis como PASS de desarrollo
- congelar N=60/M=30/.25/1.10 como candidata unica
- documentarla en v1.9
- mantener config productiva en None
- mantener fail-closed
- reservar datos posteriores a 2026-10-01 para 5b.X

**Condicion previa al freeze documental definitivo:** anadir P1/P2/P3
lower/upper CI y revisar concentracion por ticker.

---

## Conclusion

La secuencia ha producido un resultado metodologicamente util:

    5b3 (+19.2 pp) -> correccion temporal -> 5b4 (+6.4 pp,
    heterogeneidad A/B) -> rediseno de bloques -> 5b4-bis (+7.4 pp,
    +++ en P1/P2/P3)

Evidencia de desarrollo mas solida que 5b3. Pero el paso logico
siguiente no es activar el indicador. Es congelar la hipotesis,
preservar el historico como desarrollo, y esperar observacion
temporal nueva para comprobar si +7.4 pp y +++ sobreviven fuera de
la muestra de seleccion.

---

**Fin del dictamen 33.**
