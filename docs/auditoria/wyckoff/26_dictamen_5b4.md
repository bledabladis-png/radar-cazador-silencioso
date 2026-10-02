# DICTAMEN EXTERNO - Fase 5b.4 (landmark temporal)

**Documento normativo. Fuente autoritativa de la decision sobre 5b.4.**
**Fecha:** 2026-10-02.
**Documento auditado:** 25_protocolo_fase_5b4.md (borrador).
**Resultado:** Core landmark APROBADO. Protocolo completo NO aprobado.
Requiere patch normativo antes de ejecutar.

---

## 0. Veredicto ejecutivo

La correccion principal (landmark temporal) es **correcta y debe
conservarse**:

    L = t0 + M
    confirmed = SOW en ventana hasta L
    baseline  = ausencia de SOW en ventana hasta L
    outcome (ambos) = desde L

Elimina el immortal time bias de 5b.3.

Pero **no se aprueba la ejecucion del borrador tal cual**, por 6
motivos. Hay un bloqueante fundamental: **no existe holdout historico
verdaderamente ciego**.

| Punto | Dictamen |
|---|---|
| Landmark L = t0 + M | APROBADO |
| Outcome comun desde L | APROBADO |
| Grid 240 | APROBADO |
| Bloques A-D | APROBADOS como desarrollo; NO como validacion ciega |
| Holdout D 2025-09/2026-10 | NO es ciego |
| Bootstrap solo por ticker | Insuficiente para inferencia fuerte |
| D1 `n_sow_raw` | Semanticamente debil |
| D4 | Necesita muestra minima por bloque |
| Ranking/tiebreak | Debe corregirse |
| Exito de 5b.4 | No puede incluir validacion ciega historica |

---

## 1. Landmark: correcto

    t0
     |
     +-- ventana de observacion SOW
     |
     L = t0 + M
     |
     +-- outcome futuro L + H

Confirmed = SOW observado hasta L.
Baseline = ausencia de SOW hasta L.
Ambos: outcome = f(L, L+H).

M > H20 es valido. No hay leakage.

---

## 2. Correccion obligatoria: M y el intervalo [t0, L]

Problema contractual: si L = t0 + M, el intervalo [t0, L] contiene
M+1 sesiones. Hay que congelar la semantica.

**Recomendacion obligatoria:**

    M = numero de sesiones posteriores a t0 hasta el landmark
    Exposicion SOW: [t0+1, L]

Ademas: **el SOW en t0 no debe ser confirmacion**. candidate_t y
SOW_t usan informacion del mismo cierre. Para la hipotesis "SOW anade
informacion mas alla de candidate", la definicion temporal limpia es:

    confirmed = existe SOW_t=1 en [t0+1, L]
    baseline  = no existe SOW_t=1 en [t0+1, L]

**Correccion obligatoria antes de ejecutar.**

---

## 3. Bloques temporales A-D

Fronteras aprobadas:

    A  2021-10-01 -> 2022-12-31
    B  2023-01-01 -> 2024-03-31
    C  2024-04-01 -> 2025-08-31
    D  2025-09-01 -> 2026-10-01

**No describir en el contrato como "mercado bajista 2022" etc.**
Caracterizacion economica ex-post. Usar "periodo temporal 1, 2, 3, 4".
En el informe posterior si se puede describir el regimen.

---

## 4. Bloque D NO es holdout ciego

**Bloqueante principal.** El periodo 2025-03-31 -> 2026-10-01 ya fue
inspeccionado en 5b.3. Se conocieron:

- todos los 240 parametros tenian lift positivo;
- rango de lifts;
- candidatos A/B/C;
- resultados concretos.

El periodo D (2025-09-01 -> 2026-10-01) esta contenido en esa ventana.
**No existe blindaje retrospectivo dividiendo una ventana ya observada.**

"No se vio por sesion, solo agregado" no conserva ceguera estadistica.

**Decision H2:**

    A - D como holdout ciego       RECHAZADO
    B - ventana futura             APROBADO
    C - A+B+C desarrollo, D valid. RECHAZADO como validacion ciega

---

## 5. Separar 5b.4 en dos funciones

### 5b.4 = desarrollo temporal

Usar A+B+C+D para:
- evaluar correccion temporal;
- estudiar estabilidad;
- seleccionar configuracion;
- diagnosticos;
- sensibilidad por periodo.

Sin llamarlo validacion ciega.

### 5b.X = validacion final (futura)

Una vez congelados N, M, X_ATR, Y_VOL, codigo y protocolo, usar
**ventana futura no inspeccionada** para validacion final.

---

## 6. Criterio de exito de 5b.4

Actual: "generaliza en bloque de validacion". **No aplicable**: no hay
bloque historico ciego.

### Exito de 5b.4

    pasa criterios de desarrollo
    + muestra estabilidad temporal A-D
    + queda congelada

### Validacion final (5b.X)

    configuracion congelada -> nuevo periodo temporal
    -> lower_CI(lift) > 0

Solo esa segunda fase usa "validacion ciega/final".

---

## 7. Bootstrap por ticker: base correcta, no suficiente

Cluster = ticker es conceptualmente correcto para dependencia
intraticker.

**Pero hay segunda dependencia:** todos los tickers expuestos
simultaneamente a shocks de mercado. Dependencia en dos dimensiones:
ticker y fecha/tiempo.

**Dictamen H4:**

- Se aprueba ticker-cluster bootstrap como primera capa.
- Se anade comprobacion de sensibilidad temporal/common-shock.
- El informe debe declarar: IC ticker-cluster + sensibilidad
  temporal, sin presentar el primero como garantia absoluta.

---

## 8. B = 2000, semilla fija

B=1000 deja ~25 replicaciones por cola para IC95%. Recomendado:

    B = 2000
    seed = 20261002
    percentile bootstrap 95%

**H3 APROBADO con modificacion.**

---

## 9. Definir estimando del bootstrap

Si un ticker tiene 2 episodios y otro 15: cada episodio pesa igual o
cada ticker pesa igual? Congelar.

**Estimando principal:** episode-weighted lift:

    Lift = P(Y=1 | confirmed) - P(Y=1 | baseline)
    cada episodio elegible pesa 1

**Bootstrap:** remuestrear **tickers completos**, manteniendo todos
sus episodios.

Publicar: n_tickers, n_episodes, n_confirmed, n_baseline.

---

## 10. D1 `n_sow_raw >= 100`: no como criterio duro

Problema semantico: n_sow_raw no demuestra que SOW anade informacion
sobre candidate. 5000 SOW con 10 relacionados con candidate no
sirven.

**Recomendacion:**

- Mantener n_sow_raw como diagnostico.
- Criterio duro: `n_confirmed >= 20 AND n_baseline >= 20`.
- Si se conserva D1 como guardrail: definirlo como "numero de SOW
  dentro de las ventanas de exposicion de los candidate episodes
  elegibles". No el total del universo.

---

## 11. D4: 3 de 4 bloques con lift > 0

Idea buena (evita que un periodo extraordinario determine el
resultado). Pero falta condicion:

Un bloque solo es **evaluable** cuando:

    n_confirmed >= 20 AND n_baseline >= 20

D4:

    al menos 3 de 4 bloques evaluables con lift_H20 > 0
    Y los 4 bloques deben tener datos suficientes

Evita que "3 de 3" pase cuando el cuarto no tiene muestra.

---

## 12. D3 `lower_CI(lift_H20) > 0`

Como criterio de desarrollo, aceptado. Pero cambiar interpretacion:
con 240 configuraciones hay multiple testing. Que alguna tenga
lower_CI > 0 no demuestra efecto confirmatorio al 95%.

    D3 = filtro de desarrollo
    Validacion futura: IC95% > 0 = confirmacion out-of-sample

---

## 13. Ranking: cambios

Actual:

    1. mayor mediana de lift
    2. menor desviacion estandar
    3. mayor X_ATR
    4. mayor Y_VOL
    5. N ascendente
    6. M ascendente

Cambios:

- **mediana**: aprobada.
- **desviacion estandar -> MAD** (median absolute deviation). Con 4
  bloques, SD es sensible al valor extremo.
- **tiebreak "mayor X_ATR > mayor Y_VOL"**: NO aprobado. Introduce
  preferencia economica no demostrada.
- **tiebreak lexicografico**: `(N, M, X_ATR, Y_VOL)` ascendente.
  Determinista, sin hipotesis oculta.

---

## 14. Sesiones de mercado, no dias de calendario

Explicito:

    t0 + M  y  L + H  = M y H sesiones de negociacion posteriores
    NO dias naturales

Importante por fines de semana, festivos, calendarios distintos.

---

## 15. Regla de censura

Un episodio es elegible solo si existe informacion completa hasta
L + H. Regla:

    si t0 + M + H no esta disponible: episodio excluido de ese H

Misma regla para confirmed y baseline. No permitir que un grupo tenga
mas censura que otro.

Para outcomes secundarios, declarar ventana exacta.

---

## 16. Semantica de confirmed cuando candidate termina

Actual: candidate episode = desde t0 mientras candidate=True.
Confirmed = SOW en [t0+1, L].

Permite: candidate=True en t0, t1, False en t2, SOW=True en t20 ->
confirma el episodio original. Dos conceptos:

    A. Confirmacion del episodio: SOW solo mientras candidate activo
    B. Confirmacion dentro de ventana desde inicio: SOW hasta L,
       aunque candidate haya desaparecido

**Para esta hipotesis: B.** Objetivo es saber si SOW en siguientes M
dias anade informacion. Escribir explicitamente.

---

## 17. `struct_deterioration_H20` y `support[L]`

Aprobados. Sin cambios. No introducir metrica de continuidad.

---

## 18. SLB / PCG

**No forman parte del criterio estadistico de seleccion.**

Guardrail funcional / regression test: "SLB no confirma con un break
marginal" es buena prueba de regresion. Pero no "una configuracion
solo es valida si pasa SLB".

---

## 19. Contrato corregido (resumen)

    candidate episode:
        candidate_t = True
        candidate_(t-1) = False

    t0 = fecha/sesion de entrada

    L = M sesiones despues de t0

    confirmed:
        existe SOW_t = 1 en [t0+1, L]

    baseline:
        no existe SOW_t = 1 en [t0+1, L]

    elegible:
        existe informacion completa hasta L+H

    primary outcome:
        struct[L+20] < struct[L]

    lift:
        P(primary | confirmed) - P(primary | baseline)

---

## 20. Dictamen H1-H8

| Punto | Dictamen |
|---|---|
| H1 Fronteras A-D | APROBADAS como estratos de desarrollo |
| H2 Holdout ciego D | RECHAZADO (contaminado por 5b.3) |
| H3 Bootstrap B/seed | B=2000, semilla fija |
| H4 Episodios por ticker | ticker cluster + sensibilidad temporal |
| H5 D4 3/4 | Concepto aprobado, exigir muestra por bloque |
| H6 Ranking | mediana + MAD; sin preferencias X/Y; tiebreak lexicografico |
| H7 D1-D4 | Aprobar tras correcciones |
| H8 Adicionales | obligatorios: t0+1, sesiones, censura, semantica episodio |

---

## 21. Secuencia autorizada

    5b.4 DEVELOPMENT
    A + B + C + D
        - 240 configuraciones
        - landmark correcto
        - D1/D2
        - lift
        - estabilidad 4 bloques
        - bootstrap
        - seleccion ex-ante
              |
              v
        PARAMETROS CONGELADOS
              |
              v
    5b.X FUTURE VALIDATION
    datos posteriores al freeze
        - mismo codigo
        - mismos parametros
        - ninguna recalibracion
        - IC95% lift > 0

---

## 22. Veredicto final

CORE LANDMARK: APROBADO.
PROTOCOLO 5b.4 COMPLETO: NO APROBADO.
Ejecucion del grid: BLOQUEADA hasta incorporar correcciones.
5c.4 y 5d: correctamente BLOQUEADAS.

**Fin del dictamen 26.**