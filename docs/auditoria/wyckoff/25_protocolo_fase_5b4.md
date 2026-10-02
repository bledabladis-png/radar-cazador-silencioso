# PROTOCOLO FASE 5b.4 - Landmark temporal (v2)

**Documento normativo. Sustituye al borrador v1.**
**Fecha:** 2026-10-02.
**Precedente:** 24_dictamen_5b3_D3.md + 26_dictamen_5b4.md.
**Estado:** APROBADO por dictamen 26. Ejecutable.

---

## 0. Contexto

5b.3 fallo formalmente (D3=0/240) y ademas tenia immortal time bias
en el anclaje del outcome. El dictamen 24 fijo las correcciones
metodologicas. El dictamen 26 anadio 10 correcciones normativas
obligatorias.

Este protocolo implementa ambas. **Es 5b.4 = fase de desarrollo**, no
de validacion ciega. La validacion final sera una fase posterior
(5b.X) sobre ventana futura no inspeccionada.

**Objetivo de 5b.4:** determinar si el SOW anade informacion sobre
deterioro futuro mas alla de candidate, con anclaje temporal simetrico
(landmark L = t0 + M), criterio de seleccion ex-ante, e inferencia
que respete la dependencia por ticker y por tiempo.

---

## 1. Definiciones

### 1.1. Convencion temporal

**Todas las referencias temporales son sesiones de negociacion, no
dias naturales.** `t0 + M` significa "M sesiones posteriores al
indice t0 en la serie del ticker". `L + H` idem.

### 1.2. Unidad de observacion

    candidate episode = (candidate_t=True) AND (candidate_{t-1}=False)
    por ticker

Donde `candidate` se calcula con la formula del contrato v1.8 §5.4:

    struct_score_t < STRUCT_DETERIORO (-0.10)
    AND t_norm_t > -T_NORM_STRONG (-0.30)
    AND struct_max en [t-W+1, t-1] > PREC_STRUCT_STRONG (0.30)

con W = PRECEDENT_WINDOW = 60 fijo.

### 1.3. Landmark

    L = t0 + M   (M sesiones posteriores a t0)

t0 = indice de entrada al episodio. M es un parametro del grid.

### 1.4. Grupo confirmed

    confirmed = existe SOW_t = 1 para algun t en [t0+1, L]

**Nota critica:** la ventana de exposicion empieza en **t0+1**, no en
t0. El SOW en t0 usaria informacion del mismo cierre que el candidate
y no seria una confirmacion posterior. Correccion obligatoria del
dictamen 26 seccion 2.

**Semantica del episodio (dictamen 26 seccion 16):** el SOW puede
aparecer aunque el label candidate ya haya desaparecido antes de L.
El objetivo es saber si la aparicion de un SOW en las siguientes M
sesiones anade informacion. Por tanto, el episodio no requiere que
candidate siga activo hasta L.

### 1.5. Grupo baseline

    baseline = NO existe SOW_t = 1 en [t0+1, L]

### 1.6. Elegibilidad (censura)

Un episodio es elegible para el horizonte H solo si existe informacion
completa hasta L + H:

    t0 + M + H no supera el ultimo indice disponible del ticker

Misma regla para confirmed y baseline. **Ningun grupo puede tener mas
censura que el otro.** Correccion obligatoria del dictamen 26 seccion 15.

### 1.7. Outcome primario

    struct_deterioration_H20 = struct[L+20] < struct[L]

Requiere struct[L] y struct[L+20] validos. Comparacion punto-a-punto.
NO implica deterioro continuo (dictamen 24 seccion 16).

### 1.8. Outcomes secundarios

    price_weakness_H{h} = close[L+h] < close[L] * (1 - 0.02)
    below_support_H{h}  = close[L+h] < support[L]
    lower_low_H{h}      = min(close[L+1:L+h]) < min(close[L-N:L])

donde `support[L] = rolling_min(Low, N).shift(1)` evaluado en L
(ex-ante). Horizontes H10, H20, H40.

### 1.9. Lift

    Lift = P(struct_deterioration_H20 | confirmed)
         - P(struct_deterioration_H20 | baseline)

**Estimando principal: episode-weighted.** Cada episodio elegible pesa
una vez. Se publican ademas n_tickers, n_episodes, n_confirmed,
n_baseline para no confundir dimensiones (dictamen 26 seccion 9).

---

## 2. Diferencia con 5b.3

    CONFIRMED (5b.3, sesgado):
        t0 -> busqueda de SOW -> t_sow -> outcome = struct[t_sow + H]

    BASELINE (5b.3, sesgado):
        t0 -> outcome = struct[t0 + H]

Los dos grupos se median en relojes distintos. Immortal time bias.

    CONFIRMED (5b.4, corregido):
        t0 -> ventana [t0+1, L] -> outcome = struct[L + H] < struct[L]

    BASELINE (5b.4, corregido):
        t0 -> ventana [t0+1, L] -> outcome = struct[L + H] < struct[L]

Mismo reloj. La pertenencia al grupo se decide con informacion
disponible hasta L. El outcome se mide despues de L.

**Consecuencia:** ya no se exige M <= H. M=30 con H=20 es valido,
porque L = t0+30 y outcome = struct[L+20], todo posterior a L.

---

## 3. Grid

Los mismos 4 parametros del contrato v1.8, ahora con landmark:

    N     in {20, 30, 40, 60}
    M     in {5, 10, 15, 20, 30}
    X_ATR in {0.25, 0.50, 0.75, 1.00}
    Y_VOL in {1.10, 1.20, 1.50}

Grid total: 240 combinaciones.

**Constantes de control (fijas, fuera del grid):**

    WYCKOFF_ATR_WINDOW = 20
    PRECEDENT_WINDOW   = 60
    WYCKOFF_T_NORM_K   = 0.25 (congelado 5b.2)

---

## 4. Universo y bloques

### 4.1. Dataset

- Fuente: `data/stock_prices.parquet`.
- Universo: 315 tickers con >= 200 obs Close no-NaN (SPCX fuera).
- Periodo: 2021-10-01 -> 2026-10-01.

### 4.2. Bloques temporales (desarrollo)

Fronteras aprobadas (dictamen 26 seccion 3). **Sin caracterizacion
economica ex-post en el contrato:**

    Bloque A - periodo temporal 1: 2021-10-01 -> 2022-12-31
    Bloque B - periodo temporal 2: 2023-01-01 -> 2024-03-31
    Bloque C - periodo temporal 3: 2024-04-01 -> 2025-08-31
    Bloque D - periodo temporal 4: 2025-09-01 -> 2026-10-01

Estos 4 bloques son **estratos de desarrollo**. NO son holdout ciego.

En el informe posterior se puede describir el regimen de mercado
predominante en cada bloque. En el protocolo, no.

### 4.3. Asignacion de episodios a bloques

Un episodio se asigna al bloque que contiene su **landmark L**, no su
t0. Razon: L es el momento desde el que se mide el outcome, y es el
punto donde el episodio queda "definido" como confirmed o baseline.

---

## 5. Holdout (DIFERIDO a 5b.X)

**Bloqueante del dictamen 26:** no existe holdout historico
verdaderamente ciego.

El periodo 2025-03-31 -> 2026-10-01 fue inspeccionado en 5b.3
(top-10 hold, X=1.00, etc.). El bloque D esta contenido en esa
ventana. No se puede "recuperar" su ceguera cambiando la frontera.

**5b.4 NO declara validacion ciega.**

La validacion final sera **5b.X (futura)** sobre datos posteriores al
freeze de parametros. Requiere:

- misma implementacion (mismo commit del modulo y del script);
- parametros congelados;
- sin recalibracion;
- ventana temporal no inspeccionada.

Criterio de validacion final: `lower_CI(lift_H20) > 0` sobre datos
nuevos.

Hasta que exista esa ventana (o hasta que el auditor designe un
bloque historico reservado no usado en el nuevo proceso), 5b.4 se
cierra con "parametros congelados a la espera de validacion".

---

## 6. Inferencia

### 6.1. Cluster bootstrap por ticker (capa 1)

No tratar cada episodio como observacion independiente. Un ticker con
15 episodios no son 15 unidades. Se remuestrean **tickers completos**,
manteniendo todos sus episodios.

    B = 2000
    seed = 20261002
    percentile bootstrap 95% (percentiles 2.5 / 97.5)

Criterio de desarrollo: `lower_CI(lift_H20) > 0`.

### 6.2. Sensibilidad temporal / common-shock (capa 2)

Los tickers estan expuestos simultaneamente a shocks comunes
(misma fecha, mismo regimen). El bootstrap por ticker **no captura**
esa dependencia.

**El informe debe declarar explicitamente:**

- IC ticker-cluster (capa 1).
- Analisis de sensibilidad temporal: re-ejecutar el calculo del lift
  en sub-bloques temporales y reportar dispersion de lifts entre
  bloques (ver seccion 7.2).
- Conclusion no puede presentar la capa 1 como garantia absoluta de
  independencia.

No se requiere modelo econometrico complejo. Si un bloque temporal
tiene lift mucho mas alto que los demas, esa heterogeneidad **es la
evidencia** de common-shock.

---

## 7. Criterios de seleccion ex-ante

### 7.1. Criterios duros

    D1: n_confirmed >= 20  Y  n_baseline >= 20
        (sustituye al n_sow_raw >= 100; el numero bruto de SOW del
        universo no mide adecuacion del experimento. Ver dictamen 26
        seccion 10.)
    D2: lower_CI(lift_H20) > 0 (bootstrap por ticker, B=2000)
        (criterio de DESARROLLO, no de confirmacion out-of-sample)
    D3: al menos 3 de 4 bloques evaluables con lift_H20 > 0,
        donde "evaluable" = (n_confirmed >= 20 AND n_baseline >= 20).
        Los 4 bloques deben tener datos suficientes.

### 7.2. Metricas de estabilidad (informe obligatorio)

Para cada combinacion:

    lift_H20_A, lift_H20_B, lift_H20_C, lift_H20_D
    mediana_lift = median(lifts de bloques evaluables)
    MAD_lift     = median(|lift_b - mediana_lift|)
    bloques_pos  = numero de bloques evaluables con lift > 0

### 7.3. D3_diagnostic (no puerta)

Se reporta pero no selecciona:

    pct_conf_struct_H20 (nivel absoluto del grupo confirmed)
    pct_base_struct_H20 (nivel absoluto del grupo baseline)
    lift_pp = (pct_conf - pct_base) * 100

Sustituye al antiguo D3=0.50. Ya no es puerta binaria de seleccion
(dictamen 24 seccion 10).

### 7.4. Ranking entre combinaciones que pasan D1-D3

    Paso 1: mayor mediana_lift (sobre bloques evaluables)
    Paso 2: menor MAD_lift (estabilidad robusta; sustituye a SD)
    Paso 3: tiebreak lexicografico (N, M, X_ATR, Y_VOL) ascendente.

**Sin preferencia por mayor X_ATR ni mayor Y_VOL.** Esa preferencia
introducia hipotesis economica no demostrada. Sustituida por orden
determinista (dictamen 26 seccion 13).

---

## 8. SLB / PCG

No forman parte del criterio estadistico de seleccion.

**Guardrail funcional / regression test:** "SLB no confirma con un
break marginal". Se reporta como diagnostico, no como condicion de
validez.

Una configuracion que pase D1-D4 pero confirme SLB no se invalida
por ese hecho. Solo se documenta.

---

## 9. Analisis prohibidos

- Optimizar por n_confirmed o por numero de DISTRIBUTION.
- Seleccionar mirando el top de bloques sin criterio ex-ante.
- Reutilizar el holdout 2025-03-31/2026-10-01 como validacion ciega.
- Relajar D3 retroactivamente.
- Modificar este protocolo tras ver resultados.
- Anclar confirmed y baseline en fechas distintas.
- Calcular outcome sin usar el mismo L para ambos grupos.
- Permitir SOW en t0 como confirmacion.
- Tratar episodios como observaciones independientes.
- Presentar IC ticker-cluster como garantia absoluta de
  independencia estadistica sin la sensibilidad temporal.
- Introducir preferencia artificial por X_ATR / Y_VOL en el tiebreak.

---

## 10. Salida esperada

- `outputs/audit/wyckoff_5b4_grid.csv` (240 filas).
- `outputs/audit/wyckoff_5b4_summary.json`.
- `outputs/audit/wyckoff_5b4_bootstrap.csv` (IC por combinacion).
- `outputs/audit/wyckoff_5b4_run.log`.
- `27_calibracion_5b4_resultados.md`.

---

## 11. Criterios de exito de 5b.4

5b.4 tiene exito si:

- Al menos una combinacion pasa D1-D3.
- La combinacion elegida pasa D1-D3 **y** tiene:
  - mediana_lift > 0;
  - MAD_lift peque\ño;
  - al menos 3 de 4 bloques con lift > 0.

**NO se exige "generalizacion en bloque de validacion"**. Eso pertenece
a 5b.X (futura, ciega).

Si ninguna combinacion pasa D1-D3: 5b.4 FAIL. Se abre 5b.5 con otra
hipotesis, o se abandona la linea SOW como confirmador.

### 11.1. Estado al cierre de 5b.4

Si 5b.4 tiene exito:

    N, M, X_ATR, Y_VOL  ->  CONGELADOS
    codigo              ->  CONGELADO en un commit
    contrato            ->  actualizado a v1.9 (parametros fijados)

Pero **NO se declaran "validados"**. Quedan a la espera de 5b.X.

---

## 12. Estado

    v1.8              CONTRATO VIGENTE
    5b.3              FAIL
    5b.4 (protocolo)  ESTE DOCUMENTO v2
    5b.4 (ejecucion)  PENDIENTE
    5b.X              PENDIENTE (validacion futura, requiere ventana
                      temporal nueva)
    Parametros SOW    None en config hasta cierre 5b.4
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

## 13. Trazabilidad

- Dictamen 24 (5b.3/D3): inmortal time bias detectado, landmark
  requerido.
- Dictamen 26 (5b.4): core landmark aprobado; 10 correcciones
  normativas obligatorias.
- Este protocolo v2 incorpora las 10.

---

**Fin del protocolo 5b.4 (v2).**

---

## 14. Nota de trazabilidad

Este protocolo v2 incorpora las 10 correcciones del dictamen 26.

**Correccion aplicada 2026-10-02 (post-commit 709a3d3):** se elimino
un cuarto criterio duro (D2: n_tickers >= 30) que el equipo habia
anadido por sobre-ingenieria y que el dictamen 26 no habia pedido.
Los criterios duros quedan en tres (D1, D2, D3), alineados
exactamente con los dictaminados. La numeracion se ajusto (D3->D2,
D4->D3).
