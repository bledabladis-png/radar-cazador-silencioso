# PROTOCOLO FASE 5b.3 - Calibracion SOW

**Documento normativo. Define el procedimiento de calibracion de los
parametros N, M, X_ATR, Y_VOL del SOW de confirmacion.**
**Fecha:** 2026-10-02.
**Precedente:** dictamen externo v1.6 5c.3 (O1-R aprobada).
**Estado:** APROBADO por dictamen externo (2026-10-02) con correcciones.

---

## 0. Contexto

v1.6 introdujo SOW binario para confirmar DISTRIBUTION (candidate + SOW).
El experimento 5c.3 revelo 3 hallazgos:

- H1: SLB es falso positivo (SOW con -0.06% de ruptura, 1.03x volumen).
- H2: M=10 es determinante (ACS con SOW fuerte a 13 sesiones queda fuera).
- H3: PCG pasa con SOW marginal (-1.43%, 1.05x).

El auditor aprobo introducir magnitud minima **normalizada por ATR**
(O1-R) y calibrar 4 parametros con grid ex ante.

---


---

## 0bis. Correcciones del dictamen externo (2026-10-02)

El dictamen externo aprobo el protocolo condicionalmente. Cuatro
correcciones obligatorias:

**C1. ATR ex-ante.**
La version inicial usaba `ATR_t = ATR(20)`, que incluye la barra t y
contamina el denominador. Nueva formula:

    ATR_baseline_t = ATR(window_atr).shift(1)

De esta forma `break_depth_t` se evalua completamente contra
informacion disponible antes de t.

**C2. Separar struct deterioration de price deterioration.**
No mezclar ambas en un OR. Reportar por separado:

    struct_deterioration_H{h}   -> struct[t+h] < struct[t]
    price_weakness_H{h}         -> close[t+h] < close[t] * (1 - 0.02)
    below_support_H{h}          -> close[t+h] < support[t]
    lower_low_H{h}              -> min(close[t+1:t+h]) < min(close[t-N:t])

Outcome primario: `struct_deterioration_H20`.

**C3. Baseline candidate sin SOW.**
La calidad de SOW no se mide en absoluto. Se mide como **incremento
respecto a un candidate sin SOW**. Anadir:

    P20_confirmed = P(struct_deterioration_H20 | candidate + SOW)
    P20_baseline  = P(struct_deterioration_H20 | candidate sin SOW)
    incremental_lift_H20 = P20_confirmed - P20_baseline

**Obligatorio reportar `incremental_lift_H20` y `incremental_lift_H40`.**
Si SOW no aporta lift positivo, no es buen confirmador aunque el
porcentaje absoluto sea alto.

**C4. Unidad de observacion y solapamiento de SOW.**
Un SOW aislado no es una observacion independiente si otros SOW estan
dentro de la ventana N. Reportar:

    n_sow_raw       (cuenta bruta de SOW=1)
    n_sow_unique    (agrupados por ticker; SOW a menos de N sesiones
                     del anterior cuentan como 1)

Definicion ex-ante del agrupamiento: `same ticker AND |t_i - t_{i-1}| < N`.

---

## 0ter. Ajustes menores del dictamen

- **D1 (n_sow_total >= 100):** mantener como guardrail operativo, NO
  como benchmark empirico. Reportar ademas `sow_rate_per_100_ticker_years`.
- **D2 (n_confirmed >= 20):** idem, reportar `confirmed_per_100_ticker_years`.
- **D3 (>= 50%):** aceptado como umbral provisional pero **debe
  combinarse con incremental_lift**. El criterio real es "lift > 0"
  (o el umbral que se decida en el informe final), no solo D3.
- **D4 (variacion <= 30%):** reportar `absolute_delta` y
  `relative_delta`. Definicion: `relative_delta = (max - min) / mean`
  sobre el vecindario. Si `mean` < 0.01, no aplicar el criterio.
- **`WYCKOFF_ATR_WINDOW`:** mantener fijo = 20. Si ATR20 demuestra
  inadecuado tras 5b.3, abrir 5b.4.

---

## 0quater. Secuencia de ejecucion aprobada

    1. Congelar protocolo 5b.3 corregido (este doc).
    2. Implementar v1.7 (SOW con ATR.shift(1) + X_ATR + Y_VOL).
    3. Tests unitarios v1.7 (I30-I33):
       - ATR ex-ante.
       - break_depth con X_ATR.
       - SOW sin ruptura -> no dispara.
       - no-look-ahead.
    4. Ejecutar prueba pequena/sintetica.
    5. Revisar salida estructural.
    6. Ejecutar grid 240.
    7. Seleccionar candidato segun criterios.
    8. Holdout 30%.

**Regla critica del dictamen:**

> 5b.3 no debe concluir que SOW es buen confirmador simplemente porque
> los casos `candidate + SOW` continuen deteriorandose. Debe demostrar
> que `candidate + SOW` contiene **mas informacion sobre el deterioro
> futuro que `candidate` sin SOW**.

---

## 1. Formula v1.7

    support_t          = rolling_min(Low, N).shift(1)
    ATR_baseline_t     = ATR(window_atr).shift(1)      <-- ex-ante
    volume_baseline_t  = rolling_mean(Volume, N).shift(1)

    break_depth_t      = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t     = Volume_t / volume_baseline_t

    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

`WYCKOFF_ATR_WINDOW = 20` ya existe en config, se reutiliza.

**Semantica:** cuantos ATR ha penetrado el precio por debajo del
soporte, mas esfuerzo minimo de volumen.

---

## 2. Parametros a calibrar

| Parametro | Funcion | Rango propuesto |
|---|---|---|
| N | Ventana para soporte y baseline de volumen | {20, 30, 40, 60} |
| M | Max edad del SOW para confirmar | {5, 10, 15, 20, 30} |
| X_ATR | Penetracion minima (unidades de ATR) | {0.25, 0.50, 0.75, 1.00} |
| Y_VOL | Ratio minimo de volumen | {1.10, 1.20, 1.50} |

Grid total: 4 × 5 × 4 × 3 = **240 combinaciones**.

---

## 3. Dataset

- Fuente: `data/stock_prices.parquet`.
- Universo: todos los tickers con >= 200 obs validas.
- Periodo: 2021-10-01 hasta ultima sesion cerrada.
- Split: 70% calibracion / 30% holdout temporal (mismo criterio que 5b.2).

**Aviso survivorship bias:** universo actual, no historico.

---

## 4. Metricas

Para cada combinacion (N, M, X_ATR, Y_VOL):

### 4.1. Metricas de actividad

- `n_sow_total`: numero total de SOW detectados.
- `n_candidate_events`: cuantos estados candidate.
- `n_confirmed`: candidate + SOW reciente.
- `tasa_candidate_confirmed`: n_confirmed / n_candidate_events.

### 4.2. Metricas de estabilidad

- `margen_pct_confirmed`: sensibilidad a variaciones de M.
- `freq_sow_por_anio`: SOW/ano.
- `pct_candidate_sin_sow`: candidate que no confirman.

### 4.3. Metricas de calidad confirmatoria (clave)

Para cada SOW confirmado, medir comportamiento posterior en horizontes
fijos H ∈ {10, 20, 40} sesiones:

- `pct_deterioro_continuo`: % casos donde struct sigue deteriorandose.
- `pct_ma50_menor`: % casos donde MA50 sigue cayendo respecto a MA200.
- `pct_precio_bajo_soporte`: % casos donde precio se mantiene bajo
  el soporte que rompio.
- `pct_lower_lows`: % casos donde se produce un nuevo minimo.

**Definicion "deterioro continuo":**
- struct_t+H < struct_t (caida adicional).
- O bien: precio_t+H < precio_t * (1 - 0.02).

### 4.4. Estabilidad ante variaciones de N/M

Para la combinacion elegida, comprobar el vecindario:

    N ∈ {N-10, N, N+10}
    M ∈ {M-5, M, M+5}

Medir si `tasa_candidate_confirmed` cambia > 30% en el vecindario.
Si cambia mucho, la combinacion es fragil.

---

## 5. Criterios de seleccion ex ante

### 5.1. Criterios duros

    D1. n_sow_total >= 100 (frecuencia razonable en universo).
    D2. n_confirmed >= 20 (suficientes para medir calidad).
    D3. pct_deterioro_continuo(H=20) >= 50% (calidad confirmatoria).
    D4. estabilidad: variacion <= 30% en el vecindario de N/M.

### 5.2. Ranking entre aceptados

**No se elige por numero de DISTRIBUTION.** Se elige por:

1. Mayor `pct_deterioro_continuo(H=20)`.
2. Desempate: menor X_ATR (mas conservador respecto a magnitud).
3. Desempate final: valores mas estandar (N=30, M=10, X_ATR=0.50, Y_VOL=1.20).

---

## 6. Validacion holdout

Una vez elegida la combinacion sobre 70%:

1. Aplicar sobre el 30% final.
2. Medir D1-D4 sobre holdout.
3. D1-D4 obligatorios en holdout.
4. Si no pasan: combinacion no generaliza.

---

## 7. Analisis prohibidos

- Optimizar por `n_confirmed` (frecuencia artificial).
- Optimizar por numero de DISTRIBUTION total.
- Elegir parametros para que los 5 casos del dictamen 5c.3 queden
  "bien".
- Modificar criterios ex ante tras ver resultados.

---

## 8. Salida esperada

- `docs/auditoria/wyckoff/21_calibracion_5b3_resultados.md` con:
  - Tabla de grid completa (240 filas).
  - Combinacion elegida.
  - Resultados de D1-D4 en calibracion y validacion.
  - Analisis de estabilidad.
  - Comportamiento posterior al SOW.

---

## 9. Parametros ATR

`WYCKOFF_ATR_WINDOW` ya existe (=20 en config). Se mantiene fijo en
5b.3. Si en el futuro se decide calibrar, sera 5b.4.

---

## 10. Criterios de exito

5b.3 tiene exito si:

- Al menos una combinacion cumple D1-D4.
- La combinacion elegida tambien cumple D1-D4 en holdout.
- pct_deterioro_continuo(H=20) >= 50%.
- El SOW ya no captura ruido (SLB con -0.06% no deberia ser confirmado).

**Sobre SLB:** con la nueva definicion ATR-normalizada, SLB (breach
-0.06%, ATR~2-3%) da `break_depth ≈ 0.03`. Con X_ATR >= 0.25 (minimo
de la grid), SLB no confirmaria. Esperado.

---

## 11. Estado del proyecto

    v1.6                    IMPLEMENTADA (arquitectura candidate/confirmed)
    5c.3                    DIAGNOSTICO CERRADO
    v1.7                    PROPUESTA (SOW con magnitud ATR)
    5b.3                    PROTOCOLO (este doc)
    v1.7.x                  PENDIENTE
    5c.4                    PENDIENTE
    5d                      BLOQUEADA
    Legacy                  INTACTO

---

Fin del protocolo 5b.3.
