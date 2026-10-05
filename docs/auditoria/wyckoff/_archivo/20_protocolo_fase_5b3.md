# PROTOCOLO FASE 5b.3 - Calibracion SOW (v3, corregido)

**Documento normativo. Define el procedimiento de calibracion de N, M,
X_ATR, Y_VOL del SOW de confirmacion.**

**Fecha:** 2026-10-02.
**Estado:** APROBADO por dictamen externo (2026-10-02) tras expediente
`21_expediente_5b3_aclaraciones.md`. Esta version incorpora las
correcciones C1-C4, las decisiones P1-P7 y las precisiones adicionales
sobre unidad de observacion, anclaje temporal, split y holdout.

---

## 0. Contexto

v1.6 introdujo SOW binario para confirmar DISTRIBUTION (candidate + SOW).
El experimento 5c.3 revelo 3 hallazgos:

- H1: SLB es falso positivo (SOW con -0.06% de ruptura, 1.03x volumen).
- H2: M=10 es determinante (ACS con SOW fuerte a 13 sesiones queda fuera).
- H3: PCG pasa con SOW marginal (-1.43%, 1.05x).

El auditor aprobo introducir magnitud minima **normalizada por ATR**
(O1-R) y calibrar 4 parametros con grid ex ante.

El expediente 5b.3 (posterior) detecto 8 discrepancias entre protocolo,
contrato, codigo y config. El dictamen resolvio P1-P7 y añadio
precisiones metodologicas que este documento incorpora.

**Decision de gobernanza (P1):** v1.7 pasa a historico. El contrato
vigente es **v1.8** (`01_contrato_semantico_v1_8.md`), que supersede
v1.7 por inconsistencias documentales.

---

## 1. Correcciones obligatorias del dictamen externo

### C1. ATR ex-ante

La version inicial usaba `ATR_t = ATR(20)`, que incluye la barra t y
contamina el denominador. Formula vigente:

    ATR_baseline_t = ATR(WYCKOFF_ATR_WINDOW).shift(1)

`break_depth_t` se evalua completamente contra informacion disponible
antes de t. `WYCKOFF_ATR_WINDOW = 20` fijo, NO en grid.

### C2. Separar deterioro estructural de deterioro de precio

**Retirado el OR de la version previa.** Cuatro metricas separadas:

    struct_deterioration_H{h}   -> struct[t+h] < struct[t]
                                   (requiere struct[t] y struct[t+h] validos)
    price_weakness_H{h}         -> close[t+h] < close[t] * (1 - 0.02)
    below_support_H{h}          -> close[t+h] < support[t]
    lower_low_H{h}              -> min(close[t+1:t+h]) < min(close[t-N:t])

**Outcome primario:** `struct_deterioration_H20`.
**Secundarios:** H10, H40 de cada una de las cuatro.
El nombre antiguo `pct_deterioro_continuo` desaparece del protocolo.

### C3. Baseline candidate sin SOW (obligatorio)

La calidad de SOW no se mide en absoluto. Se mide como **incremento
respecto a candidate sin SOW**:

    P20_confirmed = P(struct_deterioration_H20 | candidate + SOW valido)
    P20_baseline  = P(struct_deterioration_H20 | candidate sin SOW)
    incremental_lift_H20 = P20_confirmed - P20_baseline

Idem `incremental_lift_H40`. **Obligatorio reportar ambos.**

Si SOW no aporta lift positivo, no es buen confirmador aunque el
porcentaje absoluto sea alto.

### C4. Unidad de observacion: candidate episode

Un candidate persistente en 15 sesiones NO son 15 observaciones. Es 1.

**Definicion de candidate_event:**

    candidate_t = True
    AND
    candidate_{t-1} = False

por ticker. Es decir: **entrada en un episodio de DISTRIBUTION_CANDIDATE**.

Reportar:
- `n_candidate_events` (numero de episodios).
- `n_sow_raw` (cuenta bruta de SOW=1, por trazabilidad).
- `n_sow_unique` (SOW agrupados por ticker; SOW a menos de N sesiones
  del anterior cuentan como 1). Definicion ex-ante:
  `same ticker AND |t_i - t_{i-1}| < N`.
- `n_confirmed` (episodios con al menos 1 SOW valido).
- `tasa_candidate_confirmed = n_confirmed / n_candidate_events`.

---

## 2. Anclaje temporal del outcome (precision adicional del dictamen)

Si el outcome es `struct[t+h] < struct[t]`, **`t` es la fecha del primer
SOW valido que confirma el candidate_event**. No es la fecha del
candidate ni la del ultimo SOW.

    candidate_event (entrada en episodio)
        ↓
    primer SOW valido dentro de la ventana M
        ↓
    t = fecha de ese SOW
        ↓
    H10 / H20 / H40 se miden desde t

**Regla multi-SOW:** si un episodio tiene SOW en dias 3, 6, 8, el
`confirmation_time = dia 3`. Los H10/H20/H40 se miden desde dia 3.
Los SOW posteriores **no** crean observaciones adicionales
(anti-pseudo-replicacion).

---

## 3. Boundary del split (precision adicional del dictamen)

### Split

**Una unica fecha cronologica global** para todo el universo.
Todos los tickers usan el mismo corte. NO se calcula 70% por ticker
individual (produciria fechas distintas y rompe independencia temporal).

### Eventos cercanos al limite

Para calibracion:

    si t + 20 > fecha_fin_calibracion  -> excluir para H20
    si t + 40 > fecha_fin_calibracion  -> excluir para H40
    si t + 10 > fecha_fin_calibracion  -> excluir para H10

Lo mismo para holdout respecto a su fecha de fin. **No se traslada
informacion del futuro al bloque actual.**

---
## 4. Formula v1.8 y parametros

### 4.1. Formula SOW

    support_t          = rolling_min(Low, N).shift(1)
    ATR_baseline_t     = ATR(WYCKOFF_ATR_WINDOW).shift(1)
    volume_baseline_t  = rolling_mean(Volume, N).shift(1)

    break_depth_t      = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t     = Volume_t / volume_baseline_t

    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

### 4.2. Parametros a calibrar

| Parametro | Funcion | Rango |
|---|---|---|
| N | Ventana para soporte y baseline de volumen | {20, 30, 40, 60} |
| M | Max edad del SOW para confirmar | {5, 10, 15, 20, 30} |
| X_ATR | Penetracion minima (unidades de ATR) | {0.25, 0.50, 0.75, 1.00} |
| Y_VOL | Ratio minimo de volumen | {1.10, 1.20, 1.50} |

Grid total: 4 x 5 x 4 x 3 = **240 combinaciones**.

### 4.3. Constantes de control (NO en grid)

    WYCKOFF_ATR_WINDOW = 20    fijo
    PRECEDENT_WINDOW   = 60    fijo (PROPUESTO, no calibrado; futuro 5b.4)
    WYCKOFF_T_NORM_K   = 0.25  congelado (5b.2)

Razon: 5b.3 aisla el efecto de N/M/X_ATR/Y_VOL. Añadir W o ATR_WINDOW
al grid convierte el experimento en una busqueda multi-dimensional
no controlada.

### 4.4. Fail-closed en codigo

`detect_sow` **exige** `window`, `x_atr`, `y_vol` explicitos. Si falta
alguno, lanza `ValueError`. `config/settings.py` mantiene los cuatro
parametros SOW en `None` hasta que 5b.3 cierre. Invariantes I34-I36
(contrato v1.8 §9).

---

## 5. Dataset y metricas

### 5.1. Dataset

- Fuente: `data/stock_prices.parquet`.
- Universo: todos los tickers con >= 200 obs validas.
- Periodo: 2021-10-01 hasta ultima sesion cerrada.
- Split: 70% calibracion / 30% holdout temporal, con **una unica
  fecha de corte global** (ver §3).

**Aviso survivorship bias:** universo actual, no historico.

### 5.2. Metricas de actividad

- `n_candidate_events`: numero de entradas en episodios (candidate_t=True
  AND candidate_{t-1}=False, por ticker).
- `n_confirmed`: episodios con >= 1 SOW valido dentro de la ventana M.
- `n_sow_raw`: SOW=1 brutos, por trazabilidad.
- `n_sow_unique`: SOW agrupados por ticker (|t_i - t_{i-1}| < N cuentan
  como 1).
- `tasa_candidate_confirmed = n_confirmed / n_candidate_events`.
  Si `n_candidate_events == 0` -> `N/A` (no cero).

### 5.3. Metricas de estabilidad

- `neighbour_confirmation_delta_pp`: maximo cambio absoluto en **puntos
  porcentuales** de `tasa_candidate_confirmed` frente a vecinos
  inmediatos de la grid, manteniendo constantes los otros dos
  parametros. Sustituye a `margen_pct_confirmed` (retirado por
  ambiguedad, dictamen P4).
- `freq_sow_por_anio`: SOW/anio (global).
- `sow_rate_per_100_ticker_years`: SOW por 100 ticker-anios.
- `confirmed_per_100_ticker_years`.
- `pct_candidate_sin_sow`: candidate sin confirmar.

### 5.4. Metricas de calidad confirmatoria (clave)

Para cada configuration (N, M, X_ATR, Y_VOL), sobre cada candidate
episode confirmado, anclar `t = primer SOW valido del episodio` y
medir en horizontes H in {10, 20, 40}:

- `struct_deterioration_H{h}`: `struct[t+h] < struct[t]`, con
  `struct[t]` y `struct[t+h]` validos. **Outcome primario: H20.**
- `price_weakness_H{h}`: `close[t+h] < close[t] * (1 - 0.02)`.
- `below_support_H{h}`: `close[t+h] < support[t]`.
- `lower_low_H{h}`: `min(close[t+1:t+h]) < min(close[t-N:t])`.

**Metricas de lift obligatorias (dictamen C3):**

    P20_confirmed = mean(struct_deterioration_H20 | episodios confirmados)
    P20_baseline  = mean(struct_deterioration_H20 | episodios NO confirmados)
    incremental_lift_H20 = P20_confirmed - P20_baseline

Idem `incremental_lift_H40`. **Obligatorias en el informe.**

---

## 6. Criterios de seleccion ex ante

### 6.1. Criterios duros

    D1. n_sow_total >= 100 (guardrail de muestra). Reportar ademas
        sow_rate_per_100_ticker_years.
    D2. n_confirmed >= 20 (guardrail de muestra). Reportar ademas
        confirmed_per_100_ticker_years.
    D3. struct_deterioration_H20 >= 50% (umbral provisional ex-ante).
        NO basta por si solo. Debe coexistir con incremental_lift_H20.
    D4. Estabilidad: `neighbour_confirmation_delta_pp` <= 30 pp en el
        vecindario inmediato de la grid.

**Nota D1/D2:** son guardrails de adecuacion muestral, no metricas de
calidad. Ver dictamen §15.

**Nota D3:** D3 se combina con lift. El criterio real de confirmador es
"lift > 0" (o el umbral que decida la segunda ronda de auditoria). El
informe debe presentar D3 e `incremental_lift_H20` juntos.

### 6.2. Ranking entre aceptados

    Paso 1. Pasar D1-D4.
    Paso 2. Maximizar struct_deterioration_H20.
    Paso 3. Exigir que el resultado no dependa de un vecino
            extremadamente distinto (coherente con D4).
    Paso 4. Desempate (orden fijo, determinista):
            4a. mayor X_ATR
            4b. mayor Y_VOL
            4c. N ascendente
            4d. M ascendente

**Retirado:** "valores estandar (N=30, M=10...)" como desempate.
"Estandar" no es condicion contractual (dictamen P3).

**Nota sobre H20 maximo aislado:** si dos combinaciones tienen 55% vs
54% pero una tiene muestra minima y la otra robusta, D2 evita el caso
extremo. El informe debe presentar tambien la incertidumbre/muestra.

---
## 4. Formula v1.8 y parametros

### 4.1. Formula SOW

    support_t          = rolling_min(Low, N).shift(1)
    ATR_baseline_t     = ATR(WYCKOFF_ATR_WINDOW).shift(1)
    volume_baseline_t  = rolling_mean(Volume, N).shift(1)

    break_depth_t      = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t     = Volume_t / volume_baseline_t

    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

### 4.2. Parametros a calibrar

| Parametro | Funcion | Rango |
|---|---|---|
| N | Ventana para soporte y baseline de volumen | {20, 30, 40, 60} |
| M | Max edad del SOW para confirmar | {5, 10, 15, 20, 30} |
| X_ATR | Penetracion minima (unidades de ATR) | {0.25, 0.50, 0.75, 1.00} |
| Y_VOL | Ratio minimo de volumen | {1.10, 1.20, 1.50} |

Grid total: 4 x 5 x 4 x 3 = **240 combinaciones**.

### 4.3. Constantes de control (NO en grid)

    WYCKOFF_ATR_WINDOW = 20    fijo
    PRECEDENT_WINDOW   = 60    fijo (PROPUESTO, no calibrado; futuro 5b.4)
    WYCKOFF_T_NORM_K   = 0.25  congelado (5b.2)

Razon: 5b.3 aisla el efecto de N/M/X_ATR/Y_VOL. Añadir W o ATR_WINDOW
al grid convierte el experimento en una busqueda multi-dimensional
no controlada.

### 4.4. Fail-closed en codigo

`detect_sow` **exige** `window`, `x_atr`, `y_vol` explicitos. Si falta
alguno, lanza `ValueError`. `config/settings.py` mantiene los cuatro
parametros SOW en `None` hasta que 5b.3 cierre. Invariantes I34-I36
(contrato v1.8 §9).

---

## 5. Dataset y metricas

### 5.1. Dataset

- Fuente: `data/stock_prices.parquet`.
- Universo: todos los tickers con >= 200 obs validas.
- Periodo: 2021-10-01 hasta ultima sesion cerrada.
- Split: 70% calibracion / 30% holdout temporal, con **una unica
  fecha de corte global** (ver §3).

**Aviso survivorship bias:** universo actual, no historico.

### 5.2. Metricas de actividad

- `n_candidate_events`: numero de entradas en episodios (candidate_t=True
  AND candidate_{t-1}=False, por ticker).
- `n_confirmed`: episodios con >= 1 SOW valido dentro de la ventana M.
- `n_sow_raw`: SOW=1 brutos, por trazabilidad.
- `n_sow_unique`: SOW agrupados por ticker (|t_i - t_{i-1}| < N cuentan
  como 1).
- `tasa_candidate_confirmed = n_confirmed / n_candidate_events`.
  Si `n_candidate_events == 0` -> `N/A` (no cero).

### 5.3. Metricas de estabilidad

- `neighbour_confirmation_delta_pp`: maximo cambio absoluto en **puntos
  porcentuales** de `tasa_candidate_confirmed` frente a vecinos
  inmediatos de la grid, manteniendo constantes los otros dos
  parametros. Sustituye a `margen_pct_confirmed` (retirado por
  ambiguedad, dictamen P4).
- `freq_sow_por_anio`: SOW/anio (global).
- `sow_rate_per_100_ticker_years`: SOW por 100 ticker-anios.
- `confirmed_per_100_ticker_years`.
- `pct_candidate_sin_sow`: candidate sin confirmar.

### 5.4. Metricas de calidad confirmatoria (clave)

Para cada configuration (N, M, X_ATR, Y_VOL), sobre cada candidate
episode confirmado, anclar `t = primer SOW valido del episodio` y
medir en horizontes H in {10, 20, 40}:

- `struct_deterioration_H{h}`: `struct[t+h] < struct[t]`, con
  `struct[t]` y `struct[t+h]` validos. **Outcome primario: H20.**
- `price_weakness_H{h}`: `close[t+h] < close[t] * (1 - 0.02)`.
- `below_support_H{h}`: `close[t+h] < support[t]`.
- `lower_low_H{h}`: `min(close[t+1:t+h]) < min(close[t-N:t])`.

**Metricas de lift obligatorias (dictamen C3):**

    P20_confirmed = mean(struct_deterioration_H20 | episodios confirmados)
    P20_baseline  = mean(struct_deterioration_H20 | episodios NO confirmados)
    incremental_lift_H20 = P20_confirmed - P20_baseline

Idem `incremental_lift_H40`. **Obligatorias en el informe.**

---

## 6. Criterios de seleccion ex ante

### 6.1. Criterios duros

    D1. n_sow_total >= 100 (guardrail de muestra). Reportar ademas
        sow_rate_per_100_ticker_years.
    D2. n_confirmed >= 20 (guardrail de muestra). Reportar ademas
        confirmed_per_100_ticker_years.
    D3. struct_deterioration_H20 >= 50% (umbral provisional ex-ante).
        NO basta por si solo. Debe coexistir con incremental_lift_H20.
    D4. Estabilidad: `neighbour_confirmation_delta_pp` <= 30 pp en el
        vecindario inmediato de la grid.

**Nota D1/D2:** son guardrails de adecuacion muestral, no metricas de
calidad. Ver dictamen §15.

**Nota D3:** D3 se combina con lift. El criterio real de confirmador es
"lift > 0" (o el umbral que decida la segunda ronda de auditoria). El
informe debe presentar D3 e `incremental_lift_H20` juntos.

### 6.2. Ranking entre aceptados

    Paso 1. Pasar D1-D4.
    Paso 2. Maximizar struct_deterioration_H20.
    Paso 3. Exigir que el resultado no dependa de un vecino
            extremadamente distinto (coherente con D4).
    Paso 4. Desempate (orden fijo, determinista):
            4a. mayor X_ATR
            4b. mayor Y_VOL
            4c. N ascendente
            4d. M ascendente

**Retirado:** "valores estandar (N=30, M=10...)" como desempate.
"Estandar" no es condicion contractual (dictamen P3).

**Nota sobre H20 maximo aislado:** si dos combinaciones tienen 55% vs
54% pero una tiene muestra minima y la otra robusta, D2 evita el caso
extremo. El informe debe presentar tambien la incertidumbre/muestra.

---
## 7. Validacion holdout

Una vez elegida la combinacion sobre el 70%:

1. Congelar N, M, X_ATR, Y_VOL.
2. Aplicar sobre el 30% final (misma fecha de corte global, §3).
3. Medir D1-D4 sobre holdout.
4. D1-D4 **obligatorios** en holdout. Si no pasan: no generaliza.
5. Reportar tambien `incremental_lift_H20` y `incremental_lift_H40`
   en holdout.

**Distincion de horizontes:**

    H20   PRIMARY OOS OUTCOME
    H10   SECONDARY
    H40   SECONDARY

Los tres se reportan. La decision principal se basa en H20.

---

## 8. Analisis prohibidos

- Optimizar por `n_confirmed` (frecuencia artificial).
- Optimizar por numero de DISTRIBUTION total.
- Elegir parametros para que los 5 casos del dictamen 5c.3 queden
  "bien".
- Modificar criterios ex ante tras ver resultados.
- Incluir W (`PRECEDENT_WINDOW`) o `WYCKOFF_ATR_WINDOW` en el grid.
- Usar `pct_deterioro_continuo` (nombre retirado, §1-C2).
- Contar sesiones `candidate=True` como eventos independientes
  (violacion de unidad de observacion, §1-C4).
- Reportar lift sin baseline candidate-sin-SOW (violacion C3).
- Trasladar informacion del futuro al bloque actual (§3).

---

## 9. Salida esperada

- `docs/auditoria/wyckoff/22_calibracion_5b3_resultados.md` con:
  - Tabla de grid completa (240 filas) con D1-D4 por combinacion.
  - Combinacion elegida (con desempates aplicados en orden §6.2).
  - Resultados D1-D4 en calibracion y holdout.
  - `incremental_lift_H20` y `incremental_lift_H40` para la elegida
    y, como contexto, top-10.
  - Analisis de estabilidad (`neighbour_confirmation_delta_pp`).
  - Comportamiento posterior al SOW por horizonte (H10/H20/H40) y
    por metrica (struct / price / below_support / lower_low).
  - Casos SLB y PCG explicitamente (verificacion H1/H3 del 5c.3).
  - `n_sow_raw`, `n_sow_unique`, `n_candidate_events`, `n_confirmed`.

**Nota numeracion:** el 21 esta ocupado por
`21_expediente_5b3_aclaraciones.md`. El resultado de la ejecucion es
el 22.

---

## 10. Criterios de exito

5b.3 tiene exito si:

- Al menos una combinacion cumple D1-D4 en calibracion.
- La combinacion elegida cumple D1-D4 en holdout.
- `struct_deterioration_H20 >= 50%`.
- `incremental_lift_H20 > 0` (sin umbral duro todavia; el signo es
  obligatorio).
- El SOW ya no captura ruido marginal.

**Sobre SLB:** con la nueva definicion ATR-normalizada, SLB (breach
-0.06%, ATR~2-3%) da `break_depth ≈ 0.03`. Con X_ATR >= 0.25 (minimo
de la grid), SLB no confirma. Esperado.

**Sobre PCG:** H3 del 5c.3 (-1.43%, 1.05x) debe quedar excluido si
Y_VOL >= 1.20 (grid). Verificacion explicita en informe.

---

## 11. Estado del proyecto

    v1.6                    IMPLEMENTADA (arquitectura candidate/confirmed)
    5c.3                    DIAGNOSTICO CERRADO
    v1.7                    HISTORICA (stale por A1; superseded por v1.8)
    v1.8                    CONTRATO VIGENTE
    5b.3 (protocolo)        ESTE DOCUMENTO (v3, corregido)
    5b.3 (ejecucion)        PENDIENTE
    5b.4                    PENDIENTE (calibracion W, si procede)
    5c.4                    BLOQUEADA hasta cierre 5b.3
    5d                      BLOQUEADA hasta cierre 5c.4
    Legacy                  INTACTO

**Reglas de gobernanza activas:**

- K = 0.25 congelado (5b.2). No cambiar.
- W = 60 fijo en 5b.3. No entra en grid.
- WYCKOFF_ATR_WINDOW = 20 fijo. No entra en grid.
- Parametros SOW en config = None hasta cierre 5b.3.
- `detect_sow` fail-closed: sin parametros explicitos -> ValueError.
- No migrar consumidores productivos hasta que 5b.3 cierre y el
  contrato v1.8 este verificado en produccion.

---

**Fin del protocolo 5b.3 (v3).**