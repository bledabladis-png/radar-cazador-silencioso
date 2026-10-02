# CONTRATO WYCKOFF v1.9 - Candidata SOW congelada para validacion

**Documento normativo. Supersede v1.8.**

**STATUS: FROZEN_FOR_VALIDATION. NOT_PRODUCTION.**

**Fecha:** 2026-10-02.
**Motivo:** v1.9 supersede v1.8 al congelar la candidata SOW
(N=60, M=30, X_ATR=0.25, Y_VOL=1.10) como especificacion normativa
para la validacion futura 5b.X. NO activa produccion. NO modifica
config/settings.py. NO retira fail-closed. Ver dictamen
`33_dictamen_5b4bis_v2.md`.

Historial de cambios:
- v1.2 -> v1.3: ver `07_revision_contrato_v1_3.md`.
- v1.3 -> v1.4: ver `14_diagnostico_5c.md`.
- v1.4 -> v1.5: ver `15_hallazgo_distribution_v14.md`.
- v1.5 -> v1.6: ver `18_propuesta_v16_candidate_confirmed.md`.
- v1.6 -> v1.7: ver `20_protocolo_fase_5b3.md` (SOW ATR-normalizado).
- v1.7 -> v1.8: ver `21_expediente_5b3_aclaraciones.md` + dictamen 5b.3.
- v1.8 -> v1.9: ver `33_dictamen_5b4bis_v2.md` + `34_qa_5b4bis_resultados.md`
  (candidata SOW congelada para validacion 5b.X; no activacion productiva).

---

## 1. Proposito

Modulo de identificacion de **fases estructurales Wyckoff** con
**maquina de estados selectiva**. Los estados direccionales (MARKUP,
MARKDOWN) se clasifican por su estructura actual. Las formaciones
(ACCUMULATION, DISTRIBUTION) se clasifican por estructura actual +
precedente estructural historico.

---

## 2. Estados

### 2.1. Fases de mercado (5)

    MARKUP          estado direccional alcista
    MARKDOWN        estado direccional bajista
    ACCUMULATION    formacion de base tras caida
    DISTRIBUTION    formacion de techo tras subida
    RANGE           estado residual

### 2.2. Estado de datos (1)

    INSUFFICIENT_DATA

**No es una fase.** Sin evidencia, nunca se devuelve una fase de mercado.

### 2.3. Separacion estructural / tactico

    STRUCTURAL  -> determina la fase
    TACTICAL    -> confirma / contextualiza / eventos

`tact_score` **no veta** transiciones estructurales (invariante I17).

---
## 3. Componentes primarios

### 3.1. Tendencia

    MA50, MA200
    trend = MA50 / MA200 - 1
    t_norm = tanh(trend / K)

**Cambio v1.2 -> v1.3 (P1 critico).** En v1.2 la formula era
`tanh(robust_zscore(trend, window=200, min_periods=60))`. Esa
composicion mide **desviacion del regimen de tendencia**, no nivel de
tendencia. Ver `06_hallazgo_t_norm_v1_2.md`.

**Semantica v1.3 (mantenida en v1.8):**

    trend > 0  -> t_norm > 0
    trend = 0  -> t_norm = 0
    trend < 0  -> t_norm < 0

**K (WYCKOFF_T_NORM_K) = 0.25.** CALIBRADO en Fase 5b.2 (commit
e596733, ver `12_calibracion_K_5b2.md`). Congelado.

### 3.2. Compresion

    ATR = rolling(TR, 20)
    compression = ATR / Close
    c_norm = -tanh(robust_zscore(compression, window=200, min_periods=60))

Semantica: `c_norm > 0` = compresion (volatilidad baja); `c_norm < 0`
= expansion.

### 3.3. Volumen y esfuerzo

    v_norm = tanh(robust_zscore(Volume, window=60, min_periods=20))

    effort = z_robusto(Volume)
    result = z_robusto(|Close_t / Close_{t-20} - 1|)
    e_norm = tanh(effort - result)

### 3.4. Composicion

    struct_score = 0.60*t_norm + 0.40*c_norm
    tact_score   = 0.50*v_norm + 0.50*e_norm
    combined     = 0.70*struct_score + 0.30*tact_score

**Nota:** `combined` es diagnostico. No participa en la clasificacion
de fase (invariante I17).

### 3.5. Estabilidad

    stability = 1 - 2*tanh(MAD(combined, ventana) / K)

Rango (-1, 1). No depende del nivel.

---

## 4. Precedente estructural

### 4.1. Definicion

El precedente en `t` se computa sobre la ventana `[t-W, t-1]`,
W = 60.

**W = 60 es CONTROL FIJO de la Fase 5b.3.** PROPUESTO, NO CALIBRADO.
No entra en el grid 5b.3. Su calibracion, si se decide, sera 5b.4.
Ver dictamen 5b.3 §5 (P5).

Se calcula **solo** sobre variables continuas (`struct_score`,
`t_norm`). **No** sobre etiquetas de fase previas. **No** invoca
`classify_wyckoff_phase`.

### 4.2. Variables

    struct_min  = min(struct_score[t-W : t-1])
    struct_max  = max(struct_score[t-W : t-1])
    struct_mean = mean(struct_score[t-W : t-1])
    t_norm_min  = min(t_norm[t-W : t-1])
    t_norm_max  = max(t_norm[t-W : t-1])

### 4.3. Invariantes

- **I11:** solo datos `<= t-1`.
- **I12:** extender input con datos futuros no cambia la fase en `t`.
- **I13:** no circular (no invoca la propia funcion).

### 4.4. Nota sobre `struct_max` en DISTRIBUTION

En la definicion de DISTRIBUTION (§5.4) `struct_max` se computa sobre
`[t-W+1, t-1]` con W=60. Ver §5.4 para la formula exacta y el test
de no-look-ahead.

---

## 5. Clasificacion

### 5.1. INSUFFICIENT_DATA

    si len(Close validos) < 200
       OR no hay observacion valida de struct_score / t_norm / c_norm
       OR precedente no disponible (para fases que lo requieren)
    -> INSUFFICIENT_DATA

**I7:** `ALL_NAN != RANGE`. Sin evidencia, nunca devolver fase.

**Warm-up practico:** ~460 filas (doble rolling). Series de 200-459
filas devuelven INSUFFICIENT_DATA por prudencia.

### 5.2. ACCUMULATION

**Requiere precedente** (formacion de base).

Precedente:

    struct_min < PREC_STRUCT_WEAK         (-0.20)   debilidad reciente

Estructura actual:

    t_norm_t < T_NORM_WEAK OR |t_norm_t| < T_NORM_WEAK   (debil/lateral)
    AND
    STRUCT_BASE_LOW <= struct_score_t <= STRUCT_BASE_HIGH   (-0.20..+0.20)
    AND
    c_norm_t > C_NORM_COMPRESSION         (0.30)   compresion

Confirmacion opcional (no requisito):

    tact_score_t > 0
    OR detect_spring en ultimas N_spring sesiones

**Regla de continuidad:** si `struct_max >= PREC_STRUCT_STRONG` (0.30),
un activo recientemente fuerte no puede entrar directo a ACCUMULATION.
Devuelve RANGE.

### 5.3. MARKUP

**NO requiere precedente.**

Estructura alcista actual:

    struct_score_t > STRUCT_STRONG        (0.30)
    AND
    t_norm_t > T_NORM_STRONG              (0.30)

**Sin veto por `c_norm`:** tanto regimen de compresion (subida
ordenada) como de expansion (subida volatil) son compatibles con
MARKUP.

**Sin veto por `combined`:** la fase es estructural, no tactica.

**Justificacion (dictamen):** MARKUP no es "direccion instantanea"
sino "estructura alcista actual con evidencia suficiente de expansion/
direccion". Las dos condiciones (`struct` + `t`) capturan ambas
dimensiones. La persistencia en el tiempo no invalida el estado.

---
### 5.4. DISTRIBUTION

**Contrato v1.6 (dictamen externo):** DISTRIBUTION = candidate + SOW.
`DISTRIBUTION_CANDIDATE` **no es una fase**. Es un flag binario.

**Fase `DISTRIBUTION`** (confirmada):

    candidate_conditions
    AND
    SOW reciente (en las ultimas WYCKOFF_SOW_MAX_AGE_M sesiones)

Donde:

    candidate_conditions:
        struct_score_t < STRUCT_DETERIORO (-0.10)
        AND t_norm_t > -T_NORM_STRONG (-0.30)
        AND struct_max (historico [t-W+1, t-1], W=60)
            > PREC_STRUCT_STRONG (0.30)

**SOW v1.7 (formula vigente, cambia respecto a v1.6):**

    support_t          = rolling_min(Low, N).shift(1)
    ATR_baseline_t     = ATR(WYCKOFF_ATR_WINDOW).shift(1)    <-- ex-ante
    volume_baseline_t  = rolling_mean(Volume, N).shift(1)

    break_depth_t      = (support_t - Close_t) / ATR_baseline_t
    volume_ratio_t     = Volume_t / volume_baseline_t

    SOW_t = (Close_t < support_t)
            AND (break_depth_t >= X_ATR)
            AND (volume_ratio_t >= Y_VOL)

Cambios respecto a v1.6:
- **ATR ex-ante** (`shift(1)` tras la rolling). Evita que la barra t
  contamine el denominador de `break_depth`.
- **Magnitud ATR-normalizada** (`break_depth`). Evita confirmar
  rupturas marginales (ej. SLB, -0.06%) como si fueran SOW.
- **Umbral de volumen** (Y_VOL). Evita confirmar rupturas sin esfuerzo.
- **X_ATR, Y_VOL no tienen default.** La funcion `detect_sow` exige
  parametros explicitos y lanza `ValueError` si faltan (fail-closed,
  dictamen 5b.3 §6). Los valores vivos en `config/settings.py` son
  `None` hasta que 5b.3 cierre.

**Parametros de SOW (candidata congelada para validacion):**

    WYCKOFF_SOW_WINDOW_N    = 60      # CANDIDATA (v1.9, FROZEN_FOR_VALIDATION)
    WYCKOFF_SOW_MAX_AGE_M   = 30      # CANDIDATA (v1.9, FROZEN_FOR_VALIDATION)
    WYCKOFF_SOW_X_ATR       = 0.25    # CANDIDATA (v1.9, FROZEN_FOR_VALIDATION)
    WYCKOFF_SOW_Y_VOL       = 1.10    # CANDIDATA (v1.9, FROZEN_FOR_VALIDATION)

**STATUS = FROZEN_FOR_VALIDATION. NOT_PRODUCTION.**

Estos valores son la candidata #1 seleccionada por el ranking v3 en
5b.4-bis (ver `31_calibracion_5b4bis_resultados.md`). NO estan
activados en `config/settings.py` (siguen en `None`). NO son
contrato productivo hasta que pase 5b.X.

`detect_sow` mantiene fail-closed: exige los cuatro parametros
explicitos. La validacion 5b.X los pasara explicitamente desde su
script, sin tocar config.

Ver seccion 13 para el detalle de la candidata congelada y
condiciones de validacion.

**Semantica:**

> DISTRIBUTION representa una perdida estructural de fuerza tras una
> subida previa CONFIRMADA por un evento de debilidad (SOW). NO es
> cualquier deterioro; requiere evidencia independiente de oferta.

**Deteccion de candidate sin confirmar:**

La fase devuelta por `classify_wyckoff_phase` es `RANGE` cuando se
cumplen las candidate_conditions pero no hay SOW reciente. El flag
`distribution_candidate = True` se expone via
`classify_wyckoff_phase_meta` para consumidores que lo necesiten.

**Frontera DISTRIBUTION / MARKDOWN:**

- DISTRIBUTION: candidate + SOW reciente, `struct_t` en (-0.30, -0.10).
- MARKDOWN: `struct_t < -0.30`, `t_norm < -0.30`, `c_norm < 0`.

**Formalizacion historica:** `struct_max` se calcula como
`max(struct_score[t-W+1 : t-1])` con W=60. `support_t`,
`ATR_baseline_t` y `volume_baseline_t` usan `.shift(1)`.
Estrictamente historico, sin look-ahead.

**Tests obligatorios v1.6 (mantenidos):** T1 (candidate sin SOW),
T2 (SOW sin candidate), T3 (candidate + SOW), T4 (SOW antiguo),
T5 (no look-ahead), T6 (clasifica meta con flag).

**Tests obligatorios v1.7 (nuevos, en `test_wyckoff_v1_contract.py`):**
I30 (ATR ex-ante), I31 (break_depth con X_ATR), I32 (SOW sin ruptura
-> no dispara), I33 (no look-ahead).

**Tests obligatorios v1.8 (nuevos, pre-grid):**
- `detect_sow()` sin `x_atr`/`y_vol` -> `ValueError` (fail-closed).
- `config.WYCKOFF_SOW_X_ATR is None` y `WYCKOFF_SOW_Y_VOL is None`.
- `config.WYCKOFF_SOW_WINDOW_N is None` y `WYCKOFF_SOW_MAX_AGE_M is None`.

### 5.5. MARKDOWN

**NO requiere precedente.**

Estructura bajista actual:

    struct_score_t < STRUCT_WEAK          (-0.30)
    AND
    t_norm_t < -T_NORM_STRONG             (-0.30)
    AND
    c_norm_t < 0                          expansion (no compresion)

**Asimetria respecto a MARKUP:** `c_norm < 0` **es** requisito en
MARKDOWN. Razon: markdown confirmado suele implicar expansion de
volatilidad (ruptura de rango). MARKUP puede coexistir con compresion
(subida ordenada de baja volatilidad). **No es simetria matematica
espejo; es asimetria microestructural justificada.**

### 5.6. RANGE

Cualquier caso que no cumpla las condiciones anteriores. **No es
fallback de fallo.** Es fase explicita.

---
## 6. Transiciones informativas

    MARKUP       -> DISTRIBUTION    (deterioro estructural)
    MARKUP       -> RANGE           (perdida de fuerza)
    DISTRIBUTION -> MARKDOWN        (ruptura confirmada)
    DISTRIBUTION -> RANGE           (soporte inesperado)
    MARKDOWN     -> ACCUMULATION    (estabilizacion)
    MARKDOWN     -> RANGE           (agotamiento)
    ACCUMULATION -> MARKUP          (fortalecimiento)
    ACCUMULATION -> RANGE           (base fallida)
    RANGE        -> cualquier       (transitoria)

**Regla de continuidad:** `MARKUP -> ACCUMULATION` directa **no
permitida**. Se implementa por precedente (`struct_max < PREC_STRUCT_STRONG`
como condicion de ACCUMULATION).

---

## 7. Eventos

### 7.1. Spring

    (Low_t < Low_{t-1}) & (Close_t > Open_t) & (Volume_t > 1.5 * MA20(Volume))

### 7.2. SOS

    (Close_t > max(High_{t-20..t-1})) & (Volume_t > MA20(Volume))

### 7.3. SOW

    Ver §5.4. Formula v1.7 con ATR ex-ante + X_ATR + Y_VOL.
    Requiere parametros explicitos. Sin default.

### 7.4. Invariante

Con NaN en inputs, spring y sos devuelven 0. No distinguen
"no es evento" de "no evaluable". Deuda aceptada.

SOW: si `ATR_baseline` o `volume_baseline` son NaN, la condicion
correspondiente es False (por comparacion con NaN). Misma deuda
aceptada.

---

## 8. Contrato de entrada / salida

### 8.1. Entrada

`df`: MultiIndex (field, ticker) o flat OHLCV. `ticker`: str.

`detect_sow` requiere adicionalmente `window`, `x_atr`, `y_vol`
explicitos. `atr_window` tiene default `WYCKOFF_ATR_WINDOW`.

### 8.2. Salida

    classify_wyckoff_phase(df, ticker) -> str
        uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION
                | MARKDOWN | INSUFFICIENT_DATA

    classify_wyckoff_phase_meta(df, ticker) -> dict
        {"phase": str, "distribution_candidate": bool}

    wyckoff_score(df, ticker) -> tuple[7 Series]

    wyckoff_stability(combined, window, K) -> Series

    detect_sow(df, ticker, window, x_atr, y_vol, atr_window) -> Series[int]

### 8.3. Preparacion

`build_ticker_df`: Close autoritativo. Open/High/Low con fallback a
Close. Volume con 0.

---

## 9. Invariantes normativas

**I1-I10** (heredadas de v1, verificadas):
- I1: struct = 0.60*t + 0.40*c.
- I2: tact = 0.50*v + 0.50*e.
- I3: combined = 0.70*struct + 0.30*tact.
- I4: *_norm en (-1, 1).
- I5: struct, tact, combined en (-1, 1).
- I6: stability en (-1, 1).
- I7: ALL_NAN != RANGE -> INSUFFICIENT_DATA.
- I8: determinismo.
- I9: sin look-ahead.
- I10: pesos suman 1.00.

**I11-I14** (v1.1, reforzadas):
- I11: precedente solo usa datos <= t-1.
- I12: datos futuros no cambian fase en t.
- I13: precedente no circular.
- I14: continuidad de fase (informativa).

**I15-I18** (v1.2):
- I15: MARKUP no requiere precedente.
- I16: MARKDOWN no requiere precedente.
- I17: tact_score no veta fase estructural.
- I18: precedente selectivo por semantica (solo ACCUMULATION y
  DISTRIBUTION lo requieren).

**I19-I23** (v1.3):
- I19: `trend > 0 -> t_norm > 0`; `trend < 0 -> t_norm < 0`;
  `trend = 0 -> t_norm = 0`.
- I20: `t_norm` acotado en (-1, 1).
- I21: aceleracion no invierte el signo de `t_norm`.
- I22: T1 (subida sostenida -> MARKUP) es invariante contractual
  permanente.
- I23: sin look-ahead en la nueva clasificacion.

**I24-I26** (v1.4/v1.5, K y DISTRIBUTION):
- I24: monotonicidad de t_norm respecto a trend (calibracion K).
- I25: DISTRIBUTION alcanzable (existe al menos un caso).
- I26: umbral K=0.25 congelado tras 5b.2.

**I27-I29** (v1.6):
- I27: candidate_conditions sin SOW no devuelven DISTRIBUTION
  (devuelven RANGE).
- I28: SOW sin candidate no eleva a DISTRIBUTION.
- I29: `distribution_candidate` flag coherente con la fase.

**I30-I33** (v1.7):
- I30: ATR_baseline aplica `.shift(1)`. Modificar `Close_t` no altera
  `ATR_baseline_t`.
- I31: `break_depth` usa ATR ex-ante y umbral X_ATR.
- I32: SOW sin ruptura de soporte no dispara.
- I33: SOW_t no cambia si se alteran datos > t.

**I34-I36** (v1.8, nuevas):
- I34: `detect_sow` sin `x_atr` o `y_vol` explicitos lanza
  `ValueError`. Fail-closed.
- I35: `config.WYCKOFF_SOW_X_ATR is None` y
  `config.WYCKOFF_SOW_Y_VOL is None` antes de cerrar 5b.3.
- I36: `classify_wyckoff_phase` NO emite DISTRIBUTION si no se le
  pasan los parametros SOW explicitos via argumento o via context.
  (Impl. ver §11bis.)

---

## 10. Parametros

Ver `02_proveniencia_parametros.md`. Los umbrales de precedente
(PREC_*), la ventana W=60, los limites de banda (STRUCT_BASE_LOW,
STRUCT_BASE_HIGH) y los parametros SOW (N, M, X_ATR, Y_VOL) estan
marcados como PROPUESTOS. No pasan a produccion sin calibracion.

**Regla:** los 4 casos MSFT/PLTR/INTC/AMD **no son dataset de
calibracion**. Solo sirven para detectar contradicciones del contrato.

**Control fijo 5b.3:** `PRECEDENT_WINDOW = 60` y
`WYCKOFF_ATR_WINDOW = 20`. NO entran en el grid.

**Congelado:** `WYCKOFF_T_NORM_K = 0.25` (5b.2). No cambiar.

---

## 11. Diferencias con versiones anteriores

| Aspecto | v1.5 | v1.6 | v1.7 | v1.8 |
|---|---|---|---|---|
| Distinguir candidate / confirmed | no | si | si | si |
| SOW | no | binario | ATR-normalizado + X_ATR + Y_VOL | ATR-normalizado + X_ATR + Y_VOL |
| ATR ex-ante en SOW | — | no | si | si |
| Defaults config X_ATR/Y_VOL | — | — | 0.50 / 1.20 | None (fail-closed) |
| `detect_sow` sin parametros | — | usa config | usa config | ValueError |
| Contrato coherente con codigo | — | si | NO (stale) | si |
| Invariantes | I1-I26 | I1-I29 | I1-I33 | I1-I36 |

## 11bis. Nota sobre classify_wyckoff_phase y parametros SOW

A partir de v1.8, `classify_wyckoff_phase` y
`classify_wyckoff_phase_meta` **no pueden emitir DISTRIBUTION** si no
reciben los parametros SOW explicitos (via kwarg `sow_params` o via
`detect_sow(..., x_atr=..., y_vol=..., ...)` en el mismo call-site).

Si no se pasan, el comportamiento es:

    candidate_conditions=True -> RANGE (con flag distribution_candidate=True)
    candidate_conditions=False -> comportamiento normal

Razon (dictamen 5b.3 §6): un default en config produce
"decisiones no congeladas". Fail-closed es la unica opcion coherente
con el estado PROPUESTO de los cuatro parametros hasta que 5b.3 cierre.

Consecuencia practica: mientras 5b.3 no haya congelado N/M/X/Y,
**ningun consumidor puede obtener DISTRIBUTION**. Solo
`DISTRIBUTION_CANDIDATE` (via flag). Esto es intencional.

---

## 12. Alcance

- `01_contrato_semantico_v1_9.md` (este) -> activo (FROZEN_FOR_VALIDATION).
- `01_contrato_semantico_v1_8.md` -> historico (referencia para v1.9).
- `01_contrato_semantico_v1_7.md` y anteriores -> historico.

Implementacion: `indicators/wyckoff_v1.py` implementa v1.8 con
fail-closed. NO se actualiza a v1.9 hasta que 5b.X valide la candidata.
Los cuatro parametros SOW son None en config; el contrato v1.9 los
declara como CANDIDATA para validacion futura.

---

## 13. Candidata SOW congelada para validacion

### 13.1. Estado

    STATUS = FROZEN_FOR_VALIDATION
    NOT_PRODUCTION

Los cuatro parametros SOW quedan congelados a nivel documental y
contractual, pero NO se activan en produccion.

### 13.2. Valores congelados

    N     = 60
    M     = 30
    X_ATR = 0.25
    Y_VOL = 1.10

Seleccionados por el ranking v3 del protocolo 5b.4-bis
(`30_protocolo_fase_5b4bis_v3.md`) y confirmados como ganadores
deterministas en `31_calibracion_5b4bis_resultados.md`.

### 13.3. Evidencia de desarrollo

    n_confirmed = 635
    n_baseline  = 1753
    lift_H20_ALL = +0.0740
    IC95_ALL     = [+0.0234, +0.1242]

Por bloque:

    P1: +0.093  IC95 [-0.099, +0.296]  (cruza 0)
    P2: +0.050  IC95 [-0.024, +0.122]  (cruza 0)
    P3: +0.106  IC95 [+0.038, +0.178]  (no cruza 0)

Solo P3 tiene IC95 que no cruza 0. El agregado esta arrastrado por P3.
Ver `34_qa_5b4bis_resultados.md`.

**No es validacion final. Es candidata de desarrollo.**

### 13.4. Condiciones de validacion (5b.X)

- Cutoff temporal: 2026-10-01.
- Todo dato posterior a 2026-10-01 pertenece a la validacion.
- UNA sola configuracion (la de 13.2).
- Misma implementacion (mismo commit del modulo y del script).
- Sin recalibracion.
- Sin volver a seleccionar entre 17/240 ni variantes.
- Criterio de exito: IC95 lower > 0 sobre datos nuevos.

### 13.5. Lo que NO cambia en v1.9 respecto a v1.8

- config/settings.py: los cuatro SOW siguen a None.
- detect_sow: fail-closed.
- classify_wyckoff_phase: sin sow_params no emite DISTRIBUTION.
- Legacy `wyckoff.py`: intacto.
- Consumidores del pipeline: sin migrar.
- 5c.4 y 5d: bloqueadas.

### 13.6. Que NO debe hacerse hasta 5b.X

- No activar parametros en config.
- No retirar fail-closed.
- No migrar consumidores del pipeline a wyckoff_v1.
- No abrir 5c.4 ni 5d.
- No ampliar el grid.
- No recalibrar.
- No considerar demostrado el efecto.

---

## 14. Estado normativo

    v1.8  CONTRATO PRODUCTIVO VIGENTE (hasta validacion)
    v1.9  CANDIDATA SOW CONGELADA PARA VALIDACION
          N=60 M=30 X=0.25 Y=1.10
          STATUS = FROZEN_FOR_VALIDATION / NOT_PRODUCTION

    5b.3        FAIL
    5b.4        FAIL seleccion / diagnostico valido
    5b.4-bis    PASS desarrollo / NO confirmado
    5b.X        PENDIENTE (datos posteriores a 2026-10-01)
    5c.4        BLOQUEADA
    5d          BLOQUEADA
    Legacy      INTACTO
    Config SOW  None (fail-closed)

---

**Fin del contrato v1.9 (FROZEN_FOR_VALIDATION).**