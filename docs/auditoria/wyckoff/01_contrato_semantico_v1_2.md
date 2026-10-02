# CONTRATO WYCKOFF v1.2 - Semantica y estados

**Documento normativo. Sustituye a v1.1.0.**
**Fecha:** 2026-10-02.
**Cambio v1.1 -> v1.2:** ver `05_revision_contrato_v1_2.md`.
**Referencia:** dictamen auditor externo (segunda ronda v1.2).

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
    t_norm = tanh(robust_zscore(trend, window=200, min_periods=60))

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
de fase (contrato v1.2, invariante I17).

### 3.5. Estabilidad

    stability = 1 - 2*tanh(MAD(combined, ventana) / K)

Rango (-1, 1). No depende del nivel.

---

## 4. Precedente estructural

### 4.1. Definicion

El precedente en `t` se computa sobre la ventana `[t-N, t-1]`, N = 60
(PROPUESTO, calibrar Fase 5b). Se calcula **solo** sobre variables
continuas (`struct_score`, `t_norm`). **No** sobre etiquetas de fase
previas. **No** invoca `classify_wyckoff_phase`.

### 4.2. Variables

    struct_min  = min(struct_score[t-N : t-1])
    struct_max  = max(struct_score[t-N : t-1])
    struct_mean = mean(struct_score[t-N : t-1])
    t_norm_min  = min(t_norm[t-N : t-1])
    t_norm_max  = max(t_norm[t-N : t-1])

### 4.3. Invariantes

- **I11:** solo datos `<= t`.
- **I12:** extender input con datos futuros no cambia la fase en `t`.
- **I13:** no circular (no invoca la propia funcion).

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

**Sin veto por `c_norm`:** tanto regimen de compresion (subida ordenada)
como de expansion (subida volatil) son compatibles con MARKUP.

**Sin veto por `combined`:** la fase es estructural, no tactica.

**Justificacion (dictamen):** MARKUP no es "direccion instantanea" sino
"estructura alcista actual con evidencia suficiente de expansion/
direccion". Las dos condiciones (`struct` + `t`) capturan ambas
dimensiones. La persistencia en el tiempo no invalida el estado.

### 5.4. DISTRIBUTION

**Requiere precedente** (formacion de techo).

Precedente:

    struct_max > PREC_STRUCT_STRONG       (0.30)   fortaleza reciente

Estructura actual:

    t_norm_t > 0 OR |t_norm_t| < T_NORM_WEAK   (aun positivo o neutro)
    AND
    struct_score_t < STRUCT_DETERIORO     (-0.10)   deterioro
    AND
    c_norm_t > C_NORM_COMPRESSION         (0.30)   compresion en techo

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

**Justificacion (dictamen):** MARKDOWN no es "MA50 < MA200". Requiere
(a) estructura bajista clara (struct), (b) direccion bajista (t),
(c) microestructura de expansion (c).

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

### 7.3. Invariante

Con NaN en inputs, el evento devuelve 0. No distingue "no es evento" de
"no evaluable". Deuda aceptada.

---

## 8. Contrato de entrada / salida

### 8.1. Entrada

`df`: MultiIndex (field, ticker) o flat OHLCV. `ticker`: str.

### 8.2. Salida

    classify_wyckoff_phase(df, ticker) -> str
        uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION
                | MARKDOWN | INSUFFICIENT_DATA

    wyckoff_score(df, ticker) -> tuple[7 Series]

    wyckoff_stability(combined, window, K) -> Series

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
- I11: precedente solo usa datos <= t.
- I12: datos futuros no cambian fase en t.
- I13: precedente no circular.
- I14: continuidad de fase (informativa).

**I15-I18** (v1.2, nuevas):
- I15: MARKUP no requiere precedente.
- I16: MARKDOWN no requiere precedente.
- I17: tact_score no veta fase estructural.
- I18: precedente selectivo por semantica (solo ACCUMULATION y
  DISTRIBUTION lo requieren).

---

## 10. Parametros

Ver `02_proveniencia_parametros.md`. Los umbrales de precedente
(PREC_*), la ventana N=60, y los limites de banda (STRUCT_BASE_LOW,
STRUCT_BASE_HIGH) estan marcados como PROPUESTOS. No pasan a produccion
sin calibracion en Fase 5b.

**Regla:** los 4 casos MSFT/PLTR/INTC/AMD **no son dataset de
calibracion**. Solo sirven para detectar contradicciones del contrato.

---

## 11. Diferencias con versiones anteriores

| Aspecto | v1.0 | v1.1 | v1.2 |
|---|---|---|---|
| Naturaleza | score regime | fase + precedente | fase + precedente selectivo |
| MARKUP precedente | — | `s_max < 0.30` | no requiere |
| MARKDOWN precedente | — | `s_min > -0.30` | no requiere |
| MARKDOWN `c_norm` | — | `< 0` | `< 0` (justificado) |
| ACCUMULATION `tact` | obligatorio | opcional | opcional |
| ACCUMULATION precedente | — | requerido | requerido |
| DISTRIBUTION precedente | — | requerido | requerido |
| Invariantes | I1-I10 | I1-I14 | I1-I18 |

---

## 12. Alcance

- `01_contrato_semantico_v1_2.md` (este) -> activo.
- `01_contrato_semantico_v1_1.md` -> historico.
- `01_contrato_semantico_v1.md` -> historico.
- `04_revision_contrato_v1_1.md` -> justificacion v1->v1.1.
- `05_revision_contrato_v1_2.md` -> justificacion v1.1->v1.2.

Implementacion: `indicators/wyckoff_v1.py` se actualiza a v1.2 tras
este documento.

---

Fin del contrato v1.2.
