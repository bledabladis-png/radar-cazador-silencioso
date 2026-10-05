# CONTRATO WYCKOFF v1.1 - Semantica y estados

**Documento normativo. Sustituye a v1.0.**
**Fecha:** 2026-10-02.
**Cambio v1 -> v1.1:** ver `04_revision_contrato_v1_1.md`.
**Referencia:** dictamen auditor externo 2026-10-02 (dos rondas).

---

## 1. Proposito

Modulo de identificacion de **fases estructurales Wyckoff** con
continuidad temporal, a partir de series OHLCV.

**Principio rector:**

> La fase en `t` no depende solo de las variables en `t`, sino de la
> trayectoria estructural reciente. El clasificador es una **maquina
> de estados**, no un umbral instantaneo.

---

## 2. Estados

### 2.1. Fases de mercado (5)

    ACCUMULATION
    MARKUP
    DISTRIBUTION
    MARKDOWN
    RANGE

### 2.2. Estado de datos (1)

    INSUFFICIENT_DATA

**No es una fase.** Se activa cuando no hay evidencia. Nunca devuelve
una fase de mercado.

### 2.3. Separacion estructural / tactico

    STRUCTURAL  -> determina la fase
    TACTICAL    -> confirma / contextualiza / eventos

`tact_score` no veta transiciones estructurales.

---

## 3. Componentes primarios

### 3.1. Tendencia

    MA50  = media movil de Close, ventana 50
    MA200 = media movil de Close, ventana 200
    trend = MA50 / MA200 - 1
    t_norm = tanh(robust_zscore(trend, window=200, min_periods=60))

### 3.2. Compresion

    TR_t = max(High-Low, |High-prev_Close|, |Low-prev_Close|)
    ATR  = media movil de TR, ventana 20
    compression = ATR / Close
    c_norm = -tanh(robust_zscore(compression, window=200, min_periods=60))

Semantica: `c_norm > 0` = compresion (volatilidad baja). `c_norm < 0`
= expansion.

### 3.3. Volumen y esfuerzo

    v_norm = tanh(robust_zscore(Volume, window=60, min_periods=20))

    effort = z_robusto(Volume)
    result = z_robusto(|Close_t / Close_{t-20} - 1|)
    effort_vs_result = tanh(effort - result)
    e_norm = effort_vs_result

Semantica: positivo = absorcion (mucho esfuerzo, poco resultado).

### 3.4. Composicion

    struct_score = 0.60*t_norm + 0.40*c_norm
    tact_score   = 0.50*v_norm + 0.50*e_norm
    combined     = 0.70*struct_score + 0.30*tact_score

### 3.5. Estabilidad

    score_mad = MAD(combined en ventana N_mad)
    stability = 1 - 2*tanh(score_mad / K)

Rango (-1, 1). No depende del nivel.

---

## 4. Precedente estructural

### 4.1. Definicion

Para clasificar la fase en `t`, se computa un **precedente estructural**
sobre la ventana `[t-N, t-1]`. N por definir en Fase 3.

### 4.2. Variables de precedente

    struct_min  = min(struct_score[t-N:t-1])
    struct_max  = max(struct_score[t-N:t-1])
    struct_mean = mean(struct_score[t-N:t-1])
    t_norm_min  = min(t_norm[t-N:t-1])
    t_norm_max  = max(t_norm[t-N:t-1])

### 4.3. Reglas

- El precedente se calcula **sobre variables continuas**.
- **No** se calcula sobre etiquetas de fase previas.
- Solo usa datos `<= t`.
- Su computo no invoca `classify_wyckoff_phase`.

### 4.4. Invariantes

**I11.** Sin look-ahead: solo datos `<= t`.
**I12.** Independiente de datos futuros: extender input con `> t` no
cambia la fase en `t`.
**I13.** No circular: no recursion sobre la propia funcion.

---

## 5. Clasificacion

### 5.1. INSUFFICIENT_DATA

    si len(Close validos) < 200
       OR no hay observacion valida de struct_score
    -> INSUFFICIENT_DATA

**Invariante I7:** `ALL_NAN != RANGE`. Sin evidencia, nunca se devuelve
fase de mercado.

**Hallazgo 2026-10-02 - warm-up doble.**
El minimo de 200 filas es **teorico** (MA200). El minimo **practico**
para producir un `struct_score` no vacio es ~460 filas:

    MA200 warm-up                        = 200
    robust_zscore(trend, w=200, mp=60)   = 200 + 60
    ---
    Total                                ~460

Razon: robust_zscore computa `median` **y** `MAD = |s - median|.median()`,
ambos con `min_periods=60`. El segundo requiere 60 valores no-NaN
despues de que el primero ya produjo valores. Warm-up doble.

Consecuencia:
- Series de 200-459 filas: `len(Close) >= 200` pero `struct_score` vacio
  -> INSUFFICIENT_DATA (correcto por I7).
- Series de >=460 filas: score valido.

Test de regresion: `test_warmup_real_es_doble_rolling` en
`tests/test_wyckoff_v1_contract.py`.

Accion futura: revisar si el doble warm-up es deseable o si conviene
reducir `min_periods` de robust_zscore en alguna capa. Decision en
Fase 5b (calibracion).

### 5.2. ACCUMULATION

Precedente:

    struct_min < THRESH_PREC_WEAK       (debilidad reciente)

Estado actual:

    t_norm_t < THRESH_T_WEAK
    OR |t_norm_t| < THRESH_T_SIDEWAYS
    AND
    struct_score_t en banda de base
    AND
    c_norm_t > THRESH_C_MIN              (compresion)

**Confirmacion opcional (no requisito):**

    tact_score_t > 0
    OR detect_spring en las ultimas N_spring sesiones

### 5.3. MARKUP

Precedente:

    struct_max < THRESH_PREC_MARKUP      (no era markup recientemente)

Estado actual:

    t_norm_t > THRESH_T_STRONG
    AND
    struct_score_t > THRESH_STRUCT_STRONG

**Eliminadas respecto a v1.0:**
- `c_norm < 0.30` (incompatible con bull market ordenado)
- `combined > 0.30` (veto tactico indebido)

### 5.4. DISTRIBUTION

Precedente:

    struct_max > THRESH_PREC_STRONG      (era fuerte recientemente)

Estado actual:

    t_norm_t >= THRESH_T_WEAK            (aun positivo o neutro)
    AND
    struct_score_t < THRESH_STRUCT_WEAK  (deterioro)
    AND
    c_norm_t > THRESH_C_MIN              (compresion en techo)

### 5.5. MARKDOWN

Precedente:

    struct_min > THRESH_PREC_NEG         (no era fuertemente bajista)
    OR struct_mean > THRESH_PREC_MD

Estado actual:

    t_norm_t < -THRESH_T_STRONG
    AND
    struct_score_t < -THRESH_STRUCT_STRONG
    AND
    c_norm_t < THRESH_C_MIN              (expansion, no compresion)

### 5.6. RANGE

Cualquier caso que no cumpla las condiciones anteriores. **No es
fallback de fallo:** es fase explicita.

### 5.7. Transiciones

**Permitidas (informativo):**

    ACCUMULATION -> MARKUP
    ACCUMULATION -> RANGE
    MARKUP       -> DISTRIBUTION
    MARKUP       -> RANGE
    DISTRIBUTION -> MARKDOWN
    DISTRIBUTION -> RANGE
    MARKDOWN     -> ACCUMULATION
    MARKDOWN     -> RANGE
    RANGE        -> cualquier

**Prohibida por continuidad:**

    MARKUP -> ACCUMULATION  (sin DISTRIBUTION/RANGE intermedio)

Esta prohibicion se evalua **por precedente estructural** (no por
etiqueta previa). Si `struct_max` reciente es fuerte, una clasificacion
actual de ACCUMULATION se promueve a RANGE.

**Nota:** I14 (informativa). Las violaciones no lanzan excepcion. Se
registran para diagnostico.

---

## 6. Eventos

### 6.1. Spring

    (Low_t < Low_{t-1}) & (Close_t > Open_t) & (Volume_t > 1.5 * MA20(Volume))

### 6.2. SOS

    (Close_t > max(High_{t-20..t-1})) & (Volume_t > MA20(Volume))

### 6.3. Invariante

Con NaN en cualquiera de los inputs, el evento devuelve 0. No distingue
"no es evento" de "no evaluable" hoy. Deuda aceptada.

---

## 7. Contrato de entrada / salida

### 7.1. Entrada

`df`: MultiIndex (field, ticker) o flat OHLCV. `ticker`: str.

### 7.2. Salida

    classify_wyckoff_phase(df, ticker) -> str
        uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION
                | MARKDOWN | INSUFFICIENT_DATA

    wyckoff_score(df, ticker) -> tuple[7 Series]
        (combined, struct_score, tact_score, t_norm, c_norm,
         v_norm, e_norm)

    wyckoff_stability(combined, window, K) -> Series

### 7.3. Preparacion

`build_ticker_df(df, ticker)`: mascaras de validez por campo (no
`dropna` global). Close autoritativo. Open/High/Low rellenos con Close.
Volume con 0.

---

## 8. Invariantes normativas

**I1.** `struct_score = 0.60*t_norm + 0.40*c_norm`.
**I2.** `tact_score = 0.50*v_norm + 0.50*e_norm`.
**I3.** `combined = 0.70*struct_score + 0.30*tact_score`.
**I4.** `t_norm, c_norm, v_norm, e_norm` en (-1, 1).
**I5.** `struct_score, tact_score, combined` en (-1, 1).
**I6.** `stability` en (-1, 1).
**I7.** `ALL_NAN != RANGE`. Sin evidencia -> INSUFFICIENT_DATA.
**I8.** Determinismo: mismo input -> mismo output.
**I9.** Sin look-ahead: fase en `t` usa solo datos <= t.
**I10.** Pesos suman 1.00 en cada nivel.
**I11.** Precedente sin look-ahead.
**I12.** Independiente de datos futuros.
**I13.** Precedente no circular (no invoca `classify_wyckoff_phase`).
**I14.** (Informativa) Continuidad de fase: transiciones prohibidas se
registran pero no bloquean.

---

## 9. Parametros

Todos los parametros (THRESH_*, N, N_mad, N_spring, K) estan listados en
`02_proveniencia_parametros.md`. Los que no tienen valor calibrado se
marcan como PROPUESTO o PENDIENTE.

**Regla:** ningun umbral sin proveniencia documentada pasa a produccion.

---

## 10. Diferencias con v1.0

| Aspecto | v1.0 | v1.1 |
|---|---|---|
| MARKUP `c_norm` | `c_norm < 0.30` | eliminado |
| MARKUP `combined` | `> 0.30` | sustituido por `struct_score > umbral` |
| ACCUMULATION `tact > 0` | obligatorio | opcional (informativo) |
| Precedente | no existe | variables estructurales historicas |
| Fase en t | valor de t | valor + precedente |
| tact en clasificacion | veta | no veta |
| Invariantes | I1-I10 | I1-I14 |
| Continuidad | no | regla informativa |

---

## 11. Alcance

Este contrato sustituye a v1.0. Los documentos:
- `01_contrato_semantico_v1.md` -> historico.
- `01_contrato_semantico_v1_1.md` (este) -> activo.
- `04_revision_contrato_v1_1.md` -> justificacion del cambio.

La implementacion actual `indicators/wyckoff_v1.py` implementa v1.0.
La actualizacion a v1.1 se hara tras cerrar los tests de contrato.

---

Fin del contrato v1.1.
