# CONTRATO WYCKOFF v1 - Semantica y estados

**Documento normativo. Define el modulo de analisis Wyckoff v5.**
**Fecha:** 2026-10-02.
**HEAD de partida:** 9c09e0b.
**Sustituye a:** indicators/wyckoff.py v4.2 (legacy, se congela al cierre del proyecto).
**Referencia:** dictamen auditor externo 2026-10-02 (§24, veredicto NO CERRAR).

---

## 1. Proposito

Este modulo identifica **fases estructurales Wyckoff** en instrumentos de
renta variable, a partir de series OHLCV. Su salida se consume en el
ranking WLS de sectores (SPDR USA) y de indices (USA + Europa) para
seleccionar candidatos en fase de acumulacion o markup.

**Que es:**
- Un clasificador de fases estructurales.
- Deterministico dado un snapshot de input.
- Descriptivo. No genera senales de trading.

**Que no es:**
- Un score regime con etiquetas convencionales.
- Un indicador predictivo.
- Un sistema de timing.

---

## 2. Definiciones

### 2.1. Fase Wyckoff

Estado estructural del instrumento en el momento de la ultima
observacion valida. Cinco estados posibles, mutuamente excluyentes:

    ACCUMULATION    base tras caida, compresion, absorcion
    MARKUP          tendencia alcista confirmada
    DISTRIBUTION    techo tras subida, compresion, distribucion
    MARKDOWN        tendencia bajista confirmada
    RANGE           rango sin sesgo direccional claro

Mas un estado de datos:

    INSUFFICIENT_DATA   no hay evidencia suficiente para clasificar

Este ultimo **no es una fase**. Es un estado de calidad de datos. El
consumidor debe distinguirlo y decidir (hoy: omitir del ranking).

### 2.2. Score estructural y tactico

Dos componentes numericos continuos en el rango (-1, 1):

    struct_score = 0.60 * t_norm + 0.40 * c_norm
    tact_score   = 0.50 * v_norm + 0.50 * e_norm

Donde cada `*_norm` es una transformacion acotada del z-score robusto
de la variable subyacente. El detalle esta en §3.

Score compuesto:

    combined = 0.70 * struct_score + 0.30 * tact_score

`combined` NO es una fase. Es una escala continua que se usa como input
de la clasificacion y, en algunos consumidores, como valor de ranking.

### 2.3. Estabilidad

Medida de dispersion del `combined` en una ventana reciente. NO confundir
con `combined` ni con nivel del score. Ver §3.4.

---

## 3. Componentes primarios

### 3.1. Tendencia (trend)

    MA50  = media movil de Close, ventana 50
    MA200 = media movil de Close, ventana 200
    trend = MA50 / MA200 - 1

Semantica: separacion relativa de las medias. Positivo = tendencia
alcista; negativo = bajista.

Transformacion:

    t_norm = tanh( robust_zscore(trend, window=200, min_periods=?)

`min_periods` por definir en Fase 3 (proveniencia).

### 3.2. Compresion (compression)

    TR_t = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
    ATR  = media movil de TR, ventana 20
    compression = ATR / Close

Semantica: volatilidad relativa al precio. Alto = expansion; bajo =
compresion.

Transformacion:

    c_norm = -tanh( robust_zscore(compression, window=?, min_periods=?)

Signo negativo: compresion (baja volatilidad) aporta positivo al score
estructural.

### 3.3. Volumen y esfuerzo

Volumen:

    v_norm = tanh( robust_zscore(Volume, window=60, min_periods=20) )

Effort vs result (forma clasica Wyckoff):

    effort = z_robusto(Volume)             # esfuerzo observado
    result = z_robusto(|Close_t / Close_{t-20} - 1|)   # resultado de precio
    effort_vs_result = tanh(effort - result)

Semantica:
- Positivo: esfuerzo > resultado -> **absorcion** (mucho volumen, poco precio).
- Negativo: resultado > esfuerzo -> **movimiento sin esfuerzo**.

El nombre `effort_vs_result` respeta la formulacion clasica. La version
legacy invertia el sentido (result/effort) y se retira.

    e_norm = effort_vs_result

### 3.4. Estabilidad

    score_mad = MAD(combined en las ultimas N sesiones)
    stability = 1 - 2 * tanh(score_mad / K)

Con N y K por definir en Fase 3.

Semantica:
- `stability = +1`: dispersion nula -> score perfectamente estable.
- `stability = 0`: dispersion en la banda K.
- `stability = -1`: dispersion alta -> score inestable.

Rango (-1, 1). **NO depende del nivel de `combined`** (a diferencia de
la version legacy).

---

## 4. Composicion

### 4.1. Score estructural

    struct_score = 0.60 * t_norm + 0.40 * c_norm

Pesos 0.60 / 0.40. Justificacion en Fase 3.

### 4.2. Score tactico

    tact_score = 0.50 * v_norm + 0.50 * e_norm

Pesos simetricos. Justificacion en Fase 3.

### 4.3. Score compuesto

    combined = 0.70 * struct_score + 0.30 * tact_score

Peso mayor a struct. Razon: la fase Wyckoff es una propiedad estructural
(tendencia + compresion), no tactica (volumen puntual).

---

## 5. Clasificacion de fase

La clasificacion NO es una banda de `combined`. Es una conjuncion de
condiciones estructurales.

### 5.1. INSUFFICIENT_DATA

Se activa si:

    len(Close validos) < 200
    OR
    no existe ninguna observacion valida de `combined`

**Invariante**: si no hay evidencia, NUNCA se devuelve una fase de
mercado (MARKUP / ACCUMULATION / RANGE / DISTRIBUTION / MARKDOWN).

Este estado sustituye al fallback silencioso `ALL_NAN -> RANGE` del
modulo legacy. **Razon**: RANGE es una fase de mercado, ALL_NAN es
ausencia de datos. Son categorias distintas (ver dictamen auditor §4).

### 5.2. ACCUMULATION

Condiciones conjuntas (todas deben cumplirse):

    1) tendencia previa negativa o lateral-baja
       t_norm < 0  OR  |t_norm| < 0.30

    2) compresion alta
       c_norm > 0.30

    3) evidencia de base/rango reciente
       ATR actual <= mediana(ATR, ventana 60)   [ventana por validar en Fase 3]

    4) confirmacion tactica
       tact_score > 0
       OR
       detect_spring en las ultimas N sesiones  [N por validar en Fase 3]

### 5.3. MARKUP

    1) tendencia alcista confirmada
       t_norm > 0.30

    2) sin compresion extrema
       c_norm < 0.30

    3) score compuesto positivo
       combined > 0.30

### 5.4. DISTRIBUTION

    1) tendencia previa positiva o lateral-alta
       t_norm > 0  OR  |t_norm| < 0.30

    2) compresion alta
       c_norm > 0.30

    3) score compuesto negativo
       combined < -0.10   [umbral por validar en Fase 3]

### 5.5. MARKDOWN

    1) tendencia bajista confirmada
       t_norm < -0.30

    2) sin compresion extrema
       c_norm < 0

    3) score compuesto negativo
       combined < -0.30

### 5.6. RANGE

Cualquier caso que no cumple las condiciones anteriores. NO es un
fallback de fallo; es una fase explicita (mercado sin sesgo
estructural claro).

---

## 6. Eventos

### 6.1. Spring

    condition_t = (Low_t < Low_{t-1}) & (Close_t > Open_t) & (Volume_t > 1.5 * MA20(Volume))

Semantica: perforacion de minimo previo con cierre alcista y volumen
elevado. **Es un patron inspirado en Wyckoff, no un spring estructural
completo.** Requiere contexto de rango previo para interpretarse como
spring autentico.

### 6.2. SOS (Sign of Strength)

    condition_t = (Close_t > max(High_{t-20..t-1})) & (Volume_t > MA20(Volume))

Semantica: breakout de maximo reciente con volumen por encima de la
media. Mismo aviso que spring.

### 6.3. Invariante

Un evento no cumple condicion cuando hay NaN. Se diferencia "no es
evento" (int 0) de "no evaluable". Forma de representarlo (tercer valor
o mascara) por definir en Fase 2-bis.

---

## 7. Contrato de entrada / salida

### 7.1. Entrada

`df` MultiIndex (field, ticker) o DataFrame flat con columnas OHLCV.
`ticker` string.

### 7.2. Salida

`classify_wyckoff_phase(df, ticker) -> str`
Devuelve uno de:

    MARKUP | ACCUMULATION | RANGE | DISTRIBUTION | MARKDOWN
    INSUFFICIENT_DATA

`wyckoff_score(df, ticker) -> tuple[7 Series]`

    (combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm)

Todas las series con DatetimeIndex. NaN durante warm-up.

### 7.3. Preparacion

`build_ticker_df(df, ticker) -> DataFrame flat OHLCV`.
Manejo de NaN por definir en Fase 2-bis (mascaras de validez, no
dropna global).

---

## 8. Invariantes normativas

Estas reglas se verifican en tests (Fase 4):

**I1.** struct_score = 0.60*t_norm + 0.40*c_norm (bit a bit, tolerancia 1e-12).
**I2.** tact_score = 0.50*v_norm + 0.50*e_norm.
**I3.** combined = 0.70*struct_score + 0.30*tact_score.
**I4.** Todos los `*_norm` en rango (-1, 1).
**I5.** struct_score, tact_score, combined en rango (-1, 1).
**I6.** stability en rango (-1, 1).
**I7.** `ALL_NAN ≠ RANGE`: si no hay observacion valida, la fase devuelta es INSUFFICIENT_DATA.
**I8.** Clasificacion deterministica: mismo input -> mismo output.
**I9.** Sin look-ahead: la clasificacion de t usa solo datos hasta t.
**I10.** Los pesos suman 1.00 en cada nivel.

---

## 9. Estados y transiciones

Los estados son **instantaneos**. No hay estado persistente entre
llamadas. Un mismo instrumento puede cambiar de fase entre dos
observaciones consecutivas.

Transiciones observables (no contractuales, descriptivas):

    ACCUMULATION -> MARKUP         (confirmacion de tendencia)
    ACCUMULATION -> RANGE          (base fallida)
    MARKUP       -> DISTRIBUTION   (techo)
    MARKUP       -> RANGE          (perdida de tendencia)
    DISTRIBUTION -> MARKDOWN       (confirmacion de caida)
    DISTRIBUTION -> RANGE          (soporte inesperado)
    MARKDOWN     -> ACCUMULATION   (reversion)
    MARKDOWN     -> RANGE          (agotamiento)
    RANGE        -> ACCUMULATION   (nueva base)
    RANGE        -> DISTRIBUTION   (nuevo techo)

Las transiciones no se validan en codigo. Son informacion para el
consumidor.

---

## 10. Diferencias con el modulo legacy v4.2

| Aspecto | Legacy v4.2 | Contrato v1 |
|---|---|---|
| Naturaleza | Score regime con etiquetas | Fase estructural |
| ACCUMULATION | combined > 0 | Conjuncion estructural (5.2) |
| MARKDOWN | No existe (fantasma en consumidores) | Fase real (5.5) |
| effort_vs_result | result/effort (invertido) | effort/result (clasico) |
| stability | tanh(median/mad), saturado | 1 - 2*tanh(mad/K) |
| ALL_NAN | Devuelve RANGE | Devuelve INSUFFICIENT_DATA |
| dropna | Global | Mascaras de validez por campo |
| Proveniencia parametros | No documentada | Documentada (Fase 3) |

---

## 11. Alcance de la migracion

El modulo legacy `indicators/wyckoff.py` se congela al inicio del
proyecto. Permanece operativo hasta que:

1. El modulo nuevo este implementado y testeado.
2. Los 5 consumidores migren y pasen tests.
3. Una sesion de validacion E2E compare outputs legacy vs v1 y
   documente discrepancias.

Consumidores a migrar:

    indicators/stock_leader.py
    indicators/index_leaders.py
    indicators/sector_breadth.py
    indicators/sector_wyckoff_distribution.py
    indicators/index_phase.py

---

## 12. Anexo - Vocabulario

- **t_norm**: transformacion acotada del z-score robusto de trend.
- **c_norm**: transformacion acotada (con signo negativo) del z-score robusto de compresion.
- **v_norm**: transformacion acotada del z-score robusto de volumen.
- **e_norm**: transformacion acotada de effort_vs_result.
- **struct_score**: combinacion ponderada de t_norm y c_norm.
- **tact_score**: combinacion ponderada de v_norm y e_norm.
- **combined**: combinacion final de struct y tact.
- **stability**: medida de dispersion del combined en ventana reciente.
- **fase**: uno de los 5 estados estructurales.
- **INSUFFICIENT_DATA**: estado de calidad de datos, no fase.

---

Fin del contrato v1.
