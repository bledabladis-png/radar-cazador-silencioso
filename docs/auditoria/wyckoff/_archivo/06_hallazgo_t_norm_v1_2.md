# HALLAZGO v1.2 - `t_norm` mide desviacion del regimen de tendencia, no nivel de tendencia

**Documento de auditoria. Requiere dictamen externo antes de continuar.**
**Fecha:** 2026-10-02.
**Estado:** PENDIENTE de resolucion. Frente Wyckoff congelado.

---

## 1. Contexto

Durante la implementacion de v1.2 (contrato con precedente selectivo) un
test contractual (T1 del dictamen) fallo. La investigacion descubrio un
problema **anterior y mas profundo** que los D1/D2/D3 ya corregidos:

> **`t_norm` no mide nivel de tendencia. Mide desviacion respecto del regimen reciente de tendencia.**

---

## 2. Evidencia empirica

Tres series sinteticas deterministas (n=600, sin ruido en close):

### Serie A - subida constante

    Close: 100 * exp(linspace(0, 1.2, 600))
    Precio sube +232% en 600 sesiones. Drift diario constante.

Resultado:

    trend_ult = +0.1549     (MA50/MA200 - 1 = +15.5%)
    t_norm    = +0.0009     (z-score sobre trend)

**Diagnostico:** la serie crece sostenidamente. Pero `robust_zscore(trend,
window=200)` compara el valor actual con la mediana rolling. Como el
`trend` es constante (+0.1549), la mediana = 0.1549 y el z-score = 0.
`tanh(0) = 0`. **El activo no es MARKUP segun v1.2.**

### Serie B - subida acelerada

    drift = linspace(0.001, 0.006, 600)
    Precio sube mas rapido cada dia.

Resultado:

    trend_ult = +0.4293
    t_norm    = +0.5924

**Diagnostico:** el `trend` crece. La mediana rolling va por detras del
valor actual. z-score positivo. MARKUP correcto.

### Serie C - subida desacelerada

    drift = linspace(0.006, 0.001, 600)
    Precio sube +13% pero frenando.

Resultado:

    trend_ult = +0.1288
    t_norm    = -0.5802

**Diagnostico:** el `trend` decrece. La mediana rolling va por delante
del valor actual. z-score negativo. Se acerca a MARKDOWN. **Un activo
que sigue subiendo se clasifica como bajista.**

---

## 3. Problema semantico

En Wyckoff clasico, MARKUP (Phase E) es el **movimiento alcista
sostenido** tras la salida del trading range. Una subida constante es
la esencia de MARKUP. La aceleracion no es requisito.

El contrato v1.2 pide:

    MARKUP: struct_score_t > 0.30 AND t_norm_t > 0.30

Con `t_norm = tanh(robust_zscore(trend, window=200, min_periods=60))`:

- Subida constante: `t_norm ≈ 0` -> MARKUP rechazado.
- Subida acelerada: `t_norm > 0` -> MARKUP aceptado.
- Subida desacelerada: `t_norm < 0` -> casi MARKDOWN.

**`t_norm` mide la desviacion del nivel de tendencia respecto de su mediana rolling. No mide el nivel absoluto de tendencia.**

---

## 4. Impacto

### 4.1. En el modulo Wyckoff v1.2

- La definicion de MARKUP/MARKDOWN es **incorrecta** semanticamente.
- Los umbrales `T_NORM_STRONG = 0.30` no tienen sentido: un activo con
  tendencia sostenida del +15% deberia estar por encima; hoy da 0.

### 4.2. En los consumidores

Los 5 consumidores usan `wyckoff_score`:

    indicators/stock_leader.py              (sectores SPDR USA)
    indicators/index_leaders.py             (indices USA + Europa)
    indicators/sector_breadth.py            (conteo de fases)
    indicators/sector_wyckoff_distribution.py
    indicators/index_phase.py

Todos consumen `wyckoff_score()[0]` (combined) y los `_norm` para
clasificacion y WLS.

**Consecuencia no medida:** los rankings WLS llevan semanas ponderando
`rws_z` (derivado de `wyckoff_score`) sobre un valor que mide aceleracion
y no tendencia. Activos con subida sostenida reciben score bajo;
activos con aceleracion reciben score alto.

### 4.3. En el modulo legacy

`indicators/wyckoff.py` (legacy v4.2) usa la misma formula con
`window=60` (default de `robust_zscore`). El defecto esta presente desde
el inicio del sistema. **No es regresion de v1.2.**

---

## 5. Causa tecnica

    trend = MA50 / MA200 - 1
    t_norm = tanh(robust_zscore(trend, window=200, min_periods=60))

`robust_zscore` calcula:

    median_t = rolling_median(trend, window=200)
    mad_t    = rolling_median(|trend - median|, window=200)
    z_t      = (trend_t - median_t) / (1.4826 * mad_t)

Cuando `trend` es constante en el tiempo:

    trend_t = median_t -> z_t = 0 -> t_norm = 0

Cuando `trend` cambia (sube o baja dentro de la ventana):

    trend_t != median_t -> z_t != 0

**`robust_zscore` mide desviacion respecto a la mediana rolling. En una
serie monotona y estable, z = 0.**

---

## 6. Opciones de solucion

### O1 - `t_norm = tanh(trend)` (sin z-score)

    t_norm = tanh(MA50/MA200 - 1)

**Ventajas:**
- Mide tendencia absoluta directamente.
- Un trend del +15.5% da `tanh(0.155) = 0.154` -> podria ajustarse con K.
- Determinismo sin dependencia de ventanas.

**Inconvenientes:**
- Pierde adaptatividad al regimen del activo: un activo con volatilidad
  alta y otro con volatilidad baja se miden igual.
- Requiere calibracion de K (¿cuanto es "tendencia fuerte"?).
- `tanh(0.15)` es muy bajo; umbral 0.30 seria inalcanzable.

### O2 - `t_norm = tanh(trend / K)` con K calibrado

    t_norm = tanh((MA50/MA200 - 1) / K)

Con K=0.15: trend 15% -> tanh(1.0) = 0.76. Umbral 0.30 se alcanza con
trend ≈ 4.6%.

**Ventajas:**
- Mide tendencia absoluta con escala conocida.
- Un solo parametro (K).
- Comparable entre activos (mismo K).

**Inconvenientes:**
- K no es adaptativo por activo.
- Calibracion requiere dataset (Fase 5b).

### O3 - `t_norm = tanh(robust_zscore(trend, window=LARGE))`

Aumentar la ventana a 500-1000 sesiones para que la mediana refleje
"trend historico" en lugar de "trend reciente".

**Ventajas:**
- Mantiene la forma actual del contrato.
- Requiere cambio minimo.

**Inconvenientes:**
- No resuelve el problema estructural: una serie de 1000+ sesiones con
  tendencia constante y creciente eventualmente tiene mediana cercana
  al ultimo valor.
- Con 600 sesiones de warm-up obligatorio, la ventana util se reduce
  mucho.

### O4 - cambiar la metrica: `t_norm = tanh(trend - threshold)`

Diferencia directa contra un umbral absoluto.

**Ventajas:**
- Directo: "trend > X% es alcista".

**Inconvenientes:**
- Pierde suavizado.
- Requiere threshold calibrado.

### O5 - usar la pendiente del `trend`, no el nivel

    t_norm = tanh(robust_zscore(trend.diff(window)))

Mide explicitamente la aceleracion.

**Ventajas:**
- Coherente con la implementacion actual (por accidente).

**Inconvenientes:**
- **Sigue midiendo aceleracion, no tendencia.**
- Se aleja de la semantica Wyckoff.

---

## 7. Recomendacion tecnica (a validar por auditor)

**Opcion O2** con K calibrado:

    t_norm = tanh((MA50/MA200 - 1) / K)

Razones:

1. **Mide lo que el contrato dice medir:** tendencia actual.
2. **Un solo parametro** (K), calibrable en Fase 5b.
3. **Comparable entre activos** (mismo K para todos).
4. **Coherente con Wyckoff:** MARKUP = movement sostenido con tendencia
   clara, no aceleracion.
5. **Modifica minimamente el contrato** (§3.1).

**K inicial propuesto:** 0.15 (trend 15% -> t_norm 0.76; trend 4.6%
-> t_norm 0.30). Sin calibrar, marcado como PROPUESTO.

---

## 8. Implicaciones sobre el contrato

Si se aprueba O2:

- §3.1 cambia: `t_norm = tanh(trend / K)`.
- Se elimina `robust_zscore` para `trend`. Solo se mantiene para
  `compression`, `volume`, `price_move` (donde el z-score tiene sentido
  porque mide desviacion respecto a un regimen, no tendencia).
- Se introduce parametro `T_NORM_K` en `02_proveniencia`.
- Se actualiza el contrato v1.2 -> v1.3.

Si se aprueba O3 (ventana mayor):

- Solo cambia el parametro de la ventana. Minimo cambio contractual.
- No resuelve estructuralmente.

Si se aprueba O5 (explicitar aceleracion):

- El contrato debe reconocer que mide aceleracion, no tendencia.
- Los umbrales `T_NORM_*` deben reinterpretarse.
- Se aleja del lenguaje Wyckoff.

---

## 9. Preguntas al auditor

1. ¿Se aprueba O2 (t_norm = tanh(trend / K)) con K calibrable en Fase 5b?
2. ¿O se prefiere O1 (t_norm = tanh(trend)) sin z-score ni division?
3. ¿O se aprueba O3 (ventana mayor) como mitigacion parcial?
4. ¿Se acepta O5 reconociendo que el modulo mide aceleracion, no
   tendencia? (cambio semantico del contrato)
5. ¿Que pasa con los consumidores mientras tanto? ¿Se congela la
   migracion?
6. ¿El hallazgo debe elevarse al nivel de "P1 critico" en el expediente
   original, dado que afecta a la semantica central del modulo?

---

## 10. Estado del proyecto

- **Fases 1-3:** cerradas (contratos v1, v1.1, v1.2).
- **Fase 4 (tests):** 21 passed + 3 skipped + 1 xfail (T1, pendiente de
  este hallazgo).
- **Fase 5a (modulo v1.2):** implementado pero semanticamente incorrecto.
- **Fase 5b (calibracion):** bloqueada hasta resolver el hallazgo.
- **Fase 5c (comparativa):** bloqueada.
- **Fase 5d (migracion):** bloqueada.
- **Legacy `wyckoff.py`:** intacto.

**Regla:** no avanzar a Fase 5c/5d hasta que el hallazgo se resuelva.
Un modulo con metrica semantica incorrecta no debe migrarse a produccion.

---

Fin del hallazgo v1.2.

---

# 11. DICTAMEN DEL AUDITOR EXTERNO (2026-10-02)

**Veredicto: P1 - CRITICO, confirmado.**

> Debe bloquearse la migracion de `wyckoff_v1.2` y la calibracion 5b/5c
> hasta corregir la definicion de `t_norm`.

## 11.1. Correccion terminologica

El auditor corrige el titulo: **no es "aceleracion"**. Es:

> **desviacion robustamente estandarizada del nivel actual de `trend`
> respecto de su mediana rolling.**

En terminos economicos puede comportarse como proxy de cambio de
regimen, pero no es una segunda derivada ni un `delta^2 precio`.

Titulo final aplicado: "`t_norm` mide desviacion del regimen de
tendencia, no nivel de tendencia".

## 11.2. Diagnostico del auditor

- **Serie A (subida constante):** prueba mas fuerte. Activo sube +15.5%
  y `t_norm = +0.001`. El `robust_zscore` pregunta "que excepcional es
  este nivel dentro de los ultimos 200 valores", no "es este trend
  positivo".
- **Serie C (desacelerada):** activo sigue subiendo pero `t_norm < 0`.
  Un activo en estructura alcista obtiene senal tendencial negativa
  solo porque pierde velocidad.
- **Causa:** fallo de especificacion del input transformado. Ni
  `robust_zscore` ni `tanh` estan mal considerados aisladamente. La
  composicion "nivel de tendencia -> z-score temporal -> etiquetado
  como tendencia absoluta" es el defecto.

## 11.3. Decision sobre las opciones

| Opcion | Dictamen |
|---|---|
| O1 `tanh(trend)` | No preferida (escala comprimida, umbral 0.30 inalcanzable) |
| **O2 `tanh(trend/K)`** | **APROBADA como direccion de diseno** |
| O3 ventana mayor | Rechazada como solucion; solo mitigacion |
| O4 `tanh(trend-threshold)` | Tecnicamente viable; introduce otra semantica |
| O5 explicitar aceleracion | No para `t_norm` de este modulo |

## 11.4. Correcciones obligatorias sobre K

- **K = PROPUESTO**, nunca calibrado en esta ronda.
- **K no se calibra sobre los 4 casos** (MSFT, PLTR, INTC, AMD).
- Secuencia: contrato v1.3 -> definicion de K -> proveniencia ->
  calibracion 5b -> validacion 5c.

## 11.5. Impacto elevado a nivel sistemico

El auditor eleva el hallazgo mas de lo que lo hace este documento:

> Los valores historicos de WLS derivados de `wyckoff_score` no deben
> interpretarse como si hubieran sido producidos por el contrato que
> ahora se esta definiendo. Su semantica historica debe quedar marcada
> como no comparable con una serie recalculada bajo v1.3.

Relevante para cualquier backtest, IC o comparacion IS/OOS que use
`rws_z`.

## 11.6. Legacy elevado al mismo nivel

El legacy `indicators/wyckoff.py` (mismo concepto, `window=60`) queda
marcado como:

> **deuda estructural historica del indicador.**

No debe registrarse como "bug introducido por v1.2". Registro correcto:

    causa = diseno heredado
    deteccion = validacion contractual de v1.2

## 11.7. Condiciones impuestas por el auditor (10)

1. Cambiar "aceleracion" por "desviacion del regimen de tendencia".
2. Definir `t_norm` como nivel de tendencia escalado.
3. Introducir `K` como parametro explicito.
4. Mantener `K=0.15` unicamente como PROPUESTO.
5. Calibrar `K` fuera de los 4 casos diagnosticos.
6. Mantener T1 y convertirlo en invariante contractual.
7. Bloquear fases 5c/5d hasta completar v1.3.
8. Marcar el legacy como afectado historicamente.
9. No reinterpretar los WLS historicos como si usaran la nueva
   semantica.
10. Prohibir cualquier look-ahead en la nueva clasificacion.

## 11.8. Separacion conceptual obligatoria

    TREND LEVEL        -> t_norm (nivel de tendencia escalado)
    TREND CHANGE       -> opcional/futuro, NO en este modulo
    TACTICAL           -> confirmacion / WLS

La distincion entre nivel y cambio debe quedar explicita. El concepto
anterior (desviacion respecto al regimen) puede conservarse como senal
separada si interesa en el futuro, pero no debe presentarse como
tendencia.

## 11.9. Tests obligatorios para v1.3

| Test | Invariante |
|---|---|
| `constant_uptrend_is_positive` | subida sostenida -> `t_norm > 0` |
| `constant_downtrend_is_negative` | bajada sostenida -> `t_norm < 0` |
| `flat_trend_is_zero` | `trend = 0` -> `t_norm = 0` |
| `acceleration_does_not_define_direction` | acelerar/desacelerar modifica magnitud pero no invierte el signo |
| `markup_not_rejected_by_stable_trend` | subida sostenida puede ser MARKUP (T1 conservado) |
| `future_data_no_effect` | datos posteriores a `t` no afectan fase en `t` |
| `bounds` | `-1 < t_norm < +1` |

## 11.10. Conclusion del auditor

> La solucion no consiste en buscar una ventana de `robust_zscore`
> suficientemente grande para que la subida constante "parezca
> tendencia". Eso intentaria hacer que una metrica disenada para medir
> desviacion respecto de un regimen se comporte como otra metrica
> distinta. La correccion estructural es eliminar esa transformacion
> del nivel de tendencia y conservar el z-score, si interesa, como una
> senal separada de cambio de regimen.

## 11.11. Pregunta no resuelta (a elevar en el proximo ciclo)

En el mensaje original se planteo una pregunta secundaria al auditor:

> Deberian `c_norm` (`-tanh(robust_zscore(compression))`) y `e_norm`
> (`tanh(effort - result)`) revisarse con el mismo escrutinio?

El auditor no respondio a esta pregunta en el dictamen actual. Se eleva
en la siguiente ronda tras cerrar v1.3.
