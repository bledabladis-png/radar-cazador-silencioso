# HALLAZGO v1.2 - `t_norm` mide aceleracion, no tendencia

**Documento de auditoria. Requiere dictamen externo antes de continuar.**
**Fecha:** 2026-10-02.
**Estado:** PENDIENTE de resolucion. Frente Wyckoff congelado.

---

## 1. Contexto

Durante la implementacion de v1.2 (contrato con precedente selectivo) un
test contractual (T1 del dictamen) fallo. La investigacion descubrio un
problema **anterior y mas profundo** que los D1/D2/D3 ya corregidos:

> **`t_norm` no mide tendencia. Mide cambio de tendencia.**

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

**`t_norm` mide la derivada de la tendencia, no la tendencia.**

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
