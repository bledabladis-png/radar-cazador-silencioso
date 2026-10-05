# VERIFICACION MANUAL - 10 DISTRIBUTION v1.5

**Documento de auditoria. Fase 5c.2, verificacion.**
**Fecha:** 2026-10-02.

---

## 1. Objetivo

Verificar manualmente los 10 casos clasificados como DISTRIBUTION por
v1.5, contrastando contra OHLCV crudo (MA50, MA200, precio, trend).

---

## 2. Resultado por caso

### 2.1. DISTRIBUTION claros (5/10)

| Ticker | MA50 | MA200 | MA50 vs 200 | struct | t_norm | Veredicto |
|---|---:|---:|---|---:|---:|---|
| ACS.MC | 102.4 | 109.6 | -6.6% | -0.287 | -0.257 | Bajista claro |
| ANA.MC | 214.3 | 223.3 | -4.0% | -0.139 | -0.159 | Bajista claro |
| BA | 212.6 | 220.8 | -3.7% | -0.271 | -0.147 | Bajista claro |
| PCG | 15.5 | 16.4 | -5.5% | -0.530 | -0.218 | Bajista fuerte (c_norm=-0.999, cerca de MARKDOWN) |
| TEF.MC | 3.61 | 3.66 | -1.4% | -0.178 | -0.055 | Bajista claro |

En estos 5, MA50 < MA200 confirma tendencia bajista estructural. La
clasificacion DISTRIBUTION es **correcta**: hubo fortaleza previa
(struct_max > 0.30), hay deterioro actual (struct < -0.10), pero aun no
es MARKDOWN (t_norm > -0.30).

### 2.2. DISTRIBUTION frontera (5/10)

| Ticker | MA50 | MA200 | MA50 vs 200 | Precio | struct | t_norm | Legacy |
|---|---:|---:|---|---:|---:|---:|---|
| BKNG | 189.5 | 182.1 | +4.1% | 160.3 | -0.142 | +0.161 | RANGE |
| O | 60.4 | 60.4 | ~0% | 53.5 | -0.117 | -0.005 | RANGE |
| SLB | 52.7 | 50.0 | +5.4% | 48.7 | -0.113 | +0.214 | ACCUMULATION |
| VMRK | 65.1 | 63.3 | +2.7% | 60.2 | -0.155 | +0.109 | DISTRIBUTION |
| WFC | 86.2 | 84.1 | +2.5% | 80.3 | -0.109 | +0.098 | ACCUMULATION |

En estos 5, MA50 > MA200 (tendencia estructural positiva) pero precio
bajo AMBAS medias. El `struct` esta apenas por debajo del umbral -0.10.

**Ambiguedad semantica:** con MA50 > MA200, podria ser:
- DISTRIBUTION (techo tras subida, precio cae antes del cruce lento).
- ACCUMULATION (correccion dentro de tendencia alcista, base previa).

El contrato v1.5 no distingue ambos. Decide DISTRIBUTION por `struct` <
-0.10 + `struct_max` > 0.30 + `t_norm` > -0.30.

---

## 3. Interpretacion

### 3.1. No es bug del contrato

Los 10 casos cumplen el contrato v1.5 literalmente:
- struct_score_t < -0.10.
- t_norm_t > -0.30.
- struct_max > 0.30.

El test I26 lo verifica con mock.

### 3.2. Frontera conocida del modelo

Hay dos fases "no direccionales" (ACCUMULATION y DISTRIBUTION) que
comparten el mismo espacio estructural (compresion + precedente).
El clasificador las separa por el signo de `struct`:
- ACCUMULATION: base (struct en banda -0.20 a +0.20).
- DISTRIBUTION: deterioro (struct < -0.10).

Los 5 frontera estan justo entre ambas: struct en (-0.16, -0.10), con
MA50 > MA200. El clasificador decide DISTRIBUTION porque struct < -0.10.

### 3.3. El legacy no aclara

Legacy clasifica:
- RANGE (BKNG, O): no cumple ninguna banda legacy.
- ACCUMULATION (SLB, WFC): legacy ve base.
- DISTRIBUTION (VMRK): coincide.

Pero el legacy media aceleracion, no nivel. Su opinion no es autoridad.

---

## 4. Recomendacion

**No cambiar el contrato.** Los 10 casos son defendibles:

- Los 5 claros estan confirmados.
- Los 5 frontera cumplen literalmente el contrato y la semantica
  (hubo fuerza, hay deterioro, no es MARKDOWN aun).

**Preguntas al auditor:**

1. ¿Aceptas los 5 casos frontera como DISTRIBUTION por diseno?
2. ¿O prefieres endurecer §5.4 con un criterio adicional, por ejemplo
   `MA50 < MA200` o `t_norm < 0.30`?
3. Si se endurece, ¿se convierte en v1.6?
4. ¿O se aceptan tal cual y se procede a 5d?

---

## 5. Detalle tecnico de los 5 frontera

### BKNG

    close=160.28 MA50=189.53 MA200=182.13
    trend=+0.0406 t_norm=+0.1610
    struct=-0.1421 c_norm=-0.5968
    precedente: s_max=+0.368 s_min=-0.485
    Legacy: RANGE

Precio 16% por debajo de MA50. Correccion severa con MA50 aun > MA200.
DISTRIBUTION defendible. ACCUMULATION tambien.

### O (Realty Income)

    close=53.53 MA50=60.35 MA200=60.44
    trend=-0.0014 t_norm=-0.0055
    struct=-0.1165 c_norm=-0.2830
    precedente: s_max=+0.474 s_min=-0.299
    Legacy: RANGE

MA50 ≈ MA200 (diferencia 0.1%). Tendencia practicamente plana. Precio
11% por debajo. DISTRIBUTION defendible (perdida de fuerza), RANGE
tambien.

### SLB

    close=48.66 MA50=52.73 MA200=50.01
    trend=+0.0544 t_norm=+0.2142
    struct=-0.1128 c_norm=-0.6032
    precedente: s_max=+0.679 s_min=-0.265
    Legacy: ACCUMULATION

MA50 > MA200 (+5.4%). Precio 7.7% por debajo de MA50. **Caso mas
dudoso.** El struct -0.1128 esta a 0.0128 del umbral -0.10. Correccion
dentro de tendencia alcista. Legacy ve base.

### VMRK

    close=60.17 MA50=65.07 MA200=63.33
    trend=+0.0273 t_norm=+0.1089
    struct=-0.1552 c_norm=-0.5514
    Legacy: DISTRIBUTION

MA50 > MA200 (+2.7%). Precio 7.5% por debajo de MA50. Coincide con
legacy en DISTRIBUTION. Menos dudoso.

### WFC

    close=80.25 MA50=86.18 MA200=84.10
    trend=+0.0247 t_norm=+0.0984
    struct=-0.1085 c_norm=-0.4190
    Legacy: ACCUMULATION

MA50 > MA200 (+2.5%). Precio 6.9% por debajo de MA50. **Caso muy
dudoso.** struct -0.1085 a 0.0085 del umbral. Legacy ve base.

---

Fin de la verificacion.
