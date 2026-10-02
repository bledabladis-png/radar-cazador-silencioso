# REVISION DEL CONTRATO WYCKOFF - v1.2 -> v1.3

**Documento de auditoria. Justifica el cambio v1.2 -> v1.3.**
**Fecha:** 2026-10-02.
**Aprobacion:** dictamen auditor externo P1 (critico, confirmado).

---

## 1. Contexto

El hallazgo P1 (docs/auditoria/wyckoff/06_hallazgo_t_norm_v1_2.md)
identifico que `t_norm` mide **desviacion del regimen de tendencia**,
no **nivel de tendencia**. El dictamen lo eleva a criticos y bloquea
las fases 5b, 5c y 5d.

La correccion estructural es cambiar `t_norm` de:

    t_norm = tanh(robust_zscore(trend, window=200, min_periods=60))

a:

    t_norm = tanh(trend / K)

con `K` como parametro explicito (PROPUESTO hasta calibracion 5b).

---

## 2. Diagnostico del defecto (resumen)

**Evidencia empirica (3 series deterministas):**

| Serie | `trend_ult` | `t_norm` v1.2 | Semantica v1.2 |
|---|---:|---:|---|
| Subida constante +15.5% | +0.155 | +0.001 | No MARKUP |
| Subida acelerada | +0.429 | +0.592 | MARKUP |
| Subida desacelerada +13% | +0.129 | -0.580 | Casi MARKDOWN |

**Causa:** `robust_zscore(trend)` compara el nivel actual con su mediana
rolling. Con `trend` constante, mediana = trend y z = 0. Con `trend`
creciente, z > 0. Con `trend` decreciente, z < 0.

**Consecuencia:** la metrica mide **cambio de regimen**, no **nivel**.
Incompatible con la semantica Wyckoff de MARKUP (movimiento sostenido).

---

## 3. Correccion aprobada

### 3.1. Nueva definicion de t_norm

    trend = MA50 / MA200 - 1
    t_norm = tanh(trend / K)

**Semantica:**

    trend > 0  -> t_norm > 0
    trend = 0  -> t_norm = 0
    trend < 0  -> t_norm < 0

La magnitud es interpretable: `t_norm = 0.30` equivale a
`trend = K * atanh(0.30) ≈ 0.3095 * K`.

Con K=0.15 (PROPUESTO): umbral 0.30 <-> trend ≈ 4.64%.

### 3.2. K como parametro explicito

Introducido en `config/settings.py` como `WYCKOFF_T_NORM_K`.

**Valor PROPUESTO inicial:** `0.15`.

**Razon del valor:** un trend del +15% (tendencia fuerte en equity)
produce `t_norm = tanh(1.0) = 0.76`. El umbral 0.30 se alcanza con
trend ≈ 4.64%. Marcado como PROPUESTO en `02_proveniencia`.

**Prohibicion:** K no se calibra sobre los 4 casos diagnosticos
(MSFT, PLTR, INTC, AMD). Si se calibra, sera en Fase 5b con dataset
controlado.

### 3.3. Eliminacion de `robust_zscore` para `trend`

`robust_zscore` deja de aplicarse a `trend`. Se mantiene para:

- `compression` (`c_norm`).
- `volume` (`v_norm`).
- `price_move` en `effort_vs_result` (`result`).

**Razon:** en esos casos el z-score mide desviacion respecto a un
regimen, que es lo que se pretende. En `trend` la desviacion no es la
propiedad que el contrato quiere medir.

### 3.4. Separacion conceptual obligatoria

    TREND LEVEL       -> t_norm (nuevo)
    TREND CHANGE      -> NO en este modulo (opcional futuro)
    TACTICAL          -> confirmacion / WLS

El concepto anterior (desviacion respecto al regimen de tendencia)
puede recuperarse en el futuro como senal separada. **No en v1.3.**

---

## 4. Consecuencias

### 4.1. Coherencia de struct_score

    struct_score = 0.60 * t_norm + 0.40 * c_norm

Ahora `t_norm` = nivel de tendencia; `c_norm` = compresion relativa.
La combinacion es coherente: "estructura = tendencia + compresion".

### 4.2. WLS historicos: NO comparables

**Aviso del auditor:** los valores historicos de WLS derivados de
`wyckoff_score` (via `rws_z`) no deben interpretarse como si hubieran
sido producidos por la nueva semantica. Cualquier backtest, IC o
comparacion IS/OOS debe marcar la serie historica como **no
comparable** con la nueva.

### 4.3. Legacy afectado

`indicators/wyckoff.py` (legacy) usa la misma formula con `window=60`.
**Es la misma deuda estructural historica.** No es regresion de v1.2.

Registro correcto:

    causa = diseno heredado
    deteccion = validacion contractual de v1.2

### 4.4. Consumidores bloqueados

Los 5 consumidores NO se migran hasta cerrar v1.3. Fases 5c y 5d
bloqueadas.

---

## 5. Invariantes nuevas

**I19.** `trend > 0 -> t_norm > 0`. `trend < 0 -> t_norm < 0`.
`trend = 0 -> t_norm = 0`.

**I20.** `t_norm` acotado en (-1, 1).

**I21.** Aceleracion no invierte el signo: si `trend > 0`, `t_norm > 0`
independientemente de la pendiente de `trend`.

**I22.** T1 (subida sostenida -> MARKUP) es invariante contractual
permanente.

**I23.** Sin look-ahead en la nueva clasificacion.

---

## 6. Tests obligatorios

Nuevos (dictamen §14):

- `test_constant_uptrend_is_positive`
- `test_constant_downtrend_is_negative`
- `test_flat_trend_is_zero`
- `test_acceleration_does_not_define_direction`
- `test_markup_not_rejected_by_stable_trend` (T1, sin xfail)
- `test_future_data_no_effect` (ya existe T4)
- `test_t_norm_bounds`

Los existentes I1-I18 se mantienen. T1 pasa de xfail a verde
permanente.

---

## 7. Cambios resumidos v1.2 -> v1.3

| Aspecto | v1.2 | v1.3 |
|---|---|---|
| `t_norm` definicion | `tanh(robust_zscore(trend, w=200, mp=60))` | `tanh(trend / K)` |
| `t_norm` semantica | desviacion respecto a regimen | nivel de tendencia escalado |
| K | no existia | parametro explicito, PROPUESTO = 0.15 |
| `robust_zscore(trend)` | usado | eliminado |
| WLS historicos | (implicitos) | marcados como no comparables |
| Invariantes | I1-I18 | I1-I23 |
| Tests | 20 passed + 1 xfail | +7 tests, T1 verde permanente |
| Fases 5c/5d | bloqueadas | siguen bloqueadas |
| Legacy | intacto | marcado como afectado historicamente |

---

## 8. Estado del proyecto

- Fases 1-3: cerradas.
- Fase 4 (tests): ampliada con 7 tests nuevos para v1.3.
- Fase 5a (modulo v1.3): proximo paso (codigo) tras cerrar este doc.
- Fase 5b (calibracion K): bloqueada hasta contrato v1.3 firme.
- Fase 5c/5d: bloqueadas hasta v1.3 completo + calibracion.

---

Fin de la revision v1.2 -> v1.3.
