# REVISION DEL CONTRATO WYCKOFF - v1.1 -> v1.2

**Documento de auditoria. Justifica el cambio v1.1 -> v1.2.**
**Fecha:** 2026-10-02.
**Aprobacion:** dictamen auditor externo (segunda ronda v1.2).

---

## 1. Contexto

Implementacion de v1.1 probada sobre 4 tickers (MSFT, PLTR, INTC, AMD).
Los 4 devolvieron `RANGE`. El analisis de precedente estructural
identifico **dos defectos semanticos del contrato v1.1** (no de la
implementacion).

**Valores observados:**

| Ticker | `t_last` | `c_last` | `s_last` | `s_min` | `s_max` |
|---|---:|---:|---:|---:|---:|
| MSFT | +0.810 | +0.410 | +0.650 | -0.468 | +0.695 |
| PLTR | +0.458 | +0.957 | +0.658 | -0.505 | +0.629 |
| INTC | -0.379 | +0.287 | -0.113 | -0.163 | +0.199 |
| AMD  | -0.024 | +0.546 | +0.204 | +0.031 | +0.492 |

---

## 2. Defecto D1-v1.1 - MARKUP penaliza la persistencia

**Definicion v1.1 (§5.3):** MARKUP exige `struct_max < PREC_STRUCT_STRONG`
(0.30). Es decir: "para estar en MARKUP hoy, no debiste estar fuerte los
ultimos 60 dias".

**Contradiccion semantica:** un estado direccional sostenido (bull market)
produce `struct_max` crecientes. La condicion **penaliza la persistencia
de la propia tendencia que se pretende clasificar**.

**Caso ilustrativo:** MSFT. `t=+0.81`, `s=+0.65` (direccion y estructura
alcistas claras). `s_max=+0.695` -> falla `s_max < 0.30` -> RANGE.

**Referencia externa:** Wyckoff describe Phase E (markup) como el
desarrollo del movimiento alcista una vez el precio sale del trading
range. Pueden aparecer re-acumulaciones posteriores; la persistencia no
invalida el estado.

**Correccion v1.2:** eliminar el precedente como condicion de MARKUP.

---

## 3. Defecto D2-v1.1 - Precedente indiscriminado

**Definicion v1.1:** precedente exigido en ACCUMULATION, MARKUP,
DISTRIBUTION, MARKDOWN. Tratamiento uniforme.

**Defecto:** **MARKUP y MARKDOWN son estados direccionales actuales.**
No son transiciones desde otro estado. ACCUMULATION y DISTRIBUTION **si
son transiciones** (base tras caida; techo tras subida) y por definicion
requieren contexto historico.

**Correccion v1.2:** precedente **selectivo por semantica**:

    ACCUMULATION   -> requiere precedente (agotamiento bajista)
    DISTRIBUTION   -> requiere precedente (agotamiento alcista)
    MARKUP         -> no requiere precedente
    MARKDOWN       -> no requiere precedente
    RANGE          -> estado residual (sin precedente)
    INSUFFICIENT_DATA -> estado de datos (sin precedente)

---

## 4. Preciso del dictamen: MARKUP/MARKDOWN no son "direccion instantanea"

El dictamen del auditor advierte correctamente: eliminar precedente de
MARKUP **no significa** definirlo como:

    MARKUP = (t_norm > X)

Eso degradaria el modulo a un clasificador de tendencia.

**Formulacion aprobada v1.2:**

    MARKUP:
        estructura alcista actual
        + evidencia suficiente de expansion/direccion

    MARKDOWN:
        estructura bajista actual
        + evidencia suficiente de deterioro/expansion bajista

Traducido a variables:

    MARKUP:
        struct_score_t > STRUCT_STRONG
        AND t_norm_t > T_NORM_STRONG
        (sin veto por c_norm; compresion y expansion son ambas
        compatibles con markup ordenado o volatil)

    MARKDOWN:
        struct_score_t < STRUCT_WEAK
        AND t_norm_t < -T_NORM_STRONG
        AND c_norm_t < 0
        (expansion requerida: markdown es movimiento direccional
        tras ruptura, no compresion)

El `c_norm < 0` de MARKDOWN **no es espejo simetrico** de un
`c_norm > 0` en MARKUP. Razon: la microestructura de un movimiento
bajista confirmado suele implicar expansion de volatilidad (el mercado
deja de comprimir); en cambio un markup ordenado puede coexistir con
compresion (subida sostenida de baja volatilidad). Asimetria justificada
por evidencia externa, no por simetria matematica.

---

## 5. Defecto D3-v1.1 - `tact > 0` como requisito de ACCUMULATION

**Definicion v1.1:** ACCUMULATION exige `tact > 0` obligatorio.

**Defecto:** la acumulacion es un proceso de construccion dentro de un
trading range (PS, SC, ST, spring, SOS, LPS). No existe regla que
convierta un tactical score diario positivo en condicion necesaria de
existencia de acumulacion.

**Correccion v1.2:** `tact > 0` **opcional**. Puede aumentar confianza,
pero no bloquea la fase.

---

## 6. Regla de no-look-ahead

**Invariante I11 (reforzada v1.2):** el computo del precedente en `t`
usa exclusivamente observaciones `<= t`. **Nunca** `t+1 ... t+N`.

**Invariante I12 (reforzada v1.2):** extender el input con datos
posteriores a `t` no cambia la fase en `t`.

Estas reglas se anaden al contrato §4 y se verifican con tests.

---

## 7. Sobre los 4 casos

**Regla de auditoria (dictamen):** los 4 casos **no son objetivos de
calibracion**. Sirven para detectar contradicciones del contrato. Un
resultado `RANGE` para INTC/AMD es legitimo aunque parezca "casi
acumulacion".

**Cambios NO aprobados:**
- Mover umbrales para forzar ACCUMULATION en INTC o AMD.
- Ajustar `STRUCT_BASE_HIGH` de 0.20 a 0.25 para capturar AMD.
- Reducir `C_NORM_COMPRESSION` de 0.30 a 0.28 para capturar INTC.

**Cambios SI aprobados:**
- Quitar precedente de MARKUP y MARKDOWN.
- Explicitar la asimetria `c_norm` entre MARKUP y MARKDOWN.

**Verificacion esperada tras v1.2:**

| Ticker | v1.2 esperado |
|---|---|
| MSFT | MARKUP (t=+0.81, s=+0.65) |
| PLTR | MARKUP (t=+0.46, s=+0.66) |
| INTC | RANGE (c=0.287 < 0.30; s=-0.113 no MARKDOWN) |
| AMD  | RANGE (s=+0.204 > banda base 0.20; no MARKUP) |

**No es una afirmacion de "verdad de mercado".** Es una verificacion de
que v1.2 elimina las 3 contradicciones de v1/v1.1 **sin relajar umbrales**.

---

## 8. Invariantes y tests anadidos

**Nuevas invariantes:**

- I15: MARKUP no requiere precedente.
- I16: MARKDOWN no requiere precedente.
- I17: `tact_score` no veta fase estructural.
- I18: precedente solo con datos `<= t` (reforzamiento de I11).

**Tests contractuales obligatorios (dictamen §10):**

- Test 1: mas historia alcista -> sigue siendo MARKUP.
- Test 2: tactical negativo + estructura MARKUP valida -> MARKUP.
- Test 3: tactical negativo + condiciones estructurales de ACCUMULATION -> ACCUMULATION.
- Test 4: modificacion de datos futuros no cambia fase en t.

---

## 9. Cambios resumidos v1.1 -> v1.2

| Aspecto | v1.1 | v1.2 |
|---|---|---|
| MARKUP precedente | `struct_max < 0.30` | eliminado |
| MARKDOWN precedente | `struct_min > -0.30` | eliminado |
| MARKUP `c_norm` | (no exigido) | (no exigido; ambos regimenes validos) |
| MARKDOWN `c_norm` | `< 0` | `< 0` (justificado: expansion direccional) |
| ACCUMULATION `tact > 0` | opcional | opcional (sin cambio) |
| INVARIANTES | I1-I14 | I1-I18 |
| Tests contractuales | 16 + 3 skipped | +4 obligatorios |

---

Fin de la revision v1.1 -> v1.2.
