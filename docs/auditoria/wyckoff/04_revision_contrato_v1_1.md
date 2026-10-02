# REVISION DEL CONTRATO WYCKOFF - v1 -> v1.1

**Documento de auditoria. Justifica el cambio de contrato v1 a v1.1.**
**Fecha:** 2026-10-02.
**Precedente:** implementacion inicial de v1 en `indicators/wyckoff_v1.py`
y prueba sobre 4 tickers (MSFT, PLTR, INTC, AMD).

---

## 1. Contexto

El contrato v1 fue implementado en `indicators/wyckoff_v1.py` y probado
sobre 4 tickers del universo USA. Los 4 resultaron clasificados como
`RANGE` pese a tener caracteristicas estructurales claramente distintas.

**Tabla de resultados:**

| Ticker | `t_norm` | `c_norm` | `struct` | `tact` | `combined` | Fase v1 |
|---|---:|---:|---:|---:|---:|---|
| MSFT | +0.81 | +0.41 | (positivo) | -0.40 | +0.33 | RANGE |
| PLTR | +0.46 | +0.96 | (positivo) | -0.72 | +0.25 | RANGE |
| INTC | -0.38 | +0.29 | (negativo) | -0.39 | -0.20 | RANGE |
| AMD | -0.02 | +0.55 | (neutro) | -0.73 | -0.08 | RANGE |

**No es un fallo de la implementacion.** Es un fallo del contrato v1,
como se demuestra en §2-§4.

---

## 2. Defecto D1 - MARKUP condicionado por `c_norm < 0.30`

**Definicion v1 (§5.3):**

    MARKUP: t_norm > 0.30 AND c_norm < 0.30 AND combined > 0.30

**Defecto:** `c_norm = -tanh(z(ATR/Close))`. Compresion alta (volatilidad
baja) produce `c_norm` positivo. Un bull market ordenado tiene tipicamente
baja volatilidad relativa. Por tanto:

    tendencia alcista + compresion alta

es compatible con MARKUP, no lo excluye. La condicion `c_norm < 0.30`
esta mal alineada con la semantica de `c_norm`.

**Caso ilustrativo:** MSFT con `t_norm = +0.81`, `c_norm = +0.41`. La
condicion MARKUP exige `c_norm < 0.30` -> falla -> cae a RANGE.

**Correccion v1.1:** eliminar `c_norm < 0.30` del contrato MARKUP.

---

## 3. Defecto D2 - MARKUP vetado por `combined`

**Definicion v1 (§5.3):** `combined > 0.30` como condicion adicional.

**Defecto:** `combined = 0.70 * struct + 0.30 * tact`. El tactico pesa
30%. Una estructura alcista valida con tactico bajo (volumen reciente
escaso) puede caer por debajo del umbral `combined > 0.30` pese a tener
`struct` fuerte.

**Caso ilustrativo:** PLTR con `t_norm = +0.46`, `c_norm = +0.96`
(estructura alcista clara) pero `tact = -0.72`. `combined = +0.25` ->
falla la condicion -> RANGE.

**Correccion v1.1:** la clasificacion de fase usa `struct_score` y
precedente estructural. `tact_score` actua como confirmacion, no como
veto estructural.

---

## 4. Defecto D3 - ACCUMULATION exige `tact > 0`

**Definicion v1 (§5.2):** `tact > 0` como condicion de confirmacion.

**Defecto:** la acumulacion es un proceso silencioso. El volumen
aparece al romper la base, no durante. Exigir tactico positivo excluye
el propio momento de acumulacion.

**Caso ilustrativo:** INTC (`tact = -0.39`), AMD (`tact = -0.73`).
Ambos quedan excluidos de ACCUMULATION por esta condicion.

**Correccion v1.1:** `tact > 0` pasa a ser condicion **opcional** de
confirmacion. No es requisito obligatorio.

---

## 5. Defecto estructural (mas profundo) - ausencia de semantica temporal

**Problema:** el clasificador v1 mira solo el ultimo valor de
`struct_score` y `combined`. Sin memoria, no puede distinguir:

- ACCUMULATION (base tras caida).
- RANGE en caida lateral.
- MARKDOWN ordenado.

**Todos tienen `t_norm` debil o negativo con `c_norm` positivo.**

**Correccion v1.1:** maquina de estados con **precedente estructural
historico**. La fase en `t` depende no solo de las variables en `t`
sino de la trayectoria estructural reciente.

---

## 6. Como NO implementar el precedente (dictamen del auditor)

**Rechazado:**

    phase_now = classify(df_t)
    phase_prev = classify(df_{t-N})
    if phase_prev == MARKDOWN and ...:
        return ACCUMULATION

Razones:

1. **Circularidad:** la fase en `t` depende de la fase en `t-N`, que a
   su vez depende de la fase en `t-2N`, etc.
2. **Dependencia de umbrales:** cambiar un umbral historico puede
   cambiar la etiqueta precedente aunque los datos no cambien.
3. **Memoria de decision, no de mercado:** la etiqueta precedente es
   una decision pasada del clasificador, no un hecho del mercado.

---

## 7. Como SI implementar el precedente (v1.1)

**Precedente estructural = agregacion de variables estructurales
historicas.**

Definicion conceptual:

    window = [t-N, t-1]                          (N por definir en Fase 3)
    struct_window = struct_score[window]
    t_norm_window = t_norm[window]

    struct_min  = min(struct_window)
    struct_max  = max(struct_window)
    struct_mean = mean(struct_window)
    t_norm_min  = min(t_norm_window)
    t_norm_max  = max(t_norm_window)

Estas son las **variables de precedente**. Son continuas, se calculan
sobre datos <= t, y no dependen de ninguna clasificacion previa.

**Invariante de no-look-ahead:** todos los calculos en `t` usan
exclusivamente informacion `<= t`.

**Invariante de reordenacion:** el resultado para `t` es identico si el
historial se trunca a `[0, t]` o si se extiende con datos posteriores
`> t`.

---

## 8. Maquina de estados v1.1

La fase en `t` es funcion de:

    (struct_score_t, t_norm_t, c_norm_t, precedente_estructural)

**Diagrama conceptual:**

    ACCUMULATION -> MARKUP -> DISTRIBUTION -> MARKDOWN -> ACCUMULATION

**Transiciones permitidas** (informativo, no validado en codigo):

    ACCUMULATION -> MARKUP       (confirmacion alcista)
    ACCUMULATION -> RANGE        (base fallida)
    MARKUP       -> DISTRIBUTION (techo)
    MARKUP       -> RANGE        (perdida de fuerza)
    DISTRIBUTION -> MARKDOWN     (confirmacion bajista)
    DISTRIBUTION -> RANGE        (soporte inesperado)
    MARKDOWN     -> ACCUMULATION (reversion)
    MARKDOWN     -> RANGE        (agotamiento)
    RANGE        -> cualquier    (fase transitoria)

**Transicion prohibida por continuidad:**

    MARKUP -> ACCUMULATION  (requiere DISTRIBUTION/RANGE intermedio)

Esta restriccion se evalua por precedente estructural, no por etiqueta
previa.

---

## 9. Separacion estructural / tactico

**Contrato v1.1:**

    STRUCTURAL  -> determina la fase
    TACTICAL    -> confirma / contextualiza / eventos

Consecuencias:

- `tact_score` no veta transiciones estructurales.
- `detect_spring` / `detect_sos` son informativos.
- La calidad del tactico (P3) es problema de calibracion, no de
  contrato. **P3 no bloquea P4.**

---

## 10. Invariantes anadidas

**I11. Sin look-ahead en precedente.**
La fase en `t` solo usa datos `<= t`. Test: truncar el input a `[0, t]`
y verificar misma salida en `t`.

**I12. Independencia de datos futuros.**
El resultado para `t` no cambia si el input incluye datos `> t`. Test:
extender input con ruido futuro y verificar misma salida en `t`.

**I13. No circularidad.**
El precedente estructural se calcula sobre variables continuas, no
sobre etiquetas de fase previas. Test: verificar que no hay llamada
recursiva a `classify_wyckoff_phase` dentro del precedente.

**I14. Continuidad de fase (informativa, no obligatoria).**
Las transiciones prohibidas por §8 se detectan como advertencia, no
como excepcion. Un cambio directo MARKUP -> ACCUMULATION se registra
pero no bloquea.

---

## 11. Cambios resumidos v1 -> v1.1

| Aspecto | v1 | v1.1 |
|---|---|---|
| MARKUP cond. 2 | `c_norm < 0.30` | Eliminada |
| MARKUP cond. 3 | `combined > 0.30` | `struct_score > umbral` (por calibrar) |
| ACCUMULATION cond. 4 | `tact > 0` obligatorio | Opcional (informacion) |
| Precedente | No existe | Variable estructural historica |
| Fase de t | Solo valor de t | Valor de t + precedente |
| tact en clasificacion | Veta | No veta |
| Invariantes | I1-I10 | I1-I14 |
| N (ventana precedente) | N/A | Definido en Fase 3 |

---

## 12. Lo que NO cambia

- **Frontera arquitectonica:** wyckoff.py calcula; consumidores
  interpretan.
- **Pesos de composicion:** 0.60/0.40, 0.50/0.50, 0.70/0.30 (por
  ahora; recalibracion en Fase 5+).
- **Definicion de effort_vs_result:** forma clasica (effort - result).
- **stability:** dispersion pura (1 - 2*tanh(mad/K)).
- **INSUFFICIENT_DATA:** nunca devuelve fase de mercado.
- **Determinismo:** mismo input, mismo output.

---

## 13. Estado del contrato

**v1.0:** superseded por v1.1.
**v1.1:** activo tras aprobacion de este documento.

**Archivos:**
- `01_contrato_semantico_v1.md` (archivado, historico).
- `01_contrato_semantico_v1_1.md` (activo).

---

Fin de la revision.
