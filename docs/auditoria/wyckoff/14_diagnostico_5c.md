# DIAGNOSTICO 5c - H1/H2/H3 resueltos

**Documento de auditoria. Dictamen externo 2026-10-02.**
**Fecha:** 2026-10-02.

---

## 1. H1 - DISTRIBUTION inalcanzable

**Confirmado matematicamente por el auditor.** Las condiciones
§5.4 del contrato v1.3:

    struct_score_t < -0.10
    AND (t_norm_t > 0 OR |t_norm_t| < 0.30)
    AND c_norm_t > 0.30
    AND struct_max > 0.30

son **algebraicamente imposibles**. Demostracion del dictamen:

Rama 1 (t > 0, c > 0.30): struct > 0.12, incompatible con struct < -0.10.
Rama 2 (|t| < 0.30, c > 0.30): struct > -0.06, incompatible con struct < -0.10.

**No es un estado raro; es un estado inalcanzable.**

### Correccion v1.4

Sustituir `c_norm_t > 0.30` por `c_norm_t > -0.10`.

Verificacion de alcanzabilidad (peor caso Rama 2):
- t_norm = -0.30, c_norm = -0.10.
- struct = 0.60*(-0.30) + 0.40*(-0.10) = -0.18 - 0.04 = **-0.22**.
- cumple struct < -0.10 ✓
- cumple c_norm > -0.10 ✓
- cumple |t_norm| < 0.30 ✓

**Ahora DISTRIBUTION es alcanzable.** El umbral -0.10 representa
"compresion no extrema". No es tan laxo como -0.50 (expansion fuerte).

Nueva definicion §5.4 v1.4:

    struct_score_t < -0.10
    AND (t_norm_t > 0 OR |t_norm_t| < 0.30)
    AND c_norm_t > -0.10
    AND struct_max > 0.30

Test obligatorio: `test_distribution_is_reachable` que construye
sinteticamente un caso que cumpla las 4 condiciones.

---

## 2. H2 - RANGE al 66.8%

**RANGE genuino, no artificial.**

Analisis de causas sobre los 211 tickers en RANGE:

### Causas de no-MARKUP

| Causa | Casos | Near-miss (0.10 del umbral) |
|---|---:|---:|
| t_norm <= 0.30 | 173 | 42 (con t en 0.20-0.30) |
| struct <= 0.30 | 165 | 57 (con struct en 0.20-0.30) |

### Causas de no-ACCUMULATION

| Causa | Casos | Near-miss (0.05 del umbral) |
|---|---:|---:|
| fuera de banda de base | 122 | 34 |
| sin compresion (c<=0.30) | 124 | 5 |
| sin precedente debil | 99 | 17 |
| trend no weak | 38 | — |

**Lectura:** el RANGE viene de causas repartidas. No hay un unico
cuello de botella dominante. La causa principal es "sin compresion"
(124) y "fuera de banda" (122), seguidas de "sin precedente" (99).

**Conclusion:** el contrato v1.3 es estricto pero coherente con la
semantica. RANGE al 67% es el precio de exigir conjunciones
estructurales. **No se modifica nada por H2.**

Regla aplicada: no ajustar umbrales por apariencia. Sin diagnostico que
demuestre defecto, se mantienen.

---

## 3. H3 - Verificacion manual (5 casos)

### CHTR

    close:      111.37
    MA50:       140.60
    MA200:      175.39
    trend:      -0.1984  (MA50/MA200-1 = -19.8%)
    t_norm:     -0.6604
    struct:     -0.6128
    c_norm:     -0.5415
    precedente: s_min=-0.890 s_max=-0.550 s_mean=-0.732
    FASE v1.3:  MARKDOWN
    FASE legacy:MARKUP (score +0.3764)

**Veredicto:** CHTR esta en tendencia bajista clara. Precio 111 esta
por debajo de MA50 (140) y muy por debajo de MA200 (175). El legacy
media aceleracion (la caida se estaba desacelerando) y la interpretaba
como alcista. **v1.3 tiene razon estructural.**

### PM

    close:      188.18
    MA50:       190.06
    MA200:      176.29
    trend:      +0.0781  (+7.8%)
    t_norm:     +0.3026
    struct:     +0.3915
    c_norm:     +0.5249
    precedente: s_min=-0.125 s_max=+0.522 s_mean=+0.225
    FASE v1.3:  MARKUP
    FASE legacy:DISTRIBUTION (score -0.3321)

**Veredicto:** PM tiene MA50 por encima de MA200 (+7.8%). Tendencia
alcista confirmada, aunque el precio esta ligeramente por debajo de
MA50 (freno reciente). El legacy veia "desaceleracion fuerte" y lo
clasificaba DISTRIBUTION. **v1.3 tiene razon estructural:** es MARKUP
con freno, no distribucion.

### MBG.DE

    close:      40.06
    MA50:       45.47
    MA200:      50.95
    trend:      -0.1075  (-10.8%)
    t_norm:     -0.4053
    struct:     -0.4732
    c_norm:     -0.5750
    FASE v1.3:  MARKDOWN
    FASE legacy:ACCUMULATION

**Veredicto:** MBG en tendencia bajista. **v1.3 correcto.**

### MCD

    close:      231.83
    MA50:       258.80
    MA200:      286.75
    trend:      -0.0975  (-9.8%)
    t_norm:     -0.3713
    struct:     -0.4302
    c_norm:     -0.5186
    FASE v1.3:  MARKDOWN
    FASE legacy:ACCUMULATION

**Veredicto:** MCD claramente bajista. **v1.3 correcto.**

### AZO

    close:      2819.03
    MA50:       2966.28
    MA200:      3300.17
    trend:      -0.1012  (-10.1%)
    t_norm:     -0.3840
    struct:     -0.3298
    c_norm:     -0.2486
    FASE v1.3:  MARKDOWN
    FASE legacy:ACCUMULATION

**Veredicto:** AZO bajista claro. **v1.3 correcto.**

### Conclusion H3

**Los 5 casos dan razon a v1.3.** En los 5, el legacy medía
aceleracion (desviacion del regimen) y clasificaba activos bajistas
como alcistas porque la caida/desaceleracion se estaba frenando.

**Confirmado el defecto P1 en produccion sobre el universo real.**
v1.3 lo corrige.

---

## 4. Delta 41% con |delta| > 0.30

**Aceptable.** El auditor lo confirma: es consecuencia esperada del
cambio semantico de `t_norm`. No se usa como criterio de aceptacion.

---

## 5. Decision v1.4

Unico cambio aprobado: §5.4 DISTRIBUTION.

- `c_norm_t > 0.30` -> `c_norm_t > -0.10`.
- Test nuevo obligatorio: `test_distribution_is_reachable`.
- Test nuevo obligatorio: `test_distribution_algebraic_conditions`.

Resto del contrato v1.3 sin cambios. K=0.25 congelado. t_norm v1.3.

---

## 6. Estado

- Fase 5c: CERRADA (H1, H2, H3 resueltos).
- v1.4: por implementar (fix DISTRIBUTION).
- 5d: sigue bloqueada hasta cierre v1.4 + re-ejecucion 5c.2.

---

Fin del diagnostico 5c.
