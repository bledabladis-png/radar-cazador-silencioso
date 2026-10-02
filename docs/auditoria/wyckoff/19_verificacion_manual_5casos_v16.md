# VERIFICACION MANUAL v1.6 - 5 casos (dictamen 5c.3)

**Documento de auditoria. Fase 5c.3, verificacion manual.**
**Fecha:** 2026-10-02.

---

## 1. Muestra (segun dictamen)

| Caso | Categoria | Resultado v1.6 |
|---|---|---|
| ACS.MC | claro -> cae | RANGE (candidate=True) |
| ANA.MC | claro -> cae | RANGE (candidate=True) |
| BA | claro -> pasa | DISTRIBUTION |
| PCG | claro -> pasa | DISTRIBUTION |
| SLB | frontera -> pasa | DISTRIBUTION |

---

## 2. Detalle por caso

### 2.1. ACS.MC (claro, cae a RANGE)

    close=92.05  MA50=102.40  MA200=109.59
    precio vs MA50: -10.11%   vs MA200: -16.00%
    struct_t=-0.2866   t_norm_t=-0.2565
    struct_max=+0.4844
    ultimo SOW: 2026-09-14
    distancia (sesiones): -13
    SOW: close=91.85 < support=96.30  (-4.62%)
    volumen SOW: 1.63x baseline

**Diagnostico:** candidato vigente, SOW fuerte (caida -4.62%, volumen
1.63x) pero a **13 sesiones** de distancia. Fuera de M=10.

**Si M>=15:** pasaria a DISTRIBUTION.

### 2.2. ANA.MC (claro, cae a RANGE)

    close=211.60  MA50=214.32  MA200=223.25
    precio vs MA50: -1.27%   vs MA200: -5.22%
    struct_t=-0.1387   t_norm_t=-0.1585
    struct_max=+0.4539
    ultimo SOW: 2026-08-04
    distancia (sesiones): -42
    SOW: close=214.60 < support=227.40  (-5.63%)
    volumen SOW: 3.66x baseline

**Diagnostico:** candidato vigente, SOW fuerte (caida -5.63%, volumen
3.66x) pero a **42 sesiones** de distancia. Fuera de M=10.

**Si M>=45:** pasaria a DISTRIBUTION.

### 2.3. BA (claro, pasa)

    close=192.28  MA50=212.64  MA200=220.81
    precio vs MA50: -9.58%   vs MA200: -12.92%
    struct_t=-0.2709   t_norm_t=-0.1469
    struct_max=+0.3593
    ultimo SOW: 2026-09-28
    distancia (sesiones): -3
    SOW: close=184.39 < support=194.40  (-5.15%)
    volumen SOW: 2.95x baseline

**Diagnostico:** candidato vigente, SOW fuerte reciente. DISTRIBUTION
correcto.

### 2.4. PCG (claro, pasa, marginal)

    close=12.17  MA50=15.52  MA200=16.42
    precio vs MA50: -21.56%   vs MA200: -25.90%
    struct_t=-0.5304   t_norm_t=-0.2179
    struct_max=+0.3852
    ultimo SOW: 2026-09-23
    distancia (sesiones): -6
    SOW: close=12.36 < support=12.54  (-1.43%)
    volumen SOW: 1.05x baseline

**Diagnostico:** candidato vigente, SOW DENTRO de M pero **marginal**:
caida -1.43% (justo rompe soporte) y volumen 1.05x (apenas superior a
la media). Es un SOW debil.

### 2.5. SLB (frontera, pasa, falso positivo)

    close=48.66  MA50=52.73  MA200=50.01
    precio vs MA50: -7.72%   vs MA200: -2.70%
    struct_t=-0.1128   t_norm_t=+0.2142
    struct_max=+0.6786
    ultimo SOW: 2026-10-01
    distancia (sesiones): 0
    SOW: close=48.66 < support=48.69  (-0.06%)
    volumen SOW: 1.03x baseline

**Diagnostico:** candidato vigente, SOW HOY pero **falso positivo
claro**: caida -0.06% (48.66 vs 48.69, diferencia de 3 centimos) y
volumen 1.03x. Es ruido, no un Sign of Weakness autentico.

---

## 3. Hallazgos

### H1 - SLB es falso positivo del SOW

`close < support` con diferencia de 3 centimos (-0.06%) y volumen 1.03x
dispara SOW. No hay umbral minimo de magnitud.

**El SOW actual no distingue entre:**
- Ruptura significativa de soporte.
- Oscilacion normal alrededor del soporte.

### H2 - M=10 es determinante para ACS/ANA

Ambos tienen SOW fuerte:
- ACS: caida -4.62%, volumen 1.63x. Distancia 13 sesiones.
- ANA: caida -5.63%, volumen 3.66x. Distancia 42 sesiones.

Con M=15 (o mayor), pasarian a DISTRIBUTION. Con M=10, no.

**La eleccion de M cambia directamente el conjunto de DISTRIBUTION.**

### H3 - PCG pasa con SOW marginal

Distancia OK (6 sesiones), pero caida -1.43% y volumen 1.05x.
Estaria en la frontera si se introdujera un umbral de magnitud.

---

## 4. Diagnostico de conjunto

El experimento confirma las tres preocupaciones del auditor:

1. **M es determinante.** No es un parametro secundario.
2. **N (30) es determinante para la definicion de soporte.** Un N mayor
   generaria menos rupturas y exigiria caidas mas profundas.
3. **Falta un umbral de magnitud en SOW.** Sin el, hay falsos positivos
   (SLB) y fronteras dudosas (PCG).

**El SOW actual es una condicion binaria: existe o no existe. Sin
umbral de magnitud, "existe" abarca desde -0.06% hasta -5.63%.**

---

## 5. Opciones para 5b.3 (a dictamen del auditor)

### O1 - Anadir umbrales minimos de magnitud al SOW

    SOW_t = Close_t < support_t * (1 - X_min)
            AND Volume_t > vol_baseline_t * Y_min

Con X_min (p.ej. 1%) y Y_min (p.ej. 1.2x).

**Ventaja:** elimina falsos positivos (SLB).
**Inconveniente:** introduce 2 parametros mas.

### O2 - Calibrar solo N/M (sin umbral)

Mantener el SOW binario actual, calibrar N y M.

**Ventaja:** menos parametros.
**Inconveniente:** SLB sigue siendo DISTRIBUTION en el rango M=10.

### O3 - Cambiar a un SOW continuo

En lugar de binario, calcular una magnitud normalizada del SOW:

    sow_score = -z(close / support - 1) * z(volume / baseline - 1)

Y exigir un umbral minimo sobre el score compuesto.

**Ventaja:** mas informacion, no binario.
**Inconveniente:** define nueva metrica.

---

## 6. Preguntas al auditor

1. **¿Apruebas los hallazgos H1/H2/H3?**
2. **¿Se introduce un umbral minimo de magnitud en SOW (O1)?**
3. **Si O1, ¿los umbrales X_min e Y_min se fijan ahora o en 5b.3 junto con N/M?**
4. **Sobre M=10:** con el experimento actual, ¿confirmas que debe ser
   parametro calibrable en 5b.3 (no PROPUESTO fijo)?
5. **Sobre N=30:** igual.
6. **¿5b.3 estudia solo N/M (O2), o N/M + umbrales (O1 + O2)?**

---

## 7. Estado

- v1.6: implementada, testeada.
- 8 DISTRIBUTION: 3 correctos (BA, PCG, ACS/ANA segun M), 1 falso
  positivo (SLB), resto pendiente de calibracion.
- 5c.3: ABIERTA.
- 5b.3: ABIERTA (N/M + posible umbral de magnitud).
- 5d: BLOQUEADA.

---

Fin de la verificacion manual.
