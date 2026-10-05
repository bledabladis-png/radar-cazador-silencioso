# HALLAZGO 5b - H3 depende del regimen de mercado

**Documento de auditoria. Requiere dictamen externo.**
**Fecha:** 2026-10-02.

---

## 1. Contexto

Fase 5b ejecutada con K=0.25. H1, H2, H4 pasan en calibracion y en
validacion. **H3 (`p05 <= -0.40`) falla en validacion.**

Detalle en `10_calibracion_K_resultados.md`.

---

## 2. Naturaleza del hallazgo

**No es un defecto de K.** Es un defecto del criterio H3 aplicado a un
periodo con sesgo de regimen.

**Evidencia:**

- Calibracion (2021-10 → 2025-06): incluye el bear market 2022. p05
  con K=0.25 = -0.409. Pasa H3.
- Validacion (2025-06 → 2026-09): 15 meses de bull market sostenido.
  p05 = -0.193. Falla H3.

Con K=0.40 (mas conservador), p05 en calibracion es -0.265. Tambien
fallaria H3. **Ningun K de la grid pasa H3 en el periodo de validacion.**

**El problema es H3, no K.**

---

## 3. Diagnostico

H3 e H4 miden la **amplitud absoluta de la distribucion de `t_norm`**
en el universo de tickers. Esta amplitud depende de dos factores:

1. **K** (escala).
2. **Regimen de mercado del periodo** (¿hay tickers en tendencia bajista
   durante el periodo?).

H1 y H2 (saturacion) solo dependen de K y de la **magnitud maxima** de
`trend`. No dependen de la simetria.

H3 y H4 (cola izquierda / cola derecha) **si dependen del regimen**.
Un periodo sin tendencias bajistas no puede producir p05 bajo.

**En un periodo de 15 meses con tendencia alcista generalizada, el p05
de `t_norm` satura en valores moderados, no en -0.40.**

---

## 4. Implicaciones para el protocolo

El protocolo v2 asume implicitamente que:

    el universo tiene tendencias alcistas y bajistas equilibradas en
    cualquier periodo suficientemente largo

Esto es falso en equity en general, y muy falso en periodos de 15 meses.

**Consecuencia:** H3 (y H4) son **criterios fragiles a la eleccion del
periodo de validacion**. La eleccion actual (30% final = 2025-06 →
2026-09) coincide con un bull market claro.

Si el split temporal hubiera sido distinto (p.ej. 60/40 → validacion
2024-07 → 2026-09, incluyendo correcciones de 2024), H3 habria pasado.

**El resultado de 5b no deberia depender de donde se ponga el corte
temporal.** Con el protocolo actual, sí depende.

---

## 5. Opciones de solucion (a dictamen del auditor)

### Opcion A - Reformular H3/H4 como metricas por ticker

En lugar de medir p05/p95 de la distribucion **cross-sectional** de
`t_norm`, medir la distribucion **temporal intra-ticker**:

    para cada ticker:
        calcular p05_ticker, p95_ticker de su t_norm historico
    agregar por mediana de tickers

Asi, cada ticker aporta su propio rango observado, sin depender del
regimen agregado del periodo.

**Ventaja:** robusto a regimenes.

**Inconveniente:** mide propiedad distinta (resolucion intra-ticker).

### Opcion B - Evaluar H3/H4 sobre el periodo completo (cal + val)

H1 y H2 (saturacion) evaluados sobre cal y val por separado.
H3 y H4 (amplitud) evaluados sobre la serie completa.

**Razon:** la amplitud de la distribucion de `t_norm` solo tiene sentido
sobre un periodo que contenga tanto expansiones como correcciones.

**Ventaja:** simple.

**Inconveniente:** H3/H4 dejan de ser validacion fuera de muestra.

### Opcion C - Ampliar el periodo de validacion para incluir un ciclo

Cambiar el split a 60/40 → validacion incluye 2024 (correcciones).

**Inconveniente:** la eleccion del split se convierte en calibracion
implicita (elegido por ver cual funciona).

**Descartable** segun los propios criterios del auditor.

### Opcion D - Sustituir H3/H4 por metricas robustas a regimen

Por ejemplo: medir `|p05| + p95` como simetria de amplitud, con umbral
unico (p.ej. `>= 0.80`). No separa cola izquierda y derecha.

**Ventaja:** robusto.

**Inconveniente:** pierde informacion de asimetria.

### Opcion E - Declarar H3/H4 no aplicables en validacion

Solo H1/H2 (saturacion) obligatorios en validacion. H3/H4 solo en
calibracion.

**Ventaja:** reconoce el problema.

**Inconveniente:** debilita la validacion.

---

## 6. Recomendacion tecnica (a validar)

**Opcion A: H3/H4 como metricas por ticker (temporal intra-ticker).**

Razones:

1. **Coherente con el objetivo:** `t_norm` debe tener buena resolucion
   **para cada ticker a lo largo de su historia**, no solo en cortes
   transversales de un periodo.
2. **Robusto a regimen:** cada ticker aporta su propio rango. Un ticker
   con historia 2021-2026 incluye tanto bear como bull market. Su p05
   siempre sera < -0.40 si su trend ha llegado a ser < -0.40·K.
3. **Elimina la dependencia del split temporal** para H3/H4.
4. **Separa H1/H2 (saturacion, intrinseco) de H3/H4 (resolucion,
   requiere historia).**

Con Opcion A, el criterio pasa a ser:

    mediana sobre tickers de (p05_ticker <= -0.40)
    mediana sobre tickers de (p95_ticker >= +0.40)

Es decir: al menos el 50% de los tickers debe tener su propio p05 bajo
o su propio p95 alto.

---

## 7. Preguntas al auditor

1. ¿Se aprueba Opcion A (H3/H4 por ticker)?
2. Si no, ¿que opcion prefiere (B, C, D, E)?
3. ¿Se congela K=0.25 mientras se resuelve este hallazgo? (Mi
   recomendacion: NO congelar hasta tener criterios definitivos).
4. ¿Se abre 5b.2 tras resolver H3/H4, o se ejecuta el ciclo completo
   de nuevo con los criterios corregidos?

---

## 8. Estado

- Fase 5b: **FALLA segun protocolo actual**.
- K=0.25: candidato solido, pendiente de criterios corregidos.
- 5c/5d: siguen bloqueadas.
- Legacy: intacto.

---

Fin del hallazgo 5b.

---

# 9. DICTAMEN DEL AUDITOR (2026-10-02)

## 9.1. Veredicto general

> El fallo encontrado es del criterio H3, no de K. La Fase 5b ha
> funcionado como mecanismo de auditoria: no ha encontrado un K malo,
> ha encontrado que un criterio intentaba validar una propiedad de la
> distribucion del mercado como si fuera una propiedad de la
> transformacion K.

## 9.2. Correcciones al protocolo 5b

**C1. Opcion A corregida (intra-ticker + split estricto).**
H3/H4 se calculan por ticker **dentro de cada split**:
- Calibracion: distribucion temporal de t_norm por ticker, sobre el 70%.
- Validacion: idem, sobre el 30%.
No se usa la historia completa 2021-2026 para seleccionar K. Eso
romperia la separacion cal/val.

**C2. H3/H4 fuera de validacion OOS como hard criteria.**
En validacion, H3/H4 dejan de ser obligatorios. Razon: exigen que el
periodo OOS reproduzca las dos colas observadas en calibracion, cosa
que depende del regimen del periodo, no de K.

Validacion OOS pasa a exigir:
  V1 bounds (rango acotado)
  V2 ausencia de saturacion extrema (pct_90, pct_95)
  V3 no degradacion grave de resolucion
  V4 ausencia de comportamiento patologico

Y como diagnosticos OOS (reportados, no fallo):
  p05, p95, skewness.

**C3. Nueva invariante contractual: monotonicidad.**
Para todo trend_a < trend_b: t_norm_a < t_norm_b.
Y: trend = 0 -> t_norm = 0; trend > 0 -> t_norm > 0; trend < 0 ->
t_norm < 0. Se anade como test permanente.

**C4. Nomenclatura formal.**
  Ticker-balanced  -> estimador primario (cada ticker pesa igual).
  Observation-pooled -> sensibilidad obligatoria (cada observacion pesa
  igual).
Ambos se reportan siempre. Pooled no puede generar seleccion ad-hoc
de K.

## 9.3. Estado de K=0.25

    K=0.25
    CALIBRACION: PASS
    VALIDACION: H1 PASS, H2 PASS
                H3/H4 = criterio no aplicable como fallo
    → NO CONGELAR

## 9.4. Decision sobre las 4 preguntas

| Pregunta | Dictamen |
|---|---|
| Q1 Opcion A | Aprobada SOLO en version corregida (intra-ticker + split estricto). H3/H4 no son hard OOS. |
| Q2 Congelar K=0.25 | NO |
| Q3 5b.2 o 5c | 5b.2 corregida. NO 5c. |
| Q4 Balanced/Pooled | Balanced = primario. Pooled = diagnostico obligatorio. |

## 9.5. Prohibiciones del dictamen

- No cambiar el split a 60/40 para incluir bear market (seria
  calibracion implicita).
- No usar 2021-2026 completo para obtener las dos colas (elimina OOS).
- No elegir K=0.35 "porque parece menos saturado".
- No elegir K segun los 4 tickers.
