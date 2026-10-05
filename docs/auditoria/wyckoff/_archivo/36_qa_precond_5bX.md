# QA de precondiciones 5b.X - Analisis de poder y umbrales

**Documento de consulta. NO normativo. Requiere dictamen externo.**
**Fecha:** 2026-10-02.
**Precedente:** protocolo 35_protocolo_5bX.md + contrato v1.9.
**Script QA:** scripts/qa_precond_5bX.py (read-only).
**Artefactos:** outputs/audit/wyckoff_5bX_precond_qa.json,
outputs/audit/wyckoff_5bX_precond_qa_run.log.
**Estado:** BLOQUEO de la validacion hasta dictamen sobre umbrales.

---

## 0. Proposito

El protocolo 5b.X fijo umbrales minimos para ejecutar la validacion
out-of-sample (12 meses, 50 episodios, 20 confirmed). Esos umbrales
fueron propuestos sin respaldo empirico.

Este informe mide las tasas historicas reales y analiza si los
umbrales actuales son suficientes para obtener un IC95% con capacidad
de discriminar.

**Conclusion previa:** los umbrales actuales son demasiado bajos para
tener poder estadistico. Se solicita dictamen sobre ajuste.

---

## 1. Tasas historicas medidas

Historico 2021-10-01 -> 2026-10-01 (60 meses):

    sesiones: 1293
    sesiones/mes: 21.6
    candidate starts totales: 2415
    starts/mes: 40.3
    episodios con landmark: 2388
    confirmed: 635
    tasa confirmed: 26.6%

**Al ritmo medio:**

| Horizonte | Episodios estimados | Confirmed estimados |
|---|---:|---:|
| 6 meses | ~242 | ~64 |
| 12 meses | ~483 | ~128 |
| 18 meses | ~725 | ~193 |
| 24 meses | ~966 | ~257 |
| 36 meses | ~1450 | ~386 |

Los umbrales actuales (50 ep, 20 conf) se cumplen en ~1.5 meses al
ritmo medio.

---

## 2. Variabilidad temporal: hallazgo no contemplado

La tasa de episodios **no es estable**. Sub-bloques de 6 meses:

    2021-10-01 -> 2022-04-30:    0 starts (0.0/mes)
    2022-04-30 -> 2022-10-31:   76 starts (12.6/mes)
    2022-10-31 -> 2023-04-30:  156 starts (26.2/mes)
    2023-04-30 -> 2023-10-31:  237 starts (39.2/mes)
    2023-10-31 -> 2024-04-30:  239 starts (40.0/mes)
    2024-04-30 -> 2024-10-31:  455 starts (75.3/mes)
    2024-10-31 -> 2025-04-30:  510 starts (85.8/mes)
    2025-04-30 -> 2025-10-31:  187 starts (30.9/mes)
    2025-10-31 -> 2026-04-30:  415 starts (69.8/mes)

**Rango: de 0 a 85.8 starts/mes.** Factor > 10x entre minimo y maximo.

**Implicacion:** la estimacion "12 meses -> 483 episodios" solo vale
al ritmo medio. Si el mercado entra en un regimen como 2021-2022,
12 meses pueden producir **0 episodios**. Si entra como 2024-2025,
12 meses pueden producir >1000.

**El protocolo 5b.X no contempla esta variabilidad.** Los umbrales
basados en tiempo (12 meses) pueden coincidir con un regimen esteril.

---

## 3. Analisis de poder estadistico

El criterio del protocolo 5b.X es:

    IC95% lower del lift_H20 > 0

Con la muestra de desarrollo (5b.4-bis, 635 confirmed):

    lift_point = +0.0740
    IC95% = [+0.0234, +0.1242]
    ancho = 0.1008

### Aproximacion

El ancho del IC escala aproximadamente con 1/sqrt(n_confirmed).
Para n' = k * n_actual:

    ancho(n') = ancho(actual) * sqrt(n_actual / n')
              = 0.1008 * sqrt(635 / n')

### Estimacion por horizonte

Asumiendo lift_point real = +0.0740 (el observado en desarrollo):

| Horizonte | n_conf | ancho esperado | lower_CI esperado |
|---|---:|---:|---:|
| 6 meses | 64 | 0.318 | -0.085 |
| 12 meses | 128 | 0.225 | -0.038 |
| 18 meses | 193 | 0.183 | -0.017 |
| 24 meses | 257 | 0.158 | +0.005 |
| 36 meses | 386 | 0.129 | +0.020 |

### Escenarios de lift real distinto

Si el lift real en validacion fuera distinto al observado:

| Lift real | 12m lower_CI | 24m lower_CI |
|---|---:|---:|
| +0.030 (mas bajo) | -0.083 | -0.049 |
| +0.070 (igual) | -0.042 | -0.009 |
| +0.100 (mas alto) | -0.013 | +0.021 |
| +0.150 (mucho mas alto) | +0.038 | +0.071 |

### Conclusion del analisis de poder

- **12 meses** con lift ~+0.07: lower_CI ~ -0.04. **Cruza 0. Escenario B.**
- **24 meses** con lift ~+0.07: lower_CI ~ +0.005. **Borderline.**
- **36 meses** con lift ~+0.07: lower_CI ~ +0.020. **Confirma.**

**Con el lift observado en desarrollo, 5b.X necesitaria ~24-36 meses
de datos nuevos para confirmar.** Con 12 meses, el resultado mas
probable es Escenario B (neutro), no por fracaso del SOW sino por
falta de muestra.

### Nota critica

Un Escenario B a los 12 meses **no refuta el SOW**. Solo significa
"no hay suficiente informacion aun". El protocolo actual no distingue
claramente entre:
- "el SOW no funciona" (Escenario C).
- "no hay datos suficientes" (que deberia ser un escenario propio).

Este es un hueco del protocolo 5b.X.

---

## 4. Propuesta de ajuste de umbrales

### 4.1. Sustituir umbrales por horizonte

Los umbrales actuales mezclan **tiempo** (12 meses) y **muestra**
(50 ep, 20 conf). El QA demuestra que el tiempo no garantiza muestra
por la alta variabilidad temporal.

**Propuesta:** sustituir el umbral de tiempo por **criterio basado
solo en muestra**, con un minimo de tiempo solo como guardrail:

    Ejecutar 5b.X cuando:
    - n_confirmed >= 200 en el bloque de validacion, Y
    - n_baseline  >= 500 en el bloque de validacion, Y
    - al menos 6 meses de datos posteriores al cutoff (guardrail
      minimo).

Esto garantiza poder estadistico suficiente (~2x la muestra del
desarrollo) independientemente del ritmo temporal.

### 4.2. Escenario "INSUFICIENTE_MUESTRA"

Anadir un cuarto escenario al protocolo 5b.X:

    A: IC95% lower > 0 -> confirma.
    B: IC95% cruza 0 y lift > 0.025 -> podria ser real, falta muestra.
    C: IC95% cruza 0 y lift <= 0.025 -> probablemente nulo o negativo.
    D (nuevo): INSUFICIENTE_MUESTRA -> n_confirmed < 200 o
       n_baseline < 500. No se clasifica en A/B/C.

Razon: el protocolo actual puede caer en "B neutro" sin distinguir
"falta muestra" de "efecto nulo". El escenario D explicita el caso.

### 4.3. Reporte obligatorio del ritmo temporal

Cuando se ejecute 5b.X, reportar:

- n_sessions_after_cutoff.
- n_candidate_starts_with_L_after_cutoff.
- n_confirmed, n_baseline.
- starts/mes en el bloque de validacion.

Para que se pueda contextualizar el resultado respecto a la
variabilidad historica.

---

## 5. Preguntas al auditor externo

### P1. Umbrales actuales

Los umbrales del protocolo 5b.X (12 meses, 50 ep, 20 conf) son
insuficientes para poder estadistico. Con lift ~+0.07:

    12 meses -> lower_CI ~ -0.04 (cruza 0)
    24 meses -> lower_CI ~ +0.005 (borderline)
    36 meses -> lower_CI ~ +0.020 (confirma)

**Pregunta:** sustituir los umbrales actuales por criterio basado
solo en muestra (n_confirmed >= 200, n_baseline >= 500)?

    a) Si, sustituir. Sin umbral de tiempo (solo guardrail de 6 meses).
    b) Si, sustituir, pero con umbral de tiempo mas alto (24 meses o
       36 meses).
    c) No, mantener 12 meses + 50/20. Aceptar que el resultado
       probable sea Escenario B.
    d) Otra.

### P2. Escenario D (INSUFICIENTE_MUESTRA)

Anadir un cuarto escenario para distinguir "falta muestra" de "efecto
nulo"?

    a) Si, anadir.
    b) No, con los umbrales ajustados de P1 no hace falta.
    c) Si, pero con umbral de "insuficiente" distinto. Indicar.
    d) Otra.

### P3. Umbral de lift para distinguir B de C

En la propuesta, el limite entre B y C es lift_point = +0.025.
Justificacion: ~1/3 del lift observado en desarrollo, por debajo del
cual la senal es marginal.

**Pregunta:** es razonable +0.025, o preferible otro corte?

    a) +0.025 (mantener).
    b) 0 (B si lift > 0, C si lift <= 0).
    c) +0.050 (mas estricto).
    d) Otra.

### P4. Bloque D como validacion parcial

El bloque D del desarrollo (2025-04 -> 2026-10) contiene ~293
confirmed. Fue inspeccionado en 5b.4-bis.

**Pregunta:** puede usarse como "pre-validacion" (semi-ciego, sabiendo
que ya fue visto), o se descarta completamente?

    a) Descartar completamente. Solo datos post-2026-10-01.
    b) Usar como pre-validacion informativa, sin valor confirmatorio.
    c) Usar como validacion si el resultado es contundente (positive
       o negative).
    d) Otra.

### P5. Sobre el protocolo 5b.X global

Con estos hallazgos, el protocolo 5b.X necesita reescritura
(revision v2)? O los cambios son puntuales?

    a) Reescritura completa a v2. Se necesita revisar todo el
       documento.
    b) Cambios puntuales: umbrales + escenario D + umbral de lift.
    c) Posponer cambios hasta que haya datos nuevos y ver entonces.
    d) Otra.

---

## 6. Anexo A - Reproducibilidad

    Set-Location D:\Macro_Sectorial
    py scripts\qa_precond_5bX.py

Determinista sobre el snapshot actual de data/stock_prices.parquet.

Salida: outputs/audit/wyckoff_5bX_precond_qa.json.

---

## 7. Anexo B - Artefactos disponibles

    outputs/audit/wyckoff_5bX_precond_qa.json
    outputs/audit/wyckoff_5bX_precond_qa_run.log

---

## 8. Que NO se ha tocado

- Cero cambios en config/settings.py.
- Cero cambios en el contrato v1.9.
- Cero cambios en el protocolo 5b.X (35).
- Cero cambios en codigo de produccion.
- El script QA es read-only.

---

## 9. Estado

    v1.8              CONTRATO PRODUCTIVO VIGENTE
    v1.9              CANDIDATA FROZEN_FOR_VALIDATION
    5b.X (35)         PROTOCOLO CONGELADO, con umbrales cuestionados
    36 (este)         QA de precondiciones, requiere dictamen
    Config SOW        None (fail-closed)
    5c.4, 5d          BLOQUEADAS
    Legacy            INTACTO

---

**Fin del informe 36.**
