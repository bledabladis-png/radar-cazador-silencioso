# INFORME FINAL DE AUDITORÍA — RADAR SECTORIAL v3.15
# Versión corregida: 24/07/2026 (incluye Fase 2 + revisión externa + H6)

## H6: FRECUENCIA TEMPORAL — VERIFICADA

- Intervalos totales: 259 (entre 260 observaciones)
- Intervalos de 7 días: 258 (99.6%)
- Intervalos de 14 días: 1 (0.4%)
- Gaps >14 días: 0
- Gap máximo: 14 días
- Media: 7.03 días | Mediana: 7 días

**Dictamen:** Frecuencia semanal prácticamente regular, con un único intervalo de 14 días.  
**Nota:** Para análisis de correlación o persistencia temporal, no debe asumirse espaciado perfecto sin controlar ese intervalo atípico. No afecta materialmente a H5.

## H5: ANÁLISIS DE REDUNDANCIA BREADTH/LIS — CORREGIDO

- Correlación de rangos de Spearman: ρ_s = +0.838
- Pseudo-R² basado en Spearman: ρ_s² = 70.2%
- Varianza NO explicada por relación lineal: 29.8% (no interpretable como "información única")
- Dependencia con Tactical/Structural/Persistence: |ρ| < 0.20 (baja dependencia monotónica marginal, no independencia)

## ESTADO FINAL DE LA AUDITORÍA

**🟡 COMPLETADA CON HALLAZGOS ABIERTOS**

- Breadth y LIS se mantienen en v3.15 por su distinción conceptual (amplitud vs intensidad)
- La dependencia elevada (ρ=0.838) es esperable por construcción y está documentada como riesgo controlado
- Se recomienda prueba de ablación real (H7) para v4.0
- No se requieren cambios de código urgentes
