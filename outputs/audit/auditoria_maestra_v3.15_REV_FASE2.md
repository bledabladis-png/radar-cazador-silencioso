# INFORME CONSOLIDADO DE AUDITORÍA MAESTRA v3.15 — REVISIÓN POST-FASE 2

**Fecha:** 24 de julio de 2026
**Versión:** v3.15 (post-auditoría maestra + Fase 2)
**Estado:** 🟡 COMPLETADA CON HALLAZGOS ABIERTOS (no "sin modificaciones necesarias")

---

## 1. PRECISIONES DEL AUDITOR EXTERNO INCORPORADAS

| ID | Hallazgo del auditor | Verificación | Acción |
|----|---------------------|--------------|--------|
| H1 | Cobertura SLPM: ¿3/11 o n/expected_leaders? | Código real usa `SLPM_EXPECTED_LEADERS=5`, no `total_universe=11` | Corregido en código. Informe actualizado. |
| H2 | Capa 3B: correlaciones ~0 sospechosas | Verificado: las señales pasan por `tanh` y combinación no lineal. La baja correlación es esperable. | Conclusión ajustada: "No se detecta dependencia monótona contemporánea", no "diversificación probada". |
| H3 | UNRESOLVED mezcla falta de datos y señales mixtas | El código ya distingue `UNRESOLVED_LOW_COVERAGE` vs `UNRESOLVED_MIXED_SIGNALS` vía `reason_code` | Implementado en v3.15. Informe actualizado. |
| H4 | Flow Proxy no es idéntico en todos los módulos | Verificado: `momentum.py` usa 30% flow+35% OBV+35% CMF; `stock_leader.py` usa `robust_zscore(ret*dollar_vol)` | Dependency Tracker actualizado con dos tipos de Flow. |

---

## 2. RESULTADOS FASE 2: ABLATION TEST BREADTH/LIS

- Correlación Spearman: **+0.838**
- R²: **0.702** (70.2% varianza compartida)
- **Varianza NO compartida: 29.8%**

**Conclusión:** Breadth y LIS comparten el 70.2% de su varianza pero retienen un 29.8% de información única cada uno. Esto es coherente con que Breadth mida "amplitud" (% de líderes que cumplen condiciones) y LIS mida "intensidad" (fuerza de esas condiciones). La redundancia es alta pero no total. Ambos componentes son independientes de Tactical, Structural y Persistence (correlaciones <0.20). **No se recomienda recalibrar pesos en v4.0 sin un estudio de sensibilidad previo.**

---

## 3. ANTI-DOUBLE-COUNTING ACTUALIZADO

| Variable | Tipo | Módulos |
|----------|------|---------|
| Flow Proxy sectorial (30% flow + 35% OBV + 35% CMF) | Señal compuesta | `momentum.py` → `tactical_engine.py`, `structural_engine.py` |
| Flow Signal de líderes (robust_zscore(ret*dollar_vol)) | Señal individual | `stock_leader.py` → `slpm_v12.py` (LIS, Flow Divergence) |
| RS/Flow Proxy | Alta reutilización | `stock_leader.py`, `slpm_v12.py`, `tactical_engine.py`, `structural_engine.py` |
| Breadth ↔ LIS | ρ=+0.838, R²=0.702 | Documentado. Sin acción urgente. |

---

## 4. ESTADO FINAL DE LA AUDITORÍA

| Capa | Resultado | Estado |
|------|-----------|--------|
| 1. Dependencias de datos | 10 fuentes. Flow Proxy refinado. | ✅ |
| 2. Transformaciones | Trazables. Dos tipos de Flow documentados. | ✅ |
| 3A. Pearson señales primarias | Sin redundancia >0.80 | ✅ |
| 3B. Spearman scores vs señales | Correlaciones ~0 (consistente con combinación no lineal) | ✅ (interpretación ajustada) |
| 3C. SLPM Legacy | Structural↔Breadth: +0.798 | ⚠️ No aplica al sistema activo |
| 3D. SLPM v1.2 activo | Breadth↔LIS: +0.838, R²=0.702, 29.8% varianza única | ⚠️ Documentado |
| 4. Consistencia temporal | Mezcla de horizontes documentada | ✅ |
| Fase 2 | Ablation test completado | ✅ |

---

## 5. CONCLUSIÓN REVISADA

> El Radar Sectorial v3.15 presenta una arquitectura modular, determinista y generalmente trazable, sin evidencias de errores de ejecución críticos. Las señales primarias muestran una baja dependencia lineal entre familias y no se ha detectado una dominancia evidente de una única señal en los análisis realizados.
>
> No obstante, la auditoría identifica varios puntos que requieren corrección o validación adicional antes de considerar completamente cerrada la auditoría estructural. El principal es la definición de cobertura del SLPM v1.2, donde debe aclararse la diferencia entre cobertura del universo sectorial y cobertura de líderes analizados. Asimismo, debe precisarse la reutilización real de las señales de flujo, ya que el Flow Proxy sectorial y el flujo utilizado por Stock Leader no son idénticos.
>
> La correlación elevada entre `effective_breadth` y `LIS` (ρ=0.838, R²=0.702) constituye una alerta válida de posible redundancia informativa, pero el análisis de varianza muestra un 29.8% de información única. Se recomienda documentar sin recalibrar pesos en v4.0.
>
> La Capa 3B no permite concluir por sí sola que los Tactical y Structural Scores estén correctamente diversificados, dado que las correlaciones contemporáneas extremadamente bajas entre señales y scores son consistentes con la combinación no lineal de múltiples componentes normalizados con `tanh`.
>
> **Estado correcto de la auditoría: 🟡 COMPLETADA CON HALLAZGOS ABIERTOS.**
>
> No se recomienda modificar inmediatamente la lógica central del Radar v3.15. Sí se recomienda corregir las inconsistencias documentales y ejecutar una auditoría de validación específica sobre cobertura SLPM, alineación de Capa 3B, frecuencia temporal de los históricos y diferenciación entre ausencia de datos y estado `UNRESOLVED`.

---

**Firma del ingeniero de supervisión:**  
Radar de Rotación Sectorial v3.15 — Auditoría Maestra + Fase 2  
Documento generado el 24 de julio de 2026.
