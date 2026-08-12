# INFORME MAESTRO DE AUDITORIA - Radar de Rotación Sectorial v4.2

**Fecha:** 2026-08-12  
**Estado:** Auditoría interna completada (Fases 0-5)

---

## 1. Resumen ejecutivo

El Radar de Rotación Sectorial ha sido sometido a una auditoría estructurada en cinco fases.  
Se confirma que el sistema es **funcionalmente correcto**, **determinista** y **adecuado para su propósito informativo**.  
No se han detectado errores críticos de código, pero sí **riesgos de redundancia y limitaciones metodológicas** que deben mantenerse documentados.

---

## 2. Estado de las fases

| Fase | Objetivo | Estado |
|------|----------|--------|
| 0 | Arquitectura y trazabilidad | ✅ Completada |
| 1 | Redundancia y doble conteo | ✅ Completada |
| 2 | Validación estadística de señales y thresholds | ✅ Completada |
| 3 | Frescura y proveedores de datos | ✅ Completada |
| 4 | Holdings y líderes | ✅ Completada |
| 5 | Revisión final y documentación | ✅ Completada |

---

## 3. Principales hallazgos

### 3.1 Arquitectura y trazabilidad

- 
un.py es un orquestador limpio, sin cálculos ocultos.
- src/report_generator.py solo consume resultados, no recalcula métricas.
- La mayoría de los pesos están centralizados en config/weights.py; **excepciones:** inancial_conditions.py, options_metrics.py y darkpool.py.

### 3.2 Redundancia y doble conteo

- El Tactical Score está dominado por RS20 y Momentum20 (94.8% de varianza conjunta).
- El componente Flow del Tactical Score (versión simplificada) aporta casi 0% de información incremental.
- El Flow Proxy completo de momentum.py es independiente de Wyckoff (corr = -0.04), lo que es positivo.
- Correlaciones altas: Momentum↔Tactical (0.91), RS↔Tactical (0.87), RS↔Momentum (0.71).

### 3.3 Validación estadística

- **Ninguna señal muestra poder predictivo relevante** (IC promedio entre -0.22 y +0.01).  
  Esto confirma que el sistema es descriptivo, no predictivo.
- Los umbrales Wyckoff (±0.30) cambian de fase solo en ~6% de los casos ante perturbaciones de ±0.05: **estabilidad razonable**.

### 3.4 Frescura de datos

- CBOE y Yahoo Finance: CURRENT.
- FINRA Dark Pools: ARCHIVAL (23 días); correctamente excluido de clasificación actual.
- FRED/Macro manual: STALE (21 días); en el límite, vigilar.

### 3.5 Holdings y líderes

- Duplicados corregidos en data/etf_holdings.csv (13 → 0).
- Persistencia normalizada a 0-1 en sectorial e internacional.
- Sin casos de "mejor de un grupo malo" (todos los líderes #1 tienen Wyckoff positivo).

---

## 4. Clasificación de señales según independencia

- **Primarias:** RS, Momentum.
- **Diagnósticas:** Wyckoff, Flow Proxy completo.
- **Agregaciones con redundancia interna:** Tactical Score (RS+Momentum+Flow+Breadth+Acceleration).
- **Contexto global:** VIX, PCR, Breadth equity.
- **Capa de selección:** WLS (no debe usarse como señal independiente).

---

## 5. Recomendaciones para v4.3

1. Centralizar los pesos/umbrales de inancial_conditions.py, options_metrics.py y darkpool.py.
2. Revisar la versión simplificada de Flow en 	actical_engine.py y alinearla con el Flow Proxy completo si se desea más coherencia.
3. Añadir en el reporte la advertencia de que un acuerdo alto no implica señales independientes.
4. Calibrar históricamente la confianza del MTE si se desea que sea interpretable como probabilidad.
5. Ejecutar walk-forward out-of-sample si se quiere medir robustez temporal.
6. Mantener los scripts de validación como parte del control de calidad continuo.

---

## 6. Limitaciones conocidas

- Los umbrales y pesos son heurísticos o heredados, no calibrados.
- El sistema no tiene poder predictivo; no debe usarse como backtester.
- La amplitud sectorial con 11 sectores tiene baja resolución (cada sector = 9.09%).
- FINRA puede tener desfases >21 días; el sistema lo gestiona, pero conviene monitorizar.
- Los holdings de ETFs europeos se actualizan manualmente (sin fuente automática fiable).

---

## 7. Trazabilidad del código auditado

- Código productivo: D:\Macro_Sectorial
- Scripts de validación: alidation/
- Informes de auditoría: outputs/auditoria_*.md, docs/auditoria_*.md
- Documentación automática: docs/ (generada con scripts/generate_docs.py)

---

## 8. Conclusión

El Radar está **operativo, validado y documentado**.  
Las auditorías realizadas confirman la solidez funcional y la transparencia de la arquitectura, a la vez que identifican áreas de mejora para mantener la independencia estadística de las señales.  
No se recomienda modificar la lógica de producción en esta versión.  
Los hallazgos quedan registrados para la planificación de la v4.3.

