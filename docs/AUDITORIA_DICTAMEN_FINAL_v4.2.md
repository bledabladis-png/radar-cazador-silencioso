# DICTAMEN FINAL DE AUDITORÍA — RADAR DE ROTACIÓN SECTORIAL v4.2

**Fecha:** 12 de agosto de 2026  
**Versión auditada:** v4.2  
**Auditor:** Externo / Ingeniero de supervisión  
**Estado:** Aprobado

---

## 1. Declaración formal

El Radar de Rotación Sectorial v4.2 queda **congelado como baseline oficial**.  
No se modificarán pesos, umbrales, lógica de WLS, SLPM o clasificación sectorial sin una nueva planificación bajo v4.3.

---

## 2. Estado técnico y funcional

| Dimensión                   | Estado           |
|-----------------------------|------------------|
| Operatividad                | 🟢 Operativo     |
| Validación funcional        | 🟢 Validado      |
| Transparencia               | 🟢 Alta          |
| Calidad de datos            | 🟢 Adecuada      |
| Independencia estadística   | 🟡 Limitada      |
| Capacidad predictiva        | 🔴 No demostrada |
| Robustez out-of-sample      | 🟡 Pendiente     |
| Necesidad de cambios en producción | 🟢 Ninguna |

---

## 3. Hallazgos principales

1. **Arquitectura correcta.** 
un.py es un orquestador limpio; 
eport_generator.py no recalcula métricas.
2. **Redundancia del Tactical Score.** RS20 + Momentum20 explican el 94,8% de su varianza.  
   Correlaciones: Momentum↔Tactical 0.91, RS↔Tactical 0.87, RS↔Momentum 0.71.
3. **Flow Proxy completo vs Wyckoff es independiente** (corr = -0.04). No hay redundancia empírica grave.
4. **Ninguna señal es predictiva.** IC promedio entre -0.22 y +0.01 a 20/60 días.  
   El sistema es descriptivo y diagnóstico, no predictivo.
5. **Umbrales Wyckoff estables** (±0.30). Una perturbación de ±0.05 cambia de fase solo en ~6% de los casos.
6. **Frescura de datos controlada.** CBOE/Yahoo al día; FINRA obsoleto (excluido); FRED en el límite.
7. **Holdings y líderes saneados.** 13 duplicados eliminados, persistencia normalizada a 0‑1, sin "mejor de un grupo malo".

---

## 4. Conclusión formal

> **v4.2 es un sistema descriptivo y diagnóstico funcionalmente validado, no un modelo predictivo validado. Su principal riesgo no es de programación sino de interpretación: múltiples métricas relacionadas pueden aparentar una confirmación mayor de la que realmente proporcionan.**

---

## 5. Aprobación

Este dictamen se registra como **estado de auditoría de referencia** para v4.2.  
Los riesgos cuantificados y aceptados no se reinterpretarán como fallos en el futuro.

**Firma:** Ingeniero de supervisión  
**Fecha:** 12/08/2026
