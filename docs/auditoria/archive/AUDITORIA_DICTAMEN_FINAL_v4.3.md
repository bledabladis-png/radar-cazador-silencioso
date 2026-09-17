# DICTAMEN FINAL DE AUDITORÍA — RADAR v4.3

**Fecha:** 12 de agosto de 2026  
**Baseline de partida:** v4.2 congelada (dictamen previo)  
**Versión auditada:** v4.3  
**Auditor:** Externo / Ingeniero de supervisión  
**Estado:** APROBADO

---

## 1. Declaración formal

La v4.3 queda **aprobada como versión de validación y monitorización**.  
No se han modificado pesos, umbrales ni lógica de clasificación productiva.  
La baseline v4.2 permanece intacta como referencia.

---

## 2. Estado técnico

| Dimensión | Estado |
|-----------|--------|
| Validación funcional | 🟢 10/10 |
| Centralización de parámetros | 🟢 Completada |
| Walk-forward OOS | 🟢 Completado |
| Monitorización continua | 🟢 Implementada |
| Predictividad | 🔴 No demostrada |
| Trading automático | 🟢 No aplicable |
| Riesgo principal | 🟠 Redundancia interna + interpretación excesiva de consenso |

---

## 3. Hallazgos clave

1. **No existe evidencia de capacidad predictiva OOS** (IC ≈ 0).  
   El Radar es descriptivo, no predictivo.

2. **Tactical Score muy concentrado** (RS + Momentum ≈ 94,8% de varianza conjunta).  
   No es una señal de 5 fuentes independientes.

3. **Flow simplificado vs Flow Proxy completo**: corr = 0.387.  
   Son constructos diferentes; ninguno es predictivo.  
   No cambiar producción.

4. **MTE Confidence no calibrada**: la nueva etiqueta "Confidence Score no calibrado" resuelve el problema semántico, no el estadístico.

5. **Holdings europeos**: actualización manual sin fuente automática fiable. Riesgo operativo.

6. **Breadth de 11 sectores**: baja resolución (cada sector = 9,09%). No artificialmente mejorable.

---

## 4. Regla de oro para futuras versiones

> **No modificar una señal porque no sea predictiva. Primero hay que demostrar que esa señal no cumple correctamente su función descriptiva o estructural.**

---

## 5. Conclusión

El Radar v4.3 es **funcionalmente operativo, auditado y transparente**.  
Su valor no reside en predecir, sino en diagnosticar, clasificar, contextualizar y seleccionar información para el gestor humano.

**Firma:** Ingeniero de supervisión  
**Fecha:** 12/08/2026
