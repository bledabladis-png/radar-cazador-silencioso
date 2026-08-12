# PLAN DE TRABAJO — RADAR v4.4

**Fecha:** 12 de agosto de 2026  
**Baseline:** v4.3 aprobada  
**Enfoque:** Profundizar en los puntos abiertos sin modificar lógica productiva.

---

## 1. Puntos abiertos priorizados

| Prioridad | Punto abierto | Objetivo |
|-----------|---------------|----------|
| 1 | Redundancia del Tactical Score | Determinar si se debe documentar o rediseñar conceptualmente |
| 2 | Flow simplificado vs completo | Decidir cuál representa mejor el concepto de flujo |
| 3 | Calibración MTE Confidence | Evaluar si es posible calibrar históricamente |
| 4 | Holdings europeos | Buscar fuente automática fiable o documentar el riesgo |
| 5 | Breadth 11 sectores | Documentar limitación, no intentar artificialmente aumentar resolución |

---

## 2. Fases propuestas

### Fase A — Análisis conceptual del Tactical Score
- Revisar la fórmula y su interpretación.
- Evaluar si RS+Momentum dominan por diseño o por falta de aporte real de los otros componentes.
- **Decisión:** documentar como "agregación de momentum" en lugar de "5 señales independientes".

### Fase B — Comparación conceptual de Flow
- Auditar las definiciones de ambos flujos.
- Determinar cuál es más coherente con la filosofía del sistema.
- **Sin cambio productivo**; solo propuesta para v4.5 si aplica.

### Fase C — MTE Confidence
- Investigar si hay histórico suficiente para calibrar.
- Si no es posible, mantener etiqueta "no calibrado" y cerrar el punto.

### Fase D — Holdings europeos
- Explorar fuentes automáticas alternativas.
- Si no se encuentra fiable, mantener actualización manual y documentar como limitación.

### Fase E — Monitorización ampliada
- Añadir chequeo de redundancia del Tactical Score.
- Añadir comparación automática Flow simplificado vs completo.
- Mantener checks existentes.

---

## 3. Estimación total

| Fase | Duración |
|------|----------|
| A    | 1 semana  |
| B    | 1 semana  |
| C    | 1 semana  |
| D    | 1 semana  |
| E    | 1 semana  |
| **Total** | **5 semanas** |

---

## 4. Reglas de oro

- No modificar lógica productiva sin completar la fase correspondiente.
- No introducir ML ni optimización de parámetros.
- Mantener determinismo y transparencia.
- Los cambios de interpretación son aceptables; los cambios de lógica solo con evidencia.
- v4.3 permanece como baseline aprobada durante todo el desarrollo de v4.4.
