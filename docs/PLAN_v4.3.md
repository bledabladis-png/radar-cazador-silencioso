# PLAN DE TRABAJO — RADAR DE ROTACIÓN SECTORIAL v4.3

**Fecha:** 12 de agosto de 2026  
**Baseline de partida:** v4.2 congelada  
**Enfoque:** Mejora metodológica sin convertir el sistema en predictivo ni automático.

---

## 1. Prioridades

| Prioridad | Acción                          | Objetivo |
|-----------|----------------------------------|----------|
| 1         | Walk-forward out-of-sample       | Validar robustez temporal real de señales y clasificaciones |
| 2         | Centralizar parámetros           | Mover pesos/umbrales de Financial Conditions, Options y Dark Pools a config |
| 3         | Comparar Flow simplificado vs Flow Proxy completo | Evaluar si la versión simplificada aporta valor incremental |
| 4         | Calibrar Confidence del MTE      | Dar significado probabilístico (o renombrar como "score de confianza") |
| 5         | Monitorización continua          | Automatizar tests de duplicados, NaN, frescura, correlaciones y umbrales |

---

## 2. Fases y tareas

### Fase A — Validación out-of-sample (semana 1‑2)
- Implementar alidation/walk_forward.py.
- Dividir 2016‑2024 entrenamiento / 2025‑2026 validación.
- Medir estabilidad de IC, cambios de ranking y transiciones de régimen.
- **Criterio de aceptación:** Informe con métricas OOS y comparación con in‑sample.

### Fase B — Centralización de parámetros (semana 3)
- Mover weights de inancial_conditions.py a config/weights.py.
- Mover umbrales PCR/IHR de options_metrics.py a config/settings.py.
- Mover umbrales Dark Pool de darkpool.py a config/settings.py.
- **Criterio:** Todos los parámetros auditable desde config; sin hardcode en los módulos.

### Fase C — Comparación de Flow (semana 4)
- Crear alidation/flow_comparison.py.
- Comparar low_recent (tactical_engine) vs compute_flow_proxy (momentum.py) en términos de correlación, IC y contribución incremental.
- **Criterio:** Documentar si la versión simplificada aporta valor; no modificar producción todavía.

### Fase D — Calibración MTE Confidence (semana 5)
- Auditar compute_confidence y score_scenarios.
- Evaluar si el score de confianza actual puede calibrarse con históricos.
- Si no es posible, renombrar en el reporte como "Confidence Score (no calibrado)".
- **Criterio:** Evitar interpretación probabilística no justificada.

### Fase E — Monitorización continua (semana 6)
- Consolidar scripts de auditoría en un solo alidation/run_all_audits.py.
- Incluir checks de duplicados, NaN, frescura, correlaciones, persistencia y regresión del reporte.
- **Criterio:** Un comando ejecuta todas las auditorías y genera outputs/informe_monitorizacion.md.

---

## 3. Estimación total

| Fase | Duración |
|------|----------|
| A    | 2 semanas |
| B    | 1 semana  |
| C    | 1 semana  |
| D    | 1 semana  |
| E    | 1 semana  |
| **Total** | **6 semanas** (equiv. ~1,5 meses) |

---

## 4. Reglas de oro para v4.3

- No modificar la lógica de producción sin completar la fase correspondiente.
- No introducir ML ni optimización de parámetros.
- Mantener determinismo y transparencia.
- Cualquier cambio de pesos/umbrales debe estar respaldado por evidencia OOS.
- v4.2 permanece congelada durante todo el desarrollo de v4.3.
