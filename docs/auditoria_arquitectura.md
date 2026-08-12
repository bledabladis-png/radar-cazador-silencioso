# Auditoria Fase 0 - Arquitectura y Trazabilidad

**Fecha:** 2026-08-12
**Estado:** Completada

---

## Orden de ejecucion verificado

1. Descarga de datos de mercado
2. Validacion de datos
3. Carga de datos macro manuales
4. Calculo de Cond. Financieras
5. Calculo de Liquidez Real (FRED)
6. Calculo de Volatilidad
7. Calculo de Macro Regime
8. Calculo de rankings sectoriales
9. Calculo de rankings de precio y flujo
10. Lideres sectoriales
11. Opciones (PCR)
12. Dark Pools (FINRA ATS)
13. Market Transition Engine
14. Fases Wyckoff para indices internacionales
15. Validation Gate
16. Generacion del reporte

**Conclusion:** un.py es un orquestador limpio, sin calculos ocultos.

---

## Grafo de dependencias (resumen)

- un.py importa todos los motores, indicadores y cargadores.
- src/report_generator.py solo consume resultados, no recalcula metricas.
- No se detectaron imports circulares.

---

## Matriz de centralizacion de pesos/umbrales

| Modulo | Pesos/Umbrales | Centralizado |
|--------|----------------|--------------|
| Tactical Engine | TACTICAL_WEIGHTS | Si (config/weights.py) |
| Structural Engine | STRUCTURAL_WEIGHTS | Si (config/weights.py) |
| Sector Regime | SECTOR_SCORE_WEIGHTS, SECTOR_DISPERSION_PENALTY | Si (config/weights.py) |
| SLPM | SLPM_WEIGHTS | Si (config/weights.py) |
| Financial Conditions | 0.40/0.30/0.15/0.15 | No (hardcode) |
| Options Metrics | umbrales PCR/IHR | No (hardcode) |
| Dark Pool | umbrales Z-score | No (hardcode) |

---

## Hallazgos

- La mayoria de los pesos estan bien centralizados en config/weights.py.
- inancial_conditions.py, options_metrics.py y darkpool.py tienen umbrales duros no centralizados.
- No hay contradicciones con settings.py, pero la trazabilidad es mejorable.

---

## Recomendaciones para v4.3

- Mover los pesos de inancial_conditions.py a config/weights.py.
- Mover los umbrales de options_metrics.py a config/settings.py.
- Mover los umbrales de darkpool.py a config/settings.py.

*Sin cambios en produccion en esta fase.*
