# Dictamen de Validación Out-of-Sample — Flujos del Radar

**Fecha:** 2026-08-16
**Objetivo:** Evaluar la capacidad predictiva de las capas de flujo (ETF Primary Flow, CFTC Position Flow, SEC N‑PORT Position Change) sobre retornos forward.

## Resumen de resultados

| Capa | Metodología | Resultado | Conclusión |
|------|-------------|-----------|------------|
| ETF Primary Flow | Spearman + bootstrap + Bonferroni (horizontes 5,10,20) | 1 de 36 pruebas significativa (XLB 20d, rho=-0.136); resto no significativo | Descriptivo, no predictivo |
| CFTC Position Flow | Spearman + bootstrap + Bonferroni (horizontes 5,10,20) | Ninguna correlación significativa | Descriptivo, no predictivo |
| SEC N‑PORT Position Change | Spearman + bootstrap + Bonferroni (horizontes 20,40,60) | Insuficientes datos (solo 1-2 trimestres); no evaluable | Pendiente de más historia |

## Recomendación

- **Mantener todas las capas de flujo como descriptivas/diagnósticas.** No deben alimentar un score agregado predictivo sin evidencia OOS adicional.
- **Acumular más datos N‑PORT** (al menos 4-6 trimestres) antes de repetir la validación.
- **No introducir `REAL_FLOW_SCORE`** ni `FLOW_CONFIDENCE` como señal de trading. `FLOW_CONFIDENCE` es solo una síntesis cualitativa de convergencia, no predictiva.

## Archivos de resultados

- `outputs/audit/oos_flow_results.csv`
- `outputs/audit/oos_cftc_results.csv`
- `outputs/audit/oos_nport_results.csv` (vacío por falta de datos)

*Esta validación es descriptiva y no constituye una recomendación de inversión.*
