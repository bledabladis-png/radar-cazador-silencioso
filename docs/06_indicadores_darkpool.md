# Indicadores: Dark Pools (FINRA v1.0)

## Proposito
Mide el porcentaje de volumen negociado en ATS (Alternative Trading Systems) respecto al volumen total, usando datos de FINRA.

## Z-Scores
Se calculan Z-Scores robustos para 4 ventanas: 13, 26, 52 y 104 semanas.

## Clasificacion
| Z-Score | Estado |
|---------|--------|
| >= 2.5 | Acumulacion extrema |
| 1.5 a 2.5 | Acumulacion fuerte |
| 0.5 a 1.5 | Acumulacion moderada |
| -0.5 a 0.5 | Neutral |
| -1.5 a -0.5 | Distribucion moderada |
| -2.5 a -1.5 | Distribucion fuerte |
| < -2.5 | Distribucion extrema |

## Constantes
| Constante | Valor |
|---|---|
| DARKPOOL_FULL_HISTORY_WEEKS | 104 |
| DARKPOOL_MIN_HISTORY_WEEKS | 13 |
| DARKPOOL_ZSCORE_WINDOWS | (13, 26, 52, 104) |
