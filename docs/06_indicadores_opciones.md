# Indicadores: Opciones (OMS v2.0)

## Proposito
Calcula el PCR (Put/Call Ratio) y el IHR (Institutional Hedge Ratio) a partir de datos de CBOE.

## Metricas
- PCR Total, PCR Indices, PCR Acciones, PCR ETP, PCR VIX, PCR SPX.
- IHR = PCR Indices / PCR Acciones.
- Volumen en Indices (% del total).
- Put Share / Call Share.

## Clasificacion IHR
| Rango | Clasificacion |
|-------|---------------|
| < 0.8 | Especulacion extrema |
| 0.8 - 1.2 | Especulacion alta |
| 1.2 - 1.6 | Equilibrado |
| 1.6 - 2.5 | Cobertura institucional alta |
| > 2.5 | Cobertura institucional extrema |

## Clasificacion PCR (Z-Score)
| Rango | Estado |
|-------|--------|
| >= 2.0 | Panico |
| 1.0 - 2.0 | Miedo |
| -1.0 a 1.0 | Neutral |
| -2.0 a -1.0 | Optimismo |
| < -2.0 | Euforia |