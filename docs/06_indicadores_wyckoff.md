## Proposito
Proporciona un score continuo de estructura de precios para ETFs sectoriales y acciones lideres, basado en los principios de Wyckoff (acumulacion/distribucion).

## Arquitectura

- wyckoff_structural_score(): trend + ATR (70%).
- wyckoff_tactical_score(): volume + effort (30%).
- wyckoff_score(): combinacion ponderada de ambos.
- wyckoff_structure_core(): clasifica en MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.


## Formulas

**Score Estructural:** 0.60*trend_norm + 0.40*compression_norm
**Score Tactico:** 0.50*volume_norm + 0.50*effort_norm
**Score Combinado:** 0.70*structural + 0.30*tactical


## Salidas

- Fase Wyckoff (MARKUP, ACCUMULATION, RANGE, DISTRIBUTION) en rankings sectoriales.
- Wyckoff Leadership Score (WLS) en tablas de lideres.
- Confianza y dispersion de componentes en metadatos.


## Constantes Configurables
| Constante | Valor |
|---|---|
| WYCKOFF_ATR_WINDOW | 20 |
| WYCKOFF_COMBINED_STRUCT_WEIGHT | 0.70 |
| WYCKOFF_COMBINED_TACT_WEIGHT | 0.30 |
| WYCKOFF_MIN_PERIODS | 60 |
| WYCKOFF_STRUCT_WEIGHT_COMPRESSION | 0.40 |
| WYCKOFF_STRUCT_WEIGHT_TREND | 0.60 |
| WYCKOFF_TACT_WEIGHT_EFFORT | 0.50 |
| WYCKOFF_TACT_WEIGHT_VOLUME | 0.50 |
| WYCKOFF_THRESHOLD_ACCUMULATION | 0.00 |
| WYCKOFF_THRESHOLD_DISTRIBUTION | -0.30 |
| WYCKOFF_THRESHOLD_MARKUP | 0.30 |
| WYCKOFF_TREND_FAST_MA | 50 |
| WYCKOFF_TREND_SLOW_MA | 200 |
| WYCKOFF_VOLUME_WINDOW | 20 |
| WYCKOFF_VOLUME_ZSCORE_WINDOW | 60 |
