# Modulo Wyckoff (v4.2)

## Proposito
Proporciona un score continuo de estructura de precios para ETFs sectoriales y acciones lideres.

## Arquitectura
- wyckoff_structural_score(): trend + ATR (70%).
- wyckoff_tactical_score(): volume + effort (30%).
- wyckoff_score(): combinacion ponderada de ambos.
- wyckoff_structure_core(): clasifica en MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.

## Constantes Configurables
| Constante | Valor |
|---|---|
| WYCKOFF_ATR_WINDOW | 20 |
| WYCKOFF_COMBINED_STRUCT_WEIGHT | 0.70 |
| WYCKOFF_COMBINED_TACT_WEIGHT | 0.30 |
| WYCKOFF_MIN_PERIODS | 60 |
| WYCKOFF_RANGE_WINDOW | 20 |
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
| WYCKOFF_WEIGHT_EFFORT | 0.20 |
| WYCKOFF_WEIGHT_RANGE | 0.25 |
| WYCKOFF_WEIGHT_TREND | 0.35 |
| WYCKOFF_WEIGHT_VOLUME | 0.20 |