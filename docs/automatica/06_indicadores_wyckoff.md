## Proposito
Proporciona un score continuo de estructura de precios para ETFs sectoriales y acciones lideres, basado en los principios de Wyckoff (acumulacion/distribucion).

## Arquitectura

- wyckoff_structural_score(): trend + ATR (70%).
- wyckoff_tactical_score(): volume + effort (30%).
- wyckoff_score(): combinacion ponderada de ambos.
- wyckoff_structure_core(): clasifica en MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.


## Formulas

**Score Estructural:** Construye DataFrame OHLCV limpio para un ticker.

    I2 (2026-09-18): wyckoff_structure_core degenera al fallback
    silencioso RANGE cuando recibe un df con NaN internos. La
    construccion de ticker_df con dropna() es el patron correcto,
    ya aplicado en sector_breadth.py y stock_leader.py.

    Args:
        df: DataFrame multi-ticker (MultiIndex field,ticker) o
            columnas planas si el ticker esta solo.
        ticker: ticker a extraer.

    Returns:
        DataFrame con columnas Open/High/Low/Close/Volume sin NaN.

    Raises:
        KeyError: si el ticker no existe en df.
**Score Tactico:** Construye DataFrame OHLCV limpio para un ticker.

    I2 (2026-09-18): wyckoff_structure_core degenera al fallback
    silencioso RANGE cuando recibe un df con NaN internos. La
    construccion de ticker_df con dropna() es el patron correcto,
    ya aplicado en sector_breadth.py y stock_leader.py.

    Args:
        df: DataFrame multi-ticker (MultiIndex field,ticker) o
            columnas planas si el ticker esta solo.
        ticker: ticker a extraer.

    Returns:
        DataFrame con columnas Open/High/Low/Close/Volume sin NaN.

    Raises:
        KeyError: si el ticker no existe en df.
**Score Combinado:** Construye DataFrame OHLCV limpio para un ticker.

    I2 (2026-09-18): wyckoff_structure_core degenera al fallback
    silencioso RANGE cuando recibe un df con NaN internos. La
    construccion de ticker_df con dropna() es el patron correcto,
    ya aplicado en sector_breadth.py y stock_leader.py.

    Args:
        df: DataFrame multi-ticker (MultiIndex field,ticker) o
            columnas planas si el ticker esta solo.
        ticker: ticker a extraer.

    Returns:
        DataFrame con columnas Open/High/Low/Close/Volume sin NaN.

    Raises:
        KeyError: si el ticker no existe en df.


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
