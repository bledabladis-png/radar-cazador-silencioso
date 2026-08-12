## Proposito
Mide el porcentaje de volumen negociado en ATS (Alternative Trading Systems) respecto al volumen total, usando datos de FINRA.

## Arquitectura

- compute_darkpool_signals(): orquestador principal.
- FinraProvider: descarga datos ATS semanales.
- _compute_z_for_window(): calcula Z-Score robusto para cada ventana.
- classify_darkpool(): clasifica en Acumulacion/Distribucion extrema, fuerte, moderada o Neutral.


## Formulas

- Z-Score robusto (mediana/MAD) para ventanas de 13, 26, 52 y 104 semanas.
- % ATS medio: media del porcentaje de volumen ATS entre todos los tickers.


## Salidas

- Seccion 'Actividad en ATS - Dark Pools (FINRA v1.0)' en el reporte.
- % Volumen en ATS medio, Z-Scores por ventana, Top 5 tickers.
- Advertencia de obsolescencia si los datos tienen >21 dias.


## Limitaciones Conocidas
Los datos de FINRA pueden tener un desfase de varias semanas. Si la antiguedad supera los 21 dias, los datos no se usan para clasificacion actual.

## Constantes
| Constante | Valor |
|---|---|
| DARKPOOL_FULL_HISTORY_WEEKS | 104 |
| DARKPOOL_MIN_HISTORY_WEEKS | 13 |
| DARKPOOL_ZSCORE_WINDOWS | (13, 26, 52, 104) |
