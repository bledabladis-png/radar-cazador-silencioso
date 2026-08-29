## Proposito
El Radar de Rotacion Sectorial es un sistema informativo diario que analiza flujos institucionales, contexto macro, estructura de precios (Wyckoff) y estructura del mercado de opciones para producir rankings, tablas y analisis para el gestor humano.

## Arquitectura

- 
un.py: orquestador principal.
- config/: settings, tickers, weights.
- 
egimes/: condiciones financieras, liquidez, volatilidad, macro, sector.
- indicators/: todos los indicadores y scores.
- src/: carga de datos, generacion de reporte, utilidades.
- data/: providers (yahoo, cboe, finra, fred), datos macro manuales.
- alidation/: scripts de auditoria y backtesting.


## Formulas
No aplica (modulo estructural).

## Salidas
Reporte diario en Markdown (outputs/report/reporte_diario.md).
