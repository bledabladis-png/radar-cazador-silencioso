# Arquitectura General

## Premisas Fundamentales
- **NO trading bot:** el sistema no genera ordenes, no sugiere timing, no automatiza rotacion de cartera.
- **NO sobreingenieria:** no se usa ML, optimizacion de parametros ni complejidad gratuita. Codigo determinista y transparente.
- **Toda decision final de inversion es humana.**

## Flujo Principal
1. Descarga de datos de mercado (Yahoo Finance, FRED, CBOE, FINRA).
2. Validacion de datos (NaN, cobertura).
3. Calculo de regimenes (Macro, Financial, Liquidity, Volatility, Sector).
4. Motores tactico y estructural para cada sector.
5. Indicadores: momentum, flujo, breadth, Wyckoff, opciones, Dark Pools, MTE.
6. SLPM (Structural Leadership) para auditar liderazgo del sector #1.
7. Generacion de rankings y reporte Markdown.

## Estructura de Modulos
- 
un.py: orquestador principal.
- config/: settings, tickers, weights.
- 
egimes/: condiciones financieras, liquidez, volatilidad, macro, sector.
- indicators/: todos los indicadores y scores.
- src/: carga de datos, generacion de reporte, utilidades.
- data/: providers (yahoo, cboe, finra, fred), datos macro manuales.
- alidation/: scripts de auditoria y backtesting.
