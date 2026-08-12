# Indicadores: Market Transition Engine (MTE v1.0)

## Proposito
Infiere el escenario macro que el mercado parece estar descontando, basado en 4 motores (SRS, SHS, CLS, IPS).

## Motores
- **SRS (Sector Rotation Score):** rotacion sectorial.
- **SHS (Safe Haven Score):** demanda de activos refugio.
- **CLS (Credit/Liquidity Stress Score):** estres en credito/liquidez.
- **IPS (Inflation Pressure Score):** presion inflacionaria.

## Indices
- **MSI (Market Stress Index):** SRS + SHS + CLS (0-100).
- **IPI (Inflation Pressure Index):** basado en IPS (0-100).

## Escenarios
CRISIS, RECESSION, STAGFLATION, SOFT LANDING, EXPANSION, MIXED.

## Confianza
```
Calcula el MTE completo y devuelve un diccionario con todos los resultados.
```
