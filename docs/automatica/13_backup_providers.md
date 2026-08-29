## Proposito
Describe los proveedores de respaldo multi-API y los mecanismos de resiliencia: rate limiting, circuit breaker y validación cruzada.

## Arquitectura

- data/providers/backup_providers.py: Alpha Vantage, Tiingo, Twelve Data, Finnhub, FMP.
- RateLimiter: límites diarios y por minuto por proveedor.
- CircuitBreaker: desactiva temporalmente tras fallos consecutivos.
- Validación cruzada contra caché local.
- data/providers/polygon.py: proveedor Polygon.io / Massive.


## Formulas

- Presupuesto global de respaldo: 20 llamadas por ejecución.
- Límites específicos configurados en RateLimiter por proveedor.


## Salidas

- DataFrames de OHLCV unificados (MultiIndex con ticker canónico).
- Mensajes de trazabilidad: [RESPALDO], [RATE], [CIRCUIT], [VALIDACIÓN].


## Limitaciones Conocidas
Solo se activa cuando Yahoo Finance falla. No reemplaza la fuente primaria.
