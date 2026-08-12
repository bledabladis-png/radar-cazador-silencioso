# Auditoria Fase 3 - Frescura y Proveedores de Datos

**Fecha:** 2026-08-12 17:58

## Resultados

| Fuente | Archivo | Ultimo dato | Antiguedad (dias) | Estado |
|--------|---------|-------------|-------------------|--------|
| CBOE (Opciones) | outputs/pcr_history.csv | 2026-08-11 | 1 | CURRENT |
| FINRA (Dark Pools) | outputs/darkpool_history.csv | 2026-07-20 | 23 | ARCHIVAL |
| FRED/Macro Manual | data/macro_manual | 2026-07-22 | 21 | STALE |
| Yahoo Finance (Precios) | Descarga en vivo | HOY | 0 | CURRENT |

## Configuracion de Calidad

- MAX_NAN_RATIO = 0.1
- EXPECTED_SECTOR_COUNT = 11
- MIN_VALID_SECTORS = 8
- MIN_SECTOR_COVERAGE = 0.8
- MIN_VALID_TICKERS = 5

*Esta auditoria verifica la antiguedad de los datos almacenados. La validacion completa de cobertura se realiza durante la ejecucion de run.py.*
