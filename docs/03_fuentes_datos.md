# Fuentes de Datos

| Fuente | Proveedor | Archivo | Actualizacion |
|--------|-----------|---------|---------------|
| Precios (ETFs, acciones, indices) | Yahoo Finance | data/providers/yahoo.py | Diaria (< 1 dia) |
| Datos macro (WALCL, SOFR, RRP) | FRED / archivos manuales | data/providers/fred.py, data/macro_manual/ | Semanal |
| Opciones (PCR, IHR) | CBOE | data/providers/cboe.py | Diaria (1-2 dias) |
| Dark Pools (ATS) | FINRA | data/providers/finra.py | Semanal |

## Cache
- CACHE_HOURS = 23: los datos de mercado se cachean por 23 horas.
- CACHE_TTL: por proveedor (yahoo=23h, fred=168h, cboe=24h, finra=168h).
