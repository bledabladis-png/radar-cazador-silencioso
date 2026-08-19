## Proposito
Descripcion de los proveedores de datos utilizados por el radar.

## Arquitectura

| Fuente | Proveedor | Archivo | Actualizacion |
|--------|-----------|---------|---------------|
| Precios | Yahoo Finance | data/providers/yahoo.py | Diaria |
| Opciones | CBOE | data/providers/cboe.py | Diaria |
| Dark Pools | FINRA | data/providers/finra.py | Semanal |
| Macro | FRED / manual | data/providers/fred.py, data/macro_manual/ | Semanal |


## Formulas
No aplica.

## Salidas
DataFrames de OHLCV, datos de opciones, datos ATS y series macroeconomicas.
