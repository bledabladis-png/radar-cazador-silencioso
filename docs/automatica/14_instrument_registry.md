## Proposito
Documenta el registro central de instrumentos que mapea tickers canónicos (Yahoo) a símbolos específicos de cada proveedor.

## Arquitectura

- src/instrument_registry.py: diccionario INSTRUMENTS con equivalencias por proveedor.
- Función resolve_symbol(canonical_ticker, provider) devuelve el símbolo correcto o None.
- Cobertura explícita para BRK-B, BF-B, ^GSPC, ^STOXX50E, ^VIX3M, MOGA y otros.
- Los proveedores normalizan siempre al ticker canónico en sus salidas.


## Formulas
No aplica (mapeo estático).

## Salidas
Tickers normalizados en todos los DataFrames de proveedores, evitando duplicados.

## Limitaciones Conocidas
Requiere actualización manual al añadir nuevos instrumentos o proveedores.
