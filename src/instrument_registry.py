"""
Registro de instrumentos canónicos con símbolos por proveedor.
El ticker canónico interno coincide con el de Yahoo Finance.
"""
# Registro de instrumentos con diferencias de símbolos entre proveedores.
INSTRUMENTS = {
    "BRK-B": {
        "yahoo": "BRK-B",
        "polygon": "BRK.B",
        "tiingo": "BRK.B",
        "alpha_vantage": "BRK.B",
        "finnhub": "BRK.B",
        "fmp": "BRK.B",
        "twelve_data": "BRK.B",
    },
    "BF-B": {
        "yahoo": "BF-B",
        "polygon": "BF.B",
        "tiingo": "BF.B",
        "alpha_vantage": "BF.B",
        "finnhub": "BF.B",
        "fmp": "BF.B",
        "twelve_data": "BF.B",
    },
    "MOGA": {
        "yahoo": "MOGA",
        "polygon": "MOGA",
        "tiingo": "MOGA",
        "alpha_vantage": "MOGA",
        "finnhub": "MOGA",
        "fmp": "MOGA",
        "twelve_data": "MOGA",
        "note": "Yahoo falla con MOGA, por eso mapeamos a MOG-A en stock_data_loader y data_loader"
    },
    "MOG-A": {
        "yahoo": "MOG-A",
        "polygon": "MOG-A",
        "tiingo": "MOG-A",
        "alpha_vantage": "MOG-A",
        "finnhub": "MOG-A",
        "fmp": "MOG-A",
        "twelve_data": "MOG-A",
    },
    "^GSPC": {
        "yahoo": "^GSPC",
        "polygon": "I:SPX",
        "tiingo": "^GSPC",
        "alpha_vantage": None,  # Alpha Vantage no soporta índices con ^
        "finnhub": "^GSPC",     # Finnhub usa ^GSPC
        "fmp": "^GSPC",
        "twelve_data": "SPX",
    },
    "^STOXX50E": {
        "yahoo": "^STOXX50E",
        "polygon": "I:STOXX50E",
        "tiingo": "^STOXX50E",
        "alpha_vantage": None,
        "finnhub": "^STOXX50E",
        "fmp": "^STOXX50E",
        "twelve_data": "STOXX50E",
    },
    "^VIX3M": {
        "yahoo": "^VIX3M",
        "polygon": "I:VIX3M",
        "tiingo": "^VIX3M",
        "alpha_vantage": None,
        "finnhub": "^VIX3M",
        "fmp": "^VIX3M",
        "twelve_data": "VIX3M",
    },
    # Acciones europeas con sufijo (canónico ya es con sufijo)
    "SIE.DE": {
        "yahoo": "SIE.DE",
        "polygon": "SIE.DE",
        "tiingo": "SIE.DE",
        "alpha_vantage": "SIE.DE",
        "finnhub": "SIE.DE",
        "fmp": "SIE.DE",
        "twelve_data": "SIE.DE",
    },
    "SAN.MC": {
        "yahoo": "SAN.MC",
        "polygon": "SAN.MC",
        "tiingo": "SAN.MC",
        "alpha_vantage": "SAN.MC",
        "finnhub": "SAN.MC",
        "fmp": "SAN.MC",
        "twelve_data": "SAN.MC",
    },
}

def resolve_symbol(canonical_ticker: str, provider: str):
    """Devuelve el símbolo específico del proveedor para un ticker canónico.
    Si no hay mapeo especial, asume que el ticker es el mismo y lo devuelve sin cambios.
    """
    inst = INSTRUMENTS.get(canonical_ticker)
    if inst:
        symbol = inst.get(provider)
        if symbol is None:
            # Proveedor no soporta el instrumento
            return None
        return symbol
    # Si no está en el registro, asumir que el ticker es directo (acciones USA, ETFs, etc.)
    return canonical_ticker

def is_supported(canonical_ticker: str, provider: str) -> bool:
    """Indica si un proveedor puede proporcionar datos para el ticker canónico."""
    inst = INSTRUMENTS.get(canonical_ticker)
    if inst:
        return inst.get(provider) is not None
    # Para tickers no mapeados, asumimos soporte (puede fallar pero no lo sabremos)
    return True
