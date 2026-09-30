# -*- coding: utf-8 -*-
"""Filtro de tickers de holdings (CSVs de ETF).

D11 (2026-09-30). Los parsers volcaban al CSV todo ticker no vacio:
efectivo ("-"), futuros (IXAU6, XARU6), CUSIPs (2602335D), placeholders
del proveedor (999USDZ92, ADI394XJ2). El consumidor compensaba con
INVALID_TICKERS hardcodeada, lista fragil que se queda corta cuando
SSGA renueva los futuros cada trimestre.

Regla (sin regex complejas, sin listas fragiles):
1. Primera letra: A-Z (rechaza 2602335D, 999USDZ92).
2. Ultima letra: A-Z (rechaza IXAU6, XARU6, P5N994, ADI394XJ2).
3. Longitud <= 8 (rechaza US DOLLAR despues de normalizar).
4. Caracteres permitidos: A-Z, 0-9, ".", "-".
5. Maximo un separador "." o "-" (rechaza AB.CD.EF, AB--CD).
6. Espacios normalizados a "-" antes de validar (BH A -> BH-A).

Aplicado en 4 parsers:
- scripts/update_sector_holdings.py (SSGA, 11 ETF sectoriales)
- scripts/update_index_holdings.py (SSGA SPY/DIA, Invesco QQQ,
  BlackRock IWM)
- scripts/parse_ssga_fez.py (SSGA FEZ)
- scripts/parse_blackrock_final.py (BlackRock DAXEX, ISF)

Tests: tests/test_holdings_filter.py.
"""

_MAX_LEN = 8
_ALLOWED_SEPARATORS = ".-"
_ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")


def is_valid_holding_ticker(ticker):
    """True si ticker es equity/ETF valido segun la regla D11."""
    if not isinstance(ticker, str):
        return False
    s = ticker.strip().upper().replace(" ", "-")
    if not s or len(s) > _MAX_LEN:
        return False
    if s[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return False
    if s[-1] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return False
    if not all(c in _ALLOWED_CHARS for c in s):
        return False
    n_sep = sum(1 for c in s if c in _ALLOWED_SEPARATORS)
    if n_sep > 1:
        return False
    return True
