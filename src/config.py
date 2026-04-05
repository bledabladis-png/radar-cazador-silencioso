# config.py
# Parametros del radar en formato Python

# Tickers
tickers = {
    'benchmark': 'SPY',
    'vix': '^VIX',
    'sectors': ['XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP'],
    'global': 'ACWI',
    'bonds': 'TLT'
}

# Ventanas (dias)
windows = {
    'rel_momentum': 60,
    'breadth': 200,
    'vix_change': 20
}

# Mapeo de activos del radar a nombres de mercado en CFTC (FinFutWk.txt)
CFTC_MARKETS = {
    "SPY": "S&P 500",
    "XLK": "NASDAQ-100",
    "XLE": "CRUDE OIL",
    "XLF": "S&P 500",          # proxy financiero
}
