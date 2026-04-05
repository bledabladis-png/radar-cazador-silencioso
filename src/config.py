# config.py
tickers = {
    'benchmark': 'SPY',
    'vix': '^VIX',
    'sectors': ['XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLV', 'XLU', 'XLRE'],
    'global': 'ACWI',
    'bonds': 'TLT',
    'macro': ['TLT', 'ACWI', 'QQQ']
}

windows = {
    'rel_momentum': 60,
    'breadth': 200,
    'vix_change': 20
}

# Mapeo CFTC corregido (sin errores conceptuales)
CFTC_MARKETS = {
    # Core equity (sectores con mapeo directo)
    "SPY": "E-MINI S&P 500",
    "XLK": "E-MINI S&P TECHNOLOGY INDEX",
    "XLE": "E-MINI S&P ENERGY INDEX",
    "XLF": "E-MINI S&P FINANCIAL INDEX",
    "XLI": "E-MINI S&P INDUSTRIAL INDEX",
    "XLP": "E-MINI S&P CONSU STAPLES INDEX",
    "XLV": "E-MINI S&P HEALTH CARE INDEX",
    "XLU": "E-MINI S&P UTILITIES INDEX",
    "XLRE": "DOW JONES U.S. REAL ESTATE IDX",

    # Macro layers (corregidas)
    "COMMODITIES": "BBG COMMODITY",
    "VOL": "VIX FUTURES",
    "USD": "USD INDEX",

    # Rates (tipos de interés)
    "RATES_2Y": "UST 2Y NOTE",
    "RATES_5Y": "UST 5Y NOTE",
    "RATES_10Y": "UST 10Y NOTE",
    "RATES_LONG": "ULTRA UST BOND",
    "FED": "FED FUNDS",
    "SOFR": "SOFR-3M"
}