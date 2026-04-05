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

CFTC_MARKETS = {
    "SPY": "S&P 500",
    "XLK": "NASDAQ-100",
    "XLE": "CRUDE OIL",
    "XLF": "S&P 500",
}