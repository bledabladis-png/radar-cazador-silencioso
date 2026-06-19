# Tickers de Yahoo Finance (todos gratuitos)
MARKET_TICKERS = {
    'equity': {
        'sp500': '^GSPC',
        'nasdaq100': '^NDX',
        'russell2000': '^RUT',
        'eurostoxx50': '^STOXX50E',
        'msci_em': 'EEM',
    },
    'sectors': [
        'XLK','XLF','XLV','XLE','XLY','XLP',
        'XLI','XLB','XLU','XLRE','XLC'
    ],
    'bonds': {
        '10y': '^TNX',
        '5y': '^FVX',
        '30y': '^TYX',
    },
    'credit': {
        'hyg': 'HYG',
        'lqd': 'LQD',
    },
    'volatility': {
        'vix': '^VIX',
        'vix3m': '^VIX3M',
        'vxn': '^VXN',
        'iwm': 'IWM',          # Proxy Russell 2000 (calculamos volatilidad historica)
    },
    'currencies': {
        'dxy': 'DX-Y.NYB',
        'eurusd': 'EURUSD=X',
        'usdjpy': 'USDJPY=X',
        'usdcny': 'USDCNY=X',
    },
    'commodities': {
        'gsci': '^SPGSCI',
        'gold': 'GC=F',
        'copper': 'HG=F',
        'wti': 'CL=F',
        'brent': 'BZ=F',
    },
}

# Mapeo de sectores a nombres legibles
SECTOR_NAMES = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLY': 'Consumer Discretionary',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
}

# Clasificacion de sectores
CYCLICAL_SECTORS = ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'XLE']
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE', 'XLC']