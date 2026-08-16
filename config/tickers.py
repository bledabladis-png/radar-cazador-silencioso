# -*- coding: utf-8 -*-
"""
Universo de activos del Radar de Rotación Sectorial v4.3.
"""

MARKET_TICKERS = {
    'equity': {
        'sp500': '^GSPC', 'nasdaq100': '^NDX', 'russell2000': '^RUT',
        'eurostoxx50': '^STOXX50E', 'msci_em': 'EEM', 'japan': 'EWJ',
        'spy': 'SPY', 'qqq': 'QQQ', 'iwm': 'IWM', 'msci_world': 'URTH',
        'dow': '^DJI', 'ibex': '^IBEX', 'dax': '^GDAXI', 'ftse': '^FTSE',
    },
    'sectors': ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLRE', 'XLU', 'XLC'],
    'bonds': ['BIL', 'IEF', 'TLT'],
    'credit': ['HYG', 'LQD'],
    'volatility': ['^VIX', '^VIX3M', '^VXN'],
    'currencies': ['DX-Y.NYB', 'EURUSD=X', 'USDJPY=X', 'USDCNY=X'],
    'commodities': ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F', 'BZ=F', 'NG=F'],
    'factors': ['VLUE', 'MTUM', 'QUAL'],
    'small_caps_intl': ['SCHC', 'EWX'],
    'emerging_bonds': ['EMB', 'ELD'],
    'commodity_etfs': ['GLD', 'SLV', 'USO', 'UNG'],
    'sector_proxies': ['KRE', 'SMH', 'IYT'],
    'indices': ['^GSPC', '^DJI', '^NDX', '^RUT', '^STOXX50E', '^IBEX', '^GDAXI', '^FTSE'],
}

SECTOR_NAMES = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLRE': 'Real Estate',
    'XLU': 'Utilities', 'XLC': 'Communication Services',
}

# Clasificación corregida en v3.15: XLC pasa de defensivo a cíclico.
# Communication Services tiene un comportamiento más próximo a Technology
# y Consumer Discretionary que a Utilities o Consumer Staples.
CYCLICAL_SECTORS = ['XLK', 'XLY', 'XLI', 'XLF', 'XLB', 'XLE', 'XLC']
DEFENSIVE_SECTORS = ['XLU', 'XLP', 'XLV', 'XLRE']


def validate_sector_universe():
    """
    Valida la coherencia estructural del universo sectorial.
    No descarga datos; solo comprueba consistencia de configuración.
    """
    sectors = MARKET_TICKERS['sectors']
    sector_set = set(sectors)
    names_set = set(SECTOR_NAMES.keys())
    cyclical_set = set(CYCLICAL_SECTORS)
    defensive_set = set(DEFENSIVE_SECTORS)

    errors = []

    # 1. No duplicados en la lista de sectores
    if len(sectors) != len(sector_set):
        errors.append("Duplicate sector ticker in MARKET_TICKERS['sectors']")

    # 2. Todos los sectores tienen nombre
    if sector_set != names_set:
        missing_names = sector_set - names_set
        extra_names = names_set - sector_set
        if missing_names:
            errors.append(f"Sectores sin nombre en SECTOR_NAMES: {missing_names}")
        if extra_names:
            errors.append(f"Nombres sin sector en MARKET_TICKERS: {extra_names}")

    # 3. Cíclicos y defensivos no se solapan
    overlap = cyclical_set & defensive_set
    if overlap:
        errors.append(f"Sectores clasificados como cíclicos y defensivos simultáneamente: {overlap}")

    # 4. Todos los sectores están clasificados
    classified = cyclical_set | defensive_set
    unclassified = sector_set - classified
    if unclassified:
        errors.append(f"Sectores sin clasificación (ni cíclico ni defensivo): {unclassified}")

    # 5. No hay sectores clasificados que no existan en el universo
    extra = classified - sector_set
    if extra:
        errors.append(f"Sectores clasificados que no existen en MARKET_TICKERS: {extra}")

    if errors:
        raise ValueError("Validación del universo sectorial fallida:\n- " + "\n- ".join(errors))

    return {
        'total_sectors': len(sectors),
        'cyclical': len(CYCLICAL_SECTORS),
        'defensive': len(DEFENSIVE_SECTORS),
        'status': 'PASS'
    }
