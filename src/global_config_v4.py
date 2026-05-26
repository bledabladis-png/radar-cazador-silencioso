# global_config_v4.py – Configuración del Radar Global v4.0

# Activos por módulo funcional
FLOW_ASSETS = {
    'equity': ['SPY', 'EZU', 'EWJ', 'EEM'],
    'fixed_income': ['TLT', 'HYG'],
    'commodities': ['GLD', 'DBC', 'XOP'],
    'dollar_proxy': 'UUP'
}

RISK_ASSETS = {
    'equity': ['SPY', 'EZU', 'EWJ', 'EEM'],
    'credit': ['HYG'],
    'safe_havens': ['TLT', 'GLD'],
    'commodities': ['DBC', 'XOP']
}

CROSS_REGION = {
    'americas': ['SPY'],
    'europe': ['EZU'],
    'asia_pacific': ['EWJ'],
    'emerging': ['EEM']
}

# Pesos máximos para evitar dominancia de SPY
MAX_SINGLE_ASSET_WEIGHT = 0.25

# Día de anclaje semanal (Friday close)
WEEKLY_ANCHOR_DAY = 4          # Friday (0=Mon, 4=Fri)

# Ventanas temporales
WINDOWS = {
    'volume_zscore': 60,
    'volatility': 20,
    'correlation': 4,        # semanas para coherence
    'breadth': 1,            # instantáneo (última semana)
    'ewm_halflife': 2
}

# Calidad de flow proxy por activo
FLOW_QUALITY = {
    'SPY': 1.0, 'EZU': 0.8, 'EWJ': 0.75, 'EEM': 0.7,
    'TLT': 0.9, 'HYG': 0.6, 'GLD': 0.6, 'DBC': 0.6, 'UUP': 0.7
}