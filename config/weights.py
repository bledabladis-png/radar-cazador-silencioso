# Pesos para el Macro Score combinado (jerarquía de niveles)
LEVEL_WEIGHTS = {
    'critical': 0.60,
    'important': 0.30,
    'contextual': 0.10,
}

# Sub-pesos dentro de cada nivel (v3.1 - con liquidez real)
CRITICAL_WEIGHTS = {
    'curve': 0.30,
    'credit': 0.30,
    'volatility': 0.25,
    'liquidity': 0.10,
    'real_liquidity': 0.05,
}

IMPORTANT_WEIGHTS = {
    'dollar': 0.40,
    'commodities': 0.40,
    'breadth': 0.20,
}

CONTEXTUAL_WEIGHTS = {
    'market_strength': 1.0,
}

# Pesos para Sector Score
SECTOR_SCORE_WEIGHTS = {
    'rs_mom_20': 0.25,
    'rs_mom_50': 0.15,
    'rs_mom_126': 0.10,
    'trend': 0.15,
    'volatility_inv': 0.15,
    'breadth': 0.20,
}
