# Pesos para el Macro Score combinado (jerarquía de niveles)
LEVEL_WEIGHTS = {
    'critical': 0.60,    # curva, crédito, volatilidad, liquidez
    'important': 0.30,   # dólar, materias primas, breadth
    'contextual': 0.10,  # índices bursátiles (market_strength)
}

# Sub-pesos dentro de cada nivel
CRITICAL_WEIGHTS = {
    'curve': 0.30,
    'credit': 0.30,
    'volatility': 0.25,
    'liquidity': 0.15,
}

IMPORTANT_WEIGHTS = {
    'dollar': 0.40,
    'commodities': 0.40,
    'breadth': 0.20,
}

CONTEXTUAL_WEIGHTS = {
    'market_strength': 1.0,
}

# Pesos para Sector Score (actualizados v1.4)
SECTOR_SCORE_WEIGHTS = {
    'rs_mom_20': 0.25,
    'rs_mom_50': 0.15,
    'rs_mom_126': 0.10,
    'trend': 0.15,
    'volatility_inv': 0.15,
    'breadth': 0.20,
}
