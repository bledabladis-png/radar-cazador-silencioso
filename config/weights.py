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

# ============================================================
# SLPM WEIGHTS (Leader Breadth & LIS)
# ============================================================

SLPM_WEIGHTS = {
    'leader_breadth': {
        'rs': 0.30,
        'momentum': 0.25,
        'flow': 0.25,
        'wyckoff': 0.20
    },
    'lis': {
        'rs': 0.30,
        'momentum': 0.25,
        'flow': 0.25,
        'wyckoff': 0.20
    }
}

def validate_weights():
    """Valida que todos los grupos de pesos sumen 1.0 y no tengan valores negativos."""
    groups = {
        'LEVEL_WEIGHTS': LEVEL_WEIGHTS,
        'CRITICAL_WEIGHTS': CRITICAL_WEIGHTS,
        'IMPORTANT_WEIGHTS': IMPORTANT_WEIGHTS,
        'CONTEXTUAL_WEIGHTS': CONTEXTUAL_WEIGHTS,
        'SECTOR_SCORE_WEIGHTS': SECTOR_SCORE_WEIGHTS,
        'SLPM_LEADER_BREADTH': SLPM_WEIGHTS['leader_breadth'],
        'SLPM_LIS': SLPM_WEIGHTS['lis'],
    }
    
    for name, weights in groups.items():
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f'{name} suma {total:.6f}, esperado 1.0')
        for k, v in weights.items():
            if v < 0:
                raise ValueError(f'{name}[{k}] es negativo: {v}')
    
    return True

# ============================================================
# TACTICAL ENGINE WEIGHTS
# ============================================================

TACTICAL_WEIGHTS = {
    'rs20': 0.30,
    'momentum20': 0.25,
    'flow_recent': 0.20,
    'breadth20': 0.15,
    'acceleration': 0.10,
}

# ============================================================
# STRUCTURAL ENGINE WEIGHTS
# ============================================================

STRUCTURAL_WEIGHTS = {
    'rs_structural': 0.35,
    'leader_breadth': 0.25,
    'flow_structure': 0.20,
    'persistence': 0.20,
}

# ============================================================
# SECTOR SCORE DISPERSION PENALTY
# Factor que reduce el score cuando hay desacuerdo entre
# los 6 sub-componentes del Sector Score.
# 0.0 = sin penalización, 1.0 = máxima penalización.
# ============================================================
SECTOR_DISPERSION_PENALTY = 0.5
