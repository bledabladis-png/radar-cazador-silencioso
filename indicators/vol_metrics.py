# -*- coding: utf-8 -*-
import numpy as np
from src.utils import get_col

def compute_vol_metrics(df_market):
    """
    Calcula métricas de volatilidad realizada y VRP Proxy (Implied-Realized Volatility Spread).
    Nota: VRP es una proxy que compara VIX (implícita 30d SPX) con RV (realizada SPY).
    No es una medida académicamente pura de la prima de riesgo de volatilidad.
    Retorna un diccionario con los últimos valores.
    """
    result = {}
    
    try:
        spy_close = get_col(df_market, 'SPY', 'Close')
        vix_close = get_col(df_market, '^VIX', 'Close')
        
        returns = spy_close.pct_change()
        
        # Realized Volatility (21 sesiones, corregido)
        rv_21 = returns.rolling(21).std() * np.sqrt(252)
        result['rv_21d'] = float(rv_21.iloc[-1]) if len(rv_21) > 0 else None
        
        # Realized Volatility (60 sesiones)
        rv_60 = returns.rolling(60).std() * np.sqrt(252)
        result['rv_60d'] = float(rv_60.iloc[-1]) if len(rv_60) > 0 else None
        
        # VRP Proxy (VIX - RV21) - Implied-Realized Volatility Spread
        vix_dec = vix_close / 100
        vrp_21 = vix_dec - rv_21
        result['vrp_21d'] = float(vrp_21.iloc[-1]) if len(vrp_21) > 0 else None
        
        # VRP Proxy (VIX - RV60) - Implied-Realized Volatility Spread
        vrp_60 = vix_dec - rv_60
        result['vrp_60d'] = float(vrp_60.iloc[-1]) if len(vrp_60) > 0 else None
        
    except (KeyError, IndexError):
        result = {'rv_21d': None, 'rv_60d': None, 'vrp_21d': None, 'vrp_60d': None}
    
    return result
