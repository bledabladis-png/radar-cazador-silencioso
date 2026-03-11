"""
financial_conditions_engine.py - Motor de Condiciones Financieras
Construye un índice compuesto a partir de:
- VIX (volatilidad)
- Credit spread (HYG - LQD)
- Liquidez monetaria (del motor de liquidez existente)
- (Opcional) Spread de tipos cortos (usando TLT como proxy)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class FinancialConditionsEngine:
    """
    Motor que calcula un índice de condiciones financieras.
    Valores altos = condiciones restrictivas, valores bajos = condiciones expansivas.
    """

    def __init__(self):
        self.z_window = 252  # 1 año para normalización
        self.smooth_window = 5

    def calcular_todo(self, df, liquidity_score=None):
        """
        Calcula el score de condiciones financieras.
        
        Parámetros:
        - df: DataFrame con columnas necesarias (^VIX, HYG, LQD, TLT)
        - liquidity_score: Series opcional con el score de liquidez (si no se proporciona, se calcula un proxy)
        
        Retorna DataFrame con columna 'score_financial_conditions'.
        """
        # 1. VIX (nivel)
        if '^VIX' not in df.columns:
            logger.error("No se encuentra columna ^VIX")
            return pd.DataFrame({'score_financial_conditions': 0}, index=df.index)
        
        vix = df['^VIX'].ffill()
        vix_norm = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        vix_norm = vix_norm.fillna(0).clip(-3, 3)

        # 2. Credit spread (HYG - LQD retornos)
        if 'HYG' in df.columns and 'LQD' in df.columns:
            ret_hyg = df['HYG'].ffill().pct_change(fill_method=None)
            ret_lqd = df['LQD'].ffill().pct_change(fill_method=None)
            credit_spread = ret_hyg - ret_lqd
            # Usamos el cambio del spread como indicador (spreads aumentando = tensión)
            credit_change = credit_spread.diff()
            credit_norm = (credit_change - credit_change.rolling(252).mean()) / credit_change.rolling(252).std()
            credit_norm = credit_norm.fillna(0).clip(-3, 3)
        else:
            logger.warning("No se encuentran HYG o LQD, usando 0 para crédito")
            credit_norm = pd.Series(0, index=df.index)

        # 3. Liquidez (si no se proporciona, usamos el motor de liquidez básico como proxy)
        if liquidity_score is None:
            # Calculamos un proxy simple: retorno de TLT (cuando suben los bonos largos, condiciones más laxas)
            if 'TLT' in df.columns:
                ret_tlt = df['TLT'].ffill().pct_change(fill_method=None)
                liquidity_proxy = -ret_tlt  # invertido: TLT sube = condiciones más laxas
                liquidity_norm = (liquidity_proxy - liquidity_proxy.rolling(252).mean()) / liquidity_proxy.rolling(252).std()
                liquidity_norm = liquidity_norm.fillna(0).clip(-3, 3)
            else:
                liquidity_norm = pd.Series(0, index=df.index)
        else:
            # Usamos el score de liquidez existente (invertido porque el nuestro mide apetito, no condiciones)
            liquidity_norm = -liquidity_score

        # 4. Combinar (pesos sugeridos: 40% VIX, 30% crédito, 30% liquidez)
        raw = 0.4 * vix_norm + 0.3 * credit_norm + 0.3 * liquidity_norm

        # Suavizar
        raw_smooth = raw.rolling(window=self.smooth_window, min_periods=1).mean()

        # Normalizar a score entre -1 y 1 (valores altos = condiciones restrictivas)
        # Queremos que el score final tenga media 0 y desviación 1 aproximadamente
        mean = raw_smooth.rolling(252).mean()
        std = raw_smooth.rolling(252).std()
        z = (raw_smooth - mean) / std
        z = z.fillna(0).clip(-3, 3)
        score = np.tanh(z)

        resultados = pd.DataFrame(index=df.index)
        resultados['score_financial_conditions'] = score
        resultados = resultados.ffill().fillna(0)

        return resultados


if __name__ == "__main__":
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    engine = FinancialConditionsEngine()
    res = engine.calcular_todo(df)
    print("Score Condiciones Financieras (últimos 5 días):")
    print(res.tail())