"""
regime_engine.py - Motor de régimen macroeconómico (versión mejorada)
Calcula el score de régimen usando spread de retornos SPY - TLT,
con momentum multi-horizonte (5,21) con pesos fijos [0.6, 0.4],
persistencia de 3 días y normalización robusta (MAD) con ventana 252.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class RegimeEngine:
    """
    Motor de régimen que genera un score entre -1 y 1 indicando el apetito por riesgo.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        # Parámetros actualizados: ventanas 5 y 21, pesos 0.6 y 0.4
        self.tickers = ['SPY', 'TLT']
        self.momentum_windows = [5, 21]
        self.momentum_weights = [0.6, 0.4]
        self.persistence_days = 3
        self.scaling_window = 252

    def calcular_todo(self, df):
        """
        Calcula el score de régimen para todas las fechas en df.
        
        Parámetros:
        df : DataFrame con columnas 'SPY' y 'TLT' (precios de cierre).
        
        Retorna:
        DataFrame con una columna 'score_regime' indexada igual que df.
        """
        # Verificar columnas necesarias
        required = self.tickers
        missing = [t for t in required if t not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        # 1. Calcular retornos diarios (evitando warning por fill_method)
        returns = df[required].pct_change(fill_method=None).dropna()

        # 2. Spread de retornos diarios (riesgo - refugio)
        spread = returns['SPY'] - returns['TLT']

        # 3. Momentum multi-horizonte: medias móviles de 5 y 21 días
        ma_signals = []
        for window in self.momentum_windows:
            ma = spread.rolling(window=window, min_periods=window).mean()
            ma_signals.append(ma)
        
        momentum = pd.Series(0.0, index=spread.index)
        for weight, ma in zip(self.momentum_weights, ma_signals):
            momentum += weight * ma

        # 4. Persistencia: amplificar señales sostenidas
        signo = np.sign(momentum)
        cambio = signo.diff() != 0
        grupo = cambio.cumsum()
        consecutivos = grupo.groupby(grupo).cumcount() + 1
        factor = np.minimum(1.0, consecutivos / self.persistence_days)
        momentum_persist = momentum * factor
        momentum_persist[signo == 0] = 0.0

        # 5. Normalización robusta (MAD) con ventana 252, evitando lookahead
        scaling = robust_scale(momentum_persist, window=self.scaling_window).shift(1)
        # Manejar posibles NaN en scaling (ya no hay ceros gracias al epsilon)
        scaling = scaling.ffill().fillna(0.5)
        score = np.tanh(momentum_persist / scaling)

        # Construir resultado
        resultados = pd.DataFrame(index=df.index, columns=['score_regime'])
        resultados['score_regime'] = score
        # Rellenar NaN iniciales
        resultados['score_regime'] = resultados['score_regime'].ffill().fillna(0)

        return resultados


if __name__ == "__main__":
    # Prueba rápida
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])
    engine = RegimeEngine()
    resultado = engine.calcular_todo(df)
    print("\nScore de régimen (últimos 5 días):")
    print(resultado.tail())