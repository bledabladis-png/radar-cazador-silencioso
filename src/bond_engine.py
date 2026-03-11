"""
bond_engine.py - Motor de bonos (versión mejorada)
Calcula el score de crédito usando spread de retornos JNK - LQD,
con momentum multi-horizonte (5,21) con pesos fijos [0.6, 0.4],
persistencia de 3 días y normalización robusta (MAD) con ventana 252.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class BondEngine:
    """
    Motor de bonos que mide apetito por riesgo crediticio (high yield vs investment grade).
    Genera un score entre -1 y 1.
    """
    
    def __init__(self, config_path='config/config.yaml'):
        self.tickers = ['JNK', 'LQD']
        self.momentum_windows = [5, 21]
        self.momentum_weights = [0.6, 0.4]
        self.persistence_days = 3
        self.scaling_window = 252

    def calcular_todo(self, df):
        missing = [t for t in self.tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        returns = df[self.tickers].pct_change(fill_method=None).dropna()
        spread = returns['JNK'] - returns['LQD']

        ma_signals = []
        for window in self.momentum_windows:
            ma = spread.rolling(window=window, min_periods=window).mean()
            ma_signals.append(ma)

        momentum = pd.Series(0.0, index=spread.index)
        for weight, ma in zip(self.momentum_weights, ma_signals):
            momentum += weight * ma

        signo = np.sign(momentum)
        cambio = signo.diff() != 0
        grupo = cambio.cumsum()
        consecutivos = grupo.groupby(grupo).cumcount() + 1
        factor = np.minimum(1.0, consecutivos / self.persistence_days)
        momentum_persist = momentum * factor
        momentum_persist[signo == 0] = 0.0

        scaling = robust_scale(momentum_persist, window=self.scaling_window).shift(1)
        scaling = scaling.ffill().fillna(0.5)
        score = np.tanh(momentum_persist / scaling)

        resultados = pd.DataFrame(index=df.index, columns=['score_bonds'])
        resultados['score_bonds'] = score
        resultados['score_bonds'] = resultados['score_bonds'].ffill().fillna(0)

        return resultados


if __name__ == "__main__":
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])
    engine = BondEngine()
    resultado = engine.calcular_todo(df)
    print("\nScore de bonos (últimos 5 días):")
    print(resultado.tail())