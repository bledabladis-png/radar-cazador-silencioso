"""
leadership_engine.py - Motor de liderazgo sectorial
Calcula el score de liderazgo usando spread de retornos XLK - XLF,
con momentum multi-horizonte (1,5,21) con pesos [0.5,0.3,0.2],
persistencia de 3 días y normalización robusta (MAD).
Lee los parámetros desde config.yaml para flexibilidad.
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

class LeadershipEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        cfg = self.config.get('leadership_engine', {})
        self.tickers = cfg.get('tickers', ['XLK', 'XLF'])
        self.momentum_windows = cfg.get('horizons', [1, 5, 21])
        self.momentum_weights = cfg.get('weights', [0.5, 0.3, 0.2])
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)

    def calcular_todo(self, df):
        required = self.tickers
        missing = [t for t in required if t not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        returns = df[required].pct_change(fill_method=None).dropna()
        spread = returns[self.tickers[0]] - returns[self.tickers[1]]

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

        resultados = pd.DataFrame(index=df.index, columns=['score_leadership'])
        resultados['score_leadership'] = score
        resultados['score_leadership'] = resultados['score_leadership'].ffill().fillna(0)
        return resultados

if __name__ == "__main__":
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    engine = LeadershipEngine()
    resultado = engine.calcular_todo(df)
    print(resultado.tail())
