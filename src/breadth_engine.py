"""
breadth_engine.py - Motor de amplitud de mercado
Calcula el score de breadth usando spread de retornos RSP/SPY e IWM/SPY,
con pesos 0.6 y 0.4 respectivamente.
Momentum multi-horizonte (1,5,21) con pesos [0.5,0.3,0.2], persistencia 3d.
Lee los parámetros desde config.yaml.
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

class BreadthEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        cfg = self.config.get('breadth_engine', {})
        self.tickers = cfg.get('tickers', ['RSP', 'IWM', 'SPY'])
        self.momentum_windows = cfg.get('horizons', [1, 5, 21])
        self.momentum_weights = cfg.get('weights', [0.5, 0.3, 0.2])
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)
        self.spread_weights = cfg.get('spread_weights', [0.6, 0.4])

    def calcular_todo(self, df):
        required = self.tickers
        missing = [t for t in required if t not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        # Calcular retornos
        ret_spy = df['SPY'].pct_change(fill_method=None)
        ret_rsp = df['RSP'].pct_change(fill_method=None)
        ret_iwm = df['IWM'].pct_change(fill_method=None)

        # Spreads
        spread_rsp = ret_rsp - ret_spy
        spread_iwm = ret_iwm - ret_spy

        # Combinación lineal
        raw = self.spread_weights[0] * spread_rsp + self.spread_weights[1] * spread_iwm

        # Momentum multi-horizonte
        ma_signals = []
        for window in self.momentum_windows:
            ma = raw.rolling(window=window, min_periods=window).mean()
            ma_signals.append(ma)

        momentum = pd.Series(0.0, index=raw.index)
        for weight, ma in zip(self.momentum_weights, ma_signals):
            momentum += weight * ma

        # Persistencia
        signo = np.sign(momentum)
        cambio = signo.diff() != 0
        grupo = cambio.cumsum()
        consecutivos = grupo.groupby(grupo).cumcount() + 1
        factor = np.minimum(1.0, consecutivos / self.persistence_days)
        momentum_persist = momentum * factor
        momentum_persist[signo == 0] = 0.0

        # Normalización robusta
        scaling = robust_scale(momentum_persist, window=self.scaling_window).shift(1)
        scaling = scaling.ffill().fillna(0.5)
        score = np.tanh(momentum_persist / scaling)

        resultados = pd.DataFrame(index=df.index, columns=['score_breadth'])
        resultados['score_breadth'] = score
        resultados['score_breadth'] = resultados['score_breadth'].ffill().fillna(0)
        return resultados

if __name__ == "__main__":
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    engine = BreadthEngine()
    resultado = engine.calcular_todo(df)
    print(resultado.tail())
