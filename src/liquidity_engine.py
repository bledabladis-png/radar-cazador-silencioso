"""
liquidity_engine.py - Motor de liquidez
Combina tres componentes: -DXY, TIP-TLT, y spread de curva 10y2y.
Cada componente se normaliza por separado y se promedia.
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

class LiquidityEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        cfg = self.config.get('liquidity_engine', {})
        self.tickers = cfg.get('tickers', ['DX-Y.NYB', 'TIP', 'TLT', 'spread_10y2y'])
        self.momentum_windows = cfg.get('horizons', [1, 5, 21])
        self.momentum_weights = cfg.get('weights', [0.5, 0.3, 0.2])
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)
        self.invert_dxy = cfg.get('invert_dxy', True)

    def calcular_todo(self, df):
        required = [t for t in self.tickers if t != 'spread_10y2y']
        missing = [t for t in required if t not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en df: {missing}")

        scores = []

        # Componente 1: DXY invertido
        if self.invert_dxy:
            dxy_raw = -df['DX-Y.NYB'].pct_change(fill_method=None)
        else:
            dxy_raw = df['DX-Y.NYB'].pct_change(fill_method=None)
        scores.append(dxy_raw)

        # Componente 2: TIP - TLT
        tip_ret = df['TIP'].pct_change(fill_method=None)
        tlt_ret = df['TLT'].pct_change(fill_method=None)
        tip_tlt_raw = tip_ret - tlt_ret
        scores.append(tip_tlt_raw)

        # Componente 3: spread de curva 10y2y (en niveles, normalizado)
        if 'spread_10y2y' in df.columns:
            spread_raw = df['spread_10y2y']
        else:
            logger.warning("spread_10y2y no encontrado, usando 0")
            spread_raw = pd.Series(0, index=df.index)

        # Normalizar cada componente con el mismo pipeline
        componentes_norm = []
        for raw in scores:
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
            norm = np.tanh(momentum_persist / scaling)
            componentes_norm.append(norm)

        # Añadir spread de curva normalizado
        # Para spread_10y2y (que ya es un nivel, no retorno)
        ma_signals = []
        for window in self.momentum_windows:
            ma = spread_raw.rolling(window=window, min_periods=window).mean()
            ma_signals.append(ma)
        momentum_spread = pd.Series(0.0, index=spread_raw.index)
        for weight, ma in zip(self.momentum_weights, ma_signals):
            momentum_spread += weight * ma

        signo_spread = np.sign(momentum_spread)
        cambio_spread = signo_spread.diff() != 0
        grupo_spread = cambio_spread.cumsum()
        consecutivos_spread = grupo_spread.groupby(grupo_spread).cumcount() + 1
        factor_spread = np.minimum(1.0, consecutivos_spread / self.persistence_days)
        momentum_persist_spread = momentum_spread * factor_spread
        momentum_persist_spread[signo_spread == 0] = 0.0

        scaling_spread = robust_scale(momentum_persist_spread, window=self.scaling_window).shift(1)
        scaling_spread = scaling_spread.ffill().fillna(0.5)
        norm_spread = np.tanh(momentum_persist_spread / scaling_spread)
        componentes_norm.append(norm_spread)

        # Combinar componentes (promedio simple)
        combined = pd.concat(componentes_norm, axis=1).mean(axis=1)

        resultados = pd.DataFrame(index=df.index, columns=['score_liquidity'])
        resultados['score_liquidity'] = combined
        resultados['score_liquidity'] = resultados['score_liquidity'].ffill().fillna(0)
        return resultados

if __name__ == "__main__":
    from src.data_layer import DataLayer
    logging.basicConfig(level=logging.INFO)
    dl = DataLayer()
    df = dl.load_latest()
    engine = LiquidityEngine()
    resultado = engine.calcular_todo(df)
    print(resultado.tail())
