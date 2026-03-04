"""
leadership_engine.py - Motor de liderazgo temprano
Calcula:
- Small vs Large (IWM/SPY) con factor de confianza
- Cíclico vs Defensivo (XLY/XLP) con factor de confianza
"""

import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class LeadershipEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de liderazgo
        self.max_conf = self.config['indicators']['leadership']['max_conf']
        self.slope_scale = self.config['indicators']['leadership']['slope_scale']

    def _calcular_ratio_score(self, df, num_ticker, denom_ticker):
        """
        Calcula el score para un ratio (ej. IWM/SPY) usando la metodología:
        - EMA5 y EMA200 del ratio
        - spread = EMA5 - EMA200
        - z-score del spread con std rolling 200
        - clip a ±3 y tanh
        - factor de confianza basado en pendiente de EMA20
        """
        ratio = df[num_ticker] / df[denom_ticker]

        ema5 = ratio.ewm(span=5, adjust=False).mean()
        ema200 = ratio.ewm(span=200, adjust=False).mean()
        spread = ema5 - ema200

        std_200 = ratio.rolling(200).std()
        z = spread / std_200
        z = np.clip(z, -3, 3)
        score_base = np.tanh(z)

        # Factor de confianza basado en pendiente de EMA20
        ema20 = ratio.ewm(span=20, adjust=False).mean()
        slope = ema20 - ema20.shift(10)
        conf = 0.5 + 0.5 * np.tanh(slope * self.slope_scale)
        conf = np.clip(conf, 0.5, self.max_conf)

        score = score_base * conf
        return score

    def calcular_small_vs_large(self, df):
        """Score small vs large (IWM/SPY)"""
        return self._calcular_ratio_score(df, 'IWM', 'SPY')

    def calcular_ciclico_vs_defensivo(self, df):
        """Score cíclico vs defensivo (XLY/XLP)"""
        return self._calcular_ratio_score(df, 'XLY', 'XLP')

    def calcular_todo(self, df):
        resultados = pd.DataFrame(index=df.index)
        resultados['score_small'] = self.calcular_small_vs_large(df)
        resultados['score_cyclical'] = self.calcular_ciclico_vs_defensivo(df)

        # Score de liderazgo (media simple, se puede ponderar después)
        resultados['score_leadership'] = (resultados['score_small'] + resultados['score_cyclical']) / 2

        return resultados


if __name__ == "__main__":
    from data_layer import DataLayer

    dl = DataLayer()
    df = dl.load_latest()

    engine = LeadershipEngine()
    resultados = engine.calcular_todo(df)

    print("Últimos 5 días de scores de liderazgo:")
    print(resultados.tail())