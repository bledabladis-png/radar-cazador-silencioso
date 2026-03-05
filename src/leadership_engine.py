"""
leadership_engine.py - Motor de liderazgo de mercado
Calcula el score de liderazgo usando:
- Small vs Large: IWM / SPY
- Growth vs Value: QQQ / SPY
- Cíclico vs Defensivo: XLY / XLP
Con momentum multi-horizonte (3,6,12m) y normalización.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.normalization import normalize_signal, persistence_factor

logger = logging.getLogger(__name__)

class LeadershipEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de persistencia
        self.persistence_N = self.config.get('persistence', {}).get('leadership_N', 2)
        self.persistence_enabled = True

    def _momentum_ratio(self, df, num, denom):
        """
        Calcula momentum multi-horizonte para un ratio num/denom.
        Retorna una Serie con el score normalizado entre -1 y 1.
        """
        ratio = df[num] / df[denom]
        ret_3m = ratio.pct_change(63)
        ret_6m = ratio.pct_change(126)
        ret_12m = ratio.pct_change(252)

        norm_3m = normalize_signal(ret_3m)
        norm_6m = normalize_signal(ret_6m)
        norm_12m = normalize_signal(ret_12m)

        score = 0.5 * norm_3m + 0.3 * norm_6m + 0.2 * norm_12m
        return score

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas de precios (IWM, SPY, QQQ, XLY, XLP)
        Retorna DataFrame con columna 'score_leadership'.
        """
        resultados = pd.DataFrame(index=df.index)

        # Señales individuales
        resultados['score_size']   = self._momentum_ratio(df, 'IWM', 'SPY')
        resultados['score_growth'] = self._momentum_ratio(df, 'QQQ', 'SPY')
        resultados['score_cyclicals'] = self._momentum_ratio(df, 'XLY', 'XLP')

        # Aplicar filtro de persistencia
        if self.persistence_enabled:
            resultados['score_size']   *= persistence_factor(resultados['score_size'], self.persistence_N)
            resultados['score_growth'] *= persistence_factor(resultados['score_growth'], self.persistence_N)
            resultados['score_cyclicals'] *= persistence_factor(resultados['score_cyclicals'], self.persistence_N)

        # Combinación ponderada (40% size, 30% growth, 30% cyclicals)
        raw_leadership = (0.4 * resultados['score_size'] +
                          0.3 * resultados['score_growth'] +
                          0.3 * resultados['score_cyclicals'])

        # Normalización final
        resultados['score_leadership'] = np.tanh(raw_leadership)
        # Rellenar NaN con el último valor válido y luego con 0
        resultados['score_leadership'] = resultados['score_leadership'].ffill().fillna(0)

        return resultados[['score_leadership']]


if __name__ == "__main__":
    from data_layer import DataLayer

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    engine = LeadershipEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de liderazgo (últimos 5 días):")
    print(resultado.tail())