"""
regime_engine.py - Motor de régimen macroeconómico
Calcula el score de régimen usando:
- Crédito: JNK / LQD
- Tamaño: IWM / SPY
- Consumo: XLY / XLP
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

class RegimeEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de persistencia
        self.persistence_N = self.config.get('persistence', {}).get('regime_N', 2)
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

        # Normalizar cada horizonte
        norm_3m = normalize_signal(ret_3m)
        norm_6m = normalize_signal(ret_6m)
        norm_12m = normalize_signal(ret_12m)

        # Combinación ponderada (50% 3m, 30% 6m, 20% 12m)
        score = 0.5 * norm_3m + 0.3 * norm_6m + 0.2 * norm_12m
        return score

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas de precios (debe incluir JNK, LQD, IWM, SPY, XLY, XLP)
        Retorna DataFrame con columna 'score_regime'.
        """
        resultados = pd.DataFrame(index=df.index)

        # Señales individuales
        resultados['score_credit'] = self._momentum_ratio(df, 'JNK', 'LQD')
        resultados['score_size']   = self._momentum_ratio(df, 'IWM', 'SPY')
        resultados['score_consumer'] = self._momentum_ratio(df, 'XLY', 'XLP')

        # Aplicar filtro de persistencia (opcional)
        if self.persistence_enabled:
            resultados['score_credit'] *= persistence_factor(resultados['score_credit'], self.persistence_N)
            resultados['score_size']   *= persistence_factor(resultados['score_size'], self.persistence_N)
            resultados['score_consumer'] *= persistence_factor(resultados['score_consumer'], self.persistence_N)

        # Combinación ponderada (40% crédito, 30% tamaño, 30% consumo)
        raw_regime = (0.4 * resultados['score_credit'] +
                      0.3 * resultados['score_size'] +
                      0.3 * resultados['score_consumer'])

        # Normalización final (opcional, pero recomendada)
        resultados['score_regime'] = np.tanh(raw_regime)
        # Rellenar NaN con el último valor válido (forward fill) y luego con 0 si es necesario
        resultados['score_regime'] = resultados['score_regime'].ffill().fillna(0)

        return resultados[['score_regime']]


if __name__ == "__main__":
    # Prueba del motor
    from data_layer import DataLayer

    # Configurar logging para ver mensajes
    logging.basicConfig(level=logging.INFO)

    # Cargar datos más recientes
    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    # Crear motor y calcular
    engine = RegimeEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de régimen (últimos 5 días):")
    print(resultado.tail())