"""
stress_engine.py - Motor de estrés sistémico
Combina:
- VIX (nivel normalizado e invertido)
- Drawdown de SPY
- Crédito a corto plazo (retorno 63d de JNK/LQD)
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

class StressEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de persistencia
        self.persistence_N = self.config.get('persistence', {}).get('stress_N', 2)
        self.persistence_enabled = True

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas ^VIX, SPY, JNK, LQD.
        Retorna DataFrame con columna 'score_stress'.
        """
        resultados = pd.DataFrame(index=df.index)

        # 1. Señal VIX (invertida: VIX alto → estrés alto → score negativo)
        vix = df['^VIX']
        vix_mean = vix.rolling(252).mean()
        vix_std = vix.rolling(252).std()
        vix_score = -np.tanh((vix - vix_mean) / vix_std)

        # 2. Drawdown de SPY
        spy = df['SPY']
        cummax = spy.cummax()
        drawdown = (spy - cummax) / cummax
        dd_score = np.tanh(drawdown * 5)  # escala para que drawdown del 10% dé ≈ -0.5

        # 3. Crédito a corto plazo (retorno 63d de JNK/LQD)
        credit_ratio = df['JNK'] / df['LQD']
        credit_ret = credit_ratio.pct_change(63)
        credit_score = normalize_signal(credit_ret)

        # Aplicar persistencia a las señales
        if self.persistence_enabled:
            vix_score   *= persistence_factor(vix_score, self.persistence_N)
            dd_score    *= persistence_factor(dd_score, self.persistence_N)
            credit_score *= persistence_factor(credit_score, self.persistence_N)

        # Combinación ponderada (50% VIX, 30% drawdown, 20% crédito)
        raw_stress = 0.5 * vix_score + 0.3 * dd_score + 0.2 * credit_score

        # Normalización final
        resultados['score_stress'] = np.tanh(raw_stress)
        # Rellenar NaN
        resultados['score_stress'] = resultados['score_stress'].ffill().fillna(0)

        return resultados[['score_stress']]


if __name__ == "__main__":
    from data_layer import DataLayer

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    engine = StressEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de estrés (últimos 5 días):")
    print(resultado.tail())