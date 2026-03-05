"""
bond_engine.py - Motor de bonos globales
Detecta flight‑to‑safety y ciclo de tipos mediante momentum de precios de:
- TLT (bonos largos USA)
- IBGL (bonos europeos)
- LQD (corporativos USA)
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

class BondEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de persistencia
        self.persistence_N = self.config.get('persistence', {}).get('bonds_N', 2)
        self.persistence_enabled = True

    def _momentum(self, df, ticker):
        """
        Calcula momentum multi‑horizonte para un ticker (precio).
        """
        serie = df[ticker]
        ret_3m = serie.pct_change(63)
        ret_6m = serie.pct_change(126)
        ret_12m = serie.pct_change(252)

        norm_3m = normalize_signal(ret_3m)
        norm_6m = normalize_signal(ret_6m)
        norm_12m = normalize_signal(ret_12m)

        score = 0.5 * norm_3m + 0.3 * norm_6m + 0.2 * norm_12m
        return score

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas TLT, IBGL, LQD.
        Retorna DataFrame con columna 'score_bonds'.
        """
        resultados = pd.DataFrame(index=df.index)

        # Señales individuales
        resultados['score_tlt']  = self._momentum(df, 'TLT')
        resultados['score_ibgl'] = self._momentum(df, 'IBGL')
        resultados['score_lqd']  = self._momentum(df, 'LQD')

        # --- RELLENO INCONDICIONAL DE NaN ---
        # Aseguramos que no haya NaN antes de persistencia
        for col in ['score_tlt', 'score_ibgl', 'score_lqd']:
            resultados[col] = resultados[col].ffill().fillna(0)

        # Aplicar persistencia (opcional)
        if self.persistence_enabled:
            resultados['score_tlt']  *= persistence_factor(resultados['score_tlt'], self.persistence_N)
            resultados['score_ibgl'] *= persistence_factor(resultados['score_ibgl'], self.persistence_N)
            resultados['score_lqd']  *= persistence_factor(resultados['score_lqd'], self.persistence_N)

        # Combinación ponderada (40% TLT, 30% IBGL, 30% LQD)
        raw_bonds = 0.4 * resultados['score_tlt'] + 0.3 * resultados['score_ibgl'] + 0.3 * resultados['score_lqd']
        # Rellenar posibles NaN en la combinación (por seguridad)
        raw_bonds = raw_bonds.ffill().fillna(0)

        # Normalización final
        resultados['score_bonds'] = np.tanh(raw_bonds)
        # Último relleno por si acaso
        resultados['score_bonds'] = resultados['score_bonds'].ffill().fillna(0)

        return resultados[['score_bonds']]


if __name__ == "__main__":
    from data_layer import DataLayer

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    engine = BondEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de bonos (últimos 5 días):")
    print(resultado.tail())