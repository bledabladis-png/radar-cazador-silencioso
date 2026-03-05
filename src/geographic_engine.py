"""
geographic_engine.py - Motor de rotación geográfica
Calcula scores para:
- US: SPY / ACWI
- DM ex‑US: EFA / ACWI
- EM: EEM / ACWI
- Asia: AAXJ / ACWI
Luego combina con pesos y aplica penalización por dispersión interna.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.normalization import normalize_signal, persistence_factor

logger = logging.getLogger(__name__)

class GeographicEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Parámetros de persistencia
        self.persistence_N = self.config.get('persistence', {}).get('geo_N', 3)
        self.persistence_enabled = True

        # Umbrales desde thresholds
        self.corr_threshold = self.config.get('thresholds', {}).get('geo_corr_threshold', 0.7)
        self.max_disp = self.config.get('dispersion', {}).get('max_geo', 1.0)

    def _momentum_ratio(self, df, num, denom):
        """
        Calcula momentum multi-horizonte para un ratio num/denom.
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
        df: DataFrame con columnas SPY, EFA, EEM, AAXJ, ACWI
        Retorna DataFrame con columna 'score_geo'.
        """
        resultados = pd.DataFrame(index=df.index)

        # Scores individuales por región
        resultados['score_us']   = self._momentum_ratio(df, 'SPY', 'ACWI')
        resultados['score_dm']   = self._momentum_ratio(df, 'EFA', 'ACWI')
        resultados['score_em']   = self._momentum_ratio(df, 'EEM', 'ACWI')
        resultados['score_asia'] = self._momentum_ratio(df, 'AAXJ', 'ACWI')

        # Aplicar persistencia
        if self.persistence_enabled:
            for col in ['score_us', 'score_dm', 'score_em', 'score_asia']:
                resultados[col] *= persistence_factor(resultados[col], self.persistence_N)

        # Control de correlación para DM (Spearman con SPY)
        if 'SPY' in df.columns and 'EFA' in df.columns:
            # Calcular correlación de Spearman rodante manualmente
            window = 90
            corr_series = []
            for i in range(len(df)):
                if i < window:
                    corr_series.append(np.nan)
                else:
                    x = df['EFA'].iloc[i-window:i]
                    y = df['SPY'].iloc[i-window:i]
                    corr, _ = spearmanr(x, y)
                    corr_series.append(corr)
            corr = pd.Series(corr_series, index=df.index)
            # Factor de ajuste: si correlación > umbral, peso se reduce a la mitad
            factor_dm = np.where(corr > self.corr_threshold, 0.5, 1.0)
            factor_dm = pd.Series(factor_dm, index=df.index).fillna(1.0)
            resultados['score_dm'] *= factor_dm

        # Dispersión interna
        sub_scores = resultados[['score_us', 'score_dm', 'score_em', 'score_asia']]
        dispersion = sub_scores.std(axis=1)
        # Factor de consistencia: 1 - (dispersion / max_disp), recortado a [0,1]
        factor_consistencia = 1 - (dispersion / self.max_disp)
        factor_consistencia = factor_consistencia.clip(lower=0, upper=1)

        # Combinación ponderada (30% US, 25% DM, 25% EM, 20% Asia)
        raw_geo = (0.30 * resultados['score_us'] +
                   0.25 * resultados['score_dm'] +
                   0.25 * resultados['score_em'] +
                   0.20 * resultados['score_asia'])

        # Score geográfico ajustado por dispersión
        resultados['score_geo'] = raw_geo * factor_consistencia
        # Rellenar NaN
        resultados['score_geo'] = resultados['score_geo'].ffill().fillna(0)

        # Opcional: guardar factores intermedios para auditoría
        resultados['geo_dispersion'] = dispersion
        resultados['geo_consistency_factor'] = factor_consistencia

        return resultados[['score_geo']]


if __name__ == "__main__":
    from data_layer import DataLayer

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    engine = GeographicEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore geográfico (últimos 5 días):")
    print(resultado.tail())