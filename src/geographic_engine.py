"""
geographic_engine.py - Motor de flujo geográfico
Calcula:
- Emergentes vs Mundo (EEM/ACWI)
- Desarrollados vs Mundo (EFA/ACWI) con control de correlación Spearman
"""

import pandas as pd
import numpy as np
import yaml
import logging
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

class GeographicEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.em_etf = self.config['indicators']['geographic']['em_etf']
        self.dm_etf = self.config['indicators']['geographic']['dm_etf']
        self.world_etf = self.config['indicators']['geographic']['world_etf']
        self.corr_threshold = self.config['indicators']['geographic']['corr_threshold']
        self.corr_window = self.config['indicators']['geographic']['corr_window']

    def _calcular_ratio_score(self, df, num_ticker, denom_ticker):
        """
        Calcula el score para un ratio (ej. EEM/ACWI) usando la metodología:
        - EMA5 y EMA200 del ratio
        - spread = EMA5 - EMA200
        - z-score del spread con std rolling 200
        - clip a ±3 y tanh
        """
        ratio = df[num_ticker] / df[denom_ticker]

        ema5 = ratio.ewm(span=5, adjust=False).mean()
        ema200 = ratio.ewm(span=200, adjust=False).mean()
        spread = ema5 - ema200

        std_200 = ratio.rolling(200).std()
        z = spread / std_200
        z = np.clip(z, -3, 3)
        score = np.tanh(z)
        return score

    def calcular_em_score(self, df):
        """Score de emergentes vs mundo"""
        return self._calcular_ratio_score(df, self.em_etf, self.world_etf)

    def calcular_dm_score(self, df):
        """Score de desarrollados vs mundo, con ajuste por correlación Spearman"""
        score_base = self._calcular_ratio_score(df, self.dm_etf, self.world_etf)

        # Control de correlación robusta (Spearman) entre EFA y SPY
        if 'SPY' not in df.columns or self.dm_etf not in df.columns:
            logger.warning("Faltan datos para correlación geográfica. No se aplica ajuste.")
            return score_base

        # Calcular correlación rodante de Spearman manualmente
        def rolling_spearman(series1, series2, window):
            # Devuelve una serie con la correlación rodante
            corr = pd.Series(index=series1.index, dtype=float)
            for i in range(window, len(series1) + 1):
                x = series1.iloc[i-window:i]
                y = series2.iloc[i-window:i]
                if len(x) > 1 and len(y) > 1:
                    corr.iloc[i-1] = spearmanr(x, y)[0]
                else:
                    corr.iloc[i-1] = np.nan
            return corr

        corr = rolling_spearman(df[self.dm_etf], df['SPY'], self.corr_window)
        # Factor de ajuste: si correlación > umbral, peso se reduce a la mitad
        factor = np.where(corr > self.corr_threshold, 0.5, 1.0)
        # Rellenar NaN iniciales con 1
        factor = pd.Series(factor, index=score_base.index).fillna(1.0)

        score_ajustado = score_base * factor
        return score_ajustado

    def calcular_todo(self, df):
        resultados = pd.DataFrame(index=df.index)
        resultados['score_em'] = self.calcular_em_score(df)
        resultados['score_dm'] = self.calcular_dm_score(df)

        # Score geográfico combinado (media simple, se puede ponderar después)
        resultados['score_geo'] = (resultados['score_em'] + resultados['score_dm']) / 2

        return resultados


if __name__ == "__main__":
    from data_layer import DataLayer

    dl = DataLayer()
    df = dl.load_latest()

    engine = GeographicEngine()
    resultados = engine.calcular_todo(df)

    print("Últimos 5 días de scores geográficos:")
    print(resultados.tail())