"""
regime_engine.py - Motor de régimen estructural
Calcula:
- Tendencia primaria (SPY vs MA200, pendiente)
- Riesgo crédito (ratio JNK/LQD, z-score)
- Curva de tipos (spread 10y-2y desde columna 'spread_10y2y')
- Drawdown previo SPY (bandera)
"""

import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class RegimeEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.trend_scale = self.config['indicators']['trend']['trend_scale']
        self.slope_scale = self.config['indicators']['trend']['slope_scale']
        self.ma_period = self.config['indicators']['trend']['ma_period']

        self.z_score_clip = self.config['indicators']['credit']['z_score_clip']

        self.curve_penalty = self.config['indicators']['curve']['penalizacion_curva']
        self.delta_weight = self.config['indicators']['curve']['delta_weight']
        self.curve_percentil = self.config['indicators']['curve']['percentil_bajo']
        self.curve_lookback = self.config['indicators']['curve']['lookback_days']

        self.drawdown_lookback = self.config['indicators']['drawdown']['lookback_days']
        self.drawdown_percentil = self.config['indicators']['drawdown']['percentil_drawdown']

    def calcular_tendencia(self, df):
        spy = df['SPY']
        ma200 = spy.rolling(self.ma_period).mean()
        distance = (spy - ma200) / ma200
        score_trend_base = np.tanh(distance * self.trend_scale)

        slope_ma200 = ma200 - ma200.shift(20)
        slope_factor = 0.5 + 0.5 * np.tanh(slope_ma200 * self.slope_scale)

        score_trend = score_trend_base * slope_factor
        return score_trend

    def calcular_credito(self, df):
        jnk = df['JNK']
        lqd = df['LQD']
        ratio = jnk / lqd

        mean_200 = ratio.rolling(200).mean()
        std_200 = ratio.rolling(200).std()
        z = (ratio - mean_200) / std_200
        z = np.clip(z, -self.z_score_clip, self.z_score_clip)
        score_credito = np.tanh(z)
        return score_credito

    def calcular_curva(self, df):
        """
        Usa la columna 'spread_10y2y' (calculada desde los CSV del Tesoro).
        """
        if 'spread_10y2y' not in df.columns:
            logger.warning("Columna 'spread_10y2y' no encontrada. Curva = 0")
            return pd.Series(0, index=df.index)

        spread = df['spread_10y2y']

        # Penalización por percentil bajo
        umbral_bajo = spread.rolling(self.curve_lookback).quantile(self.curve_percentil / 100.0)
        penalizacion = (spread < umbral_bajo).astype(float) * self.curve_penalty

        # Delta suavizado
        delta_20d = spread - spread.shift(20)
        delta_suavizado = delta_20d.rolling(5).mean()  # EMA5 aproximada
        aporte_delta = delta_suavizado * self.delta_weight

        score_curva = penalizacion + aporte_delta
        return score_curva

    def calcular_drawdown_previo(self, df):
        spy = df['SPY']
        rolling_max = spy.rolling(self.drawdown_lookback).max()
        drawdown = (spy - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(self.drawdown_lookback).min()

        threshold = np.percentile(max_drawdown.dropna(), self.drawdown_percentil)
        drawdown_flag = (max_drawdown < threshold).astype(float)
        return drawdown_flag

    def calcular_todo(self, df):
        resultados = pd.DataFrame(index=df.index)
        resultados['score_tendencia'] = self.calcular_tendencia(df)
        resultados['score_credito'] = self.calcular_credito(df)
        resultados['score_curva'] = self.calcular_curva(df)
        resultados['flag_drawdown'] = self.calcular_drawdown_previo(df)

        # Score de régimen (media simple de tendencia y crédito, la curva ya está incluida)
        # Según la documentación, el peso de curva es parte del régimen, pero por ahora lo promediamos todo
        resultados['score_regime'] = (resultados['score_tendencia'] + resultados['score_credito'] + resultados['score_curva']) / 3

        return resultados

if __name__ == "__main__":
    from data_layer import DataLayer

    dl = DataLayer()
    df = dl.load_latest()

    engine = RegimeEngine()
    resultados = engine.calcular_todo(df)

    print("Últimos 5 días de scores de régimen:")
    print(resultados.tail())