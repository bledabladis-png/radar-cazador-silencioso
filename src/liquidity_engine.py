"""
liquidity_engine.py - Motor de liquidez (versión simplificada)
Calcula el score de liquidez usando el retorno diario del índice dólar (DX-Y.NYB) con signo invertido,
con momentum multi-horizonte, persistencia y normalización unificada.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class LiquidityEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Cargar configuración específica del motor de liquidez
        cfg = full_config.get('liquidity_engine', {})
        self.tickers = cfg.get('tickers', ['DX-Y.NYB'])
        self.horizons = cfg.get('horizons', [1, 5, 21])
        self.weights_config = cfg.get('weights', {'type': 'adaptive', 'fixed': [0.3, 0.3, 0.4]})
        self.vol_window = cfg.get('volatility_window', 20)
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)
        self.invert_signal = cfg.get('invert_signal', True)  # True para que subida del dólar = liquidez ajustada = score negativo

    def calcular_todo(self, df):
        """
        df: DataFrame con columna de precio (DX-Y.NYB)
        Retorna DataFrame con columna 'score_liquidity'.
        """
        # Verificar que tenemos los tickers necesarios
        required = self.tickers
        for ticker in required:
            if ticker not in df.columns:
                raise ValueError(f"DataFrame no contiene la columna {ticker}")

        # 1. Calcular retornos diarios (solo un ticker)
        ret = df[required[0]].pct_change().dropna()

        # 2. Calcular raw_score combinando horizontes (vectorial) usando el retorno como señal
        raw_score = self._calculate_raw_score_vector(ret)

        # 3. Invertir señal si está configurado
        if self.invert_signal:
            raw_score = -raw_score

        # 4. Aplicar persistencia (vectorial)
        raw_score_persist = self._apply_persistence_vector(raw_score)

        # 5. Normalizar con tanh usando scaling rodante
        normalized = self._normalize_vector(raw_score_persist)

        # Crear DataFrame resultado
        resultados = pd.DataFrame(index=df.index)
        resultados['score_liquidity'] = normalized
        # Rellenar posibles NaN al inicio (por falta de datos)
        resultados['score_liquidity'] = resultados['score_liquidity'].ffill().fillna(0)

        return resultados[['score_liquidity']]

    def _calculate_raw_score_vector(self, ret_series):
        """
        Calcula raw_score para todas las fechas de forma vectorial.
        ret_series: Serie con retornos diarios del activo único.
        """
        # Calcular medias móviles del retorno para cada horizonte
        ma_signals = {}
        for h in self.horizons:
            ma = ret_series.rolling(window=h, min_periods=h).mean()
            ma_signals[h] = ma

        # Volatilidad del activo (para pesos adaptativos)
        vol = ret_series.rolling(window=self.vol_window).std().shift(1) * np.sqrt(252)
        vol = vol.fillna(0.2)  # valor por defecto

        umbral_vol = 0.2  # 20% anualizado
        n_horizons = len(self.horizons)

        if self.weights_config['type'] == 'fixed':
            pesos = np.array(self.weights_config['fixed'])
            pesos = pesos / pesos.sum()
            pesos_df = pd.DataFrame({h: pesos[i] for i, h in enumerate(self.horizons)}, index=ret_series.index)
        else:  # adaptive
            indices = np.arange(n_horizons)
            mask_alta = vol > umbral_vol
            # Inicializar con pesos iguales
            pesos_array = np.ones((len(ret_series), n_horizons)) / n_horizons
            # Pesos para volatilidad alta: proporcionales a (1 + i)
            raw_weights_altos = 1 + indices
            pesos_altos = raw_weights_altos / raw_weights_altos.sum()
            for i, h in enumerate(self.horizons):
                pesos_array[mask_alta, i] = pesos_altos[i]
            pesos_df = pd.DataFrame(pesos_array, index=ret_series.index, columns=self.horizons)

        # Combinar señales ponderadas
        raw = pd.Series(0.0, index=ret_series.index)
        for h in self.horizons:
            raw += ma_signals[h] * pesos_df[h]

        return raw

    def _apply_persistence_vector(self, raw_series):
        """
        Aplica persistencia: multiplica raw por factor basado en días consecutivos con mismo signo.
        """
        signo = np.sign(raw_series)
        cambio = signo.diff() != 0
        grupo = cambio.cumsum()
        consecutivos = grupo.groupby(grupo).cumcount() + 1
        factor = np.minimum(1.0, consecutivos / self.persistence_days)
        resultado = raw_series * factor
        resultado[signo == 0] = 0.0
        return resultado

    def _normalize_vector(self, raw_series):
        """
        Normaliza usando tanh(raw / scaling), con scaling rodante (excluyendo el día actual).
        """
        scaling = raw_series.rolling(window=self.scaling_window, min_periods=1).std().shift(1)
        scaling = scaling.fillna(0.5)
        scaling = scaling.replace(0, 0.5)
        normalized = np.tanh(raw_series / scaling)
        return normalized


if __name__ == "__main__":
    # Prueba del motor
    from src.data_layer import DataLayer  # Ajusta la ruta según tu estructura

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])
    print("Columnas disponibles:", df.columns.tolist())

    engine = LiquidityEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de liquidez (últimos 5 días):")
    print(resultado.tail())