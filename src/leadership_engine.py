"""
leadership_engine.py - Motor de liderazgo de mercado (versión simplificada)
Calcula el score de liderazgo usando spread de retornos XLK - XLF,
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

class LeadershipEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Cargar configuración específica del motor de liderazgo
        cfg = full_config.get('leadership_engine', {})
        self.tickers = cfg.get('tickers', ['XLK', 'XLF'])
        self.horizons = cfg.get('horizons', [1, 5, 21])
        self.weights_config = cfg.get('weights', {'type': 'adaptive', 'fixed': [0.3, 0.3, 0.4]})
        self.vol_window = cfg.get('volatility_window', 20)
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas de precios (debe incluir XLK, XLF)
        Retorna DataFrame con columna 'score_leadership'.
        """
        # Verificar que tenemos los tickers necesarios
        required = self.tickers
        for ticker in required:
            if ticker not in df.columns:
                raise ValueError(f"DataFrame no contiene la columna {ticker}")

        # 1. Calcular retornos diarios
        returns = df[required].pct_change().dropna()

        # 2. Spread de retornos diarios (XLK - XLF)
        spread = returns[self.tickers[0]] - returns[self.tickers[1]]

        # 3. Calcular raw_score combinando horizontes (vectorial)
        raw_score = self._calculate_raw_score_vector(spread, returns)

        # 4. Aplicar persistencia (vectorial)
        raw_score_persist = self._apply_persistence_vector(raw_score)

        # 5. Normalizar con tanh usando scaling rodante
        normalized = self._normalize_vector(raw_score_persist)

        # Crear DataFrame resultado
        resultados = pd.DataFrame(index=df.index)
        resultados['score_leadership'] = normalized
        # Rellenar posibles NaN al inicio (por falta de datos)
        resultados['score_leadership'] = resultados['score_leadership'].ffill().fillna(0)

        return resultados[['score_leadership']]

    def _calculate_raw_score_vector(self, spread, returns):
        """
        Calcula raw_score para todas las fechas de forma vectorial.
        spread: Serie con el spread diario (índice de fechas)
        returns: DataFrame con retornos (necesario para volatilidad del activo de riesgo)
        """
        # Calcular medias móviles de spread para cada horizonte
        ma_signals = {}
        for h in self.horizons:
            ma = spread.rolling(window=h, min_periods=h).mean()
            ma_signals[h] = ma

        # Volatilidad del activo de riesgo (primer ticker) para pesos adaptativos
        ret_risk = returns[self.tickers[0]]
        # Volatilidad rolling (desviación estándar) de los últimos vol_window días, finalizando el día anterior
        vol_risk = ret_risk.rolling(window=self.vol_window).std().shift(1) * np.sqrt(252)
        vol_risk = vol_risk.fillna(0.2)  # valor por defecto

        umbral_vol = 0.2  # 20% anualizado
        n_horizons = len(self.horizons)

        if self.weights_config['type'] == 'fixed':
            pesos = np.array(self.weights_config['fixed'])
            pesos = pesos / pesos.sum()
            pesos_df = pd.DataFrame({h: pesos[i] for i, h in enumerate(self.horizons)}, index=spread.index)
        else:  # adaptive
            indices = np.arange(n_horizons)
            mask_alta = vol_risk > umbral_vol
            # Inicializar con pesos iguales
            pesos_array = np.ones((len(spread), n_horizons)) / n_horizons
            # Pesos para volatilidad alta: proporcionales a (1 + i)
            raw_weights_altos = 1 + indices
            pesos_altos = raw_weights_altos / raw_weights_altos.sum()
            for i, h in enumerate(self.horizons):
                pesos_array[mask_alta, i] = pesos_altos[i]
            pesos_df = pd.DataFrame(pesos_array, index=spread.index, columns=self.horizons)

        # Combinar señales ponderadas
        raw = pd.Series(0.0, index=spread.index)
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

    engine = LeadershipEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de liderazgo (últimos 5 días):")
    print(resultado.tail())