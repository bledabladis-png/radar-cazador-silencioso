"""
regime_engine.py - Motor de régimen macroeconómico (versión simplificada)
Calcula el score de régimen usando spread de retornos SPY - IEF,
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

class RegimeEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        # Cargar configuración específica del motor de régimen
        cfg = full_config.get('regime_engine', {})
        self.tickers = cfg.get('tickers', ['SPY', 'IEF'])
        self.horizons = cfg.get('horizons', [1, 5, 21])
        self.weights_config = cfg.get('weights', {'type': 'adaptive', 'fixed': [0.3, 0.3, 0.4]})
        self.vol_window = cfg.get('volatility_window', 20)
        self.persistence_days = cfg.get('persistence_days', 3)
        self.scaling_window = cfg.get('scaling_window', 252)

    def calcular_todo(self, df):
        """
        df: DataFrame con columnas de precios (debe incluir SPY, IEF)
        Retorna DataFrame con columna 'score_regime'.
        """
        # Verificar que tenemos los tickers necesarios
        required = self.tickers
        for ticker in required:
            if ticker not in df.columns:
                raise ValueError(f"DataFrame no contiene la columna {ticker}")

        # 1. Calcular retornos diarios
        returns = df[required].pct_change().dropna()

        # 2. Spread de retornos diarios (risk - refuge)
        spread = returns[self.tickers[0]] - returns[self.tickers[1]]

        # 3. Calcular raw_score combinando horizontes (vectorial)
        raw_score = self._calculate_raw_score_vector(spread, returns)

        # 4. Aplicar persistencia (vectorial)
        raw_score_persist = self._apply_persistence_vector(raw_score)

        # 5. Normalizar con tanh usando scaling rodante
        normalized = self._normalize_vector(raw_score_persist)

        # Crear DataFrame resultado
        resultados = pd.DataFrame(index=df.index)
        resultados['score_regime'] = normalized
        # Rellenar posibles NaN al inicio (por falta de datos)
        resultados['score_regime'] = resultados['score_regime'].ffill().fillna(0)

        return resultados[['score_regime']]

    def _calculate_raw_score_vector(self, spread, returns):
        """
        Calcula raw_score para todas las fechas de forma vectorial.
        spread: Serie con el spread diario (índice de fechas)
        returns: DataFrame con retornos (necesario para volatilidad de SPY)
        """
        # Calcular medias móviles de spread para cada horizonte
        # Usaremos expanding window hasta la fecha actual para evitar lookahead
        # pero necesitamos el valor de la media móvil en cada fecha usando solo datos pasados.
        # Para ello, usamos rolling con center=False (por defecto) que usa valores pasados.
        ma_signals = {}
        for h in self.horizons:
            # rolling mean: en cada fecha, media de los últimos h días (incluye la fecha actual)
            # Esto es correcto porque estamos calculando el valor en esa fecha usando datos hasta ella.
            ma = spread.rolling(window=h, min_periods=h).mean()
            ma_signals[h] = ma

        # Determinar pesos adaptativos por fecha
        # Necesitamos la volatilidad de SPY hasta la fecha (excluyendo la actual? O incluyendo?)
        # Para evitar lookahead, usamos la volatilidad calculada con datos hasta el día anterior.
        # Podemos usar rolling con shift(1) para que la ventana termine el día anterior.
        ret_spy = returns[self.tickers[0]]
        # Volatilidad rolling (desviación estándar) de los últimos vol_window días, finalizando el día anterior
        vol_spy = ret_spy.rolling(window=self.vol_window).std().shift(1) * np.sqrt(252)
        # Si hay NaN, rellenar con un valor por defecto (0.2)
        vol_spy = vol_spy.fillna(0.2)

        # Umbral de volatilidad (ajustable)
        umbral_vol = 0.2

        # Crear pesos para cada horizonte según la volatilidad
        n_horizons = len(self.horizons)
        if self.weights_config['type'] == 'fixed':
            # Pesos fijos (constantes para todas las fechas)
            pesos = np.array(self.weights_config['fixed'])
            # Normalizar a suma 1
            pesos = pesos / pesos.sum()
            # Expandir a DataFrame con las mismas fechas que spread
            pesos_df = pd.DataFrame({h: pesos[i] for i, h in enumerate(self.horizons)}, index=spread.index)
        else:  # adaptive
            # Para cada fecha, pesos proporcionales a (1 + i) si vol > umbral, si no iguales.
            # Construimos una lista de arrays de pesos por fecha.
            indices = np.arange(n_horizons)
            # Crear máscara de volatilidad alta
            mask_alta = vol_spy > umbral_vol
            # Inicializar matriz de pesos (filas = fechas, columnas = horizontes)
            pesos_array = np.ones((len(spread), n_horizons)) / n_horizons  # por defecto iguales
            # Donde mask_alta es True, asignar pesos crecientes
            # raw_weights = 1 + i (por ejemplo)
            raw_weights_altos = 1 + indices
            pesos_altos = raw_weights_altos / raw_weights_altos.sum()
            # Aplicar donde corresponda
            for i, h in enumerate(self.horizons):
                pesos_array[mask_alta, i] = pesos_altos[i]
            # Convertir a DataFrame
            pesos_df = pd.DataFrame(pesos_array, index=spread.index, columns=self.horizons)

        # Combinar señales ponderadas: para cada fecha, suma de (ma_signal * peso)
        raw = pd.Series(0.0, index=spread.index)
        for h in self.horizons:
            raw += ma_signals[h] * pesos_df[h]

        return raw

    def _apply_persistence_vector(self, raw_series):
        """
        Aplica persistencia: multiplica raw por factor basado en días consecutivos con mismo signo.
        raw_series: Serie con raw_score para cada fecha.
        Devuelve Serie con persistencia aplicada.
        """
        # Determinar signo
        signo = np.sign(raw_series)
        # Identificar cambios de signo
        cambio = signo.diff() != 0
        # Asignar un grupo a cada racha consecutiva del mismo signo
        grupo = cambio.cumsum()
        # Calcular días consecutivos dentro de cada grupo (contador desde 1)
        consecutivos = grupo.groupby(grupo).cumcount() + 1
        # Factor de persistencia: min(1, consecutivos / persistence_days)
        factor = np.minimum(1.0, consecutivos / self.persistence_days)
        # Aplicar factor a raw (si signo es 0, dejar 0)
        resultado = raw_series * factor
        # Si signo es 0, el resultado debe ser 0 (el factor podría ser algo, pero mejor 0)
        resultado[signo == 0] = 0.0
        return resultado

    def _normalize_vector(self, raw_series):
        """
        Normaliza usando tanh(raw / scaling), donde scaling es la desviación típica
        rodante de raw_series con ventana scaling_window (sin incluir la fecha actual).
        """
        # Calcular scaling rodante: desviación estándar de los últimos scaling_window valores,
        # excluyendo el valor actual para evitar lookahead (shift)
        scaling = raw_series.rolling(window=self.scaling_window, min_periods=1).std().shift(1)
        # Si hay NaN (primeros días), usar un valor por defecto (por ejemplo, 0.5)
        scaling = scaling.fillna(0.5)
        # Evitar división por cero
        scaling = scaling.replace(0, 0.5)
        # Normalizar
        normalized = np.tanh(raw_series / scaling)
        return normalized


if __name__ == "__main__":
    # Prueba del motor
    from src.data_layer import DataLayer  # Ajusta la ruta según tu estructura

    # Configurar logging para ver mensajes
    logging.basicConfig(level=logging.INFO)

    # Cargar datos más recientes
    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])
    print("Columnas disponibles:", df.columns.tolist())

    # Crear motor y calcular
    engine = RegimeEngine()
    resultado = engine.calcular_todo(df)

    print("\nScore de régimen (últimos 5 días):")
    print(resultado.tail())