"""
stress_engine.py - Motor de estrés y compresión de volatilidad
Calcula:
- Ratio VIX / (ATR20/SPY * 100) y alerta de percentil alto
- Compresión de volatilidad (bandera)
- Drawdown de crédito (penalización)
"""

import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class StressEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.vix_ticker = self.config['indicators']['stress']['vix_ticker']
        self.atr_period = self.config['indicators']['stress']['atr_period']
        self.vix_percentil_alto = self.config['indicators']['stress']['vix_percentil_alto']
        self.compression_ema_period = self.config['indicators']['stress']['compression']['ema_period']
        self.compression_percentil_bajo = self.config['indicators']['stress']['compression']['percentil_bajo']
        self.price_move_threshold = self.config['indicators']['stress']['compression']['price_move_threshold']
        self.drawdown_umbral = self.config['indicators']['stress']['drawdown_credit']['umbral']
        self.drawdown_ventana = self.config['indicators']['stress']['drawdown_credit']['ventana']
        self.drawdown_retorno_periodo = self.config['indicators']['stress']['drawdown_credit']['retorno_periodo']

    def calcular_atr(self, df, ticker, period):
        """
        Calcula el Average True Range usando High y Low de Stooq si están disponibles.
        Si no, usa una aproximación con desviación estándar.
        """
        high_col = f"{ticker}_High"
        low_col = f"{ticker}_Low"
        close_col = ticker  # precio de cierre de la fuente principal

        if high_col in df.columns and low_col in df.columns:
            # ATR verdadero
            high = df[high_col]
            low = df[low_col]
            close_prev = df[close_col].shift(1)
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            return atr
        else:
            # Fallback a aproximación (desviación estándar)
            logger.debug(f"OHLC no disponible para {ticker}, usando aproximación.")
            return df[close_col].rolling(period).std()

    def calcular_stress(self, df):
        """Calcula el score de estrés combinando los tres subindicadores"""
        resultados = pd.DataFrame(index=df.index)

        # 1. Ratio VIX / (ATR20/SPY * 100)
        spy = df['SPY']
        # Buscar columna de VIX (puede llamarse ^VIX, VIX, etc.)
        vix_col = None
        for col in df.columns:
            if 'VIX' in col.upper():
                vix_col = col
                break
        if vix_col is not None:
            vix = df[vix_col]
            logger.info(f"VIX encontrado en columna: {vix_col}")
        else:
            vix = None
            logger.warning("No se encontró ninguna columna que contenga 'VIX'.")        
        if vix is not None:
            atr_spy = self.calcular_atr(df, 'SPY', self.atr_period)
            ratio_vix = vix / (atr_spy / spy * 100)
            # Alerta de percentil alto
            umbral_alto = ratio_vix.rolling(252*10).quantile(self.vix_percentil_alto / 100.0)
            alert_vix = (ratio_vix > umbral_alto).astype(float)
            resultados['alert_vix'] = alert_vix
        else:
            logger.warning("VIX no disponible, alert_vix = 0")
            resultados['alert_vix'] = 0.0

        # 2. Compresión de volatilidad
        # vol_comp = EMA10(ATR20/SPY)
        atr_spy = self.calcular_atr(df, 'SPY', self.atr_period)
        vol_comp_ratio = atr_spy / spy
        vol_comp = vol_comp_ratio.ewm(span=self.compression_ema_period, adjust=False).mean()
        umbral_bajo = vol_comp.rolling(252*10).quantile(self.compression_percentil_bajo / 100.0)
        # Condición: vol_comp < percentil bajo, y movimiento absoluto 20d < 5%, y score_credito > 0
        # Necesitamos score_credito; lo calcularemos aquí o lo tomaremos de df? Mejor lo calculamos provisionalmente
        # Podemos calcular score_credito simple para esta condición
        jnk = df['JNK'] if 'JNK' in df.columns else None
        lqd = df['LQD'] if 'LQD' in df.columns else None
        if jnk is not None and lqd is not None:
            ratio_credito = jnk / lqd
            mean_200 = ratio_credito.rolling(200).mean()
            std_200 = ratio_credito.rolling(200).std()
            z = (ratio_credito - mean_200) / std_200
            z = np.clip(z, -3, 3)
            score_credito = np.tanh(z)
        else:
            score_credito = pd.Series(0, index=df.index)

        price_move_20d = spy.pct_change(20).abs()
        cond1 = vol_comp < umbral_bajo
        cond2 = price_move_20d < self.price_move_threshold
        cond3 = score_credito > 0
        compresion_flag = (cond1 & cond2 & cond3).astype(float)
        resultados['compresion_flag'] = compresion_flag

        # 3. Drawdown de crédito
        # Si la media móvil de 3 días del retorno a 10 días de JNK es menor que umbral, se resta el umbral al score de estrés
        if jnk is not None:
            ret_10d = jnk / jnk.shift(self.drawdown_retorno_periodo) - 1
            media_3d = ret_10d.rolling(self.drawdown_ventana).mean()
            penalizacion = (media_3d < self.drawdown_umbral).astype(float) * abs(self.drawdown_umbral)
            resultados['drawdown_credit_penalty'] = penalizacion
        else:
            resultados['drawdown_credit_penalty'] = 0.0

        # Score de estrés combinado (podría ser la media de las alertas/penalizaciones, pero según la documentación se integra en el score global con peso 10% dinámico)
        # Por ahora devolvemos los componentes; luego scoring.py los combinará.
        # Definimos un score_stress como combinación lineal simple (p.ej., media de alert_vix, compresion_flag y drawdown_credit_penalty)
        # Pero la documentación indica que stress tiene su propio score. Podemos usar la media de los tres.
        resultados['score_stress'] = (resultados['alert_vix'] + resultados['compresion_flag'] + resultados['drawdown_credit_penalty']) / 3

        return resultados


if __name__ == "__main__":
    from data_layer import DataLayer

    dl = DataLayer()
    df = dl.load_latest()

    engine = StressEngine()
    resultados = engine.calcular_stress(df)

    print("Últimos 5 días de scores de estrés:")
    print(resultados.tail())