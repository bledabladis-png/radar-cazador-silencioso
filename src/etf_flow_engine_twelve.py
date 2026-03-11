"""
etf_flow_engine_twelve.py - Motor de flujos de ETFs usando Twelve Data API.
Lee la API key desde un archivo de texto (por defecto 'twelvedata_key.txt').
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import yaml
import time
from datetime import datetime, timedelta
from twelvedata import TDClient
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class EtfFlowEngineTwelve:
    def __init__(self, config_path='config/config.yaml'):
        # Cargar configuración
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Leer la ruta del archivo de clave de Twelve Data desde config.yaml
        key_file = self.config['data'].get('twelvedata_key_file', 'twelvedata_key.txt')
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                self.api_key = f.read().strip()
        else:
            logger.error(f"No se encuentra el archivo de clave {key_file}")
            raise ValueError("Falta la API key de Twelve Data. Crea el archivo con tu clave.")

        self.tickers = ['SPY', 'EEM', 'JNK', 'LQD', 'HYG', 'IWM', 'QQQ', 'XLK', 'XLF', 'TLT', 'SHY']
        self.z_window = 252
        self.smooth_window = 5
        self.td = TDClient(apikey=self.api_key)

    def _fetch_twelve_data(self, ticker, start_date, end_date, max_retries=3):
        """
        Descarga datos históricos de Twelve Data para un ticker.
        Maneja rate limit y reintentos con backoff.
        Devuelve DataFrame con columnas: fecha, close, volume.
        """
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        for attempt in range(max_retries):
            try:
                ts = self.td.time_series(
                    symbol=ticker,
                    interval="1day",
                    start_date=start_str,
                    end_date=end_str,
                    outputsize=5000
                )
                df = ts.as_pandas()
                if df.empty:
                    logger.debug(f"No hay datos para {ticker} en el rango")
                    return pd.DataFrame()

                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'fecha'})

                if 'close' not in df.columns or 'volume' not in df.columns:
                    logger.debug(f"El DataFrame de {ticker} no tiene las columnas 'close' o 'volume'")
                    return pd.DataFrame()

                df = df[['fecha', 'close', 'volume']]
                return df
            except Exception as e:
                error_msg = str(e)
                if "run out of API credits" in error_msg or "rate limit" in error_msg.lower():
                    wait_time = 60 * (attempt + 1)
                    logger.debug(f"Rate limit alcanzado para {ticker}. Esperando {wait_time}s (intento {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"Error descargando {ticker} de Twelve Data: {error_msg}")
                    return pd.DataFrame()
        logger.debug(f"Se agotaron los reintentos para {ticker}")
        return pd.DataFrame()

    def calcular_flows(self, start_date=None, end_date=None):
        """
        Calcula flujos proxy (precio*volumen) para todos los tickers.
        Respeta el límite de 8 peticiones por minuto con pausas.
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        all_data = []
        for i, ticker in enumerate(self.tickers):
            logger.debug(f"Procesando {ticker} ({i+1}/{len(self.tickers)})...")
            df = self._fetch_twelve_data(ticker, start_date, end_date)
            if not df.empty:
                df['ticker'] = ticker
                df['flujo_proxy'] = df['close'] * df['volume']
                all_data.append(df[['fecha', 'ticker', 'flujo_proxy']])
            else:
                logger.debug(f"No se pudo obtener datos para {ticker}")

            # Pausa entre peticiones para no exceder el límite de 8/minuto
            if i < len(self.tickers) - 1:
                time.sleep(8)

        if not all_data:
            logger.debug("No se pudo obtener datos de ningún ticker")
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True).sort_values('fecha')
        return df

    def agregar_flows(self):
        """
        Calcula z-scores del flujo proxy y los combina en un score compuesto.
        """
        df = self.calcular_flows()
        if df.empty:
            logger.debug("No hay datos de flujos, se devuelve 0")
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='D')
            return pd.DataFrame({'score_etf_flow': 0}, index=fechas)

        pivot = df.pivot_table(index='fecha', columns='ticker', values='flujo_proxy', aggfunc='mean')
        pivot = pivot.fillna(0)

        for col in pivot.columns:
            mean = pivot[col].rolling(self.z_window).mean()
            std = pivot[col].rolling(self.z_window).std()
            pivot[f'{col}_z'] = (pivot[col] - mean) / std.replace(0, np.nan)
            pivot[f'{col}_z'] = pivot[f'{col}_z'].fillna(0)

        z_cols = [c for c in pivot.columns if c.endswith('_z')]
        if not z_cols:
            logger.debug("No se pudieron calcular z-scores")
            return pd.DataFrame({'score_etf_flow': 0}, index=pivot.index)

        pivot['score_raw'] = pivot[z_cols].mean(axis=1)
        pivot['score_smooth'] = pivot['score_raw'].ewm(span=self.smooth_window).mean()

        scaling = robust_scale(pivot['score_smooth'], window=self.z_window).shift(1)
        scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
        pivot['score_etf_flow'] = np.tanh(pivot['score_smooth'] / scaling)

        fecha_inicio = pivot.index.min()
        fecha_fin = pd.Timestamp.today()
        fechas_diarias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        score = pivot['score_etf_flow'].reindex(fechas_diarias, method='ffill')

        return pd.DataFrame({'score_etf_flow': score})

    def calcular_todo(self):
        return self.agregar_flows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = EtfFlowEngineTwelve()
    res = engine.calcular_todo()
    print("Score ETF Flow (Twelve Data) - últimos 5 días:")
    print(res.tail())
