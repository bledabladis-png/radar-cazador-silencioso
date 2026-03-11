"""
ishares_flow_engine.py - Motor de flujos de ETFs usando shares de iShares y precios de Yahoo.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
import yfinance as yf
from src.data_layer import DataLayer
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class iSharesFlowEngine:
    def __init__(self):
        self.dl = DataLayer()
        self.tickers = ['IVV'] # ['IVV', 'EEM', 'LQD', 'IWM']
        self.z_window = 252
        self.smooth_window = 5

    def _get_ishares_shares(self, ticker):
        """Obtiene shares outstanding diarias desde iShares."""
        return self.dl.download_ishares_fund_data(ticker)

    def _get_yahoo_prices(self, ticker, start_date, end_date):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df.empty:
                return pd.DataFrame()
            return df[['Close']].rename(columns={'Close': 'precio'})
        except Exception as e:
            logger.error(f"Error descargando precios de {ticker}: {e}")
            return pd.DataFrame()

    def calcular_flows_individuales(self):
        all_flows = []
        for ticker in self.tickers:
            df_shares = self._get_ishares_shares(ticker)
            if df_shares.empty or len(df_shares) < 2:
                logger.warning(f"Datos insuficientes de iShares para {ticker}")
                continue

            fecha_min = df_shares['fecha'].min()
            fecha_max = df_shares['fecha'].max()
            df_precios = self._get_yahoo_prices(ticker, fecha_min, fecha_max)

            if df_precios.empty:
                logger.warning(f"No hay precios para {ticker}")
                continue

            # Asegurar que df_precios tenga una columna 'precio' (ya debería)
            if 'precio' not in df_precios.columns:
                # Intentar buscar una columna que pueda ser el precio (Close)
                if 'Close' in df_precios.columns:
                    df_precios = df_precios[['Close']].rename(columns={'Close': 'precio'})
                else:
                    logger.error(f"No se encontró columna de precio en datos de Yahoo para {ticker}")
                    continue

            # Crear columna fecha a partir del índice
            df_precios['fecha'] = df_precios.index
            df_precios['fecha'] = pd.to_datetime(df_precios['fecha'])
            # Resetear índice para evitar problemas (ya no lo necesitamos como índice)
            df_precios = df_precios.reset_index(drop=True)

            df_shares['fecha'] = pd.to_datetime(df_shares['fecha'])

            # Merge
            df = pd.merge(df_shares, df_precios, on='fecha', how='inner')
            if df.empty:
                logger.warning(f"No hay fechas comunes entre shares y precios para {ticker}")
                continue

            df = df.sort_values('fecha')

            if len(df) < 2:
                logger.warning(f"Menos de 2 fechas para {ticker}, no se pueden calcular flujos")
                continue

            # Calcular flujo
            df['flow'] = df['shares'].diff() * df['precio']
            df['aum'] = df['shares'] * df['precio']
            df['flow_intensity'] = df['flow'] / df['aum']

            # Z-score solo si hay suficientes datos
            if len(df) >= self.z_window:
                df['flow_z'] = (df['flow'] - df['flow'].rolling(self.z_window).mean()) / df['flow'].rolling(self.z_window).std()
            else:
                df['flow_z'] = 0

            df['outlier'] = df['flow_z'].abs() > 3
            df['ticker'] = ticker

            if df['outlier'].any():
                logger.warning(f"Outliers en {ticker}: {df[df['outlier']]['fecha'].tolist()}")

            all_flows.append(df[['fecha', 'ticker', 'flow', 'flow_intensity', 'flow_z', 'outlier']])

        if not all_flows:
            logger.error("No se pudo calcular ningún flujo")
            return pd.DataFrame()

        return pd.concat(all_flows).sort_values('fecha')

    def agregar_flows(self):
        df = self.calcular_flows_individuales()
        if df.empty:
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='D')
            return pd.DataFrame({'score_etf_flow': 0}, index=fechas)

        df_clean = df[~df['outlier']].copy()
        pivot = df_clean.pivot_table(index='fecha', columns='ticker', values='flow_intensity', aggfunc='mean')
        pivot = pivot.fillna(0)

        pivot['score_raw'] = pivot.mean(axis=1)
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
    engine = iSharesFlowEngine()
    res = engine.calcular_todo()
    print("Score ETF Flow (últimos 5 días):")
    print(res.tail())