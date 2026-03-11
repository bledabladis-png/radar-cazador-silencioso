"""
etf_flow_engine.py - Motor de flujos de ETFs
Utiliza scraping de ETF.com como fuente principal y Yahoo Finance como respaldo.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from src.data_layer import DataLayer
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class EtfFlowEngine:
    def __init__(self):
        self.dl = DataLayer()
        self.tickers = ['SPY', 'EEM', 'JNK', 'LQD', 'HYG', 'IWM', 'QQQ', 'XLK', 'XLF', 'TLT', 'SHY']
        self.z_window = 252
        self.smooth_window = 5

    def _get_flows_range(self, start_date, end_date):
        """
        Intenta obtener flujos de ETF.com para cada día en el rango.
        Si falla, usa Yahoo como respaldo para todo el rango.
        """
        fechas = pd.date_range(start=start_date, end=end_date, freq='D')
        all_flows = []

        # Intentar ETF.com para cada fecha
        for fecha in fechas:
            fecha_str = fecha.strftime('%Y-%m-%d')
            df_day = self.dl.scrape_etf_flows(target_date=fecha_str, tickers=self.tickers)
            if not df_day.empty:
                all_flows.append(df_day)

        if all_flows:
            df_etf = pd.concat(all_flows).sort_values('fecha')
            return df_etf
        else:
            logger.warning("No se obtuvieron datos de ETF.com, usando Yahoo Finance como respaldo")
            df_yahoo = self.dl._fetch_etf_flows_yahoo_fallback(self.tickers, start_date, end_date)
            if df_yahoo.empty:
                logger.error("Yahoo también falló")
                return pd.DataFrame()
            return df_yahoo

    def agregar_flows(self):
        """
        Calcula flujos, intensidad y z-score, y agrega en un score compuesto.
        """
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=365)

        df = self._get_flows_range(start_date, end_date)
        if df.empty:
            logger.error("No se pudo calcular ningún flujo")
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='D')
            return pd.DataFrame({'score_etf_flow': 0}, index=fechas)

        df = df.sort_values(['fecha', 'ticker'])

        # Pivotar
        pivot = df.pivot_table(index='fecha', columns='ticker', values='flujo_neto', aggfunc='mean')
        pivot = pivot.fillna(0)

        # Z-score por ticker
        for col in pivot.columns:
            mean = pivot[col].rolling(self.z_window).mean()
            std = pivot[col].rolling(self.z_window).std()
            pivot[f'{col}_z'] = (pivot[col] - mean) / std

        z_cols = [c for c in pivot.columns if c.endswith('_z')]
        if not z_cols:
            logger.error("No se pudieron calcular z-scores")
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
    engine = EtfFlowEngine()
    res = engine.calcular_todo()
    print("Score ETF Flow (últimos 5 días):")
    print(res.tail())