"""
breadth_advanced_engine.py - Motor de Breadth Avanzado mejorado (Fase 10)
Combina:
- Indicadores de StockCharts ($NYAD, $NYHL, $NYMO, $NYA50R, $NYA200R)
- Datos de NASDAQ (advances/declines, new highs/lows)
- Ratio RSP/SPY (participaciÃ³n equal-weight vs cap-weight)
Genera un score de amplitud de mercado entre -1 y 1.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from src.stockcharts_scraper import StockChartsScraper
from src.nasdaq_scraper import NasdaqScraper
from src.utils.normalization import robust_scale

logger = logging.getLogger(__name__)

class BreadthAdvancedEngine:
    def __init__(self):
        self.stockcharts = StockChartsScraper()
        self.nasdaq = NasdaqScraper()
        self.z_window = 252
        self.smooth_window = 5
        
    def _load_data(self):
        """Carga y combina datos de StockCharts y NASDAQ. Si no hay, devuelve DataFrame vacío."""
        sc_df = self.stockcharts.load_history()
        nd_df = self.nasdaq.load_history()

        if sc_df.empty and nd_df.empty:
            logger.debug("No hay datos de breadth disponibles")
            return pd.DataFrame()

        combined = pd.DataFrame()
        if not sc_df.empty:
            sc_df['date'] = pd.to_datetime(sc_df['date'])
            sc_df.set_index('date', inplace=True)
            combined = combined.join(sc_df, how='outer')
        if not nd_df.empty:
            nd_df['date'] = pd.to_datetime(nd_df['date'])
            nd_df.set_index('date', inplace=True)
            combined = combined.join(nd_df, how='outer')

        return combined.ffill().bfill()    
    def calcular_todo(self, df_principal=None):
        """
        Calcula el score de breadth avanzado.
        Si se proporciona df_principal (con columnas 'RSP' y 'SPY'), se aÃ±ade el ratio RSP/SPY.
        """
        # 1. Cargar datos de fuentes externas
        df_ext = self._load_data()
        
        # 2. Preparar DataFrame base con fechas
        if df_ext.empty:
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='D')
            combined = pd.DataFrame(index=fechas)
        else:
            combined = df_ext.copy()
        
        # 3. AÃ±adir ratio RSP/SPY si estÃ¡ disponible
        if df_principal is not None and 'RSP' in df_principal.columns and 'SPY' in df_principal.columns:
            rsp_spy = df_principal['RSP'] / df_principal['SPY']
            rsp_spy = rsp_spy.ffill().dropna()
            # Normalizar el ratio (usamos log-returns para estabilidad)
            rsp_ret = np.log(rsp_spy).diff().fillna(0)
            scaling = robust_scale(rsp_ret, window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            rsp_score = np.tanh(rsp_ret / scaling)
            combined['rsp_spy'] = rsp_score.reindex(combined.index, method='ffill')
            logger.info("Ratio RSP/SPY aÃ±adido al breadth avanzado")
        
        # 4. Calcular seÃ±ales individuales
        signals = pd.DataFrame(index=combined.index)
        
        # StockCharts
        if 'nyad' in combined.columns:
            nyad_delta = combined['nyad'].diff()
            scaling = robust_scale(nyad_delta, window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            signals['nyad_score'] = np.tanh(nyad_delta / scaling)
        
        if 'nyhl' in combined.columns:
            nyhl_delta = combined['nyhl'].diff()
            scaling = robust_scale(nyhl_delta, window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            signals['nyhl_score'] = np.tanh(nyhl_delta / scaling)
        
        if 'nymo' in combined.columns:
            scaling = robust_scale(combined['nymo'], window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            signals['nymo_score'] = np.tanh(combined['nymo'] / scaling)
        
        if 'nya50r' in combined.columns:
            signals['nya50r_score'] = (combined['nya50r'] - 50) / 50
        if 'nya200r' in combined.columns:
            signals['nya200r_score'] = (combined['nya200r'] - 50) / 50
        
        # NASDAQ
        if 'ad_line_delta' in combined.columns:
            scaling = robust_scale(combined['ad_line_delta'], window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            signals['ad_delta_score'] = np.tanh(combined['ad_line_delta'] / scaling)
        
        if 'ad_ratio' in combined.columns:
            signals['ad_ratio_score'] = (combined['ad_ratio'] - 0.5) * 2
        
        if 'hl_net' in combined.columns:
            scaling = robust_scale(combined['hl_net'], window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            signals['hl_net_score'] = np.tanh(combined['hl_net'] / scaling)
        
        if 'hl_ratio' in combined.columns:
            signals['hl_ratio_score'] = (combined['hl_ratio'] - 0.5) * 2
        
        # RSP/SPY
        if 'rsp_spy' in combined.columns:
            signals['rsp_spy_score'] = combined['rsp_spy']
        
        if signals.empty:
            logger.debug("No se pudo calcular ninguna seÃ±al de breadth avanzado")
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='D')
            return pd.DataFrame({'score_breadth_advanced': 0}, index=fechas)
        
        # Suavizar
        signals = signals.rolling(window=self.smooth_window, min_periods=1).mean()
        
        # Combinar (promedio simple)
        signals['score_breadth_advanced'] = signals.mean(axis=1)
        
        # NormalizaciÃ³n final
        score = signals['score_breadth_advanced']
        scaling = robust_scale(score, window=self.z_window).shift(1)
        scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
        final_score = np.tanh(score / scaling)
        
        # Reindexar a diario hasta hoy
        fecha_inicio = final_score.index.min()
        fecha_fin = pd.Timestamp.today()
        fechas_diarias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        final_score = final_score.reindex(fechas_diarias, method='ffill')
        
        resultados = pd.DataFrame({'score_breadth_advanced': final_score})
        resultados = resultados.ffill().fillna(0)
        
        return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data_layer import DataLayer
    dl = DataLayer()
    df = dl.load_latest()
    engine = BreadthAdvancedEngine()
    res = engine.calcular_todo(df_principal=df)
    print("Score Breadth Avanzado (Ãºltimos 5 dÃ­as):")
    print(res.tail())
