"""
global_liquidity_engine.py - Motor de Liquidez Global (GLI)
Basado en datos del BIS (Global Liquidity Indicators) y ajustado por composición de reservas COFER.
Calcula un índice compuesto de liquidez en USD, EUR, JPY y CNY con pesos por moneda de reserva.
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

class GlobalLiquidityEngine:
    """
    Motor que calcula el Global Liquidity Index (GLI) a partir de datos del BIS.
    Pesos ajustados por composición de reservas según datos COFER del FMI.
    """
    
    def __init__(self):
        self.data_layer = DataLayer()
        self.lookback_quarters = 4  # 4 trimestres ≈ 1 año para variación
        self.z_window = 8  # 8 trimestres = 2 años para normalización
        # Pesos por moneda basados en composición de reservas (aproximación)
        self.currency_weights = {
            'USD': 0.55,
            'EUR': 0.25,
            'JPY': 0.15,
            'CNY': 0.05
        }
    
    def _get_credit_growth_series(self, currency):
        """
        Obtiene la serie de crecimiento de crédito para una moneda.
        Para USD y EUR, los datos ya vienen como tasas de crecimiento interanual (UNIT_MEASURE='771').
        Para JPY, obtenemos niveles y calculamos crecimiento.
        Para CNY, intentamos obtener datos; si no, usamos un proxy o excluimos.
        """
        df = self.data_layer.load_bis_gli()
        if df.empty:
            return pd.Series()
        
        if currency in ['USD', 'EUR']:
            # Buscar la serie con UNIT_MEASURE='771' (year-on-year changes)
            mask = (
                (df['CURR_DENOM'] == currency) &
                (df['BORROWERS_SECTOR'] == 'N') &  # Non-banks total
                (df['LENDERS_SECTOR'] == 'A') &    # All sectors
                (df['BORROWERS_CTY'] == '4W') &    # All countries
                (df['UNIT_MEASURE'] == '771')      # Year-on-year changes
            )
        elif currency == 'JPY':
            # Buscar la serie con UNIT_MEASURE='JPY' (niveles)
            mask = (
                (df['CURR_DENOM'] == 'JPY') &
                (df['BORROWERS_SECTOR'] == 'N') &
                (df['LENDERS_SECTOR'] == 'A') &
                (df['BORROWERS_CTY'] == '4W') &
                (df['UNIT_MEASURE'] == 'JPY')
            )
        elif currency == 'CNY':
            # Intentar obtener datos para CNY (pueden no estar disponibles)
            mask = (
                (df['CURR_DENOM'] == 'CNY') &
                (df['BORROWERS_SECTOR'] == 'N') &
                (df['LENDERS_SECTOR'] == 'A') &
                (df['BORROWERS_CTY'] == '4W')
            )
        else:
            logger.error(f"Moneda no soportada: {currency}")
            return pd.Series()
        
        serie_df = df[mask]
        if serie_df.empty:
            logger.warning(f"No se encontró serie para {currency}")
            return pd.Series()
        
        # Tomar la primera fila
        serie = serie_df.iloc[0]
        
        # Extraer columnas de fechas
        fecha_cols = [col for col in df.columns if col[0].isdigit() or col.startswith('19') or col.startswith('20')]
        
        valores = []
        fechas = []
        for col in fecha_cols:
            try:
                año_q = col.split('-')
                año = int(año_q[0])
                trimestre = int(año_q[1][1])
                if trimestre == 1:
                    fecha = pd.Timestamp(year=año, month=3, day=31)
                elif trimestre == 2:
                    fecha = pd.Timestamp(year=año, month=6, day=30)
                elif trimestre == 3:
                    fecha = pd.Timestamp(year=año, month=9, day=30)
                else:
                    fecha = pd.Timestamp(year=año, month=12, day=31)
                
                valor = serie[col]
                if pd.notna(valor):
                    valores.append(valor)
                    fechas.append(fecha)
            except:
                continue
        
        serie_temporal = pd.Series(valores, index=fechas).sort_index()
        
        # Si es JPY, convertir niveles a crecimiento interanual
        if currency == 'JPY':
            growth = (serie_temporal / serie_temporal.shift(4) - 1) * 100
            return growth.dropna()
        elif currency == 'CNY' and not serie_temporal.empty:
            # Intentar calcular crecimiento si hay datos
            if serie_temporal.index[0] < pd.Timestamp.now() - pd.DateOffset(years=2):
                growth = (serie_temporal / serie_temporal.shift(4) - 1) * 100
                return growth.dropna()
            else:
                logger.warning(f"Datos insuficientes para CNY, usando 0")
                return pd.Series()
        else:
            # Para USD y EUR, ya es crecimiento interanual
            return serie_temporal
    
    def calcular_todo(self):
        """
        Calcula el score de liquidez global ponderado por composición de reservas.
        Retorna un DataFrame con columna 'score_global_liquidity'.
        """
        # 1. Obtener series de crecimiento para cada moneda
        usd_growth = self._get_credit_growth_series('USD')
        eur_growth = self._get_credit_growth_series('EUR')
        jpy_growth = self._get_credit_growth_series('JPY')
        cny_growth = self._get_credit_growth_series('CNY')
        
        # Verificar qué monedas tenemos disponibles
        available = []
        growths = {}
        if not usd_growth.empty:
            available.append('USD')
            growths['USD'] = usd_growth
        if not eur_growth.empty:
            available.append('EUR')
            growths['EUR'] = eur_growth
        if not jpy_growth.empty:
            available.append('JPY')
            growths['JPY'] = jpy_growth
        if not cny_growth.empty:
            available.append('CNY')
            growths['CNY'] = cny_growth
        
        if not available:
            logger.error("No se pudo cargar ninguna serie de crecimiento")
            fechas_hoy = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='QE')
            return pd.DataFrame({'score_global_liquidity': 0}, index=fechas_hoy)
        
        # 2. Alinear índices (intersección de fechas)
        fechas_comunes = None
        for g in growths.values():
            if fechas_comunes is None:
                fechas_comunes = g.index
            else:
                fechas_comunes = fechas_comunes.intersection(g.index)
        
        # 3. Crear DataFrame con las tasas de crecimiento alineadas
        growth_df = pd.DataFrame(index=fechas_comunes)
        for curr in available:
            growth_df[curr] = growths[curr].loc[fechas_comunes]
        
        # 4. Normalizar pesos según monedas disponibles
        total_weight = sum(self.currency_weights[curr] for curr in available)
        if total_weight == 0:
            logger.error("Pesos totales cero")
            return pd.DataFrame({'score_global_liquidity': 0}, index=fechas_comunes)
        
        # 5. Calcular índice ponderado
        gli_raw = pd.Series(0.0, index=fechas_comunes)
        for curr in available:
            weight = self.currency_weights[curr] / total_weight  # renormalizar
            gli_raw += weight * growth_df[curr]
        
        # 6. Normalizar con z-score a 2 años (8 trimestres)
        gli_z = (gli_raw - gli_raw.rolling(self.z_window).mean()) / gli_raw.rolling(self.z_window).std()
        
        # 7. Aplicar tanh para obtener score entre -1 y 1
        gli_score = np.tanh(gli_z)
        
        # 8. Reindexar a frecuencia diaria
        fecha_inicio = gli_score.index.min()
        fecha_fin = pd.Timestamp.today()
        fechas_diarias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        gli_score_diario = gli_score.reindex(fechas_diarias, method='ffill')
        
        resultados = pd.DataFrame({'score_global_liquidity': gli_score_diario})
        resultados = resultados.ffill().fillna(0)
        
        return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = GlobalLiquidityEngine()
    res = engine.calcular_todo()
    print("Score de liquidez global (últimos 5 días):")
    print(res.tail())