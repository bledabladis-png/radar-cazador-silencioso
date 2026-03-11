"""
cftc_engine.py - Motor de posicionamiento de futuros basado en datos CFTC (Commitment of Traders)
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

class CftcEngine:
    """
    Motor que analiza los informes COT para generar un score de posicionamiento de especuladores.
    """

    def __init__(self):
        self.data_layer = DataLayer()
        # Definir los contratos de interÃ©s: (clave, texto a buscar en Market_and_Exchange_Names)
        self.contracts = {
            'CAD': 'CANADIAN DOLLAR',
            'EUR': 'EURO FX',
            'JPY': 'JAPANESE YEN',
            'GBP': 'BRITISH POUND',
            'CHF': 'SWISS FRANC',
            'MXN': 'MEXICAN PESO',
            'AUD': 'AUSTRALIAN DOLLAR',
            '10Y': '10 YEAR U.S. TREASURY NOTES',
            '5Y': '5 YEAR U.S. TREASURY NOTES',
            '30Y': '30 YEAR U.S. TREASURY BONDS',
            'ES': 'E-MINI S&P 500',
            'NQ': 'NASDAQ-100'
        }
        # Ventana de suavizado (en semanas) para las posiciones netas
        self.smooth_window = 4
        # Ventana para z-score (en semanas) - 2 aÃ±os aprox
        self.z_window = 104

    def _load_data(self):
        """Carga y concatena todos los años de CFTC que existen en el directorio."""
        # Obtener lista de años disponibles (directorios extracted_*)
        cftc_dir = self.data_layer.raw_dir / 'cftc'
        if not cftc_dir.exists():
            logger.error("Directorio de CFTC no encontrado")
            return pd.DataFrame()

        years = []
        for path in cftc_dir.glob("extracted_*"):
            try:
                y = int(path.name.replace("extracted_", ""))
                years.append(y)
            except:
                pass

        if not years:
            logger.error("No se encontraron años extraídos de CFTC")
            return pd.DataFrame()

        # Cargar datos de cada año
        dfs = []
        for year in sorted(years):
            try:
                df = self.data_layer.load_cftc_data(year)
                if not df.empty:
                    dfs.append(df)
                    logger.debug(f"Datos CFTC {year} cargados")
                else:
                    logger.debug(f"Año {year} sin datos")
            except Exception as e:
                logger.debug(f"No se pudo cargar año {year}: {e}")
                continue

        if not dfs:
            logger.error("No se pudo cargar ningún dato de CFTC")
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def calcular_todo(self):
        """
        Calcula el score de posicionamiento agregado.
        Retorna un DataFrame con columna 'score_cftc' indexada por fecha (diaria, forward-filled).
        """
        df = self._load_data()
        if df.empty:
            # Devolver DataFrame con ceros
            fechas = pd.date_range(end=pd.Timestamp.today(), periods=10, freq='W-WED')
            return pd.DataFrame({'score_cftc': 0}, index=fechas)

        # Asegurar tipo de fecha
        df['fecha'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])
        df = df.sort_values('fecha')

        # Diccionario para almacenar los scores individuales por contrato
        contract_scores = {}

        for key, name in self.contracts.items():
            # Filtrar por nombre de mercado (contiene el nombre)
            mask = df['Market_and_Exchange_Names'].str.contains(name, na=False)
            contract_df = df[mask].copy()
            if contract_df.empty:
                logger.debug(f"No se encontraron datos para {name}")
                continue

            # Calcular posición neta para "Asset Managers" y "Leveraged Money"
            contract_df['net_spec'] = (
                contract_df['Asset_Mgr_Positions_Long_All'] - contract_df['Asset_Mgr_Positions_Short_All'] +
                contract_df['Lev_Money_Positions_Long_All'] - contract_df['Lev_Money_Positions_Short_All']
            ).astype(float)

            # Ordenar por fecha y quedarse con la última observación por fecha
            contract_df = contract_df.sort_values('fecha').drop_duplicates(subset=['fecha'], keep='last')
            contract_df = contract_df.set_index('fecha')['net_spec'].sort_index()

            # Suavizar con media móvil de 4 semanas
            smoothed = contract_df.rolling(window=self.smooth_window, min_periods=1).mean()

            # Normalización robusta
            scaling = robust_scale(smoothed, window=self.z_window).shift(1)
            scaling = scaling.replace(0, np.nan).ffill().fillna(0.5)
            score = np.tanh(smoothed / scaling)

            contract_scores[key] = score
        if not contract_scores:
            logger.error("No se pudo calcular ningÃºn score de CFTC")
            return pd.DataFrame({'score_cftc': 0}, index=pd.date_range(end=pd.Timestamp.today(), periods=10, freq='W-WED'))

        # Combinar todos los scores en un DataFrame con fechas comunes
        # Primero, aseguramos que cada serie tenga un Ã­ndice Ãºnico y ordenado
        for key in contract_scores:
            contract_scores[key] = contract_scores[key].sort_index()

        # Unir todas las series usando concat y luego reindexar a un rango continuo
        combined = pd.DataFrame(contract_scores)
        # Crear un rango de fechas diario desde la mÃ­nima hasta la mÃ¡xima de los datos
        all_dates = pd.date_range(start=combined.index.min(), end=combined.index.max(), freq='D')
        combined = combined.reindex(all_dates).ffill().bfill()

        # Promedio simple (podrÃ­amos ponderar por importancia)
        combined['score_cftc'] = combined.mean(axis=1)

        # Reindexar hasta hoy
        fecha_inicio = combined.index.min()
        fecha_fin = pd.Timestamp.today()
        fechas_diarias = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='D')
        score_diario = combined['score_cftc'].reindex(fechas_diarias, method='ffill')

        resultados = pd.DataFrame({'score_cftc': score_diario})
        resultados = resultados.ffill().fillna(0)

        return resultados


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = CftcEngine()
    res = engine.calcular_todo()
    print("Score CFTC (Ãºltimos 5 dÃ­as):")
    print(res.tail())

