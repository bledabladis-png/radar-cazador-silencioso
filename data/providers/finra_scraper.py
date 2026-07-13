import requests
import pandas as pd
from datetime import datetime
from io import BytesIO
from .base import MarketDataProvider

class FinraScraperProvider(MarketDataProvider):
    def __init__(self):
        self.name = "FINRA Scraper"
        # URLs fijas (actualizar cada trimestre)
        self.urls = {
            '2026-Q1': {
                'tier1': 'https://www.finra.org/sites/default/files/2026-04/finra-ats-1q2026-tier1.xlsx',
                'tier2': 'https://www.finra.org/sites/default/files/2026-04/finra-ats-1q2026-tier2.xlsx',
                'nms': 'https://www.finra.org/sites/default/files/2026-04/finra-ats-1q2026-nms.xlsx',
                'otc': 'https://www.finra.org/sites/default/files/2026-04/finra-ats-1q2026-otc.xlsx',
            }
        }
        # Seleccionar el trimestre más reciente (última clave del diccionario)
        self.current_quarter = list(self.urls.keys())[-1]

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            url = self.urls[self.current_quarter]['tier1']
            resp = requests.head(url, timeout=10)
            return resp.status_code == 200
        except:
            return False

    def get_ats_data(self):
        """Descarga el archivo NMS (All NMS Stocks) del trimestre actual y devuelve volumen ATS."""
        url = self.urls[self.current_quarter]['tier1']
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                return pd.DataFrame()
            df = pd.read_excel(BytesIO(resp.content), engine='openpyxl')
            # Buscar columnas de símbolo y volumen
            symbol_col = None
            vol_col = None
            for col in df.columns:
                if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                    symbol_col = col
                if 'share' in str(col).lower() or 'volume' in str(col).lower():
                    vol_col = col
            if symbol_col and vol_col:
                df = df[[symbol_col, vol_col]].dropna()
                df.columns = ['symbol', 'ats_volume']
                df['ats_volume'] = pd.to_numeric(df['ats_volume'], errors='coerce')
                return df
            else:
                # Si no encontramos las columnas, imprimir las columnas disponibles para depurar
                print(f'Columnas encontradas: {df.columns.tolist()}')
                return pd.DataFrame()
        except:
            return pd.DataFrame()

    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError

    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError

    def get_fed_data(self, index=None):
        raise NotImplementedError

    def get_options_data(self, index=None):
        raise NotImplementedError
