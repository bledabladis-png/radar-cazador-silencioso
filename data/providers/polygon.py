import requests
import pandas as pd
from datetime import datetime, timedelta
from .base import MarketDataProvider

class PolygonProvider(MarketDataProvider):
    def __init__(self, api_key=None):
        self.name = "Polygon.io"
        self.api_key = api_key
        if self.api_key is None:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            self.api_key = os.getenv("POLYGON_API_KEY")

    def get_name(self): 
        return self.name

    def is_available(self):
        if not self.api_key:
            return False
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/prev?adjusted=true&apiKey={self.api_key}"
        try:
            resp = requests.get(url, timeout=10)
            return resp.status_code == 200
        except:
            return False

    def get_prices(self, tickers, start=None, end=None, period=None):
        raise NotImplementedError("Usar Yahoo Finance para precios OHLCV")

    def get_treasury_yields(self, maturities=None, index=None):
        raise NotImplementedError("Usar FRED para yields del Tesoro")

    def get_fed_data(self, index=None):
        raise NotImplementedError("Usar FRED para datos de la Fed")

    def get_options_data(self, index=None):
        """Obtiene volúmenes de puts y calls del mercado agregado usando Polygon.io."""
        today = datetime.now().strftime('%Y-%m-%d')
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        # Intentar obtener datos de SPY como proxy del mercado general
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/day/{two_years_ago}/{today}?adjusted=true&sort=asc&limit=5000&apiKey={self.api_key}"
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return pd.DataFrame()
            
            data = resp.json()
            if 'results' not in data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('date', inplace=True)
            
            # Calcular volumen como proxy (no es put/call real, es volumen total)
            # Polygon no proporciona PCR directamente en el tier gratuito
            # Devolvemos el volumen total como indicador de actividad
            result = pd.DataFrame(index=df.index)
            result['volume'] = df['v']  # volumen total de SPY
            result['close'] = df['c']   # precio de cierre
            
            return result
        except Exception as e:
            print(f"Error descargando datos de Polygon: {e}")
            return pd.DataFrame()
