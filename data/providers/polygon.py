import requests
import pandas as pd
from datetime import datetime, timedelta
from .base import MarketDataProvider

class PolygonProvider(MarketDataProvider):
    def __init__(self, api_key=None):
        self.name = "Polygon.io"
        self.api_key = api_key
        if self.api_key is None:
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv()
                self.api_key = os.getenv("POLYGON_API_KEY")
            except ImportError:
                self.api_key = None

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

    def _period_to_dates(self, period: str):
        """Convierte period ('5y', '1y', etc.) a fechas de inicio y fin."""
        end = datetime.now()
        if period.endswith('y'):
            years = int(period[:-1])
            start = end - timedelta(days=years * 365)
        elif period.endswith('mo'):
            months = int(period[:-2])
            start = end - timedelta(days=months * 30)
        elif period.endswith('d'):
            days = int(period[:-1])
            start = end - timedelta(days=days)
        else:
            start = end - timedelta(days=5*365)  # 5 años por defecto
        return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

    def get_prices(self, tickers, start=None, end=None, period=None):
        if not self.api_key:
            return pd.DataFrame()

        # Si no se especifican fechas, usar period o 5y por defecto
        if start is None or end is None:
            if period:
                start_str, end_str = self._period_to_dates(period)
            else:
                start_str, end_str = self._period_to_dates('5y')
        else:
            start_str = start
            end_str = end

        frames = []
        for t in tickers:
            # Polygon no usa símbolos con ^ para índices; para acciones/ETFs se usa tal cual
            ticker_poly = t
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker_poly}/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
            try:
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    print(f"  Polygon: {t} HTTP {resp.status_code}")
                    continue
                data = resp.json()
                results = data.get('results')
                if not results:
                    print(f"  Polygon: {t} sin resultados")
                    continue
                df = pd.DataFrame(results)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['o','h','l','c','v']]
                df.columns = ['Open','High','Low','Close','Volume']
                df.columns = pd.MultiIndex.from_product([df.columns, [t]])
                frames.append(df)
            except Exception as e:
                print(f"  Polygon: error descargando {t}: {e}")
                continue

        if frames:
            data = pd.concat(frames, axis=1)
            return data
        return pd.DataFrame()

    def get_treasury_yields(self, maturities=None):
        raise NotImplementedError("Usar FRED para yields del Tesoro")

    def get_fed_data(self, series=None):
        raise NotImplementedError("Usar FRED para datos de la Fed")

    def get_options_data(self, index=None):
        today = datetime.now().strftime('%Y-%m-%d')
        two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
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
            result = pd.DataFrame(index=df.index)
            result['volume'] = df['v']
            result['close'] = df['c']
            return result
        except Exception as e:
            print(f"Error descargando datos de Polygon: {e}")
            return pd.DataFrame()
