import os
import pandas as pd
import requests
from datetime import datetime, timedelta

class BackupProvider:
    """Respaldo multi-proveedor usando APIs gratuitas."""
    def __init__(self):
        self.tiingo_key = os.environ.get('TIINGO_API_KEY')
        self.twelve_key = os.environ.get('TWELVE_DATA_API_KEY')
        self.alpha_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.environ.get('FINNHUB_API_KEY')
        self.fmp_key = os.environ.get('FMP_API_KEY')
        self.daily_budget = 20
        self.calls = 0

    def _can_call(self):
        return self.calls < self.daily_budget

    def get_prices(self, tickers: list, period: str = '5y') -> pd.DataFrame:
        if not self._can_call():
            print("  [RESPALDO] Presupuesto de respaldo agotado.")
            return pd.DataFrame()

        frames = []
        for t in tickers:
            if not self._can_call():
                break

            df = self._tiingo_daily(t)
            if df is not None:
                print(f"  [RESPALDO] Tiingo suministró datos para {t}")
                frames.append(df)
                self.calls += 1
                continue

            df = self._twelve_data_daily(t)
            if df is not None:
                print(f"  [RESPALDO] Twelve Data suministró datos para {t}")
                frames.append(df)
                self.calls += 1
                continue

            df = self._alpha_vantage_daily(t)
            if df is not None:
                print(f"  [RESPALDO] Alpha Vantage suministró datos para {t}")
                frames.append(df)
                self.calls += 1
                continue

            df = self._finnhub_daily(t)
            if df is not None:
                print(f"  [RESPALDO] Finnhub suministró datos para {t}")
                frames.append(df)
                self.calls += 1
                continue

            df = self._fmp_daily(t)
            if df is not None:
                print(f"  [RESPALDO] FMP suministró datos para {t}")
                frames.append(df)
                self.calls += 1
                continue

        if frames:
            data = pd.concat(frames, axis=1)
            if not isinstance(data.columns, pd.MultiIndex):
                data.columns = pd.MultiIndex.from_tuples(data.columns)
            return data
        return pd.DataFrame()

    # --- Tiingo ---
    def _tiingo_daily(self, ticker: str) -> pd.DataFrame:
        if not self.tiingo_key:
            return None
        end = datetime.now().date()
        start = end - timedelta(days=5*365)
        url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
        headers = {'Authorization': f'Token {self.tiingo_key}'}
        params = {'startDate': start.isoformat(), 'endDate': end.isoformat(), 'format': 'json'}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data:
                return None
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df[['open','high','low','close','volume']]
            df.columns = ['Open','High','Low','Close','Volume']
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except Exception:
            return None

    # --- Twelve Data ---
    def _twelve_data_daily(self, ticker: str) -> pd.DataFrame:
        if not self.twelve_key:
            return None
        url = 'https://api.twelvedata.com/time_series'
        params = {
            'symbol': ticker,
            'interval': '1day',
            'outputsize': '1825',
            'apikey': self.twelve_key
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if data.get('status') == 'error':
                return None
            values = data.get('values')
            if not values:
                return None
            df = pd.DataFrame(values)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df = df[['open','high','low','close','volume']]
            df = df.astype(float)
            df.index.name = 'Date'
            df.columns = ['Open','High','Low','Close','Volume']
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except Exception:
            return None

    # --- Alpha Vantage ---
    def _alpha_vantage_daily(self, ticker: str) -> pd.DataFrame:
        if not self.alpha_key:
            return None
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'compact',
            'apikey': self.alpha_key
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            ts = data.get('Time Series (Daily)')
            if not ts:
                return None
            df = pd.DataFrame.from_dict(ts, orient='index', dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }, inplace=True)
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=5)
            df = df[df.index >= cutoff]
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except Exception:
            return None

    # --- Finnhub ---
    def _finnhub_daily(self, ticker: str) -> pd.DataFrame:
        if not self.finnhub_key:
            return None
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=5*365)).timestamp())
        url = 'https://finnhub.io/api/v1/stock/candle'
        params = {
            'symbol': ticker,
            'resolution': 'D',
            'from': start,
            'to': end,
            'token': self.finnhub_key
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            if data.get('s') != 'ok':
                return None
            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            }).set_index('Date')
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except Exception:
            return None

    # --- Financial Modeling Prep ---
    def _fmp_daily(self, ticker: str) -> pd.DataFrame:
        if not self.fmp_key:
            return None
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={self.fmp_key}'
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                return None
            data = r.json()
            history = data.get('historical')
            if not history:
                return None
            df = pd.DataFrame(history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df = df[['open','high','low','close','volume']]
            df.columns = ['Open','High','Low','Close','Volume']
            df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
            return df
        except Exception:
            return None
