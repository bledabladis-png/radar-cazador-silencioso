import pandas as pd
import yfinance as yf
from .base import MarketDataProvider

class YahooProvider(MarketDataProvider):
    def __init__(self):
        self.name = "Yahoo Finance"

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            test = yf.download("^GSPC", period="5d")
            return not test.empty
        except:
            return False

    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        data = yf.download(tickers, start=start, end=end, period=period, auto_adjust=True)
        if not isinstance(data.columns, pd.MultiIndex):
            data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    def get_treasury_yields(self, maturities: list = None) -> pd.DataFrame:
        # Yahoo Finance no proporciona rendimientos del Tesoro directamente
        raise NotImplementedError("Yahoo Finance no tiene rendimientos del Tesoro. Usa FRED.")

    def get_fed_data(self, series: list = None) -> pd.DataFrame:
        raise NotImplementedError("Yahoo Finance no tiene datos de la Fed. Usa FRED.")
