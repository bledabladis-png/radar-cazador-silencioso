import pandas as pd
from .base import MarketDataProvider

class StooqProvider(MarketDataProvider):
    def __init__(self):
        self.name = "Stooq"

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            test = pd.read_csv("https://stooq.com/q/d/l/?s=^spx&i=d", limit=5)
            return not test.empty
        except:
            return False

    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        # Stooq requiere mapeo de símbolos; implementación simplificada como fallback
        raise NotImplementedError("Stooq no está completamente implementado. Usa Yahoo Finance como principal.")

    def get_treasury_yields(self, maturities: list = None) -> pd.DataFrame:
        raise NotImplementedError("Stooq no tiene rendimientos del Tesoro.")

    def get_fed_data(self, series: list = None) -> pd.DataFrame:
        raise NotImplementedError("Stooq no tiene datos de la Fed.")
