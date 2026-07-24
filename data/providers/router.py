from .yahoo import YahooProvider
from .fred import FredProvider
from .stooq import StooqProvider
from .polygon import PolygonProvider

class DataRouter:
    def __init__(self):
        self.providers = {
            "yahoo": YahooProvider(),
            "fred": FredProvider(),
            "stooq": StooqProvider(),
            "polygon": PolygonProvider(),
        }
        self.preferred_order = ["yahoo", "fred", "stooq", "polygon"]

    def get_market_data(self, tickers: list, period: str = "10y"):
        for name in self.preferred_order:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    print(f"Usando {provider.get_name()} para datos de mercado...")
                    return provider.get_prices(tickers, period=period)
                except:
                    print(f"{provider.get_name()} falló, intentando siguiente...")
                    continue
        raise RuntimeError("Ningún proveedor de datos de mercado disponible.")

    def get_treasury_data(self):
        for name in ["fred", "yahoo"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    return provider.get_treasury_yields()
                except:
                    continue
        return None

    def get_fed_data(self):
        for name in ["fred", "yahoo"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    return provider.get_fed_data()
                except:
                    continue
        return None

    def get_options_data(self):
        for name in ["polygon", "fred"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    data = provider.get_options_data()
                    if data is not None and not data.empty:
                        return data
                except:
                    continue
        return None
