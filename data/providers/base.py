from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    """Interfaz que deben implementar todos los proveedores de datos."""

    @abstractmethod
    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        """Descarga precios OHLCV para una lista de tickers."""

    @abstractmethod
    def get_treasury_yields(self, maturities: list = None) -> pd.DataFrame:
        """Obtiene rendimientos del Tesoro (para FRED principalmente)."""

    @abstractmethod
    def get_fed_data(self, series: list = None) -> pd.DataFrame:
        """Obtiene datos de la Reserva Federal (balance, SOFR, etc.)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Indica si el proveedor está operativo."""

    @abstractmethod
    def get_name(self) -> str:
        """Nombre legible del proveedor."""
