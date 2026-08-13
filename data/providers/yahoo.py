import pandas as pd
import yfinance as yf
import time
from pathlib import Path
from .base import MarketDataProvider

class YahooProvider(MarketDataProvider):
    def __init__(self):
        self.name = "Yahoo Finance"
        self.max_retries = 2
        self.backoff_seconds = 5

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        try:
            test = yf.download("^GSPC", period="5d", progress=False)
            return not test.empty
        except:
            return False

    def _download_with_retries(self, tickers, **kwargs):
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"  Intento {attempt}/{self.max_retries} descargando {len(tickers)} tickers...")
                data = yf.download(tickers, progress=False, **kwargs)
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                last_exc = e
                print(f"  Error en intento {attempt}: {e}")
            time.sleep(self.backoff_seconds * attempt)
        raise RuntimeError(f"No se pudieron descargar datos tras {self.max_retries} intentos. Ultimo error: {last_exc}")

    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        kwargs = {}
        if start and end:
            kwargs['start'] = start
            kwargs['end'] = end
        else:
            kwargs['period'] = period
        try:
            data = self._download_with_retries(tickers, **kwargs)
            if not isinstance(data.columns, pd.MultiIndex):
                data.columns = pd.MultiIndex.from_tuples([(c, '') for c in data.columns])
            return data
        except Exception as e:
            print(f"  Yahoo fallo definitivamente: {e}")
            print("  Intentando fallback a cache local...")
            return self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        cache_path = Path("data/market_data_cache.csv")
        if cache_path.exists():
            try:
                data = pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)
                print(f"  Cache local cargado: {cache_path} ({len(data)} filas)")
                return data
            except Exception as e:
                print(f"  Error leyendo cache local: {e}")
        raise RuntimeError("No hay cache local disponible. Descarga fallida.")

    def get_treasury_yields(self, maturities: list = None) -> pd.DataFrame:
        raise NotImplementedError("Yahoo Finance no tiene rendimientos del Tesoro. Usa FRED.")

    def get_fed_data(self, series: list = None) -> pd.DataFrame:
        raise NotImplementedError("Yahoo Finance no tiene datos de la Fed. Usa FRED.")
