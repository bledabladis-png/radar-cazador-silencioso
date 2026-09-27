import pandas as pd
import pandas_datareader.data as web
from .base import MarketDataProvider

FRED_SERIES = {
    "fed_balance": "WALCL",
    "sofr": "SOFR",
    "reverse_repo": "RRPONTSYD",
    "fed_funds": "FEDFUNDS",
    "treasury_3m": "DTB3",
    "treasury_2y": "DGS2",
    "treasury_5y": "DGS5",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
}

class FredProvider(MarketDataProvider):
    def __init__(self):
        self.name = "FRED (Federal Reserve)"
        self._cache = {}

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return True

    def _download_series(self, series_id: str, start="2000-01-01", index=None) -> pd.Series:
        try:
            data = web.DataReader(series_id, "fred", start=start)
            s = data.iloc[:, 0].sort_index()
            # Si se proporciona un índice externo, reindexar y rellenar
            if index is not None:
                s = s.reindex(index).ffill()
            else:
                # Extender el índice hasta hoy y rellenar
                # F5.6-06: fecha de ejecucion (extremo de rejilla
                # para reindexar). No es fecha de observacion: la
                # observacion real es s.index[-1]. El consumidor
                # (regimes/liquidity.py) usa dropna(how='all').
                # index[-1], no la longitud del indice.
                today = pd.Timestamp.today().normalize()
                full_index = pd.date_range(start=s.index[0], end=today, freq='D')
                s = s.reindex(full_index).ffill()
            return s
        except (KeyError, ValueError, TypeError) as e:
            print(f"FRED {series_id}: {e}")
            return pd.Series(dtype=float)

    def get_prices(self, tickers: list, start: str = None, end: str = None, period: str = "10y") -> pd.DataFrame:
        raise NotImplementedError("FRED no proporciona precios de acciones. Usa Yahoo Finance.")

    def get_treasury_yields(self, maturities: list = None, index=None) -> pd.DataFrame:
        if maturities is None:
            maturities = ["treasury_3m", "treasury_2y", "treasury_10y"]
        df = pd.DataFrame()
        for key in maturities:
            if key in FRED_SERIES:
                series = self._download_series(FRED_SERIES[key], index=index)
                if not series.empty:
                    df[key] = series
        return df

    def get_fed_data(self, index=None) -> pd.DataFrame:
        series_ids = ["fed_balance", "sofr", "reverse_repo", "fed_funds"]
        df = pd.DataFrame()
        for key in series_ids:
            if key in FRED_SERIES:
                s = self._download_series(FRED_SERIES[key], index=index)
                if not s.empty:
                    df[key] = s
        return df

