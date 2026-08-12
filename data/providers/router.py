from .yahoo import YahooProvider
from .fred import FredProvider
from .stooq import StooqProvider
from .polygon import PolygonProvider
import pandas as pd
import os
from pathlib import Path

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
                except Exception as e:
                    print(f"{provider.get_name()} fallo: {e}; intentando siguiente...")
                    continue
        # Fallback global
        print("Todos los proveedores fallaron. Intentando cache local global...")
        return self._load_cache(tickers)

    def _load_cache(self, tickers):
        cache_path = Path("data/market_data_cache.csv")
        if cache_path.exists():
            data = pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)
            missing = [t for t in tickers if t not in data.columns.get_level_values(1)]
            if missing:
                print(f"  Cache no contiene {len(missing)} tickers.")
                # Filtrar a los disponibles
                available = [t for t in tickers if t in data.columns.get_level_values(1)]
                if available:
                    subset = data.loc[:, data.columns.get_level_values(1).isin(available)]
                    print(f"  Usando {len(available)} tickers del cache.")
                    return subset
                raise RuntimeError("Cache no tiene tickers solicitados.")
            print(f"  Cache local cargado: {len(data)} filas.")
            return data
        raise RuntimeError("Ningun proveedor disponible y no hay cache local.")

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
        macro_dir = 'data/macro_manual'
        if os.path.exists(macro_dir):
            try:
                df = self._load_macro_manual(macro_dir)
                if df is not None and not df.empty:
                    print("Usando datos macro manuales locales para liquidez.")
                    return df
            except Exception:
                pass

        for name in ["fred", "yahoo"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    return provider.get_fed_data()
                except:
                    continue
        return None

    def _load_macro_manual(self, data_dir):
        dfs = []
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv'):
                path = os.path.join(data_dir, fname)
                try:
                    df = pd.read_csv(path)
                    if 'date' not in df.columns:
                        continue
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    prefix = os.path.splitext(fname)[0]
                    df = df.add_prefix(f'{prefix}_')
                    dfs.append(df)
                except Exception:
                    pass

        if not dfs:
            return None

        combined = pd.concat(dfs, axis=1)
        rename_map = {}
        for col in combined.columns:
            if col.startswith('walcl_'):
                rename_map[col] = 'fed_balance'
            elif col.startswith('rrpp_'):
                rename_map[col] = 'reverse_repo'
            elif col.startswith('sofr_'):
                rename_map[col] = 'sofr'
            elif col.startswith('discount_rate_'):
                rename_map[col] = 'fed_funds'
            elif col.startswith('iorb_'):
                rename_map[col] = 'iorb'
        combined.rename(columns=rename_map, inplace=True)
        target_cols = ['fed_balance', 'reverse_repo', 'sofr', 'fed_funds', 'iorb']
        existing = [c for c in target_cols if c in combined.columns]
        if existing:
            combined = combined[existing]
        else:
            return None
        return combined

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
