from .yahoo import YahooProvider
from .fred import FredProvider
from .polygon import PolygonProvider
import pandas as pd
import os
from pathlib import Path
from config.settings import CACHE_MARKET_PATH

class DataRouter:
    def __init__(self):
        self.providers = {
            "yahoo": YahooProvider(),
            "fred": FredProvider(),
            "polygon": PolygonProvider(),
        }
        # A5-10: Polygon es equity (prioridad); FRED es macro (ultimo recurso).
        self.preferred_order = ["yahoo", "polygon", "fred"]

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
        """Carga cache local. A5-11: NO devuelve subset silencioso.

        Si algun ticker solicitado no esta en el cache, raise RuntimeError.
        El caller (data_loader) captura y pasa al backup provider.
        Principio PROMPT Seccion 2: "si no hay suficiente -> N/D u omitir.
        No imputar." Devolver un subset parcial es equivalente a imputar
        silenciosamente ausencia de datos.
        """
        cache_path = Path(CACHE_MARKET_PATH)
        if not cache_path.exists():
            raise RuntimeError("Ningun proveedor disponible y no hay cache local.")
        data = pd.read_parquet(cache_path)
        available = data.columns.get_level_values(1)
        missing = [t for t in tickers if t not in available]
        if missing:
            raise RuntimeError(
                f"Cache local no contiene {len(missing)}/{len(tickers)} "
                f"tickers solicitados. No se devuelve subset parcial."
            )
        print(f"  Cache local cargado: {len(data)} filas.")
        return data

    def get_treasury_data(self):
        for name in ["fred", "yahoo"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    return provider.get_treasury_yields()
                except Exception as e:
                    print(f"  [WARN] router: {provider.get_name()} get_treasury_yields fallo: {e}")
                    continue
        return None

    def get_fed_data(self):
        # 1) Intentar primero proveedor automático FRED
        for name in ["fred", "yahoo"]:
            provider = self.providers[name]
            if provider.is_available():
                try:
                    data = provider.get_fed_data()
                    if data is not None and not data.empty:
                        print(f"Usando {provider.get_name()} para datos de liquidez.")
                        return data
                except Exception:
                    continue

        # 2) Fallback a CSVs manuales
        macro_dir = 'data/macro_manual'
        if os.path.exists(macro_dir):
            try:
                df = self._load_macro_manual(macro_dir)
                if df is not None and not df.empty:
                    print("Usando datos macro manuales locales para liquidez.")
                    return df
            except Exception as e:
                print(f"  [WARN] router: macro_manual fallback: {e}")

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
                except Exception as e:
                    print(f"  [WARN] router: _load_macro_manual: {e}")

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

