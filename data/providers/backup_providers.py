import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from src.instrument_registry import resolve_symbol
from config.settings import CACHE_MARKET_PATH, CACHE_STOCKS_PATH

class RateLimiter:
    """Controla llamadas por minuto y por día para un proveedor."""
    def __init__(self, daily_limit=None, minute_limit=None):
        self.daily_limit = daily_limit
        self.minute_limit = minute_limit
        self.calls_today = 0
        self.last_call_time = None
        self.last_day = datetime.now().date()

    def reset_if_new_day(self):
        today = datetime.now().date()
        if today != self.last_day:
            self.last_day = today
            self.calls_today = 0

    def can_call(self):
        self.reset_if_new_day()
        if self.daily_limit is not None and self.calls_today >= self.daily_limit:
            return False
        if self.minute_limit is not None and self.last_call_time is not None:
            elapsed = (datetime.now() - self.last_call_time).total_seconds()
            if elapsed < 60.0 / self.minute_limit:
                return False
        return True

    def record_call(self):
        self.calls_today += 1
        self.last_call_time = datetime.now()

class CircuitBreaker:
    """Desactiva un proveedor temporalmente tras varios fallos consecutivos."""
    def __init__(self, failure_threshold=5, reset_timeout=900):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = 'ACTIVE'
        self.last_failure_time = None

    def allow_call(self):
        if self.state == 'OPEN':
            if self.last_failure_time and (datetime.now() - self.last_failure_time).total_seconds() >= self.reset_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        return True

    def record_success(self):
        self.state = 'ACTIVE'
        self.failures = 0
        self.last_failure_time = None

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = datetime.now()
        if self.failures >= self.failure_threshold:
            self.state = 'OPEN'

class BackupProvider:
    """Respaldo multi-proveedor con rate limiting, circuit breaker y validación cruzada."""
    def __init__(self):
        self.providers = {
            'tiingo': {
                'key': os.environ.get('TIINGO_API_KEY'),
                'limiter': RateLimiter(daily_limit=200, minute_limit=1),
                'breaker': CircuitBreaker(),
            },
            'twelve_data': {
                'key': os.environ.get('TWELVE_DATA_API_KEY'),
                'limiter': RateLimiter(daily_limit=800, minute_limit=8),
                'breaker': CircuitBreaker(),
            },
            'alpha_vantage': {
                'key': os.environ.get('ALPHA_VANTAGE_API_KEY'),
                'limiter': RateLimiter(daily_limit=25, minute_limit=5),
                'breaker': CircuitBreaker(),
            },
            'finnhub': {
                'key': os.environ.get('FINNHUB_API_KEY'),
                'limiter': RateLimiter(daily_limit=None, minute_limit=60),
                'breaker': CircuitBreaker(),
            },
            'fmp': {
                'key': os.environ.get('FMP_API_KEY'),
                'limiter': RateLimiter(daily_limit=250, minute_limit=None),
                'breaker': CircuitBreaker(),
            },
        }
        self.daily_budget = 20
        self.calls = 0
        # FU-002 (2026-09-15): cache + status (VALID/INVALID/UNAVAILABLE).
        self.reference_cache, self.reference_cache_status = self._load_reference_cache()
        print(f"  [REF-CACHE] status={self.reference_cache_status}")

    def _can_call_global(self):
        return self.calls < self.daily_budget

    def _load_reference_cache(self):
        """Carga caches locales con verificacion de manifest (FU-002, 2026-09-15).

        Cada parquet se verifica contra su manifest (<parquet>.manifest.json).
        Estados por parquet:
          VALID               -> parquet + manifest OK, quality.status == VALID
          VALID_WITH_MISSING  -> calidad aceptable con huecos legitimos
                                 (close_nan > 0 sin duplicacion). FU-002-evo.
          INVALID             -> sha256 mismatch, schema_version desconocido,
                                 o manifest quality.status == INVALID
          UNAVAILABLE         -> parquet o manifest ausente, o error de lectura

        Combinacion (conservadora):
          - Cualquier INVALID -> INVALID global.
          - Ambos UNAVAILABLE -> UNAVAILABLE global.
          - Resto -> VALID (parcial).

        Returns:
            (DataFrame, status) con status en {VALID, INVALID, UNAVAILABLE}.
        """
        import json as _json
        import hashlib as _hashlib

        frames = []
        statuses = []
        for path in [CACHE_MARKET_PATH, CACHE_STOCKS_PATH]:
            p = Path(path)
            manifest_path = Path(str(p) + '.manifest.json')

            if not p.exists():
                statuses.append('UNAVAILABLE')
                continue
            if not manifest_path.exists():
                print(f"  [REF-CACHE] {path}: sin manifest -> UNAVAILABLE")
                statuses.append('UNAVAILABLE')
                continue
            try:
                manifest = _json.loads(manifest_path.read_text(encoding='utf-8'))
            except Exception as e:
                print(f"  [REF-CACHE] {path}: manifest ilegible ({e}) -> UNAVAILABLE")
                statuses.append('UNAVAILABLE')
                continue
            if manifest.get('schema_version') != 1:
                print(f"  [REF-CACHE] {path}: schema_version desconocido -> INVALID")
                statuses.append('INVALID')
                continue
            try:
                h = _hashlib.sha256()
                with open(p, 'rb') as fh:
                    for chunk in iter(lambda: fh.read(65536), b''):
                        h.update(chunk)
                actual_sha = h.hexdigest()
            except Exception as e:
                print(f"  [REF-CACHE] {path}: error sha256 ({e}) -> UNAVAILABLE")
                statuses.append('UNAVAILABLE')
                continue
            expected_sha = manifest.get('artifact', {}).get('sha256', '')
            if actual_sha != expected_sha:
                print(f"  [REF-CACHE] {path}: sha256 mismatch -> INVALID")
                statuses.append('INVALID')
                continue
            quality_status = manifest.get('quality', {}).get('status', '')
            if quality_status == 'INVALID':
                print(f"  [REF-CACHE] {path}: manifest quality=INVALID -> INVALID")
                statuses.append('INVALID')
                continue
            # FU-002-evo (2026-09-15): VALID_WITH_MISSING es aceptable
            # (huecos legitimos preservados por B1, no corrupcion).
            if quality_status == 'VALID_WITH_MISSING':
                print(f"  [REF-CACHE] {path}: quality=VALID_WITH_MISSING (huecos legitimos)")
            elif quality_status != 'VALID':
                print(f"  [REF-CACHE] {path}: quality='{quality_status}' -> UNAVAILABLE")
                statuses.append('UNAVAILABLE')
                continue
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    frames.append(df)
                statuses.append('VALID')
            except Exception as e:
                print(f"  [REF-CACHE] {path}: error lectura ({e}) -> UNAVAILABLE")
                statuses.append('UNAVAILABLE')

        if 'INVALID' in statuses:
            combined_status = 'INVALID'
        elif statuses and all(s == 'UNAVAILABLE' for s in statuses):
            combined_status = 'UNAVAILABLE'
        else:
            combined_status = 'VALID'

        if frames:
            combined = pd.concat(frames, axis=1)
            if combined.columns.duplicated().any():
                combined = combined.loc[:, ~combined.columns.duplicated(keep='first')]
        else:
            combined = pd.DataFrame()

        return combined, combined_status

    def _validate_ohlcv(self, df):
        """Validación básica de respuesta: columnas y filas suficientes."""
        if df is None or df.empty:
            return False
        # Comprobar que hay al menos 10 filas
        if len(df) < 10:
            return False
        # Comprobar que existen las columnas Open, High, Low, Close, Volume
        required = {'Open','High','Low','Close','Volume'}
        for col in required:
            if col not in df.columns.get_level_values(0):
                return False
        return True

    def _validate_with_cache(self, ticker, df):
        """Compara ultimo cierre con cache (FU-002, 2026-09-15).

        Returns:
            True  -> comparacion paso contra referencia VALID.
            False -> discrepancia >5% contra referencia VALID.
            None  -> referencia no disponible (UNAVAILABLE) o invalidada
                     (INVALID). El dato del proveedor se acepta pero NO
                     se marca como validado.
        """
        if self.reference_cache_status in ('INVALID', 'UNAVAILABLE'):
            return None
        if self.reference_cache.empty:
            return None
        try:
            if ticker not in self.reference_cache.columns.get_level_values(1):
                return None
            close_cache = self.reference_cache.loc[:, ('Close', ticker)].dropna()
            if close_cache.empty:
                return None
            if isinstance(close_cache, pd.DataFrame):
                close_cache = close_cache.iloc[:, 0]
            ref_close = float(close_cache.iloc[-1])
            new_close = float(df[('Close', ticker)].iloc[-1])
            if ref_close == 0:
                return None
            diff = abs(new_close - ref_close) / abs(ref_close)
            if diff > 0.05:
                print(f"  [VALIDACION] {ticker}: discrepancia >5% con cache ({ref_close:.2f} vs {new_close:.2f}). Dato rechazado.")
                return False
            return True
        except Exception as e:
            print(f"  [WARN] backup validate_with_cache {ticker}: {e}")
            return None

    def get_prices(self, tickers: list, period: str = '5y') -> pd.DataFrame:
        if not self._can_call_global():
            print("  [RESPALDO] Presupuesto global agotado.")
            return pd.DataFrame()

        frames = []
        for t in tickers:
            if not self._can_call_global():
                break

            # Intentar con cada proveedor en orden
            for provider_name, config in self.providers.items():
                if not config['key']:
                    continue
                if not config['limiter'].can_call():
                    print(f"  [RATE] {provider_name}: límite alcanzado, saltando {t}")
                    continue
                if not config['breaker'].allow_call():
                    print(f"  [CIRCUIT] {provider_name}: circuito abierto, saltando {t}")
                    continue

                # Obtener símbolo específico del proveedor
                provider_symbol = resolve_symbol(t, provider_name)
                if provider_symbol is None:
                    print(f"  [COBERTURA] {provider_name}: no soporta {t}")
                    continue

                try:
                    # Llamar al método correspondiente con el símbolo del proveedor
                    method = getattr(self, f"_{provider_name}_daily")
                    df = method(provider_symbol)
                    if df is not None and self._validate_ohlcv(df):
                        # Renombrar columna ticker externo -> canónico
                        df.columns = pd.MultiIndex.from_product([df.columns.get_level_values(0), [t]])
                        # Validacion cruzada con cache (FU-002: True/False/None)
                        result = self._validate_with_cache(t, df)
                        if result is False:
                            print(f"  [VALIDACION] {provider_name}: dato rechazado para {t}")
                            config['breaker'].record_failure()
                            break  # no probar mas proveedores para este ticker
                        # True (validado) o None (sin referencia confiable): aceptar
                        tag = 'validado' if result is True else 'sin referencia'
                        print(f"  [RESPALDO] {provider_name} suministro datos para {t} (simbolo {provider_symbol}, {tag})")
                        frames.append(df)
                        self.calls += 1
                        config['limiter'].record_call()
                        config['breaker'].record_success()
                        break  # pasar al siguiente ticker
                    else:
                        # Proveedor falló
                        config['breaker'].record_failure()
                except Exception as e:
                    print(f"  [ERROR] {provider_name}: {e}")
                    config['breaker'].record_failure()
            # Fin bucle de proveedores

        if frames:
            data = pd.concat(frames, axis=1)
            if not isinstance(data.columns, pd.MultiIndex):
                data.columns = pd.MultiIndex.from_tuples(data.columns)
            return data
        return pd.DataFrame()

    # ================= MÉTODOS DE DESCARGA =================
    def _tiingo_daily(self, ticker: str) -> pd.DataFrame:
        key = self.providers['tiingo']['key']
        end = datetime.now().date()
        start = end - timedelta(days=5*365)
        url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
        headers = {'Authorization': f'Token {key}'}
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

    def _twelve_data_daily(self, ticker: str) -> pd.DataFrame:
        key = self.providers['twelve_data']['key']
        url = 'https://api.twelvedata.com/time_series'
        params = {
            'symbol': ticker,
            'interval': '1day',
            'outputsize': '1825',
            'apikey': key
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

    def _alpha_vantage_daily(self, ticker: str) -> pd.DataFrame:
        key = self.providers['alpha_vantage']['key']
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': ticker,
            'outputsize': 'compact',
            'apikey': key
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

    def _finnhub_daily(self, ticker: str) -> pd.DataFrame:
        key = self.providers['finnhub']['key']
        end = int(datetime.now().timestamp())
        start = int((datetime.now() - timedelta(days=5*365)).timestamp())
        url = 'https://finnhub.io/api/v1/stock/candle'
        params = {
            'symbol': ticker,
            'resolution': 'D',
            'from': start,
            'to': end,
            'token': key
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

    def _fmp_daily(self, ticker: str) -> pd.DataFrame:
        key = self.providers['fmp']['key']
        url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={key}'
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
