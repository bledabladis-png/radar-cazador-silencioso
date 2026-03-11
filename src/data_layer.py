"""
data_layer.py - Capa de datos del Radar Macro RotaciÃ³n Global
Se encarga de descargar, validar, cachear y proporcionar los datos de mercado.
Incluye metadatos para auditorÃ­a (checksum, fuente, versiÃ³n).
"""

import os
import pandas as pd
import numpy as np
import requests
import hashlib
import yaml
import time
import zipfile
import shutil
import re
import logging
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO, BytesIO
import yfinance as yf
from tiingo import TiingoClient
from bs4 import BeautifulSoup

# ConfiguraciÃ³n del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLayer:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Mapeo de ticker a product_id para iShares (versiÃ³n .com)
        self.ishares_product_ids = {
            'IVV': '239726',
            'EEM': '239637',
            'LQD': '239714',
            'IWM': '239710',
        }

# Leer clave de Tiingo desde variable de entorno (prioritario) o archivo
        self.tiingo_key = os.getenv('TIINGO_KEY')
        if not self.tiingo_key:
            tiingo_key_file = self.config['data']['tiingo_key_file']
            if os.path.exists(tiingo_key_file):
                with open(tiingo_key_file, 'r') as f:
                    self.tiingo_key = f.read().strip()
            else:
                self.tiingo_key = None
                logger.warning("No se encontró clave Tiingo. Se usará Yahoo Finance.")
            try:
                self.tiingo_client = TiingoClient({'api_key': self.tiingo_key})
            except Exception as e:
                logger.error(f"Error al crear cliente Tiingo: {e}. Se usarÃ¡ Yahoo Finance.")
                self.tiingo_client = None
        else:
            self.tiingo_client = None

        # Crear carpetas necesarias
        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed')
        self.cache_dir = Path('data/cache')
        self.treasury_dir = self.raw_dir / 'treasury'
        for d in [self.raw_dir, self.processed_dir, self.cache_dir, self.treasury_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Lista de tickers principales (precios de cierre)
        self.tickers = self.config['data']['tickers']['macro']

        # Lista de tickers para descarga OHLC (desde Stooq)
        self.ohlc_tickers = self.config['data'].get('ohlc_tickers', [])

        # Rate limiting para Twelve Data (8 peticiones por minuto)
        self._twelve_last_call = 0
        self._twelve_min_interval = 60 / 8  # 7.5 segundos

    # ------------------------------------------------------------------
    # VerificaciÃ³n de integridad (Fase 12)
    # ------------------------------------------------------------------
    def verify_integrity(self, file_path):
        if not file_path.exists():
            return False, f"Archivo no encontrado: {file_path}"
        try:
            df = pd.read_parquet(file_path)
            stored_checksum = df.attrs.get('checksum', None)
            if stored_checksum is None:
                return False, "El archivo no tiene checksum almacenado (versiÃ³n antigua)"
            current_checksum = self._calculate_checksum(df)
            if current_checksum == stored_checksum:
                return True, "Checksum verificado"
            else:
                return False, f"Checksum no coincide: esperado {stored_checksum}, obtenido {current_checksum}"
        except Exception as e:
            return False, f"Error al leer archivo: {e}"

    # ------------------------------------------------------------------
    # Descarga de datos de breadth avanzado (tickers especiales de Yahoo)
    # ------------------------------------------------------------------
    def download_breadth_ticker(self, ticker, force=False):
        cache_file = self.raw_dir / f"{ticker.replace('^', '')}.parquet"
        if cache_file.exists() and not force:
            logger.info(f"Cargando {ticker} desde cachÃ©.")
            return pd.read_parquet(cache_file)
        try:
            df = yf.download(ticker, period="max", progress=False, auto_adjust=False)
            if df.empty:
                logger.warning(f"No se obtuvieron datos para {ticker}")
                return pd.DataFrame()
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_parquet(cache_file)
            logger.info(f"{ticker} guardado en cachÃ©")
            return df
        except Exception as e:
            logger.error(f"Error descargando {ticker}: {e}")
            return pd.DataFrame()

    def load_breadth_data(self, ticker):
        cache_file = self.raw_dir / f"{ticker.replace('^', '')}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        else:
            logger.warning(f"No hay datos en cachÃ© para {ticker}. Ejecuta download_breadth_ticker primero.")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Descarga de datos de la CFTC (Commitment of Traders)
    # ------------------------------------------------------------------
    def download_cftc_year(self, year, force=False):
        cftc_dir = self.raw_dir / 'cftc'
        cftc_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
        zip_path = cftc_dir / f"fut_fin_txt_{year}.zip"
        if zip_path.exists() and not force:
            logger.info(f"CFTC {year} ya existe en cachÃ©.")
            return
        logger.info(f"Descargando datos CFTC anio {year}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Descargado {zip_path.name}")
        except Exception as e:
            logger.error(f"Error descargando {year}: {e}")
            return
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cftc_dir / f"extracted_{year}")
            logger.info(f"ExtraÃ­do {zip_path.name}")
        except Exception as e:
            logger.error(f"Error extrayendo {year}: {e}")

    def download_cftc_all(self, start_year=2000, force=False):
        current_year = datetime.now().year
        for year in range(start_year, current_year + 1):
            self.download_cftc_year(year, force=force)
        logger.info("Descarga de CFTC completada.")

    def load_cftc_data(self, year=None):
        cftc_dir = self.raw_dir / 'cftc'
        if not cftc_dir.exists():
            logger.error("No hay datos de CFTC. Ejecuta download_cftc_all() primero.")
            return pd.DataFrame()
        if year is None:
            years = []
            for path in cftc_dir.glob("extracted_*"):
                try:
                    y = int(path.name.replace("extracted_", ""))
                    years.append(y)
                except:
                    pass
            if not years:
                logger.error("No se encontraron anios extraÃ­dos.")
                return pd.DataFrame()
            year = max(years)
        extract_path = cftc_dir / f"extracted_{year}"
        if not extract_path.exists():
            logger.error(f"No existe el directorio extraÃ­do para el anio {year}")
            return pd.DataFrame()
        csv_files = list(extract_path.glob("*.txt"))
        if not csv_files:
            csv_files = list(extract_path.glob("*.csv"))
        if not csv_files:
            logger.error(f"No se encontraron archivos de datos en {extract_path}")
            return pd.DataFrame()
        df = pd.read_csv(csv_files[0], delimiter=',', quotechar='"', encoding='latin1')
        logger.info(f"Datos CFTC {year} cargados desde {csv_files[0].name}")
        return df

    def load_cftc_all_years(self, start_year=2000):
        current_year = datetime.now().year
        dfs = []
        for year in range(start_year, current_year + 1):
            df = self.load_cftc_data(year)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            logger.error("No se pudo cargar ningÃºn dato de CFTC")
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Funciones de descarga desde diferentes fuentes
    # ------------------------------------------------------------------
    def _fetch_tiingo(self, ticker, start_date, end_date):
        if not self.tiingo_client:
            return None
        try:
            data = self.tiingo_client.get_ticker_price(ticker,
                                                        fmt='json',
                                                        startDate=start_date.strftime('%Y-%m-%d'),
                                                        endDate=end_date.strftime('%Y-%m-%d'),
                                                        frequency='daily')
            df = pd.DataFrame(data)
            if df.empty:
                return None
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df.set_index('date', inplace=True)
            df = df[['adjClose']].rename(columns={'adjClose': ticker})
            return df
        except Exception as e:
            logger.error(f"Error descargando {ticker} de Tiingo: {e}")
            return None

    def _fetch_yahoo(self, ticker, start_date, end_date):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df.empty:
                return None
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                logger.error(f"Yahoo: no se encontrÃ³ columna de precio para {ticker}")
                return None
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            result = pd.DataFrame({ticker: series})
            return result
        except Exception as e:
            logger.error(f"Error descargando {ticker} de Yahoo: {e}")
            return None

    def _fetch_yahoo_ohlc(self, ticker, start_date, end_date):
        """
        Descarga datos OHLC (Open, High, Low, Close) desde Yahoo Finance.
        """
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df.empty:
                return None
            # Asegurar ?ndice de fechas sin zona horaria
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Seleccionar y renombrar columnas OHLC
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in df.columns for col in ohlc_cols):
                df_ohlc = df[ohlc_cols].copy()
                df_ohlc.columns = [f"{ticker}_{col}" for col in ohlc_cols]
                return df_ohlc
            else:
                logger.warning(f"Yahoo OHLC: no se encontraron todas las columnas para {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error descargando OHLC de {ticker} desde Yahoo: {e}")
            return None

    # ------------------------------------------------------------------
    # Descarga de datos OHLC desde Stooq (proveedor alternativo)
    # ------------------------------------------------------------------
    def _fetch_stooq_ohlc(self, ticker, start_date, end_date):
        """
        Descarga datos OHLC desde Stooq.
        """
        try:
            # Construir URL para Stooq (formato: ticker.us para ETFs americanos)
            stooq_ticker = ticker.lower().replace('^', '').replace('-', '').replace('=', '')
            if not stooq_ticker.endswith('.us'):
                stooq_ticker = stooq_ticker + '.us'
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
            if 'Date' not in df.columns:
                logger.error(f"Stooq: formato inesperado para {ticker}")
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            # Renombrar columnas
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error descargando OHLC de {ticker} desde Stooq: {e}")
            return None

    # ------------------------------------------------------------------
    # Descarga de datos de Twelve Data con rate limiting
    # ------------------------------------------------------------------
    def _fetch_twelve_data(self, ticker, start_date, end_date, interval='1day'):
        """
        Descarga datos de Twelve Data (precios OHLC y volumen) con rate limiting.
        Retorna DataFrame con columnas: open, high, low, close, volume.
        """
        if not self.twelve_key:
            logger.warning(f"No hay clave Twelve Data, no se puede descargar {ticker}")
            return None

        # Rate limiting
        now = time.time()
        elapsed = now - self._twelve_last_call
        if elapsed < self._twelve_min_interval:
            sleep_time = self._twelve_min_interval - elapsed
            logger.info(f"Rate limiting Twelve Data: esperando {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        url = "https://api.twelvedata.com/time_series"
        params = {
            'symbol': ticker,
            'interval': interval,
            'apikey': self.twelve_key,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'format': 'JSON'
        }
        try:
            response = requests.get(url, params=params, timeout=15)
            self._twelve_last_call = time.time()
            response.raise_for_status()
            data = response.json()
            if 'values' not in data:
                logger.error(f"Twelve Data: respuesta sin 'values' para {ticker}: {data.get('message', '')}")
                return None
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            # Convertir a numÃ©rico
            for col in ['open','high','low','close','volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # Renombrar columnas
            df.columns = [f"{ticker}_{col.capitalize()}" for col in df.columns]
            logger.info(f"Twelve Data: {ticker} descargado correctamente")
            return df
        except Exception as e:
            logger.error(f"Error en Twelve Data para {ticker}: {e}")
            return None

    # ------------------------------------------------------------------
    # Descarga de datos del Bank for International Settlements (BIS)
    # ------------------------------------------------------------------
    def download_bis_data(self, force=False):
        bis_dir = self.raw_dir / 'bis'
        bis_dir.mkdir(parents=True, exist_ok=True)
        datasets = {
            'gli': 'https://data.bis.org/static/bulk/WS_GLI_csv_col.zip',
            'credit': 'https://www.bis.org/statistics/full_credit_csv.zip',
            'lbs': 'https://www.bis.org/statistics/full_lbs_csv.zip'
        }
        for name, url in datasets.items():
            zip_path = bis_dir / f"{name}.zip"
            extract_path = bis_dir / name
            if not zip_path.exists() or force:
                logger.info(f"Descargando {name} desde BIS...")
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        with open(zip_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                    logger.info(f"Descargado {name}.zip")
                except Exception as e:
                    logger.error(f"Error descargando {name}: {e}")
                    continue
            if not extract_path.exists() or force:
                logger.info(f"Extrayendo {name}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    logger.info(f"ExtraÃ­do {name} en {extract_path}")
                except Exception as e:
                    logger.error(f"Error extrayendo {name}: {e}")
                    continue
        logger.info("Datos del BIS actualizados.")

    def load_bis_gli(self):
        gli_dir = self.raw_dir / 'bis' / 'gli'
        if not gli_dir.exists():
            logger.error("No se encuentran datos del BIS. Ejecuta download_bis_data primero.")
            return pd.DataFrame()
        csv_files = list(gli_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No hay archivos CSV en el directorio gli")
            return pd.DataFrame()
        csv_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        df = pd.read_csv(csv_files[0])
        logger.info(f"BIS GLI cargado desde {csv_files[0].name}")
        return df

    # ------------------------------------------------------------------
    # ValidaciÃ³n y checksum
    # ------------------------------------------------------------------
    def _validate_data(self, df, ticker):
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        if not df.index.is_monotonic_increasing:
            logger.warning(f"{ticker}: Ã­ndice no monÃ³tono. Ordenando.")
            df.sort_index(inplace=True)
        if (df[ticker] < 0).any():
            raise ValueError(f"{ticker}: precios negativos detectados.")
        nan_pct = df[ticker].isna().mean()
        if nan_pct > 0.05:
            logger.warning(f"{ticker}: {nan_pct:.2%} NaNs. Se rellenarÃ¡n.")
        df[ticker] = df[ticker].ffill().bfill()
        return df

    def _calculate_checksum(self, df):
        data_string = df.to_json().encode('utf-8')
        return hashlib.sha256(data_string).hexdigest()

    # ------------------------------------------------------------------
    # Carga de curva del Tesoro desde archivos CSV locales
    # ------------------------------------------------------------------
    def _read_treasury_csv(self):
        csv_files = sorted(self.treasury_dir.glob("par-yield-curve-rates-*.csv"))
        if not csv_files:
            logger.error("No se encontraron archivos CSV del Tesoro en data/raw/treasury/")
            return pd.DataFrame(columns=['spread_10y2y'])
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if 'Date' not in df.columns:
                    continue
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                col_2y = None
                col_10y = None
                for col in df.columns:
                    if '2 Yr' in col or '2 yr' in col:
                        col_2y = col
                    if '10 Yr' in col or '10 yr' in col:
                        col_10y = col
                if col_2y is None or col_10y is None:
                    continue
                df_spread = pd.DataFrame(index=df.index)
                df_spread['spread_10y2y'] = df[col_10y] - df[col_2y]
                dfs.append(df_spread)
            except Exception as e:
                logger.error(f"Error procesando {file.name}: {e}")
                continue
        if not dfs:
            return pd.DataFrame(columns=['spread_10y2y'])
        df_combined = pd.concat(dfs)
        df_combined.sort_index(inplace=True)
        df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
        return df_combined

    def _fetch_treasury_curve(self, force=False):
        cache_file = self.raw_dir / "treasury_curve.parquet"
        if cache_file.exists() and not force:
            logger.info("Cargando curva del Tesoro desde cachÃ©.")
            return pd.read_parquet(cache_file)
        df_result = self._read_treasury_csv()
        df_result.to_parquet(cache_file)
        logger.info("Curva del Tesoro guardada en cachÃ© desde CSV locales.")
        return df_result

    # ------------------------------------------------------------------
    # Descarga de un ticker individual con cachÃ©
    # ------------------------------------------------------------------
    def download_ticker(self, ticker, start_date, end_date, force=False):
        cache_file = self.raw_dir / f"{ticker}.parquet"
        if cache_file.exists() and not force:
            logger.info(f"Cargando {ticker} desde cachÃ©.")
            df = pd.read_parquet(cache_file)
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if df.index[0] <= pd.Timestamp(start_date) and df.index[-1] >= pd.Timestamp(end_date):
                return df
        df = self._fetch_tiingo(ticker, start_date, end_date)
        source = 'tiingo'
        if df is None:
            logger.warning(f"Tiingo fallÃ³ para {ticker}, intentando Yahoo.")
            df = self._fetch_yahoo(ticker, start_date, end_date)
            source = 'yahoo'
            if df is None:
                raise Exception(f"No se pudo descargar {ticker} de ninguna fuente.")
        df = self._validate_data(df, ticker)
        checksum = self._calculate_checksum(df)
        df.attrs['checksum'] = checksum
        df.attrs['source'] = source
        df.attrs['last_updated'] = datetime.now().isoformat()
        df.attrs['version'] = '1.0'
        df.attrs['ticker'] = ticker
        df.to_parquet(cache_file)
        logger.info(f"{ticker} guardado en cachÃ© (fuente: {source})")
        return df

    # ------------------------------------------------------------------
    # Descarga de todos los tickers y combinaciÃ³n
    # ------------------------------------------------------------------
    def download_all(self, start_date, end_date, force=False):
        dfs = []
        for ticker in self.tickers:
            try:
                df = self.download_ticker(ticker, start_date, end_date, force)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error crÃ­tico con {ticker}: {e}")
                continue
        if not dfs:
            raise ValueError("No se pudo descargar ningÃºn ticker. Revisa la conexiÃ³n o los datos.")
        from functools import reduce
        df_combined = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
        df_combined.sort_index(inplace=True)
        for ticker in self.ohlc_tickers:
            try:
                logger.info(f"Descargando OHLC de {ticker} desde Stooq...")
                df_ohlc = self._fetch_stooq_ohlc(ticker, start_date, end_date)
                if df_ohlc is not None:
                    df_combined = df_combined.join(df_ohlc, how='left')
                    logger.info(f"OHLC de {ticker} aÃ±adido.")
                else:
                    logger.warning(f"No se pudo obtener OHLC de {ticker}")
            except Exception as e:
                logger.error(f"Error con OHLC de {ticker}: {e}")
        if all(col in df_combined.columns for col in ['SPY_High', 'SPY_Low', 'SPY_Close']):
            high = df_combined['SPY_High']
            low = df_combined['SPY_Low']
            close = df_combined['SPY_Close'].shift(1)
            tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
            atr20 = tr.rolling(20).mean()
            df_combined['SPY_ATR20'] = atr20
            logger.info("ATR20 de SPY calculado y aÃ±adido.")
        else:
            logger.warning("No se encontraron columnas OHLC para SPY, se crea columna SPY_ATR20 con NaN")
            df_combined['SPY_ATR20'] = np.nan
        spreads_df = self.calcular_spreads(df_combined)
        if not spreads_df.empty:
            df_combined = pd.concat([df_combined, spreads_df], axis=1)
            logger.info("Spreads calculados y aÃ±adidos.")
        else:
            logger.warning("No se pudieron calcular spreads.")
        try:
            df_treasury = self._fetch_treasury_curve(force=force)
            if not df_treasury.empty:
                df_combined = df_combined.join(df_treasury, how='left')
                logger.info("Spread de curva del Tesoro aÃ±adido.")
            else:
                logger.warning("No se obtuvo spread de la curva, se crea columna vacÃ­a.")
                df_combined['spread_10y2y'] = np.nan
        except Exception as e:
            logger.error(f"Error al aÃ±adir spread del Tesoro: {e}")
            df_combined['spread_10y2y'] = np.nan
        checksum = self._calculate_checksum(df_combined)
        df_combined.attrs['checksum'] = checksum
        df_combined.attrs['tickers'] = self.tickers
        df_combined.attrs['start_date'] = start_date.strftime('%Y-%m-%d')
        df_combined.attrs['end_date'] = end_date.strftime('%Y-%m-%d')
        df_combined.attrs['last_updated'] = datetime.now().isoformat()
        df_combined.attrs['version'] = '1.0'
        year_month = datetime.now().strftime('%Y/%m')
        out_file = self.processed_dir / year_month / f"data_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_parquet(out_file)
        logger.info(f"Datos combinados guardados en {out_file}")
        return df_combined

    # ------------------------------------------------------------------
    # Cargar los datos mÃ¡s recientes (Ãºltimo archivo procesado)
    # ------------------------------------------------------------------
    def load_latest(self):
        pattern = "data_*.parquet"
        files = list(self.processed_dir.glob(f"**/{pattern}"))
        if not files:
            raise FileNotFoundError("No hay datos procesados disponibles.")
        latest = max(files, key=lambda f: f.stat().st_mtime)
        df = pd.read_parquet(latest)
        logger.info(f"Datos cargados desde {latest}")
        return df

    # ------------------------------------------------------------------
    # Obtener spreads reales desde Yahoo Finance (bid/ask)
    # ------------------------------------------------------------------
    def obtener_spread_yahoo(self, ticker):
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            bid = info.get('bid')
            ask = info.get('ask')
            if bid and ask and ask > 0:
                spread = (ask - bid) / ask
                if spread > 0.1:
                    spread = 0.1
                return round(spread, 6)
            else:
                logger.warning(f"No se pudo obtener bid/ask para {ticker}")
                return None
        except Exception as e:
            logger.error(f"Error obteniendo spread de {ticker}: {e}")
            return None

    def calcular_spreads(self, df):
        """
        Calcula spreads (High-Low)/Close para todos los tickers con datos OHLC.
        Si no existen columnas OHLC, asigna NaN en lugar de omitir.
        """
        spreads = pd.DataFrame(index=df.index)
        
        # Primero, identificar tickers con columnas OHLC
        tickers_con_ohlc = set()
        for col in df.columns:
            if col.endswith('_High'):
                ticker = col.replace('_High', '')
                tickers_con_ohlc.add(ticker)
        
        # Para cada ticker en ohlc_tickers (configurados), asegurar columna aunque sea NaN
        all_tickers = set(self.ohlc_tickers)
        for ticker in all_tickers:
            high_col = f"{ticker}_High"
            low_col = f"{ticker}_Low"
            close_col = f"{ticker}_Close"
            
            if high_col in df.columns and low_col in df.columns and close_col in df.columns:
                # Calcular spread real
                spread = (df[high_col] - df[low_col]) / df[close_col]
                spread = spread.clip(lower=0, upper=0.1)
                spread = spread.replace([np.inf, -np.inf], np.nan)
                spreads[f"spread_{ticker}"] = spread
                logger.debug(f"Spread {ticker}: calculado")
            else:
                # Si no hay datos OHLC, asignar NaN
                spreads[f"spread_{ticker}"] = np.nan
                logger.debug(f"Spread {ticker}: sin datos OHLC, asignado NaN")
        
        return spreads

    # ------------------------------------------------------------------
    # Funciones para datos del BIS (Global Liquidity Indicators)
    # ------------------------------------------------------------------
    def get_global_credit_series(self, currency='USD', borrowers_sector='N', lenders_sector='A', borrowers_cty='4W'):
        df = self.load_bis_gli()
        if df.empty:
            return pd.Series()
        # Verificar que las columnas existan
        required_cols = ['CURR_DENOM', 'BORROWERS_SECTOR', 'LENDERS_SECTOR', 'BORROWERS_CTY', 'UNIT_MEASURE']
        if not all(col in df.columns for col in required_cols):
            logger.error("El archivo BIS GLI no tiene las columnas esperadas.")
            return pd.Series()
        mask = (
            (df['CURR_DENOM'] == currency) &
            (df['BORROWERS_SECTOR'] == borrowers_sector) &
            (df['LENDERS_SECTOR'] == lenders_sector) &
            (df['BORROWERS_CTY'] == borrowers_cty) &
            (df['UNIT_MEASURE'].isin([currency, 'USD' if currency != 'USD' else 'USD']))
        )
        serie_df = df[mask]
        if serie_df.empty:
            logger.warning(f"No se encontrÃ³ serie para {currency}, {borrowers_sector}, {lenders_sector}, {borrowers_cty}")
            return pd.Series()
        serie = serie_df.iloc[0]
        fecha_cols = [col for col in df.columns if col[0].isdigit() or col.startswith('19') or col.startswith('20')]
        valores = []
        fechas = []
        for col in fecha_cols:
            try:
                anio_q = col.split('-')
                anio = int(anio_q[0])
                trimestre = int(anio_q[1][1])
                if trimestre == 1:
                    fecha = pd.Timestamp(year=anio, month=3, day=31)
                elif trimestre == 2:
                    fecha = pd.Timestamp(year=anio, month=6, day=30)
                elif trimestre == 3:
                    fecha = pd.Timestamp(year=anio, month=9, day=30)
                else:
                    fecha = pd.Timestamp(year=anio, month=12, day=31)
                valor = serie[col]
                if pd.notna(valor):
                    valores.append(valor)
                    fechas.append(fecha)
            except:
                continue
        serie_temporal = pd.Series(valores, index=fechas).sort_index()
        return serie_temporal

    # ------------------------------------------------------------------
    # Obtener URL de descarga de iShares desde la pÃ¡gina del producto
    # ------------------------------------------------------------------
    def _get_ishares_download_url_from_page(self, ticker):
        """
        Obtiene la URL de descarga del archivo XLS (fund data) desde la pÃ¡gina del producto iShares.
        Busca el enlace con clase 'icon-xls-export' y extrae el href.
        """
        if ticker not in self.ishares_product_ids:
            logger.error(f"Ticker {ticker} no tiene product_id definido")
            return None

        product_id = self.ishares_product_ids[ticker]
        page_url = f"https://www.ishares.com/us/products/{product_id}/"

        try:
            logger.info(f"Buscando enlace de descarga en {page_url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
            }
            response = requests.get(page_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            link = soup.find('a', class_='icon-xls-export')
            if not link or not link.has_attr('href'):
                logger.error("No se encontrÃ³ el enlace de descarga (icon-xls-export) en la pÃ¡gina")
                return None

            href = link['href']
            if href.startswith('/'):
                download_url = f"https://www.ishares.com{href}"
            else:
                download_url = href
            logger.info(f"URL de descarga encontrada: {download_url}")
            return download_url
        except Exception as e:
            logger.error(f"Error al obtener URL de descarga para {ticker}: {e}")
            return None

    # ------------------------------------------------------------------
    # Descarga de datos diarios de fondo de iShares (NAV, shares, AUM)
    # ------------------------------------------------------------------
    def download_ishares_fund_data(self, ticker, force=False):
        """
        Descarga el archivo de holdings (CSV) para ETFs de iShares.
        Extrae el total de shares outstanding por fecha y lo combina con precios de Yahoo Finance.
        VersiÃ³n robusta que maneja comas en los campos.
        """
        cache_file = self.raw_dir / f"ishares_{ticker}_fund.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists() and not force:
            logger.info(f"Cargando datos de fondo de {ticker} desde cachÃ©")
            return pd.read_parquet(cache_file)

        # Obtener URL dinÃ¡mica
        url = self._get_ishares_download_url_from_page(ticker)
        if not url:
            logger.error(f"No se pudo obtener URL de descarga para {ticker}")
            return pd.DataFrame()

        try:
            logger.info(f"Descargando datos de holdings de {ticker} desde iShares")
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
            })
            resp = session.get(url, timeout=30)
            resp.raise_for_status()

            # Leer el contenido como texto
            lines = resp.text.splitlines()

            # Buscar la lÃ­nea que contiene los encabezados de la tabla
            header_line = None
            data_lines = []
            for i, line in enumerate(lines):
                if line.startswith('Ticker,') or 'Ticker' in line:
                    header_line = line
                    data_lines = lines[i+1:]
                    break

            if not header_line:
                logger.error(f"No se encontrÃ³ la lÃ­nea de encabezados en el archivo de {ticker}")
                return pd.DataFrame()

            # Procesar el archivo correctamente con pandas, especificando que las comillas dobles protegen los campos
            import csv
            csv_content = '\n'.join([header_line] + data_lines)
            csv_reader = csv.reader(StringIO(csv_content), quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
            rows = list(csv_reader)

            if not rows:
                logger.error(f"No se pudieron leer filas del CSV de {ticker}")
                return pd.DataFrame()

            headers = rows[0]
            data_rows = rows[1:]

            df = pd.DataFrame(data_rows, columns=headers)

            # Buscar columnas de fecha y cantidad
            fecha_col = next((c for c in df.columns if 'Date' in c), None)
            if not fecha_col:
                logger.error(f"No se encontrÃ³ columna de fecha. Columnas: {df.columns.tolist()}")
                return pd.DataFrame()

            quantity_col = next((c for c in df.columns if 'Quantity' in c or 'Shares' in c), None)
            if not quantity_col:
                logger.error(f"No se encontrÃ³ columna de cantidad. Columnas: {df.columns.tolist()}")
                return pd.DataFrame()

            # Convertir fecha (sin formato fijo para mayor robustez)
            df['fecha'] = pd.to_datetime(df[fecha_col], errors='coerce')

            # Limpiar cantidad: quitar comas de miles y convertir a float
            df['shares_raw'] = df[quantity_col].astype(str).str.replace(',', '', regex=False)
            df['shares'] = pd.to_numeric(df['shares_raw'], errors='coerce')

            df = df.dropna(subset=['fecha', 'shares'])

            shares_diario = df.groupby('fecha')['shares'].sum().reset_index()
            shares_diario = shares_diario.sort_values('fecha')

            shares_diario.to_parquet(cache_file)
            logger.info(f"Datos de holdings de {ticker} guardados en cachÃ© con {len(shares_diario)} fechas")
            return shares_diario

        except Exception as e:
            logger.error(f"Error descargando datos de holdings de iShares para {ticker}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Scraping de flujos de ETFs desde los artÃ­culos diarios de ETF.com
    # ------------------------------------------------------------------
    def scrape_etf_flows(self, target_date=None, tickers=None, force=False):
        """
        Descarga el artÃ­culo diario de ETF.com y extrae la tabla de flujos.
        target_date: fecha en formato 'YYYY-MM-DD' (por defecto hoy).
        tickers: lista de tickers a filtrar (por defecto todos los que encuentre).
        force: si True, fuerza la descarga aunque exista cachÃ©.
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        if tickers is None:
            tickers = ['SPY', 'EEM', 'JNK', 'LQD', 'HYG', 'IWM', 'QQQ', 'XLK', 'XLF', 'TLT', 'SHY']

        url = f"https://www.etf.com/sections/daily-etf-flows/{target_date}"

        cache_filename = f"etf_flows_{target_date}.parquet"
        cache_file = self.raw_dir / 'etf_flows' / cache_filename
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists() and not force:
            logger.info(f"Cargando ETF flows desde cachÃ©: {cache_file}")
            df = pd.read_parquet(cache_file)
            if tickers:
                df = df[df['ticker'].isin(tickers)]
            return df

        try:
            logger.info(f"Descargando artÃ­culo de flujos para {target_date}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Buscar la tabla de flujos
            table = None
            table = soup.find('table', class_=re.compile(r'flows|fund-flow', re.I))
            if not table:
                for tbl in soup.find_all('table'):
                    if tbl.find('th', string=re.compile(r'Flows', re.I)):
                        table = tbl
                        break
            if not table:
                tables = soup.find_all('table')
                max_numbers = 0
                for tbl in tables:
                    text = tbl.get_text()
                    numbers = re.findall(r'[\$\-]?\d+\.?\d*[MB]?', text)
                    if len(numbers) > max_numbers:
                        max_numbers = len(numbers)
                        table = tbl
                if max_numbers < 5:
                    table = None

            if not table:
                logger.error("No se pudo encontrar la tabla de flujos en el artÃ­culo")
                return pd.DataFrame()

            df = pd.read_html(str(table))[0]
            df.columns = [col.strip() for col in df.columns]

            # Identificar columnas clave
            ticker_col = None
            flow_col = None
            name_col = None
            for col in df.columns:
                col_lower = col.lower()
                if 'ticker' in col_lower or 'symbol' in col_lower:
                    ticker_col = col
                elif 'net' in col_lower and 'flow' in col_lower:
                    flow_col = col
                elif 'fund' in col_lower or 'name' in col_lower:
                    name_col = col

            if not ticker_col or not flow_col:
                logger.error(f"No se encontraron las columnas necesarias. Columnas: {df.columns.tolist()}")
                return pd.DataFrame()

            df = df.rename(columns={ticker_col: 'ticker', flow_col: 'flujo_neto'})
            if name_col:
                df = df.rename(columns={name_col: 'nombre'})

            # Limpiar flujo neto
            df['flujo_neto'] = df['flujo_neto'].astype(str).str.replace('$', '', regex=False)
            df['flujo_neto'] = df['flujo_neto'].str.replace('M', 'e6', regex=False)
            df['flujo_neto'] = df['flujo_neto'].str.replace('B', 'e9', regex=False)
            df['flujo_neto'] = df['flujo_neto'].str.replace(',', '', regex=False)
            df['flujo_neto'] = pd.to_numeric(df['flujo_neto'], errors='coerce')

            df['fecha'] = pd.to_datetime(target_date)

            if tickers:
                df = df[df['ticker'].isin(tickers)]

            df.to_parquet(cache_file)
            logger.info(f"ETF flows guardados en cachÃ© para {target_date}: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Error en scrape_etf_flows para {target_date}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Respaldo de flujos usando Yahoo Finance (volumen como proxy)
    # ------------------------------------------------------------------
    def _fetch_etf_flows_yahoo_fallback(self, tickers, start_date, end_date):
        """
        Obtiene datos de volumen diario de Yahoo Finance como proxy de flujos.
        Devuelve DataFrame con columnas: fecha, ticker, flujo_neto (volumen).
        """
        import time
        all_data = []
        for ticker in tickers:
            try:
                time.sleep(1)  # Evitar rate limiting
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty:
                    continue
                df = df[['Volume']].rename(columns={'Volume': 'flujo_neto'})
                df['ticker'] = ticker
                df['fecha'] = df.index
                all_data.append(df[['fecha', 'ticker', 'flujo_neto']])
            except Exception as e:
                logger.warning(f"Error descargando {ticker} de Yahoo: {e}")
                continue
        if not all_data:
            return pd.DataFrame()
        return pd.concat(all_data).sort_values('fecha')


# ----------------------------------------------------------------------
# Bloque de prueba (solo si se ejecuta directamente)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dl = DataLayer()
    end = datetime.now()
    start = end - timedelta(days=730)  # 2 anios para pruebas
    df = dl.download_all(start_date=start, end_date=end, force=False)
    print(df.head())
    print("\nColumnas disponibles:", df.columns.tolist())


