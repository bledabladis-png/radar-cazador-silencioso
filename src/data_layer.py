"""
data_layer.py - Capa de datos del Radar Macro Rotación Global
Se encarga de descargar, validar, cachear y proporcionar los datos de mercado.
"""

import os
import pandas as pd
import numpy as np
import requests
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
from tiingo import TiingoClient
import logging

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLayer:
    def __init__(self, config_path='config/config.yaml'):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Leer clave de Tiingo si existe
        tiingo_key_file = self.config['data']['tiingo_key_file']
        if os.path.exists(tiingo_key_file):
            with open(tiingo_key_file, 'r') as f:
                self.tiingo_key = f.read().strip()
        else:
            self.tiingo_key = None
            logger.warning("Archivo de clave Tiingo no encontrado. Se usará Yahoo Finance.")

        # Inicializar cliente Tiingo si hay clave
        if self.tiingo_key:
            self.tiingo_client = TiingoClient({'api_key': self.tiingo_key})
        else:
            self.tiingo_client = None

        # Crear carpetas necesarias
        self.raw_dir = Path('data/raw')
        self.processed_dir = Path('data/processed')
        self.cache_dir = Path('data/cache')
        self.treasury_dir = self.raw_dir / 'treasury'
        for d in [self.raw_dir, self.processed_dir, self.cache_dir, self.treasury_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Lista de tickers principales
        self.tickers = self.config['data']['tickers']['macro']

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
            # Eliminar zona horaria del índice
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Buscar columna de precio (Adj Close o Close)
            if 'Adj Close' in df.columns:
                series = df['Adj Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                logger.error(f"Yahoo: no se encontró columna de precio para {ticker}")
                return None
            # Si es DataFrame, extraer primera columna
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            result = pd.DataFrame({ticker: series})
            return result
        except Exception as e:
            logger.error(f"Error descargando {ticker} de Yahoo: {e}")
            return None

    def _fetch_stooq_ohlc(self, ticker, start_date, end_date):
        """Descarga OHLC desde Stooq para tickers como SPY, JNK, LQD."""
        try:
            stooq_ticker = ticker.lower().replace('^', '') + '.us'
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
            df = pd.read_csv(url)
            if df.empty:
                return None
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            df.columns = [f"{ticker}_{col}" for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error descargando {ticker} de Stooq: {e}")
            return None

    # ------------------------------------------------------------------
    # Validación y checksum
    # ------------------------------------------------------------------
    def _validate_data(self, df, ticker):
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)

        if not df.index.is_monotonic_increasing:
            logger.warning(f"{ticker}: índice no monótono. Ordenando.")
            df.sort_index(inplace=True)

        if (df[ticker] < 0).any():
            raise ValueError(f"{ticker}: precios negativos detectados.")

        nan_pct = df[ticker].isna().mean()
        if nan_pct > 0.05:
            logger.warning(f"{ticker}: {nan_pct:.2%} NaNs. Se rellenarán.")
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
                # Buscar columnas de 2 años y 10 años
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
            logger.info("Cargando curva del Tesoro desde caché.")
            return pd.read_parquet(cache_file)
        df_result = self._read_treasury_csv()
        df_result.to_parquet(cache_file)
        logger.info("Curva del Tesoro guardada en caché desde CSV locales.")
        return df_result

    # ------------------------------------------------------------------
    # Descarga de un ticker individual con caché
    # ------------------------------------------------------------------
    def download_ticker(self, ticker, start_date, end_date, force=False):
        cache_file = self.raw_dir / f"{ticker}.parquet"
        if cache_file.exists() and not force:
            logger.info(f"Cargando {ticker} desde caché.")
            df = pd.read_parquet(cache_file)
            # Asegurar índice naive
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # Verificar rango
            if df.index[0] <= pd.Timestamp(start_date) and df.index[-1] >= pd.Timestamp(end_date):
                return df

        df = self._fetch_tiingo(ticker, start_date, end_date)
        source = 'tiingo'
        if df is None:
            logger.warning(f"Tiingo falló para {ticker}, intentando Yahoo.")
            df = self._fetch_yahoo(ticker, start_date, end_date)
            source = 'yahoo'
            if df is None:
                raise Exception(f"No se pudo descargar {ticker} de ninguna fuente.")

        df = self._validate_data(df, ticker)
        checksum = self._calculate_checksum(df)
        df.attrs['checksum'] = checksum
        df.attrs['source'] = source
        df.attrs['last_updated'] = datetime.now().isoformat()
        df.to_parquet(cache_file)
        logger.info(f"{ticker} guardado en caché (fuente: {source})")
        return df

    # ------------------------------------------------------------------
    # Descarga de todos los tickers y combinación
    # ------------------------------------------------------------------
    def download_all(self, start_date, end_date, force=False):
        dfs = []
        for ticker in self.tickers:
            try:
                df = self.download_ticker(ticker, start_date, end_date, force)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error crítico con {ticker}: {e}")
                continue

        if not dfs:
            raise ValueError("No se pudo descargar ningún ticker. Revisa la conexión o los datos.")

        from functools import reduce
        df_combined = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), dfs)
        df_combined.sort_index(inplace=True)

        # Descargar OHLC adicionales desde Stooq para SPY, JNK, LQD
        ohlc_tickers = ['SPY', 'JNK', 'LQD']
        for ticker in ohlc_tickers:
            try:
                logger.info(f"Descargando OHLC de {ticker} desde Stooq...")
                df_ohlc = self._fetch_stooq_ohlc(ticker, start_date, end_date)
                if df_ohlc is not None:
                    df_combined = df_combined.join(df_ohlc, how='left')
                    logger.info(f"OHLC de {ticker} añadido.")
                else:
                    logger.warning(f"No se pudo obtener OHLC de {ticker}")
            except Exception as e:
                logger.error(f"Error con OHLC de {ticker}: {e}")

        # Añadir curva del Tesoro
        try:
            df_treasury = self._fetch_treasury_curve(force=force)
            if not df_treasury.empty:
                df_combined = df_combined.join(df_treasury, how='left')
                logger.info("Spread de curva del Tesoro añadido.")
            else:
                logger.warning("No se obtuvo spread de la curva, se crea columna vacía.")
                df_combined['spread_10y2y'] = np.nan
        except Exception as e:
            logger.error(f"Error al añadir spread del Tesoro: {e}")
            df_combined['spread_10y2y'] = np.nan

        # Calcular checksum total y guardar
        checksum = self._calculate_checksum(df_combined)
        df_combined.attrs['checksum'] = checksum
        df_combined.attrs['tickers'] = self.tickers
        df_combined.attrs['start_date'] = start_date.strftime('%Y-%m-%d')
        df_combined.attrs['end_date'] = end_date.strftime('%Y-%m-%d')
        df_combined.attrs['last_updated'] = datetime.now().isoformat()

        # Guardar en carpeta procesada por año/mes
        year_month = datetime.now().strftime('%Y/%m')
        out_file = self.processed_dir / year_month / f"data_{datetime.now().strftime('%Y%m%d')}.parquet"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_parquet(out_file)
        logger.info(f"Datos combinados guardados en {out_file}")

        return df_combined

    # ------------------------------------------------------------------
    # Cargar los datos más recientes (último archivo procesado)
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


# ----------------------------------------------------------------------
# Bloque de prueba (solo si se ejecuta directamente)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dl = DataLayer()
    end = datetime.now()
    start = end - timedelta(days=730)  # 2 años para pruebas
    df = dl.download_all(start_date=start, end_date=end, force=False)
    print(df.head())
    print("\nColumnas disponibles:", df.columns.tolist())