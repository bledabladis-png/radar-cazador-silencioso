import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from data.providers.euronext_provider import EuronextProvider
from data.providers.xetra_provider import XetraProvider
import os
import time

CACHE_HOURS = 23

YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
}

def normalize_yahoo_ticker(t):
    """Convierte tickers problemáticos al formato que acepta Yahoo Finance."""
    return YAHOO_TICKER_MAP.get(t, t)

def get_usa_tickers():
    """Obtiene solo tickers de sectores USA (data/etf_holdings.csv)."""
    tickers = []
    try:
        import pandas as pd
        df_sect = pd.read_csv('data/etf_holdings.csv')
        if 'weight' in df_sect.columns:
            df_sect = df_sect.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_sect.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass
    # Eliminar duplicados
    seen = set()
    result = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def get_stock_list():
    """Obtiene lista de tickers para descargar: top 20 de cada sector y de cada índice."""
    tickers = []

    # 1) Sectores USA (data/etf_holdings.csv)
    try:
        df_sect = pd.read_csv('data/etf_holdings.csv')
        if 'weight' in df_sect.columns:
            df_sect = df_sect.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_sect.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # 2) Índices americanos/europeos (data/index_holdings.csv)
    try:
        df_idx = pd.read_csv('data/index_holdings.csv')
        if 'weight' in df_idx.columns:
            df_idx = df_idx.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_idx.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception:
        pass

    # Eliminar duplicados preservando orden
    seen = set()
    result = []
    for t in tickers:
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result

def _get_yf_session():
    try:
        from curl_cffi import requests as curl_requests
        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None

def _classify_ticker(ticker, df):
    """Clasifica un ticker según la disponibilidad y frescura de sus datos.

    Devuelve: 'OK', 'PARTIAL', 'STALE', 'FAILED'.
    """
    if df is None or df.empty:
        return 'FAILED'
    # Buscar columna Close para el ticker
    close_col = None
    if isinstance(df.columns, pd.MultiIndex):
        # Intentar ('Close', ticker)
        if ('Close', ticker) in df.columns:
            close_col = ('Close', ticker)
        else:
            # Quizás solo hay una columna sin MultiIndex (caso individual)
            pass
    else:
        if 'Close' in df.columns:
            close_col = 'Close'
    if close_col is None:
        return 'FAILED'

    series = df[close_col]
    if series.dropna().empty:
        return 'FAILED'

    # Si la ÚLTIMA FILA del DataFrame tiene NaN, considerar FAILED
    # (independientemente de que haya datos previos válidos).
    # Esto es clave para detectar tickers europeos con sesión no cerrada.
    if pd.isna(series.iloc[-1]):
        return 'FAILED'

    last_date = series.dropna().index[-1]
    days_since = (pd.Timestamp.now() - last_date).days
    if days_since <= 5:
        return 'OK'
    elif days_since <= 15:
        return 'PARTIAL'
    else:
        return 'STALE'

# Nota: sin @retry global. El bucle por lotes ya gestiona fallos:
# los tickers fallidos se reintentan individualmente tras el bucle principal.
def download_stock_prices():
    cache_path = 'data/stock_prices.csv'
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            return pd.read_csv(cache_path, header=[0,1], index_col=0, parse_dates=True)

    tickers = get_stock_list()
    if not tickers:
        return None

    print(f"Descargando precios para {len(tickers)} tickers...")

    # Una única sesión para todas las descargas
    session = _get_yf_session()
    all_data = []
    classification = {'OK': [], 'PARTIAL': [], 'STALE': [], 'FAILED': []}
    failed_tickers = []

    batch_size = 10  # lote mayor para eficiencia
    delay = 2

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Lote {i//batch_size + 1}: {batch}")
        try:
            if session:
                data_batch = yf.download(batch, period='5y', auto_adjust=True, session=session)
            else:
                data_batch = yf.download(batch, period='5y', auto_adjust=True)
            if not data_batch.empty:
                all_data.append(data_batch)
                # Clasificar cada ticker del lote
                for ticker in batch:
                    status = _classify_ticker(ticker, data_batch)
                    classification[status].append(ticker)
                    if status == 'FAILED':
                        failed_tickers.append(ticker)
            else:
                # lote vacío: todos fallidos
                classification['FAILED'].extend(batch)
                failed_tickers.extend(batch)
        except Exception as e:
            print(f"Error en lote {batch}: {e}")
            classification['FAILED'].extend(batch)
            failed_tickers.extend(batch)

        if i + batch_size < len(tickers):
            time.sleep(delay)

    # Reintentar fallidos individualmente (misma sesión)
    if failed_tickers:
        print(f"Reintentando {len(failed_tickers)} tickers fallidos individualmente...")
        for ticker in failed_tickers[:]:  # copia para iterar y modificar listas
            try:
                # Usar lista con un solo elemento para forzar MultiIndex
                if session:
                    data_single = yf.download([ticker], period='5y', auto_adjust=True, session=session)
                else:
                    data_single = yf.download([ticker], period='5y', auto_adjust=True)

                if data_single.empty:
                    continue

                # Asegurar MultiIndex con ticker en nivel 1
                if not isinstance(data_single.columns, pd.MultiIndex):
                    data_single.columns = pd.MultiIndex.from_product([data_single.columns, [ticker]])

                # Solo añadir si el ticker tiene datos válidos reales
                close_key = ('Close', ticker)
                if close_key not in data_single.columns:
                    continue
                if not data_single[close_key].notna().any():
                    continue

                all_data.append(data_single)
                status = _classify_ticker(ticker, data_single)
                if status != 'FAILED':
                    if ticker in classification['FAILED']:
                        classification['FAILED'].remove(ticker)
                    classification[status].append(ticker)
                    if ticker in failed_tickers:
                        failed_tickers.remove(ticker)
            except Exception:
                pass

    # Cascada europea: se ejecuta SIEMPRE para los tickers cubiertos por Euronext/Xetra.
    # Los datos de estos proveedores son más fiables y frescos que Yahoo para esos tickers,
    # así que reemplazamos completamente sus columnas de Yahoo.
    euronext = EuronextProvider()
    xetra = XetraProvider()

    eu_candidates = [t for t in tickers if euronext.supports(t)]
    xe_candidates = [t for t in tickers if xetra.supports(t)]
    cascade_frames = []

    if eu_candidates:
        print(f"Intentando Euronext para {len(eu_candidates)} tickers")
        try:
            eu_data = euronext.get_prices(eu_candidates, nb_session=300, use_cache=True)
            if eu_data is not None and not eu_data.empty:
                cascade_frames.append(eu_data)
        except Exception as e:
            print(f"  Euronext falló: {e}")

    if xe_candidates:
        print(f"Intentando Xetra para {len(xe_candidates)} tickers")
        try:
            xe_data = xetra.get_prices(xe_candidates, use_cache=True)
            if xe_data is not None and not xe_data.empty:
                cascade_frames.append(xe_data)
        except Exception as e:
            print(f"  Xetra falló: {e}")

    if cascade_frames:
        cascade_data = pd.concat(cascade_frames, axis=1)
        if not isinstance(cascade_data.columns, pd.MultiIndex):
            cascade_data.columns = pd.MultiIndex.from_tuples(cascade_data.columns)
        if cascade_data.columns.duplicated().any():
            cascade_data = cascade_data.loc[:, ~cascade_data.columns.duplicated(keep='last')]

        # Quitar de all_data las columnas cuyo ticker esté en cascade_data
        cascade_ticker_set = set(c[1] for c in cascade_data.columns)
        new_all_data = []
        for frame in all_data:
            if isinstance(frame.columns, pd.MultiIndex):
                keep_cols = [c for c in frame.columns if c[1] not in cascade_ticker_set]
            else:
                keep_cols = list(frame.columns)
            if keep_cols:
                new_all_data.append(frame[keep_cols])
        all_data = new_all_data
        all_data.append(cascade_data)

        # Actualizar clasificación
        for ticker in cascade_ticker_set:
            for cat in ['FAILED', 'PARTIAL', 'STALE']:
                if ticker in classification[cat]:
                    classification[cat].remove(ticker)
            if ticker not in classification['OK']:
                classification['OK'].append(ticker)
            if ticker in failed_tickers:
                failed_tickers.remove(ticker)

        print(f"  Cascada cubrió {len(cascade_ticker_set)} tickers europeos")

    # Reportar sin cobertura
    if failed_tickers:
        print(f"  Sin cobertura europea (quedan FAILED): {len(failed_tickers)} tickers")

    # Imprimir resumen de clasificación
    print("=== Clasificación de tickers ===")
    print(f"OK ({len(classification['OK'])}): {classification['OK']}")
    print(f"PARTIAL ({len(classification['PARTIAL'])}): {classification['PARTIAL']}")
    print(f"STALE ({len(classification['STALE'])}): {classification['STALE']}")
    print(f"FAILED ({len(classification['FAILED'])}): {classification['FAILED']}")

    if not all_data:
        return None

    data = pd.concat(all_data, axis=1)
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    # Deduplicar columnas.
    # keep='last': Euronext se añade después de Yahoo para los mismos tickers,
    # por lo que queremos conservar la versión más reciente (Euronext con datos frescos).
    if data.columns.duplicated().any():
        n_dup = int(data.columns.duplicated().sum())
        print(f"  AVISO: {n_dup} columnas duplicadas detectadas, deduplicando (keep=last)")
        data = data.loc[:, ~data.columns.duplicated(keep='last')]

    data.to_csv(cache_path)
    return data
