import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config.settings import CACHE_HOURS
from data.providers.euronext_provider import EuronextProvider
from data.providers.xetra_provider import XetraProvider
from data.providers.bme_provider import BMEProvider
import os
import time


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

    # Si la ULTIMA FILA del DataFrame tiene NaN, considerar FAILED.
    # Nota: los datos han sido previamente ffill(limit=3), asi que
    # un NaN aqui indica un hueco real >3 dias (suspension, delisting, etc.),
    # no un simple festivo. Esto ya no descarta tickers por festivos.
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

    all_tickers = get_stock_list()
    if not all_tickers:
        return None

    # === Europa primero ===
    # Los tickers cubiertos por Euronext, Xetra o BME se descargan SIEMPRE
    # desde fuentes oficiales europeas. Yahoo NO los toca -> menos carga
    # sobre Yahoo, mejor trazabilidad y datos mas frescos.
    euronext = EuronextProvider()
    xetra = XetraProvider()
    bme = BMEProvider()
    european_tickers = [t for t in all_tickers
                        if euronext.supports(t) or xetra.supports(t) or bme.supports(t)]
    european_set = set(european_tickers)
    tickers = [t for t in all_tickers if t not in european_set]

    print(f"Europa primero: {len(european_tickers)} tickers europeos (Euronext+Xetra+BME)")
    print(f"Yahoo: {len(tickers)} tickers no-europeos")
    print(f"Descargando precios para {len(tickers)} tickers via Yahoo...")

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
                # Rellenar huecos de hasta 3 dias (festivos) con ultimo valor real.
                # No inventa datos: replica el ultimo dia operado.
                data_batch = data_batch.ffill(limit=3)
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

                # Rellenar huecos de hasta 3 dias (festivos) antes de clasificar.
                data_single = data_single.ffill(limit=3)

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

    # Cascada europea: descarga los europeos desde fuentes oficiales.
    # Yahoo ya NO los ha descargado (Europa primero), asi que no hay
    # solapamiento ni sustitucion: los europeos vienen exclusivamente
    # de Euronext/Xetra. Si un proveedor falla, sus tickers quedan sin
    # datos ese dia (sin fallback a Yahoo).
    eu_candidates = [t for t in european_tickers if euronext.supports(t)]
    xe_candidates = [t for t in european_tickers if xetra.supports(t)]
    bm_candidates = [t for t in european_tickers if bme.supports(t)]
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

    if bm_candidates:
        print(f"Intentando BME para {len(bm_candidates)} tickers")
        try:
            bm_data = bme.get_prices(bm_candidates, use_cache=True)
            if bm_data is not None and not bm_data.empty:
                cascade_frames.append(bm_data)
        except Exception as e:
            print(f"  BME falló: {e}")

    cascade_covered_set = set()
    if cascade_frames:
        cascade_data = pd.concat(cascade_frames, axis=1)
        if not isinstance(cascade_data.columns, pd.MultiIndex):
            cascade_data.columns = pd.MultiIndex.from_tuples(cascade_data.columns)
        if cascade_data.columns.duplicated().any():
            cascade_data = cascade_data.loc[:, ~cascade_data.columns.duplicated(keep='last')]

        # Con Europa primero, Yahoo ya no descargo estos tickers.
        # Anadimos directamente sin necesidad de quitar nada de all_data.
        cascade_covered_set = set(c[1] for c in cascade_data.columns)
        all_data.append(cascade_data)

        # Actualizar clasificacion
        for ticker in cascade_covered_set:
            for cat in ['FAILED', 'PARTIAL', 'STALE']:
                if ticker in classification[cat]:
                    classification[cat].remove(ticker)
            if ticker not in classification['OK']:
                classification['OK'].append(ticker)
            if ticker in failed_tickers:
                failed_tickers.remove(ticker)

        print(f"  Cascada cubrio {len(cascade_covered_set)} tickers europeos")

    # Reportar europeos SIN cobertura (sin fallback a Yahoo)
    european_missing = [t for t in european_tickers if t not in cascade_covered_set]
    if european_missing:
        print(f"  SIN COBERTURA EUROPEA: {len(european_missing)} tickers -> {european_missing}")

    # Reportar fallos Yahoo residuales
    if failed_tickers:
        print(f"  Yahoo sin cobertura: {len(failed_tickers)} tickers")

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

    # Relleno global defensivo de huecos <=3 dias. Idempotente: si ya
    # se aplico en batch/retry, no tiene efecto adicional. Pero garantiza
    # que un ticker con ultimo NaN (festivo) no se considere FAILED.
    data = data.ffill(limit=3)

    # Deduplicar columnas (defensivo: con Europa primero ya no hay solapamiento
    # Yahoo/Euronext, pero lo dejamos como red de seguridad).
    if data.columns.duplicated().any():
        n_dup = int(data.columns.duplicated().sum())
        print(f"  AVISO: {n_dup} columnas duplicadas detectadas, deduplicando (keep=last)")
        data = data.loc[:, ~data.columns.duplicated(keep='last')]

    data.to_csv(cache_path)
    return data
