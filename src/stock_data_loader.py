import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config.settings import CACHE_HOURS, CACHE_VALIDATE_TRADING_DATE
from data.providers.euronext_provider import EuronextProvider
from data.providers.xetra_provider import XetraProvider
from data.providers.bme_provider import BMEProvider
from src.market_calendar import last_expected_market_date, is_market_day
import os
import time


YAHOO_TICKER_MAP = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "MOGA": "MOG-A",
    "MOG A": "MOG-A",
    "GEF B": "GEF-B",
    "CRD A": "CRD-A",
    "BH A": "BH-A",
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
    except Exception as e:
        print(f"  [WARN] get_usa_tickers: etf_holdings.csv: {e}")
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
    except Exception as e:
        print(f"  [WARN] get_stock_list: etf_holdings.csv: {e}")

    # 2) Índices americanos/europeos (data/index_holdings.csv)
    try:
        df_idx = pd.read_csv('data/index_holdings.csv')
        if 'weight' in df_idx.columns:
            df_idx = df_idx.sort_values(['etf', 'weight'], ascending=[True, False])
        for etf, group in df_idx.groupby('etf'):
            tickers.extend([normalize_yahoo_ticker(t) for t in group['ticker'].head(20).tolist()])
    except Exception as e:
        print(f"  [WARN] get_stock_list: index_holdings.csv: {e}")

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

def _fill_holes_respecting_sessions(df, reference_date):
    """Rellena NaN solo en fechas que NO son sesion NYSE.

    Cualquier NaN en una fecha que SI es sesion NYSE se preserva: representa
    un hueco real del proveedor y no debe imputarse silenciosamente.

    Args:
        df: DataFrame con columnas MultiIndex (Close/High/Low/Volume, ticker).
        reference_date: fecha de referencia (datetime o Timestamp).

    Returns:
        (df_filled, diagnostics_dict) con:
            n_nan_pre_fill: total NaN en columnas Close
            n_nan_preserved: NaN preservados (eran sesion NYSE)
            affected_tickers: tickers con NaN preservado
    """
    expected_session = last_expected_market_date(reference_date)
    diag = {
        'n_nan_pre_fill': 0,
        'n_nan_preserved': 0,
        'affected_tickers': [],
        'expected_session': expected_session,
    }
    if df is None or df.empty:
        return df, diag

    close_cols = [c for c in df.columns if len(c) == 2 and c[0] == 'Close']
    if not close_cols:
        return df, diag

    df = df.copy()
    for col in close_cols:
        ticker = col[1]
        series = df[col]
        nan_mask = series.isna()
        if not nan_mask.any():
            continue
        nan_dates = series.index[nan_mask]
        diag['n_nan_pre_fill'] += int(nan_mask.sum())

        session_nan_dates = []
        for d in nan_dates:
            d_norm = pd.Timestamp(d).normalize()
            if is_market_day(d_norm.date()):
                diag['n_nan_preserved'] += 1
                session_nan_dates.append(d)

        if session_nan_dates:
            diag['affected_tickers'].append(ticker)

        filled = series.ffill(limit=3)
        for d in session_nan_dates:
            filled.loc[d] = float('nan')
        df[col] = filled

    return df, diag


def _log_yahoo_raw_diagnostics(df_raw, reference_date, batch_label):
    """Loguea estado crudo del batch ANTES de cualquier imputacion.

    Solo observabilidad. No decide ni modifica.
    """
    expected_session = last_expected_market_date(reference_date)
    if df_raw is None or df_raw.empty:
        print(f"[YAHOO_RAW] batch={batch_label} empty=True")
        return

    raw_last = df_raw.index[-1]
    raw_last_norm = pd.Timestamp(raw_last).normalize().date()
    expected_present = (raw_last_norm == expected_session)

    close_cols = [c for c in df_raw.columns if len(c) == 2 and c[0] == 'Close']
    total = len(close_cols)
    if total == 0:
        nan_count = 0
        affected = []
    else:
        last_row = df_raw[close_cols].iloc[-1]
        nan_mask = last_row.isna()
        nan_count = int(nan_mask.sum())
        affected = [col[1] for col, is_nan in zip(close_cols, nan_mask) if is_nan][:20]

    pct = 100.0 * nan_count / total if total > 0 else 0.0

    print(f"[YAHOO_RAW] batch={batch_label}")
    print(f"  expected_session={expected_session}")
    print(f"  raw_last_date={raw_last_norm}")
    print(f"  expected_session_present={expected_present}")
    print(f"  close_nan={nan_count}/{total} ({pct:.1f}%)")
    print("  before_ffill=True")
    print(f"  affected_tickers={affected}")


def _classify_ticker(ticker, df, expected_session):
    """Clasifica un ticker segun disponibilidad y frescura.

    Devuelve: (status, reason) con status en
    {'OK', 'PARTIAL', 'STALE', 'FAILED', 'DATA_ISSUE'} y reason str o None.
    """
    if df is None or df.empty:
        return 'FAILED', None
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
        return 'FAILED', None

    series = df[close_col]
    if series.dropna().empty:
        return 'FAILED', None

    # B1 (2026-09-12): chequeos de integridad de sesion esperada.
    observed_last = pd.Timestamp(series.index[-1]).normalize().date()
    expected_norm = pd.Timestamp(expected_session).normalize().date()

    if expected_norm > observed_last:
        return 'DATA_ISSUE', 'EXPECTED_SESSION_ABSENT'

    if expected_norm == observed_last and pd.isna(series.iloc[-1]):
        return 'DATA_ISSUE', 'MISSING_CLOSE_EXPECTED_SESSION'

    # Comprobacion original: NaN residual (hueco >3 dias o proveedor stale).
    if pd.isna(series.iloc[-1]):
        return 'FAILED', 'STALE_LAST_VALUE'

    last_date = series.dropna().index[-1]
    days_since = (pd.Timestamp.now() - last_date).days
    if days_since <= 5:
        return 'OK', None
    elif days_since <= 15:
        return 'PARTIAL', None
    else:
        return 'STALE', None

# Nota: sin @retry global. El bucle por lotes ya gestiona fallos:
# los tickers fallidos se reintentan individualmente tras el bucle principal.
def download_stock_prices(reference_date=None):
    # B1 (2026-09-12): reference_date se normaliza UNA vez al inicio.
    if reference_date is None:
        reference_date = datetime.now()
    _expected_session = last_expected_market_date(reference_date)

    cache_path = 'data/stock_prices.csv'
    parquet_path = 'data/stock_prices.parquet'
    # D3 Fase 2: elegir cache disponible (parquet o csv, el mas reciente)
    _candidates = []
    if os.path.exists(parquet_path):
        _candidates.append((parquet_path, 'parquet'))
    if os.path.exists(cache_path):
        _candidates.append((cache_path, 'csv'))
    if _candidates:
        _path, _fmt = max(_candidates, key=lambda t: os.path.getmtime(t[0]))
        mtime = datetime.fromtimestamp(os.path.getmtime(_path))
        if datetime.now() - mtime < timedelta(hours=CACHE_HOURS):
            _df = None
            if _fmt == 'parquet':
                try:
                    _df = pd.read_parquet(_path)
                except Exception as e:
                    print(f'  [WARN] Error leyendo Parquet: {e}')
            else:
                _df = pd.read_csv(_path, header=[0,1], index_col=0, parse_dates=True)
            # CACHE_VALIDATE_TRADING_DATE: verificar que el cache cubre
            # el ultimo dia de mercado esperado. Si no, forzar descarga.
            if _df is not None and CACHE_VALIDATE_TRADING_DATE and len(_df) > 0:
                _last_exp = last_expected_market_date()
                _df_last = _df.index[-1].date() if hasattr(_df.index[-1], 'date') else _df.index[-1]
                if _df_last < _last_exp:
                    print(f'  [CACHE] datos hasta {_df_last}, esperado >= {_last_exp}. Forzando descarga.')
                else:
                    return _df
            elif _df is not None:
                return _df

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
    classification = {'OK': [], 'PARTIAL': [], 'STALE': [], 'FAILED': [], 'DATA_ISSUE': []}
    classification_reasons = {}
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
                # B1: observabilidad del estado crudo ANTES de imputar.
                _log_yahoo_raw_diagnostics(data_batch, reference_date,
                                           f"batch_{i//batch_size + 1}")
                # B1: relleno condicional. NaN en sesion NYSE se preserva.
                data_batch, _fill_diag = _fill_holes_respecting_sessions(
                    data_batch, reference_date)
                all_data.append(data_batch)
                # Clasificar cada ticker del lote
                for ticker in batch:
                    status, reason = _classify_ticker(ticker, data_batch,
                                                     _expected_session)
                    classification[status].append(ticker)
                    if status == 'FAILED':
                        failed_tickers.append(ticker)
                    if status == 'DATA_ISSUE' and reason:
                        classification_reasons[ticker] = reason
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

                # B1: observabilidad + relleno condicional (retry individual).
                _log_yahoo_raw_diagnostics(data_single, reference_date,
                                           f"retry_{ticker}")
                data_single, _fill_diag = _fill_holes_respecting_sessions(
                    data_single, reference_date)

                # Solo añadir si el ticker tiene datos válidos reales
                close_key = ('Close', ticker)
                if close_key not in data_single.columns:
                    continue
                if not data_single[close_key].notna().any():
                    continue

                all_data.append(data_single)
                status, reason = _classify_ticker(ticker, data_single,
                                                 _expected_session)
                if status != 'FAILED':
                    if ticker in classification['FAILED']:
                        classification['FAILED'].remove(ticker)
                    classification[status].append(ticker)
                    if ticker in failed_tickers:
                        failed_tickers.remove(ticker)
                    if status == 'DATA_ISSUE' and reason:
                        classification_reasons[ticker] = reason
            except Exception as e:
                print(f'  [WARN] stock_data_loader: {ticker}: {e}')

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
    if classification['DATA_ISSUE']:
        print(f"DATA_ISSUE ({len(classification['DATA_ISSUE'])}): {classification['DATA_ISSUE']}")
        for t in classification['DATA_ISSUE'][:20]:
            print(f"  {t}: {classification_reasons.get(t, 'unknown')}")

    if not all_data:
        return None

    data = pd.concat(all_data, axis=1)
    if not isinstance(data.columns, pd.MultiIndex):
        data.columns = pd.MultiIndex.from_tuples(data.columns)

    # B1: sanity check final. NO aplica calendario NYSE: este merge puede
    # contener series con otros calendarios (Euronext/Xetra/BME). La
    # proteccion real de B1 se aplica en el pipeline Yahoo USA (batch/retry).
    # Idempotente sobre los datos ya procesados.
    data = data.ffill(limit=3)

    # Deduplicar columnas (defensivo: con Europa primero ya no hay solapamiento
    # Yahoo/Euronext, pero lo dejamos como red de seguridad).
    if data.columns.duplicated().any():
        n_dup = int(data.columns.duplicated().sum())
        print(f"  AVISO: {n_dup} columnas duplicadas detectadas, deduplicando (keep=last)")
        data = data.loc[:, ~data.columns.duplicated(keep='last')]

    # D3 Fase 2c: solo Parquet (CSV ya no se escribe)
    try:
        data.to_parquet(parquet_path)
    except Exception as e:
        print(f'  [WARN] Error escribiendo Parquet: {e}')
    return data
