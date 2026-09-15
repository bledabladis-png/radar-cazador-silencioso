import numpy as np
import pandas as pd


def append_dedup(hist_df, new_df, subset):
    """Concatena dos DataFrames, normaliza fecha a YYYY-MM-DD y elimina duplicados por subset.

    FU-009 (2026-09-13): si hist_df o new_df estan vacios, evitar pd.concat
    (genera FutureWarning en pandas 2.x y rompera en pandas 3.0).
    """
    h_empty = hist_df is None or len(hist_df) == 0
    n_empty = new_df is None or len(new_df) == 0
    if h_empty and n_empty:
        return pd.DataFrame()
    if h_empty:
        combined = new_df.copy()
    elif n_empty:
        combined = hist_df.copy()
    else:
        combined = pd.concat([hist_df, new_df], ignore_index=True)
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    if combined.empty:
        return combined
    return combined.drop_duplicates(subset=subset, keep='last')

def _observation_date_from_df(df, col=None):
    """Extrae la ultima fecha de observacion valida de un DataFrame.

    C4-code (2026-09-12): helper compartido por writers de historicos.
    NO valida calendario. NO decide si publicar. NUNCA usa now() como
    fallback.

    Contrato:
        df None/vacio         -> None
        col especificada      -> KeyError si no existe; max(df[col]) tras
                                 convertir a datetime; None si todo NaT
        sin col               -> ultima posicion no-NaT del DatetimeIndex;
                                 None si no hay ninguna
        sin fecha valida      -> None

    Args:
        df: DataFrame con DatetimeIndex o columna de fecha.
        col: nombre de la columna de fecha. Si None, usa el index.

    Returns:
        pd.Timestamp normalizado o None.
    """
    if df is None:
        return None
    try:
        if len(df) == 0:
            return None
    except Exception:
        return None

    if col is not None:
        if col not in df.columns:
            raise KeyError(
                f"_observation_date_from_df: columna '{col}' no encontrada."
            )
        s = pd.to_datetime(df[col], errors='coerce').dropna()
        if s.empty:
            return None
        return pd.Timestamp(s.max()).normalize()

    # Modo index: aceptar DatetimeIndex o indices no-numericos convertibles.
    # Un RangeIndex/Int64Index se convierte silenciosamente a 1970-01-01 por
    # pd.to_datetime; eso es una fecha inventada, no observacion. Rechazar.
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        if pd.api.types.is_numeric_dtype(df.index):
            return None
        try:
            idx = pd.to_datetime(df.index, errors='coerce')
        except Exception:
            return None
        idx = pd.DatetimeIndex(idx)
    idx = idx.dropna()
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx[-1]).normalize()


def robust_zscore(series, window=60, min_periods=None):
    """Z-score robusto (mediana/MAD) tolerante a huecos cortos.

    Args:
        series: pd.Series con posibles NaN (tipicamente por festivos).
        window: ventana rolling (default 60).
        min_periods: minimo de observaciones validas requeridas.
                     Si None -> max(window//3, 10).

    Nota (fix 2026-09-11): antes usaba min_periods=window (default pandas),
    lo que devolvia NaN si habia UN SOLO NaN en la ventana. Como ^VIX tiene
    ~91 NaN por festivos USA, robust_zscore devolvia solo 18/2606 filas
    validas -> corrompia todos los scores del sistema. Se anade ffill(limit=3)
    para replicar el ultimo dia operativo en festivos puntuales, y min_periods
    relajado para permitir el calculo con huecos residuales."""
    if min_periods is None:
        min_periods = max(window // 3, 10)
    s = series.ffill(limit=3)
    median = s.rolling(window, min_periods=min_periods).median()
    mad = (s - median).abs().rolling(window, min_periods=min_periods).median()
    z = (s - median) / (1.4826 * mad + 1e-9)
    return z.clip(-5, 5)

def rolling_percentile(series, window=120):
    return series.rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=True
    )

def tanh_normalize(series):
    z = robust_zscore(series)
    return np.tanh(z)

def sigmoid(x):
    return (np.tanh(x) + 1) / 2

def ema_smooth(series, span=10):
    return series.ewm(span=span, min_periods=5).mean()

def winsorize(series, limits=(0.01, 0.99)):
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower, upper)

def standardize_series(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def get_col(df, ticker, field='Close'):
    if isinstance(df.columns, pd.MultiIndex):
        # Recorrer todas las columnas y buscar la que coincida
        for col in df.columns:
            if len(col) != 2:
                continue
            if str(col[0]).lower() == field.lower() and str(col[1]).lower() == ticker.lower():
                series = df[col]
                return series
        raise KeyError(f'Columna MultiIndex ({field}, {ticker}) no encontrada')
    else:
        # Columnas planas: intentar TICKER_Field
        col = f'{ticker}_{field}'
        if col in df.columns:
            series = df[col]
            return series
        # Si no existe, intentar directamente el nombre del campo (para DataFrames de una sola acción)
        if field in df.columns:
            series = df[field]
            return series
        raise KeyError(f'Columna {col} o {field} no encontrada')

def trim_to_last_valid_date(df, min_coverage=0.5):
    """
    Recorta un DataFrame MultiIndex a la última fila donde al menos
    min_coverage de las columnas 'Close' tienen dato no nulo.
    Evita operar con filas festivas/vacías.
    """
    if df is None or df.empty:
        return df
    close_cols = [c for c in df.columns if c[0] == 'Close']
    if not close_cols:
        return df
    coverage = df[close_cols].notna().sum(axis=1) / len(close_cols)
    valid_rows = coverage[coverage >= min_coverage]
    if valid_rows.empty:
        return df
    last_valid_date = valid_rows.index[-1]
    return df.loc[:last_valid_date]

def trim_to_last_valid_date_for_tickers(df, tickers, min_coverage=0.8):
    """Recorta a la última fecha donde los tickers indicados tengan cobertura >= min_coverage."""
    if df is None or df.empty:
        return df
    close_cols = [c for c in df.columns if c[0] == 'Close' and c[1] in tickers]
    if not close_cols:
        return df
    coverage = df[close_cols].notna().sum(axis=1) / len(close_cols)
    valid_rows = coverage[coverage >= min_coverage]
    if valid_rows.empty:
        return df
    last_valid_date = valid_rows.index[-1]
    return df.loc[:last_valid_date]

def safe_mean(values):
    """Media de valores no nulos; 0.0 si no hay ninguno."""
    s = pd.Series(list(values) if values is not None else [])
    s = s.dropna()
    return float(s.mean()) if not s.empty else 0.0

def safe_std(values):
    """Desviación estándar de valores no nulos; 0.0 si no hay al menos 2."""
    s = pd.Series(list(values) if values is not None else [])
    s = s.dropna()
    return float(s.std()) if len(s) > 1 else 0.0

def clean_oil_prices(df):
    try:
        close_cl = get_col(df, 'CL=F', 'Close')
        if (close_cl <= 0).any():
            min_positive = close_cl[close_cl > 0].min()
            df[('Close', 'CL=F')] = close_cl.clip(lower=min_positive)
    except KeyError:
        pass
    return df
# ----------------------------------------------------------------------
# Cross-Module Conflict Detector (Fase 3)
# ----------------------------------------------------------------------
def detect_cross_module_conflict(macro_regime, financial_regime, volatility_regime, liquidity_regime, mte_scenario=None):
    """
    Detecta contradicciones entre los modulos de diagnostico.
    Clasifica en: CONSENSUS, MIXED, CONFLICT, DIVERGENCE.
    Separa estres financiero de presion inflacionaria.
    """
    # Mapeo de estados a tipo de sesgo
    financial_stress_states = ['CRISIS', 'HIGH_STRESS', 'ESTRECHA', 'STRESS']
    inflation_stress_states = ['STAGFLATION', 'INFLATION SHOCK']
    expansion_states = ['EXPANSION', 'RECOVERY', 'LATE EXPANSION', 'GOLDILOCKS', 'ABUNDANTE', 'LOW']
    
    def bias_financial(state):
        if state is None:
            return 0
        s = str(state).upper()
        if any(x in s for x in financial_stress_states):
            return -1
        if any(x in s for x in expansion_states):
            return 1
        return 0
    
    def bias_inflation(state):
        if state is None:
            return 0
        s = str(state).upper()
        if any(x in s for x in inflation_stress_states):
            return -1
        return 0
    
    modules = {
        'macro': macro_regime,
        'financial': financial_regime,
        'volatility': volatility_regime,
        'liquidity': liquidity_regime
    }
    if mte_scenario:
        modules['mte'] = mte_scenario
    
    biases_fin = {k: bias_financial(v) for k, v in modules.items()}
    biases_inf = {k: bias_inflation(v) for k, v in modules.items()}
    
    fin_stress_count = sum(1 for b in biases_fin.values() if b == -1)
    fin_exp_count = sum(1 for b in biases_fin.values() if b == 1)
    inf_stress_count = sum(1 for b in biases_inf.values() if b == -1)
    total = len(modules)
    
    # Determinar nivel de conflicto
    if fin_stress_count == total:
        level = 'CONSENSUS'
        msg = 'Todos los modulos coinciden en entorno de estres financiero.'
    elif fin_exp_count == total:
        level = 'CONSENSUS'
        msg = 'Todos los modulos coinciden en entorno expansivo.'
    elif fin_stress_count >= 2 and fin_exp_count >= 2:
        level = 'CONFLICT'
        msg = f'Contradiccion significativa: {fin_stress_count} modulos indican estres, {fin_exp_count} indican expansion.'
    elif fin_stress_count >= 2 and inf_stress_count >= 1:
        level = 'DIVERGENCE'
        msg = f'Estres financiero concentrado en {fin_stress_count} modulo(s). Presion inflacionaria detectada en {inf_stress_count} modulo(s). los módulos no presentan clasificación uniforme.'
    elif fin_stress_count >= 1 and inf_stress_count >= 1:
        level = 'MIXED'
        msg = 'Senhales mixtas: estres financiero localizado con presion inflacionaria.'
    else:
        level = 'MIXED'
        msg = 'Sin direccion clara entre los modulos.'
    
    # Detalle por bloques
    blocks = []
    if fin_stress_count > 0:
        blocks.append(f'Financial Stress: {fin_stress_count}/{total}')
    if inf_stress_count > 0:
        blocks.append(f'Inflation Pressure: {inf_stress_count}/{total}')
    if not blocks:
        blocks.append('Sin bloque de estres dominante')
    
    return {
        'conflict_level': level,
        'message': msg,
        'blocks': ' | '.join(blocks),
        'details': {name: {'state': modules[name], 'bias_financial': biases_fin[name], 'bias_inflation': biases_inf[name]} for name in modules}
    }


def confidence_from_range(scores_df, divisor=2.0):
    """Confianza = 1 - (max - min) / divisor.

    Mide DISAGREEMENT EXTREMO entre componentes. Politica conservadora:
    un solo outlier puede penalizar fuerte la confianza global.

    Args:
        scores_df: pd.DataFrame (columnas=componentes) o pd.Series.
        divisor: 2.0 por defecto (componentes normalizados a [-1, +1]).

    Returns:
        pd.Series en [0, 1] si scores_df es DataFrame.
        float si scores_df es Series.

    Reglas:
        - NaN por fila se excluyen del calculo.
        - <2 componentes validos -> 0.5 (evidencia insuficiente).
        - range = max - min sobre componentes validos.
        - conf = clip(1 - range/divisor, 0, 1).

    Nota (auditoria 2026-09-11):
        El rango es una estadistica de orden. Su distribucion PUEDE
        depender del numero de componentes aunque la formula no incluya
        N explicitamente. Se eligio por baja sensibilidad empirica en
        pruebas 3 vs 4 componentes y por su interpretabilidad.
    """
    if isinstance(scores_df, pd.Series):
        return _confidence_range_row(scores_df, divisor)

    valid = scores_df.dropna(axis=1, how='all')
    if valid.shape[1] < 2:
        return pd.Series(0.5, index=scores_df.index)
    rng = valid.max(axis=1) - valid.min(axis=1)
    return (1.0 - rng / divisor).clip(0, 1).fillna(0.5)


def _confidence_range_row(row, divisor=2.0):
    """Version escalar de confidence_from_range (uso interno)."""
    valid = row.dropna()
    if len(valid) < 2:
        return 0.5
    rng = valid.max() - valid.min()
    return max(0.0, min(1.0, 1.0 - rng / divisor))


# ============================================================
# FU-002 (2026-09-15) - Manifest de artefacto (parquet + JSON)
# ============================================================

def _try_cleanup(*paths):
    """Borra ficheros temporales silenciosamente (uso interno FU-002)."""
    import os as _os
    for p in paths:
        try:
            if _os.path.exists(p):
                _os.remove(p)
        except Exception:
            pass


def write_artifact_with_manifest(df, parquet_path, source,
                                  reference_date, run_id,
                                  schema_version=1):
    """Escribe parquet + manifest de forma atomica (FU-002, 2026-09-15).

    Pasos:
      1. Escribe parquet a <parquet>.tmp.<run_id>
      2. Calcula sha256 del temporal
      3. Calcula metadatos (content + quality) incl. expected_session
      4. Escribe manifest a <parquet>.manifest.json.tmp.<run_id>
      5. os.replace(tmp_parquet, parquet_path)
      6. os.replace(tmp_manifest, manifest_path)

    Args:
        df: DataFrame a escribir.
        parquet_path: path destino del parquet (y base del manifest).
        source: 'yahoo' | 'yahoo_european_cascade' | otro identificador.
        reference_date: datetime ya resuelto en run.py:main().
        run_id: 'YYYYMMDD_HHMMSS' generado en run.py:main().
        schema_version: version del contrato del manifest.

    Returns:
        dict con el manifest escrito, o {} si fallo.
    """
    import json
    import os as _os
    import hashlib
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    from config.settings import MANIFEST_DUP_THRESHOLD
    from src.market_calendar import last_expected_market_date

    if df is None or df.empty:
        print(f"  [WARN] manifest: df vacio para {parquet_path}. Se omite.")
        return {}

    path = _Path(parquet_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = str(path) + '.manifest.json'
    tmp_parquet = f"{parquet_path}.tmp.{run_id}"
    tmp_manifest = f"{manifest_path}.tmp.{run_id}"

    try:
        df.to_parquet(tmp_parquet)

        h = hashlib.sha256()
        with open(tmp_parquet, 'rb') as fh:
            for chunk in iter(lambda: fh.read(65536), b''):
                h.update(chunk)
        sha256 = h.hexdigest()
        n_bytes = _os.path.getsize(tmp_parquet)

        rows = int(len(df))
        cols = int(df.shape[1])
        if isinstance(df.columns, pd.MultiIndex):
            n_tickers = int(len(df.columns.get_level_values(1).unique()))
        else:
            n_tickers = cols

        date_min = date_max = None
        last_date = None
        if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
            dates = df.index.dropna()
            if len(dates) > 0:
                date_min = dates.min().strftime('%Y-%m-%d')
                date_max = dates.max().strftime('%Y-%m-%d')
                last_date = dates.max().date()

        pct_dup_last = 0.0
        close_nan_last = 0
        if isinstance(df.columns, pd.MultiIndex) and len(df) >= 2:
            close_cols = [c for c in df.columns if c[0] == 'Close']
            if close_cols:
                last_row = df[close_cols].iloc[-1]
                prev_row = df[close_cols].iloc[-2]
                both = last_row.notna() & prev_row.notna()
                if int(both.sum()) > 0:
                    dup = int((last_row[both] == prev_row[both]).sum())
                    pct_dup_last = float(dup) / float(both.sum())
                close_nan_last = int(last_row.isna().sum())

        expected_dt = last_expected_market_date(reference_date)
        expected_session = expected_dt.strftime('%Y-%m-%d')
        last_date_is_expected = (last_date == expected_dt) if last_date else False

        if last_date is None:
            print(f"  [WARN] manifest: {parquet_path} sin last_date. Se omite.")
            _try_cleanup(tmp_parquet, tmp_manifest)
            return {}
        # FU-002-evo2 (2026-09-15): la decision de estado NO depende de
        # que la ultima fila coincida con la sesion esperada. Se evalua
        # siempre el contenido de la ultima fila (pct_dup y close_nan).
        # Bug previo: si last_date != expected_session (p.ej. fila parcial
        # por calendarios heterogeneos), la regla caia en el else y
        # declaraba VALID ignorando NaN reales. Detectado en run
        # 2026-09-15 14:20 con close_nan=33, status=VALID.
        if pct_dup_last > MANIFEST_DUP_THRESHOLD:
            # Corrupcion confirmada: precios duplicados por ffill u otra causa.
            status = 'INVALID'
        elif close_nan_last > 0:
            # Ausencia legitima de observacion (B1 preservo NaN).
            status = 'VALID_WITH_MISSING'
        else:
            status = 'VALID'

        # FU-002-evo2: metadata de cobertura de la ultima fila.
        n_tickers_expected_last = int(n_tickers)
        n_tickers_with_close_last = int(n_tickers - close_nan_last)
        n_tickers_missing_close_last = int(close_nan_last)
        coverage_pct_last = (
            n_tickers_with_close_last / n_tickers_expected_last
            if n_tickers_expected_last > 0 else 0.0
        )
        last_row_is_partial = (close_nan_last > 0)

        manifest = {
            'schema_version': schema_version,
            'artifact': {
                'path': str(parquet_path),
                'sha256': sha256,
                'bytes': n_bytes,
                'written_at': _dt.now().isoformat(),
            },
            'producer': {
                'module': 'src.utils',
                'function': 'write_artifact_with_manifest',
                'run_id': run_id,
                'source': source,
            },
            'content': {
                'rows': rows,
                'cols': cols,
                'n_tickers': n_tickers,
                'date_min': date_min,
                'date_max': date_max,
            },
            'quality': {
                'last_date': last_date.strftime('%Y-%m-%d') if last_date else None,
                'expected_session': expected_session,
                'last_date_is_expected_session': last_date_is_expected,
                'pct_dup_last': round(pct_dup_last, 6),
                'close_nan_last': close_nan_last,
                # FU-002-evo2 (2026-09-15): metadata de cobertura ultima fila.
                'n_tickers_expected_last': n_tickers_expected_last,
                'n_tickers_with_close_last': n_tickers_with_close_last,
                'n_tickers_missing_close_last': n_tickers_missing_close_last,
                'coverage_pct_last': round(coverage_pct_last, 6),
                'last_row_is_partial': last_row_is_partial,
                'status': status,
            },
        }

        with open(tmp_manifest, 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

        _os.replace(tmp_parquet, str(path))
        _os.replace(tmp_manifest, manifest_path)

        # FU-002-bug (2026-09-15): observabilidad ampliada.
        # Detectar filas parciales (multi-calendario) sin tener que
        # descargar el parquet. Anade close_nan, cobertura y flag de
        # sesion esperada al log del workflow.
        _n_close_observed = int(len(close_cols) - close_nan_last) if close_cols else 0
        _coverage_pct = (_n_close_observed / n_tickers) if n_tickers > 0 else 0.0
        print(f"  [MANIFEST] {parquet_path}:")
        print(f"    status={status}")
        print(f"    rows={rows}, tickers={n_tickers}")
        print(f"    pct_dup={pct_dup_last:.3f}, close_nan={close_nan_last}, n_close_observed={_n_close_observed}")
        print(f"    coverage_pct={_coverage_pct:.4f}")
        print(f"    last_date={date_max}, expected={expected_session}")
        print(f"    last_date_is_expected_session={last_date_is_expected}")
        return manifest

    except Exception as e:
        print(f"  [WARN] manifest {parquet_path}: {e}")
        _try_cleanup(tmp_parquet, tmp_manifest)
        return {}
