import numpy as np
import pandas as pd


def append_dedup(hist_df, new_df, subset):
    """Concatena dos DataFrames, normaliza fecha a YYYY-MM-DD y elimina duplicados por subset."""
    combined = pd.concat([hist_df, new_df], ignore_index=True)
    if 'date' in combined.columns:
        combined['date'] = pd.to_datetime(combined['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    if combined.empty:
        return combined
    return combined.drop_duplicates(subset=subset, keep='last')

def robust_zscore(series, window=60):
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    z = (series - median) / (1.4826 * mad + 1e-9)
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
