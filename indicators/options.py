import numpy as np
import pandas as pd
from datetime import datetime
from data.providers.cboe import CboeProvider

def robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return 0.0
    return (series.iloc[-1] - median) / (1.4826 * mad)

def classify_pcr(z):
    if z >= 2.0:
        return "Pánico"
    elif z >= 1.0:
        return "Miedo"
    elif z > -1.0:
        return "Neutral"
    elif z > -2.0:
        return "Optimismo"
    else:
        return "Euforia"

def compute_pcr_signals():
    provider = CboeProvider()
    if not provider.is_available():
        return None

    # Obtener datos del CBOE
    data = provider.get_options_data()
    if not data or 'total_pcr' not in data:
        return None

    # Cargar historial desde CSV local (si existe)
    try:
        hist = pd.read_csv('outputs/pcr_history.csv', parse_dates=['date'], index_col='date')
    except:
        hist = pd.DataFrame(columns=['total_pcr'])

    # Añadir dato de hoy
    today = pd.Timestamp(data['date'])
    if today not in hist.index:
        new_row = pd.DataFrame({'total_pcr': [data['total_pcr']]}, index=[today])
        hist = pd.concat([hist, new_row])
        hist.sort_index(inplace=True)
        hist.to_csv('outputs/pcr_history.csv')

    # Pipeline
    pcr_series = hist['total_pcr']
    if len(pcr_series) < 5:
        return {
            'status': 'LOW_HISTORY',
            'total_pcr': data['total_pcr'],
            'index_pcr': data.get('index_pcr', np.nan),
            'equity_pcr': data.get('equity_pcr', np.nan),
            'etp_pcr': data.get('etp_pcr', np.nan),
            'spx_pcr': data.get('spx_pcr', np.nan),
            'vix_pcr': data.get('vix_pcr', np.nan),
            'last_date': data['date'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    # Suavizado EWMA(5)
    pcr_ewm = pcr_series.ewm(span=5).mean()

    # Robust Z-Score (ventana 252 días o todo el historial si es menor)
    window = min(252, len(pcr_ewm))
    if window >= 20:
        z_series = pcr_ewm.rolling(window, min_periods=20).apply(lambda x: robust_zscore(pd.Series(x)), raw=False)
        z = z_series.iloc[-1]
        momentum = z_series.ewm(span=5).mean().iloc[-1]
        percentile = (pcr_ewm.iloc[-window:] < pcr_ewm.iloc[-1]).mean() * 100
        state = classify_pcr(z)
        score = np.tanh(z / 2)
    else:
        z = np.nan
        momentum = np.nan
        percentile = np.nan
        state = "Sin historial suficiente"
        score = np.nan

    return {
        'status': 'OK',
        'total_pcr': data['total_pcr'],
        'pcr_ewm': pcr_ewm.iloc[-1],
        'z_score': z,
        'momentum': momentum,
        'percentile': percentile,
        'state': state,
        'score': score,
        'index_pcr': data.get('index_pcr', np.nan),
        'equity_pcr': data.get('equity_pcr', np.nan),
        'etp_pcr': data.get('etp_pcr', np.nan),
        'spx_pcr': data.get('spx_pcr', np.nan),
        'vix_pcr': data.get('vix_pcr', np.nan),
        'last_date': data['date'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
