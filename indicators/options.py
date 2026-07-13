import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize as scipy_winsorize
from datetime import datetime, timedelta
from data.providers.polygon import PolygonProvider
from src.utils import robust_zscore, tanh_normalize

def compute_pcr_signals():
    polygon = PolygonProvider()
    options_data = polygon.get_options_data()

    if options_data is None or options_data.empty:
        return None

    # --- 1. CONTROL DE CALIDAD DE DATOS ---
    status = 'OK'
    issues = []

    # Verificar que existen datos validos
    if options_data.empty:
        return {'status': 'DATA ISSUE', 'issues': ['Datos vacios']}

    # Verificar frescura del dato
    last_date = options_data.index[-1]
    days_since = (datetime.now() - last_date).days
    freshness_ok = days_since <= 3
    
    if not freshness_ok:
        status = 'STALE DATA'
        issues.append(f'Ultimo dato: {last_date.date()} ({days_since} dias de retraso)')

    # Verificar historial suficiente
    if len(options_data) < 60:
        if status == 'OK':
            status = 'LOW HISTORY'
        issues.append(f'Solo {len(options_data)} observaciones (minimo 252)')

    # Verificar NaNs
    if options_data['volume'].isna().mean() > 0.10:
        if status == 'OK':
            status = 'DATA ISSUE'
        issues.append('NaNs > 10%')

    # --- 2. CALCULO DE METRICAS ---
    volume_series = options_data['volume']
    close_series = options_data['close']
    
    # Z-Score del volumen
    volume_z = robust_zscore(volume_series, window=60)
    current_z = volume_z.iloc[-1]

    # Senhal de actividad: volumen alto = mercado activo
    sentiment_raw = current_z / 3.0
    sentiment_series = volume_z / 3.0
    sentiment_ewma = sentiment_series.ewm(span=3, min_periods=1).mean()
    sentiment = sentiment_ewma.iloc[-1]

    # --- 3. PERCENTILES HISTORICOS ---
    vol_percentile_3y = np.nan
    vol_percentile_10y = np.nan
    
    if len(volume_series) >= 756:
        vol_percentile_3y = (volume_series.iloc[-756:] < volume_series.iloc[-1]).mean()
    
    if len(volume_series) >= 2520:
        vol_percentile_10y = (volume_series < volume_series.iloc[-1]).mean()

    # --- 4. RESULTADO ---
    return {
        'status': status,
        'issues': issues,
        'pcr_total': close_series.iloc[-1],
        'z_score': current_z,
        'percentile_3y': vol_percentile_3y,
        'percentile_10y': vol_percentile_10y,
        'pcr_equity': np.nan,
        'pcr_index': np.nan,
        'sentiment': sentiment,
        'divergence_flag': 'No divergence',
        'extreme_flag': 'No extreme' if abs(current_z) < 2 else 'High Activity',
        'lectura_contrarian': 'Debil (solo se considera fuerte en extremos)',
        'last_date': last_date,
        'days_since': days_since,
        'coverage': 1.0,
        'available_series': ['volume'],
        'required_series': ['volume'],
        'vix_correlation': None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
