import numpy as np
from config.settings import PCR_THRESHOLDS, IHR_THRESHOLDS

# ---------- RATIOS INSTITUCIONALES ----------

def institutional_hedge_ratio(index_pcr, equity_pcr):
    if equity_pcr is None or not np.isfinite(equity_pcr) or equity_pcr <= 0:
        return None
    if not np.isfinite(index_pcr):
        return None
    return index_pcr / equity_pcr

def index_volume_share(index_volume, total_volume):
    if total_volume is None or not np.isfinite(total_volume) or total_volume <= 0:
        return None
    if not np.isfinite(index_volume):
        return None
    return index_volume / total_volume

# ---------- DESCOMPOSICIÓN PUT/CALL ----------

def put_share(put_volume, total_volume):
    if total_volume is None or not np.isfinite(total_volume) or total_volume <= 0:
        return None
    if not np.isfinite(put_volume):
        return None
    return put_volume / total_volume

def call_share(call_volume, total_volume):
    if total_volume is None or total_volume <= 0:
        return None
    return call_volume / total_volume

def volume_put_call_ratio(put_volume, call_volume):
    if call_volume is None or not np.isfinite(call_volume) or call_volume <= 0:
        return None
    if not np.isfinite(put_volume):
        return None
    return put_volume / call_volume

def oi_put_call_ratio(put_oi, call_oi):
    if call_oi is None or not np.isfinite(call_oi) or call_oi <= 0:
        return None
    if not np.isfinite(put_oi):
        return None
    return put_oi / call_oi

# ---------- CAMBIOS TEMPORALES ----------

def relative_volume(today_volume, avg20_volume):
    if avg20_volume is None or not np.isfinite(avg20_volume) or avg20_volume <= 0:
        return None
    if not np.isfinite(today_volume):
        return None
    return today_volume / avg20_volume

def oi_change(today_oi, yesterday_oi):
    if yesterday_oi is None or not np.isfinite(yesterday_oi) or yesterday_oi <= 0:
        return None
    if not np.isfinite(today_oi):
        return None
    return (today_oi - yesterday_oi) / yesterday_oi

# ---------- CLASIFICACIÓN ----------

def classify_pcr(z):
    if z is None or not np.isfinite(z):
        return "Sin historial suficiente"
    if z >= PCR_THRESHOLDS['panico']:
        return "Pánico"
    elif z >= PCR_THRESHOLDS['miedo']:
        return "Miedo"
    elif z > PCR_THRESHOLDS['neutral']:
        return "Neutral"
    elif z > PCR_THRESHOLDS['optimismo']:
        return "Optimismo"
    else:
        return "Euforia"

def classify_ihr(ihr):
    if ihr is None or np.isnan(ihr):
        return "N/A"
    if ihr >= IHR_THRESHOLDS['cobertura_extrema']:
        return "Cobertura institucional extrema"
    elif ihr >= IHR_THRESHOLDS['cobertura_alta']:
        return "Cobertura institucional alta"
    elif ihr >= IHR_THRESHOLDS['equilibrado']:
        return "Equilibrado"
    elif ihr >= IHR_THRESHOLDS['especulacion_alta']:
        return "Especulación alta"
    else:
        return "Especulación extrema"
