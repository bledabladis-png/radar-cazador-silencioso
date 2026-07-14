import numpy as np

# ---------- RATIOS INSTITUCIONALES ----------

def institutional_hedge_ratio(index_pcr, equity_pcr):
    if equity_pcr is None or equity_pcr <= 0:
        return None
    return index_pcr / equity_pcr

def index_volume_share(index_volume, total_volume):
    if total_volume is None or total_volume <= 0:
        return None
    return index_volume / total_volume

# ---------- DESCOMPOSICIÓN PUT/CALL ----------

def put_share(put_volume, total_volume):
    if total_volume is None or total_volume <= 0:
        return None
    return put_volume / total_volume

def call_share(call_volume, total_volume):
    if total_volume is None or total_volume <= 0:
        return None
    return call_volume / total_volume

def volume_put_call_ratio(put_volume, call_volume):
    if call_volume is None or call_volume <= 0:
        return None
    return put_volume / call_volume

def oi_put_call_ratio(put_oi, call_oi):
    if call_oi is None or call_oi <= 0:
        return None
    return put_oi / call_oi

# ---------- CAMBIOS TEMPORALES ----------

def relative_volume(today_volume, avg20_volume):
    if avg20_volume is None or avg20_volume <= 0:
        return None
    return today_volume / avg20_volume

def oi_change(today_oi, yesterday_oi):
    if yesterday_oi is None or yesterday_oi <= 0:
        return None
    return (today_oi - yesterday_oi) / yesterday_oi

# ---------- CLASIFICACIÓN ----------

def classify_pcr(z):
    if z is None or np.isnan(z):
        return "Sin historial suficiente"
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

def classify_ihr(ihr):
    if ihr is None or np.isnan(ihr):
        return "N/A"
    if ihr >= 2.5:
        return "Cobertura institucional extrema"
    elif ihr >= 1.8:
        return "Cobertura institucional alta"
    elif ihr >= 1.2:
        return "Equilibrado"
    elif ihr >= 0.8:
        return "Especulación alta"
    else:
        return "Especulación extrema"
