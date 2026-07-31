import pandas as pd
import numpy as np
from data.providers.router import DataRouter
from src.utils import robust_zscore, tanh_normalize

def compute_liquidity_score():
    router = DataRouter()
    fed_data = router.get_fed_data()
    if fed_data is None or fed_data.empty:
        return None, None, None

    # Rellenar hacia adelante para evitar que NaN en la ultima fila anule las senales
    fed_data = fed_data.ffill()

    signals = {}

    if 'fed_balance' in fed_data.columns:
        z = robust_zscore(fed_data['fed_balance'], window=60)
        signals['fed_balance'] = tanh_normalize(z).iloc[-1]

    if 'reverse_repo' in fed_data.columns:
        z = robust_zscore(fed_data['reverse_repo'], window=60)
        val = -tanh_normalize(z).iloc[-1]
        if pd.notna(val):
            signals['reverse_repo'] = val

    if 'sofr' in fed_data.columns:
        z = robust_zscore(fed_data['sofr'], window=60)
        val = -tanh_normalize(z).iloc[-1]
        if pd.notna(val):
            signals['sofr'] = val

    if 'fed_funds' in fed_data.columns:
        z = robust_zscore(fed_data['fed_funds'], window=60)
        val = -tanh_normalize(z).iloc[-1]
        if pd.notna(val):
            signals['fed_funds'] = val

    # Eliminar posibles NaN que haya en las senales
    signals = {k: v for k, v in signals.items() if pd.notna(v)}

    if not signals:
        return None, None, None

    weights = {'fed_balance': 0.35, 'reverse_repo': 0.25, 'sofr': 0.20, 'fed_funds': 0.20}
    available = [k for k in weights if k in signals]
    if not available:
        return None, None, None

    w_sum = sum(weights[k] for k in available)
    score = sum(signals[k] * weights[k] / w_sum for k in available)

    if score > 0.3:
        regime = 'ABUNDANTE'
    elif score > 0:
        regime = 'NEUTRAL'
    elif score > -0.3:
        regime = 'ESTRECHA'
    else:
        regime = 'CRISIS'

    sig_vals = [signals[k] for k in available]
    if len(sig_vals) > 1:
        confidence = 1 - np.std(sig_vals) / 2
    elif len(sig_vals) == 1:
        confidence = 0.5  # una sola senal -> confianza neutra
    else:
        confidence = 0.0

    # Guardar score para calculo de delta en la siguiente ejecucion
    previous_score = None
    try:
        import json, os
        delta_file = 'outputs/liquidity_state.json'
        if os.path.exists(delta_file):
            with open(delta_file, 'r') as f:
                prev = json.load(f)
                previous_score = prev.get('score', None)
        with open(delta_file, 'w') as f:
            json.dump({'score': float(score), 'date': str(fed_data.index[-1].date())}, f)
    except:
        pass

    return pd.Series(score, index=[fed_data.index[-1]]), regime, confidence, previous_score
