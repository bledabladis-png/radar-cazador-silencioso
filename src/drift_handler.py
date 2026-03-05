import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging
import os

logger = logging.getLogger(__name__)

def calcular_z_score(serie, valor):
    if len(serie) < 2:
        return 0.0
    mean = serie.mean()
    std = serie.std()
    if std == 0:
        return 0.0
    return (valor - mean) / std

def ks_test_reciente(serie, ultimos_n=60):
    if len(serie) < ultimos_n * 2:
        return 1.0
    reciente = serie.iloc[-ultimos_n:]
    historico = serie.iloc[:-ultimos_n]
    if len(historico) == 0:
        return 1.0
    return ks_2samp(reciente, historico)[1]

def handle_drift(df_historial, ultimo_score, config, contexto):
    cfg = config.get('drift', {})
    z_thresh = cfg.get('z_threshold', 3.0)
    ks_thresh = cfg.get('ks_threshold', 0.01)
    ks_window = cfg.get('ks_window', 60)

    z = calcular_z_score(df_historial['score_global'], ultimo_score)
    ks_p = ks_test_reciente(df_historial['score_global'], ks_window)

    logger.info(f"Drift check: z={z:.2f}, ks_p={ks_p:.4f}")

    if abs(z) > z_thresh or ks_p < ks_thresh:
        contexto['operations_freeze'] = True
        logger.critical(f"DRIFT detectado: z={z:.2f}, ks_p={ks_p:.4f}. Freeze activado.")
        with open('freeze_drift.txt', 'w') as f:
            f.write(f"Drift activado el {pd.Timestamp.now()} por z={z:.2f} o ks={ks_p:.4f}")
    else:
        contexto['operations_freeze'] = False
        if os.path.exists('freeze_drift.txt'):
            os.remove('freeze_drift.txt')

    return contexto