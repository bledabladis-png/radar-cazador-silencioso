"""
flow_attribution.py – Flow Attribution Engine (versión final)
- Rolling robust z-score (sin look-ahead)
- Rolling orthogonalization (correlación dinámica)
- Rolling percentiles para clasificación estable
- Centrado de percentiles a [-1,1]
No genera señales de trading; solo información.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------
# Utilidades rolling
# ------------------------------------------------------------

def rolling_robust_zscore(series, window=60):
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    return (series - median) / (1.4826 * mad + 1e-9)

def rolling_orthogonalize(base, target, window=60):
    """Elimina la correlación rolling entre target y base."""
    corr = base.rolling(window).corr(target)
    return target - corr * base

def rolling_percentile(series, window=120):
    """Percentil rolling (0-1) basado en ventana fija."""
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

def dynamic_threshold(series, window=120, q=0.7):
    """Umbral rolling (percentil q)."""
    return series.rolling(window).quantile(q)

def center_percentile(series):
    """Centra el percentil en 0 (rango [-1,1])."""
    return (series - 0.5) * 2

# ------------------------------------------------------------
# Clase principal
# ------------------------------------------------------------

class FlowAttributionEngine:
    def __init__(self, window=20):
        self.window = window

    def compute(self, ret, dollar_vol):
        """
        Calcula las tres métricas ortogonales (persistencia, intensidad, irregularidad)
        usando rolling robust z-score y ortogonalización dinámica.
        Retorna DataFrame con columnas: flow, persistence_orth, intensity_orth, irregularity_orth,
        y flow_type (clasificación).
        """
        df = pd.DataFrame({"ret": ret, "dv": dollar_vol})
        df["flow"] = df["ret"] * df["dv"]
        w = self.window

        # Métricas base (correlacionadas)
        df["sign"] = np.sign(df["flow"])
        df["persistence"] = df["sign"].rolling(w).mean()

        rolling_mean = df["flow"].rolling(w).mean()
        rolling_std = df["flow"].rolling(w).std()
        df["intensity"] = rolling_mean / (rolling_std + 1e-9)

        residual = df["flow"] - rolling_mean
        df["irregularity"] = (
            residual.abs().rolling(w).mean() /
            (df["flow"].abs().rolling(w).mean() + 1e-9)
        )

        # Normalización rolling robusta (sin look-ahead)
        for col in ["persistence", "intensity", "irregularity"]:
            df[col] = rolling_robust_zscore(df[col], window=60)

        # Ortogonalización rolling (Gram-Schmidt dinámico)
        p = df["persistence"]
        i = df["intensity"]
        ir = df["irregularity"]

        i_orth = rolling_orthogonalize(p, i, window=60)
        ir_adj = rolling_orthogonalize(p, ir, window=60)
        ir_orth = rolling_orthogonalize(i_orth, ir_adj, window=60)

        df["persistence_orth"] = p
        df["intensity_orth"] = i_orth
        df["irregularity_orth"] = ir_orth

        # Centrar percentiles (para que la señal combinada tenga media cero)
        for col in ["persistence_orth", "intensity_orth", "irregularity_orth"]:
            df[col] = center_percentile(df[col])

        # Clasificación usando umbrales rolling (estable)
        p70_persist = dynamic_threshold(df["persistence_orth"], window=120, q=0.7)
        p70_intensity = dynamic_threshold(df["intensity_orth"], window=120, q=0.7)
        p70_irregular = dynamic_threshold(df["irregularity_orth"], window=120, q=0.7)

        last_idx = df.index[-1]
        last = df.loc[last_idx]

        if (last["persistence_orth"] > p70_persist.loc[last_idx] and
            last["intensity_orth"] > p70_intensity.loc[last_idx]):
            flow_type = "CONVICTION"
        elif last["irregularity_orth"] > p70_irregular.loc[last_idx]:
            flow_type = "MECHANICAL"
        elif last["intensity_orth"] > p70_intensity.loc[last_idx]:
            flow_type = "PASSIVE"
        else:
            flow_type = "NEUTRAL"

        df["flow_type"] = flow_type
        return df

    def classify_last(self, df):
        return df["flow_type"].iloc[-1]