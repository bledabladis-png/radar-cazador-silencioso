"""
flow_attribution.py – Flow Attribution Engine (versión ortogonal)
Descompone el flujo total en tres dimensiones independientes:
- Persistencia (convicción real)
- Intensidad (impacto)
- Irregularidad (AP / hedging / ruido)
No genera señales de trading; solo información.
"""

import pandas as pd
import numpy as np

class FlowAttributionEngine:
    def __init__(self, window=20):
        self.window = window

    def compute(self, ret, dollar_vol):
        """
        Calcula las tres métricas ortogonales y clasifica el tipo de flujo.
        Retorna un DataFrame con columnas: flow, persistence, intensity, irregularity, flow_type.
        """
        df = pd.DataFrame({
            "ret": ret,
            "dv": dollar_vol
        })
        df["flow"] = df["ret"] * df["dv"]
        w = self.window

        # 1. Persistencia (convicción real) – estabilidad del signo
        df["sign"] = np.sign(df["flow"])
        df["persistence"] = df["sign"].rolling(w).mean()

        # 2. Intensidad (impacto real) – magnitud relativa
        rolling_mean = df["flow"].rolling(w).mean()
        rolling_std = df["flow"].rolling(w).std()
        df["intensity"] = rolling_mean / (rolling_std + 1e-9)

        # 3. Irregularidad (AP / hedging) – residuo respecto a la media móvil
        residual = df["flow"] - rolling_mean
        df["irregularity"] = (
            residual.abs().rolling(w).mean() /
            (df["flow"].abs().rolling(w).mean() + 1e-9)
        )

        # Normalización robusta (rango percentil)
        for col in ["persistence", "intensity", "irregularity"]:
            df[col] = df[col].rank(pct=True)

        # Clasificación sin umbrales fijos (usando percentiles)
        # En lugar de umbrales fijos, usamos la propia distribución para decidir
        # Se aplica sobre la última fila (la más reciente)
        # Para simplificar, devolvemos las tres series; la clasificación se hará en run.py
        # o se puede hacer aquí usando percentiles adaptativos.
        # Dejamos la clasificación para la capa superior.
        return df

    def classify_last(self, df):
        """
        Clasifica el tipo de flujo para la última fila usando percentiles.
        """
        last = df.iloc[-1]
        # Umbrales adaptativos (percentil 70 de la serie completa)
        p70_persist = df["persistence"].quantile(0.7)
        p70_intensity = df["intensity"].quantile(0.7)
        p70_irregular = df["irregularity"].quantile(0.7)

        if last["persistence"] > p70_persist and last["intensity"] > p70_intensity:
            return "CONVICTION"
        elif last["irregularity"] > p70_irregular:
            return "MECHANICAL"
        elif last["intensity"] > p70_intensity:
            return "PASSIVE"
        else:
            return "NEUTRAL"