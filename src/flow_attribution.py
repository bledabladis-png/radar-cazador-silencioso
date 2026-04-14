"""
flow_attribution.py – Flow Attribution Engine
Descompone el flujo total (ret × dollar_volume) en tres componentes:
- Pasivo (estable, baja volatilidad)
- Mecánico (spikes, reversión rápida)
- Convicción (persistente, direccional)
No genera señales de trading; solo información.
"""

import pandas as pd
import numpy as np

class FlowAttributionEngine:
    def __init__(self, window=20):
        self.window = window

    def compute_features(self, df, price_col, volume_col):
        """Calcula retornos y dólar volumen."""
        price = df[price_col]
        volume = df[volume_col]
        ret = price.pct_change()
        dollar_vol = price * volume
        return ret, volume, dollar_vol

    def total_flow(self, ret, dollar_vol):
        """Flujo total observado (proxy)."""
        return ret * dollar_vol

    def passive_flow_proxy(self, flow):
        """Estima flujo pasivo (estable, bajo ruido)."""
        mean = flow.rolling(self.window).mean()
        std = flow.rolling(self.window).std()
        stability = 1 / (std + 1e-9)
        return stability * np.sign(mean)

    def mechanical_flow_proxy(self, flow):
        """Estima flujo mecánico (spikes, reversiones rápidas)."""
        z = (flow - flow.rolling(self.window).mean()) / (flow.rolling(self.window).std() + 1e-9)
        return np.abs(z)

    def conviction_flow(self, flow):
        """Estima flujo de convicción (persistente y direccional)."""
        sign_consistency = flow.rolling(self.window).apply(lambda x: np.mean(np.sign(x)))
        persistence = flow.rolling(self.window).mean()
        return sign_consistency * persistence

    def classify_flow_state(self, df, price_col, volume_col):
        """
        Retorna el estado de flujo para la última fila del DataFrame.
        """
        ret, vol, dollar_vol = self.compute_features(df, price_col, volume_col)
        flow = self.total_flow(ret, dollar_vol)

        passive = self.passive_flow_proxy(flow).iloc[-1]
        mechanical = self.mechanical_flow_proxy(flow).iloc[-1]
        conviction = self.conviction_flow(flow).iloc[-1]

        # Reglas de clasificación (umbrales empíricos)
        if conviction > 0.5:
            return "CONVICTION_ACCUMULATION"
        if mechanical > 0.8:
            return "HEDGING_NOISE"
        if passive > 0:
            return "PASSIVE_RISK_ON"
        return "NEUTRAL"