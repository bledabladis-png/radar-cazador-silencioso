# -*- coding: utf-8 -*-
"""
persistence.py -- Persistence Engine v1.0
Calcula la persistencia de señales en ventanas semanales.
"""
import pandas as pd
import numpy as np

def compute_persistence(series, threshold=0.0, lookback=12):
    """
    Calcula la prevalencia direccional: % de observaciones que superan el umbral en la ventana lookback.
    Mide frecuencia de observaciones positivas, NO continuidad temporal (persistencia en sentido estricto).
    Si no hay suficientes datos, retorna None.
    Retorna un float entre 0.0 y 1.0 cuando hay datos suficientes.
    """
    if series is None or len(series.dropna()) < lookback:
        return None
    
    recent = series.dropna().iloc[-lookback:]
    if len(recent) == 0:
        return None
    
    positive = (recent > threshold).sum()
    return float(positive / len(recent))
