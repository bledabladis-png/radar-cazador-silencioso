# -*- coding: utf-8 -*-
# indicators/index_phase.py - Calcula fases Wyckoff para indices internacionales
import pandas as pd
from indicators.wyckoff import wyckoff_structure_core
from config.index_tickers import INDEX_CONFIG

def compute_index_phases(df_market):
    """
    Calcula la fase Wyckoff para cada indice en INDEX_CONFIG.
    Retorna un diccionario {nombre_indice: fase}.
    """
    phases = {}
    for nombre, config in INDEX_CONFIG.items():
        ticker = config['index_ticker']
        try:
            fase = wyckoff_structure_core(df_market, ticker)
            phases[nombre] = fase
        except Exception as e:
            print(f"  Indice {nombre} ({ticker}): error al calcular fase - {e}")
            phases[nombre] = 'ERROR'
    return phases
