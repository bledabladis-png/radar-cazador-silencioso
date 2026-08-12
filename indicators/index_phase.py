# -*- coding: utf-8 -*-
# indicators/index_phase.py - Calcula fases Wyckoff para indices internacionales
import pandas as pd
from indicators.wyckoff import wyckoff_structure_core
from config.index_tickers import INDEX_CONFIG
from data.providers.router import DataRouter

def compute_index_phases(df_market):
    router = DataRouter()
    all_index_tickers = [cfg['index_ticker'] for cfg in INDEX_CONFIG.values()]
    
    # Intentar obtener desde df_market primero
    missing_tickers = []
    phases = {}
    index_data = None
    
    for nombre, config in INDEX_CONFIG.items():
        ticker = config['index_ticker']
        try:
            fase = wyckoff_structure_core(df_market, ticker)
            phases[nombre] = fase
        except:
            missing_tickers.append(ticker)
    
    # Si faltan indices, descargarlos directamente
    if missing_tickers:
        print(f"  Descargando datos para indices faltantes: {missing_tickers}")
        try:
            index_data = router.get_market_data(missing_tickers, period='5y')
            for nombre, config in INDEX_CONFIG.items():
                ticker = config['index_ticker']
                if ticker in missing_tickers:
                    try:
                        fase = wyckoff_structure_core(index_data, ticker)
                        phases[nombre] = fase
                    except Exception as e:
                        print(f"  Indice {nombre} ({ticker}): error al calcular fase - {e}")
                        phases[nombre] = 'ERROR'
        except Exception as e:
            print(f"  Error al descargar indices: {e}")
            for nombre, config in INDEX_CONFIG.items():
                if config['index_ticker'] in missing_tickers:
                    phases[nombre] = 'ERROR'
    
    return phases, index_data  # devuelve tambien los datos descargados
