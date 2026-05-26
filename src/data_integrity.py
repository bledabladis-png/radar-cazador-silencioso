# data_integrity.py – Validación de integridad de datos descargados
import pandas as pd

REQUIRED_US_TICKERS = [
    'SPY', 'XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLV', 'XLU', 'XLRE',
    '^VIX', 'HYG', 'LQD', '^TNX', '^IRX'
]

REQUIRED_GLOBAL_TICKERS = [
    # Core
    'SPY', 'EZU', 'EWJ', 'EEM',
    'TLT', 'HYG',
    'GLD', 'DBC', 'XOP',
    'UUP',
    'EURUSD=X', 'JPYUSD=X',
    # Benchmark
    'ACWI',
    # Contexto extendido (deseables pero no críticos)
    'VGK', 'IWM', 'FXI', 'LQD'
]

def validate_market_data(df, required_tickers=REQUIRED_US_TICKERS, min_days=200):
    """
    Verifica que todos los tickers requeridos existen en el DataFrame
    y tienen al menos `min_days` días de datos válidos (no NaN).
    Retorna (passed, missing_list, details_dict).
    """
    missing = []
    details = {}
    for ticker in required_tickers:
        # Buscar la columna de precio (puede ser solo el ticker o con sufijo _close)
        if ticker in df.columns:
            col = ticker
        elif f"{ticker}_close" in df.columns:
            col = f"{ticker}_close"
        else:
            missing.append(ticker)
            details[ticker] = 'Columna no encontrada'
            continue
        
        valid_count = df[col].dropna().shape[0]
        if valid_count < min_days:
            missing.append(ticker)
            details[ticker] = f'Solo {valid_count} días válidos (mínimo {min_days})'
        else:
            details[ticker] = f'OK ({valid_count} días)'
    
    return len(missing) == 0, missing, details

def validate_global_data(df_global, required_tickers=REQUIRED_GLOBAL_TICKERS, min_days=200):
    """Análogo para los datos globales."""
    missing = []
    details = {}
    for ticker in required_tickers:
        col = f"{ticker}_close"
        if col in df_global.columns:
            valid_count = df_global[col].dropna().shape[0]
            if valid_count < min_days:
                missing.append(ticker)
                details[ticker] = f'Solo {valid_count} días válidos (mínimo {min_days})'
            else:
                details[ticker] = f'OK ({valid_count} días)'
        else:
            missing.append(ticker)
            details[ticker] = 'Columna no encontrada'
    
    return len(missing) == 0, missing, details