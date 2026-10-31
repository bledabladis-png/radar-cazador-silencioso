import pandas as pd

# -*- coding: utf-8 -*-
"""
breadth_core.py -- Funciones comunes de amplitud (NH/NL, cobertura, avances/descensos).
Centraliza la logica compartida entre breadth.py y breadth_equity.py.
"""

def compute_new_highs_lows(prices, window=252):
    """
    Calcula nuevos maximos y minimos de 52 semanas POR TICKER.

    Fix 2026-10-15 (H7): la version anterior hacia shift(1).rolling(N,
    min_periods=N) sobre el DataFrame completo. Con NaN dispersos
    (festivos USA/EU, gaps), ninguna ventana consecutiva cumplia -> 0/0
    en todo el historico (289/329 tickers). Ahora cada ticker se evalua
    sobre su serie dropna y los conteos se agregan por fecha.

    Args:
        prices: DataFrame con precios (columnas = tickers, indice = fechas).
        window: ventana en sesiones validas por ticker (default 252).

    Returns:
        nh: Series con el numero de nuevos maximos por fecha.
        nl: Series con el numero de nuevos minimos por fecha.
    """
    nh_counts = pd.Series(0, index=prices.index, dtype=int)
    nl_counts = pd.Series(0, index=prices.index, dtype=int)

    for col in prices.columns:
        s = prices[col].dropna()
        if len(s) <= window:
            continue
        prev_high = s.shift(1).rolling(window, min_periods=window).max()
        prev_low = s.shift(1).rolling(window, min_periods=window).min()
        is_nh = (s >= prev_high).reindex(prices.index, fill_value=False).astype(int)
        is_nl = (s <= prev_low).reindex(prices.index, fill_value=False).astype(int)
        nh_counts = nh_counts + is_nh
        nl_counts = nl_counts + is_nl

    return nh_counts, nl_counts


def compute_advances_declines(prices):
    """
    Calcula avances, descensos y sin cambio a partir de un DataFrame de precios.
    
    Args:
        prices: DataFrame con precios de cierre (columnas = tickers).
    
    Returns:
        advances, declines, unchanged: Series con conteos diarios.
    """
    daily_change = prices.diff()
    
    advances = (daily_change > 0).sum(axis=1)
    declines = (daily_change < 0).sum(axis=1)
    unchanged = (daily_change == 0).sum(axis=1)
    
    return advances, declines, unchanged


def validate_coverage(prices, expected_count, module_name="Breadth"):
    """
    Verifica que todos los tickers esperados tengan datos.
    Emite advertencia si la cobertura es incompleta.
    
    Returns:
        valid_count: numero de tickers con datos en la ultima fecha.
    """
    # H1 B6 (2026-10-31): guard df vacio/None. Antes .iloc[-1] lanzaba IndexError
    # cuando df_market no tenia filas (fallo masivo de proveedor, primer run).
    if prices is None or len(prices) == 0:
        return 0
    valid_count = int(prices.notna().sum(axis=1).iloc[-1])
    if valid_count < expected_count:
        missing = prices.columns[prices.iloc[-1].isna()].tolist()
        print(f"    WARN {module_name}: cobertura parcial ({valid_count}/{expected_count}). Faltan: {missing}")
    return valid_count
