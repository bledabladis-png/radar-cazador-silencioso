# -*- coding: utf-8 -*-
"""
breadth_core.py -- Funciones comunes de amplitud (NH/NL, cobertura, avances/descensos).
Centraliza la logica compartida entre breadth.py y breadth_equity.py.
"""

def compute_new_highs_lows(prices, window=252):
    """
    Calcula nuevos maximos y minimos de 52 semanas.
    Usa shift(1) para comparar contra las 252 sesiones PREVIAS (no incluye el dia actual).
    
    Args:
        prices: DataFrame con precios (columnas = tickers, indice = fechas).
        window: ventana en sesiones (default 252).
    
    Returns:
        nh: Series con el numero de nuevos maximos por fecha.
        nl: Series con el numero de nuevos minimos por fecha.
    """
    previous_high = prices.shift(1).rolling(window, min_periods=window).max()
    previous_low = prices.shift(1).rolling(window, min_periods=window).min()
    
    nh = (prices >= previous_high).sum(axis=1)
    nl = (prices <= previous_low).sum(axis=1)
    
    return nh, nl


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
    valid_count = int(prices.notna().sum(axis=1).iloc[-1])
    if valid_count < expected_count:
        missing = prices.columns[prices.iloc[-1].isna()].tolist()
        print(f"    WARN {module_name}: cobertura parcial ({valid_count}/{expected_count}). Faltan: {missing}")
    return valid_count
