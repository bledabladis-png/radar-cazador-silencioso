import pandas as pd

def ema_series(close, window):
    return close.ewm(span=window, min_periods=window).mean()

def trend_position(close):
    """
    Devuelve la posicion con respecto a EMAs 20,50,100,200.
    +1 si esta sobre cada una, -1 si esta debajo. Promedio normalizado a [-1,1].
    """
    emas = {'ema20': 20, 'ema50': 50, 'ema100': 100, 'ema200': 200}
    positions = pd.DataFrame()
    for name, w in emas.items():
        ema = ema_series(close, w)
        positions[name] = ((close > ema).astype(int) * 2 - 1)  # +1 o -1
    return positions.mean(axis=1)  # promedio de senales
