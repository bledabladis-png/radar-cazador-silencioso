import pandas as pd
from src.utils import get_col
from config import settings

def volatility_regime(returns, window=20):
    vol = returns.rolling(window).std()
    vol_median = vol.rolling(settings.VOLATILITY_BASELINE_WINDOW, min_periods=252).median()
    vol_mad = (vol - vol_median).abs().rolling(settings.VOLATILITY_BASELINE_WINDOW, min_periods=252).median()
    z = (vol - vol_median) / (1.4826 * vol_mad + 1e-9)
    if isinstance(z, pd.DataFrame):
        return z.mean(axis=1)
    return z

def atr(df, ticker, window=14):
    """ATR sobre OHLC con dropna previo.

    F3-05-sexies (2026-09-30): df_market es multi-mercado. Un ticker
    USA tiene NaN en cada festivo NYSE; uno europeo en cada festivo
    local. Sin dropna, un solo NaN en la ventana de 14 contamina 14
    sesiones consecutivas del ATR. Verificado 2026-09-30: XLB con
    tr NaN en 2026-09-07 -> atr14 con 14 NaN en tail(30). Con dropna
    previo: 0 NaN, y vol_inv (componente del score sectorial) cambia
    hasta 0.30 en XLF.

    Unico consumidor productivo: regimes/sector_regime.py:68.
    wyckoff.py tiene su propio atr_normalized inline sobre ticker_df
    ya limpio (I2, 2026-09-18).
    """
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    sub = pd.DataFrame({'High': high, 'Low': low, 'Close': close}).dropna()
    if len(sub) < window:
        return pd.Series(dtype=float)
    prev_close = sub['Close'].shift(1)
    tr = pd.concat([
        sub['High'] - sub['Low'],
        (sub['High'] - prev_close).abs(),
        (sub['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()
