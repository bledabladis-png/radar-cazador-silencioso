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
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def beta(returns, benchmark_returns, window=60):
    cov = returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var()
    return cov / (var + 1e-9)


