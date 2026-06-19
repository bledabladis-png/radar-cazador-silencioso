import numpy as np
import pandas as pd

def robust_zscore(series, window=60):
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    z = (series - median) / (1.4826 * mad + 1e-9)
    return z.clip(-5, 5)

def rolling_percentile(series, window=120):
    return series.rolling(window).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5,
        raw=True
    )

def tanh_normalize(series):
    z = robust_zscore(series)
    return np.tanh(z)

def sigmoid(x):
    return (np.tanh(x) + 1) / 2

def ema_smooth(series, span=10):
    return series.ewm(span=span, min_periods=5).mean()

def winsorize(series, limits=(0.01, 0.99)):
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower, upper)

def standardize_series(series):
    return (series - series.mean()) / (series.std() + 1e-9)

def get_col(df, ticker, field='Close'):
    if isinstance(df.columns, pd.MultiIndex):
        col = (field, ticker)
        if col in df.columns:
            series = df[col].ffill(limit=5)
            return series
        raise KeyError(f'Columna {col} no encontrada')
    else:
        col = f'{ticker}_{field}'
        if col in df.columns:
            series = df[col].ffill(limit=5)
            return series
        raise KeyError(f'Columna {col} no encontrada')

def clean_oil_prices(df):
    try:
        close_cl = get_col(df, 'CL=F', 'Close')
        if (close_cl <= 0).any():
            min_positive = close_cl[close_cl > 0].min()
            df[('Close', 'CL=F')] = close_cl.clip(lower=min_positive)
    except KeyError:
        pass
    return df
