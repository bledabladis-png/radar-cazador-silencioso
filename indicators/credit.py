import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col

def credit_spread_signal(df):
    try:
        hyg = get_col(df, 'HYG', 'Close')
        lqd = get_col(df, 'LQD', 'Close')
        ratio = hyg / lqd
        z = robust_zscore(ratio, 60)
        return np.tanh(z)
    except KeyError:
        return pd.Series(0, index=df.index)
