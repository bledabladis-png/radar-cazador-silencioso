import pandas as pd
import numpy as np

def evaluate_signal(signal_series, forward_returns, threshold=0.7, lag=1):
    signals = signal_series.shift(lag)
    high_risk = signals > threshold
    n_signals = high_risk.sum()
    
    if n_signals == 0:
        return {
            'alpha': np.nan,
            'hit_ratio': np.nan,
            'n_signals': 0,
            'message': 'No hubo señales con probabilidad > {:.1f} en el periodo analizado.'.format(threshold)
        }
    
    avg_return = forward_returns[high_risk].mean()
    avg_benchmark = forward_returns.mean()
    alpha = avg_return - avg_benchmark
    hit_ratio = (forward_returns[high_risk] > 0).mean()
    
    return {
        'alpha': alpha,
        'hit_ratio': hit_ratio,
        'n_signals': n_signals,
        'message': None
    }
