import pandas as pd
from indicators.persistence import compute_persistence

def test_compute_persistence_all_positive():
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    assert compute_persistence(s, threshold=0.0, lookback=12) == 1.0

def test_compute_persistence_all_negative():
    s = pd.Series([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2])
    assert compute_persistence(s, threshold=0.0, lookback=12) == 0.0

def test_compute_persistence_half():
    s = pd.Series([1,1,1,1,1,1,-1,-1,-1,-1,-1,-1])
    assert compute_persistence(s, threshold=0.0, lookback=12) == 0.5

def test_compute_persistence_threshold():
    s = pd.Series([1,2,3,4,5,6,7,8,9,10,11,12])
    assert compute_persistence(s, threshold=5.0, lookback=12) == 7/12

def test_compute_persistence_insufficient():
    s = pd.Series([1,2,3])
    assert compute_persistence(s, threshold=0.0, lookback=12) is None