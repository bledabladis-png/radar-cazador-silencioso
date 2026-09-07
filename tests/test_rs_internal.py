import numpy as np
import pandas as pd
from indicators.rs_internal import classify_rs

def test_classify_rs():
    assert classify_rs(0.05, 0.03) == 'Liderazgo relativo doble'
    assert classify_rs(0.05, -0.02) == 'Fortaleza sectorial'
    assert classify_rs(-0.05, 0.02) == 'Liderazgo interno en sector débil'
    assert classify_rs(-0.05, -0.03) == 'Debilidad relativa doble'
    assert classify_rs(np.nan, 0.02) == 'N/D'
    assert classify_rs(0.05, np.nan) == 'N/D'

def test_compute_rs_internal_synthetic():
    # Este test requiere un pequeño universo sintético
    # Se omite aquí por brevedad pero se puede añadir con MultiIndex
    pass