import pandas as pd
import numpy as np
from indicators.sector_dispersion import compute_sector_dispersion

def test_compute_sector_dispersion_ok():
    data = [('XLK', 0.02), ('XLF', 0.01), ('XLV', -0.01), ('XLE', 0.04),
            ('XLY', -0.02), ('XLP', 0.03), ('XLI', 0.00), ('XLB', -0.03),
            ('XLU', 0.05), ('XLRE', -0.04), ('XLC', 0.06)]
    df = compute_sector_dispersion(data)
    assert not df.empty
    assert df.iloc[0]['n_total'] == 11
    assert df.iloc[0]['n_valid'] == 11
    assert df.iloc[0]['coverage'] == 1.0
    # std debe ser > 0
    assert df.iloc[0]['std_pp'] > 0

def test_compute_sector_dispersion_insufficient():
    data = [('XLK', 0.02), ('XLF', 0.01)]  # solo 2 sectores
    df = compute_sector_dispersion(data)
    assert not df.empty
    assert df.iloc[0]['dispersion_reading'] == 'N/D'
    assert pd.isna(df.iloc[0]['range_pp'])