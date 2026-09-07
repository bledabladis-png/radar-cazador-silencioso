import numpy as np
import pandas as pd
from indicators.sector_flow_characteristics import _regime
from indicators.price_flow_divergence import detect_price_flow_divergence

def test_regime_confirmation():
    assert _regime(0.02, 1000) == 'Confirmación'
    assert _regime(-0.02, -1000) == 'Confirmación de debilidad'
    assert _regime(0.02, -1000) == 'Divergencia bajista'
    assert _regime(-0.02, 1000) == 'Absorción potencial'
    assert _regime(0.0, 1000) == 'Neutral / sin confirmación'
    assert _regime(np.nan, 1000) == 'N/D'
    assert _regime(0.02, np.nan) == 'N/D'

def test_detect_price_flow_divergence_states():
    assert detect_price_flow_divergence(0.10, 0.05)['status'] == 'PRICE_STRONG_FLOW_UNCONFIRMED'
    assert detect_price_flow_divergence(-0.10, 0.40)['status'] == 'PRICE_WEAK_FLOW_SUPPORTIVE'
    assert detect_price_flow_divergence(0.02, 0.20)['status'] == 'ALIGNED'
    assert detect_price_flow_divergence(None, 0.20)['status'] == 'UNAVAILABLE'
    assert detect_price_flow_divergence(0.02, None)['status'] == 'UNAVAILABLE'