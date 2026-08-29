import pandas as pd
from indicators.volatility import volatility_regime

def compute_volatility_regime(returns):
    z = volatility_regime(returns)
    last = z.iloc[-1] if not z.empty else 0
    
    if last < -0.5:
        regime = 'LOW'
    elif last < 0.5:
        regime = 'NORMAL'
    elif last < 1.5:
        regime = 'ELEVATED'
    else:
        regime = 'STRESS'
    
    confidence = min(abs(last) / 2, 1.0) if pd.notna(last) else 0.0
    return z, regime, confidence
