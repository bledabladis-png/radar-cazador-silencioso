import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.context.percentile_engine import PercentileEngine
from src.narrative.scenario_engine import ScenarioEngine

def test_percentile_engine():
    """Prueba básica del motor de percentiles."""
    df = pd.DataFrame({'valor': np.random.randn(100)})
    engine = PercentileEngine(df)
    engine.add_metric('test', df['valor'])
    pct = engine.percentile('test', 0)
    assert 0 <= pct <= 100
    print("✅ PercentileEngine OK")

def test_scenario_engine():
    """Prueba básica del motor de escenarios."""
    engine = ScenarioEngine()
    snapshot = {
        'score_global': 0.5,
        'score_stress': -0.1,
        'score_breadth': 0.2,
        'score_liquidity': 0.3,
        'score_riesgo_sistemico': 0.1,
        'dispersion': 0.4,
        'pend_5d': 0.02
    }
    scenarios = engine.generate_scenarios(snapshot)
    assert 'base' in scenarios
    assert 'alternativo' in scenarios
    assert 'adverso' in scenarios
    probs = [scenarios[k]['probabilidad'] for k in scenarios]
    assert abs(sum(probs) - 1.0) < 0.01
    print("✅ ScenarioEngine OK")

if __name__ == "__main__":
    test_percentile_engine()
    test_scenario_engine()
    print("🎉 Todas las pruebas pasaron.")
