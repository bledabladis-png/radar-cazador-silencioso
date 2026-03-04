from src.data_layer import DataLayer
from src.regime_engine import RegimeEngine
from src.leadership_engine import LeadershipEngine
from src.geographic_engine import GeographicEngine
from src.stress_engine import StressEngine
from src.scoring import ScoringEngine
import pandas as pd

# Cargar datos
dl = DataLayer()
df = dl.load_latest()
print("Últimas fechas en los datos:", df.index[-5:].tolist())
print()

# Calcular motores
regime = RegimeEngine()
regime_df = regime.calcular_todo(df)
leadership = LeadershipEngine()
leadership_df = leadership.calcular_todo(df)
geo = GeographicEngine()
geo_df = geo.calcular_todo(df)
stress = StressEngine()
stress_df = stress.calcular_stress(df)

# Mostrar últimas filas de cada motor
print("Últimas 5 filas de regime_df:")
print(regime_df[['score_tendencia', 'score_credito', 'score_curva', 'score_regime']].tail())
print()

print("Últimas 5 filas de leadership_df:")
print(leadership_df[['score_small', 'score_cyclical', 'score_leadership']].tail())
print()

print("Últimas 5 filas de geo_df:")
print(geo_df[['score_em', 'score_dm', 'score_geo']].tail())
print()

print("Últimas 5 filas de stress_df:")
print(stress_df[['alert_vix', 'compresion_flag', 'drawdown_credit_penalty', 'score_stress']].tail())
print()

# Calcular scoring
scoring = ScoringEngine()
resultados_df = scoring.calcular_todo(df, regime_df, leadership_df, geo_df, stress_df)

print("Últimas 5 filas de scoring (score_global, dispersion, accumulation_signal):")
print(resultados_df[['score_global', 'dispersion', 'accumulation_signal']].tail())
print()

# Verificar si la fecha de hoy está presente
hoy = pd.Timestamp.now().normalize()
if hoy in df.index:
    print(f"La fecha {hoy.date()} SÍ está en los datos.")
    # Mostrar valores de los ETFs para hoy
    print("Valores de ETFs para hoy:")
    etfs = ['SPY', 'JNK', 'LQD', 'IWM', 'XLY', 'XLP', 'EEM', 'EFA', 'ACWI', '^VIX']
    for etf in etfs:
        if etf in df.columns:
            print(f"{etf}: {df.loc[hoy, etf]}")
        else:
            print(f"{etf}: no disponible")
else:
    print(f"La fecha {hoy.date()} NO está en los datos.")