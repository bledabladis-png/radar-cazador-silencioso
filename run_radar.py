"""
run_radar.py - Script de integración del Radar Macro Rotación Global v2.1
Ejecuta todo el pipeline:
- Carga datos más recientes
- Calcula scores de cada motor
- Obtiene score global, dispersión y señal de acumulación
- Aplica gestión de riesgo
- Muestra resumen y guarda exposición para el próximo día
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

# Importar módulos del sistema
from src.data_layer import DataLayer
from src.regime_engine import RegimeEngine
from src.leadership_engine import LeadershipEngine
from src.geographic_engine import GeographicEngine
from src.stress_engine import StressEngine
from src.scoring import ScoringEngine
from src.risk import RiskManager

def main():
    print("=" * 60)
    print("RADAR MACRO ROTACIÓN GLOBAL v2.1")
    print("=" * 60)

    # 1. Cargar datos
    print("\n[1] Cargando datos más recientes...")
    dl = DataLayer()
    df = dl.load_latest()
    ultima_fecha = df.index[-1]
    print(f"    Última fecha disponible: {ultima_fecha.date()}")

    # 2. Ejecutar motores
    print("\n[2] Calculando motores...")
    regime = RegimeEngine()
    regime_df = regime.calcular_todo(df)
    print("    - Régimen: OK")

    leadership = LeadershipEngine()
    leadership_df = leadership.calcular_todo(df)
    print("    - Liderazgo: OK")

    geo = GeographicEngine()
    geo_df = geo.calcular_todo(df)
    print("    - Geográfico: OK")

    stress = StressEngine()
    stress_df = stress.calcular_stress(df)
    print("    - Estrés: OK")

    # 3. Scoring
    print("\n[3] Calculando score global...")
    scoring = ScoringEngine()
    resultados_df = scoring.calcular_todo(df, regime_df, leadership_df, geo_df, stress_df)
    
    # Obtener valores del último día
    ultimo = resultados_df.iloc[-1]
    dispersion = ultimo['dispersion']
    score_global = ultimo['score_global']
    accum_signal = ultimo['accumulation_signal']
    
    # Obtener componentes individuales (opcional)
    score_regime = ultimo['score_regime']
    score_leadership = ultimo['score_leadership']
    score_geo = ultimo['score_geo']
    score_stress = ultimo['score_stress']
    
    print(f"    Score global: {score_global:.4f}")
    print(f"    Dispersión: {dispersion:.4f}")
    print(f"    Señal acumulación: {accum_signal:.2f}")

    # 4. Preparar contexto para RiskManager
    print("\n[4] Aplicando gestión de riesgo...")
    
    # Obtener VIX del último día
    vix = df.loc[ultima_fecha, '^VIX'] if '^VIX' in df.columns else 20.0
    if pd.isna(vix):
        vix = 20.0  # valor por defecto
    
    # Dispersión ya la tenemos
    
    # Spreads (simulados por ahora, en el futuro podrías obtenerlos de una API)
    spreads = {
        'SPY': 0.001,
        'EEM': 0.004,
        'JNK': 0.006,
        'LQD': 0.002,
        'IWM': 0.002,
        'XLY': 0.002,
        'XLP': 0.002,
        'EFA': 0.003,
        'ACWI': 0.002
    }
    
    # Exposición anterior (leer de archivo si existe)
    exp_prev = 0.5  # valor por defecto
    exp_file = 'ultima_exposicion.txt'
    if os.path.exists(exp_file):
        try:
            with open(exp_file, 'r') as f:
                exp_prev = float(f.read().strip())
            print(f"    Exposición anterior leída: {exp_prev:.4f}")
        except:
            print("    No se pudo leer exposición anterior, usando 0.5")
    
    capital = 100000  # capital fijo de ejemplo
    
    contexto = {
        'vix': vix,
        'dispersion': dispersion,
        'spreads': spreads,
        'exp_prev': exp_prev,
        'capital': capital
    }
    
    # Aplicar reglas de riesgo
    rm = RiskManager()
    exp_final, penalizaciones = rm.aplicar_reglas_riesgo(score_global, contexto)
    
    print(f"    Exposición final: {exp_final:.4f}")
    print(f"    Factores aplicados:")
    print(f"      - VIX: {penalizaciones['vix_factor']:.3f}")
    print(f"      - Dispersión: {penalizaciones['dispersion_factor']:.3f}")
    print(f"      - Turnover: {penalizaciones['turnover']:.3f}")
    print(f"      - Comisión: {penalizaciones['comision']:.6f}")

    # 5. Guardar exposición para el próximo día
    with open(exp_file, 'w') as f:
        f.write(str(exp_final))
    print(f"\n[5] Exposición guardada en {exp_file} para la próxima ejecución.")

    # 6. Resumen final
    print("\n" + "=" * 60)
    print("RESUMEN EJECUTIVO")
    print("=" * 60)
    print(f"Fecha: {ultima_fecha.date()}")
    print(f"Score global: {score_global:.4f}")
    print(f"Dispersión: {dispersion:.4f}")
    print(f"Señal acumulación: {accum_signal:.2f}")
    print(f"Exposición recomendada: {exp_final:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    main()