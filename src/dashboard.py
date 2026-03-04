"""
dashboard.py - Visualización de resultados del Radar Macro Rotación Global v2.1
Genera gráficos de evolución del score global, dispersión, contribuciones y exposición.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# Importar módulos del sistema
from src.data_layer import DataLayer
from src.regime_engine import RegimeEngine
from src.leadership_engine import LeadershipEngine
from src.geographic_engine import GeographicEngine
from src.stress_engine import StressEngine
from src.scoring import ScoringEngine
from src.risk import RiskManager

def calcular_historial(dias=500):
    """
    Ejecuta el pipeline para un histórico de días y devuelve un DataFrame con los resultados.
    """
    print(f"Calculando histórico para los últimos {dias} días...")
    
    dl = DataLayer()
    df = dl.load_latest()
    
    # Seleccionar un subconjunto de fechas (últimos 'dias')
    fechas = df.index[-dias:]
    df = df.loc[fechas]
    
    # Calcular motores para todo el período
    regime = RegimeEngine()
    regime_df = regime.calcular_todo(df)
    
    leadership = LeadershipEngine()
    leadership_df = leadership.calcular_todo(df)
    
    geo = GeographicEngine()
    geo_df = geo.calcular_todo(df)
    
    stress = StressEngine()
    stress_df = stress.calcular_stress(df)
    
    scoring = ScoringEngine()
    resultados_df = scoring.calcular_todo(df, regime_df, leadership_df, geo_df, stress_df)
    
    # Para la exposición, necesitamos aplicar riesgo día a día con un contexto aproximado
    # Usaremos un riesgo simplificado: solo basado en score y dispersión (sin spreads ni turnover real)
    # Pero para visualización, podemos calcular una exposición base sin costes
    rm = RiskManager()
    exposiciones = []
    vix_vals = df['^VIX'] if '^VIX' in df.columns else pd.Series(20, index=df.index)
    
    for fecha in resultados_df.index:
        score = resultados_df.loc[fecha, 'score_global']
        dispersion = resultados_df.loc[fecha, 'dispersion']
        vix = vix_vals.loc[fecha] if not pd.isna(vix_vals.loc[fecha]) else 20
        
        # Contexto simplificado (sin spreads ni turnover histórico)
        contexto = {
            'vix': vix,
            'dispersion': dispersion,
            'spreads': {'SPY': 0.001, 'EEM': 0.004, 'JNK': 0.006},  # valores fijos
            'exp_prev': 0.5,  # no tenemos histórico, usamos valor fijo
            'capital': 100000
        }
        exp, _ = rm.aplicar_reglas_riesgo(score, contexto)
        exposiciones.append(exp)
    
    resultados_df['exposicion'] = exposiciones
    return resultados_df

def graficar(resultados_df):
    """
    Genera gráficos a partir del DataFrame de resultados.
    """
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Radar Macro Rotación Global v2.1 - Dashboard', fontsize=16)
    
    # 1. Score global
    ax = axes[0, 0]
    ax.plot(resultados_df.index, resultados_df['score_global'], color='blue', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_title('Score Global')
    ax.set_ylabel('Score')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # 2. Dispersión
    ax = axes[0, 1]
    ax.plot(resultados_df.index, resultados_df['dispersion'], color='orange', linewidth=1.5)
    ax.set_title('Dispersión entre Componentes')
    ax.set_ylabel('Dispersión')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Contribuciones de los motores (apiladas)
    ax = axes[1, 0]
    ax.stackplot(resultados_df.index,
                 resultados_df['score_regime'],
                 resultados_df['score_leadership'],
                 resultados_df['score_geo'],
                 resultados_df['score_stress'],
                 labels=['Régimen', 'Liderazgo', 'Geográfico', 'Estrés'],
                 alpha=0.7)
    ax.set_title('Contribuciones por Motor')
    ax.set_ylabel('Score')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    ax.legend(loc='upper left')
    
    # 4. Señal de acumulación
    ax = axes[1, 1]
    ax.fill_between(resultados_df.index, resultados_df['accumulation_signal'], color='green', alpha=0.5)
    ax.set_title('Señal de Acumulación')
    ax.set_ylabel('Señal (0-1)')
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # 5. Exposición recomendada
    ax = axes[2, 0]
    ax.plot(resultados_df.index, resultados_df['exposicion'] * 100, color='red', linewidth=1.5)
    ax.set_title('Exposición Recomendada')
    ax.set_ylabel('Exposición (%)')
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    # 6. Score global vs exposición (doble eje)
    ax = axes[2, 1]
    color1 = 'blue'
    color2 = 'red'
    ax.plot(resultados_df.index, resultados_df['score_global'], color=color1, label='Score Global')
    ax.set_ylabel('Score', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.tick_params(axis='x', rotation=45)
    
    ax2 = ax.twinx()
    ax2.plot(resultados_df.index, resultados_df['exposicion'] * 100, color=color2, label='Exposición')
    ax2.set_ylabel('Exposición (%)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.tight_layout()
    plt.savefig('dashboard.png', dpi=150)
    print("Gráfico guardado como 'dashboard.png'")
    plt.show()

if __name__ == "__main__":
    # Calcular histórico (últimos 500 días)
    resultados = calcular_historial(dias=500)
    # Graficar
    graficar(resultados)