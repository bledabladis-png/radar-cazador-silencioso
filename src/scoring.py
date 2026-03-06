"""
scoring.py - Motor de scoring global (versión simplificada)
Combina los scores de los módulos con pesos fijos.
Calcula la dispersión y el factor de exposición basado en estrés y dispersión.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

class ScoringEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Pesos base de los módulos (sección weights.base)
        weights = self.config.get('weights', {}).get('base', {})
        self.base_weights = {
            'regime': weights.get('regime', 0.33),
            'leadership': weights.get('leadership', 0.24),
            'geo': weights.get('geo', 0.19),
            'bonds': weights.get('bonds', 0.09),
            'stress': weights.get('stress', 0.09),
            'liquidity': weights.get('liquidity', 0.06)
        }

        # Parámetros para penalizaciones (sección exposicion o similar)
        # Usaremos valores por defecto razonables si no están definidos
        penalty_cfg = self.config.get('exposicion', {})
        self.disp_factor = penalty_cfg.get('dispersion_penal_factor', 0.5)   # factor para penalización por dispersión
        self.stress_factor = penalty_cfg.get('stress_penalty_factor', 0.5)  # factor para penalización por estrés

        # Módulos a incluir en el cálculo de dispersión (por defecto los principales sin liquidez)
        self.disp_modules = ['regime', 'leadership', 'geo', 'bonds', 'stress']

    def calcular_todo(self, regime_df, leadership_df, geo_df, bond_df, stress_df, liquidity_df=None, vix_series=None):
        """
        Entradas: DataFrames con una columna cada uno (score_regime, score_leadership, etc.)
        liquidity_df es opcional (si no se pasa, se ignora).
        vix_series no se usa en esta versión simplificada, se mantiene por compatibilidad.
        """
        # Construir lista de DataFrames disponibles con sus columnas renombradas
        dfs = [
            regime_df[['score_regime']].rename(columns={'score_regime': 'regime'}),
            leadership_df[['score_leadership']].rename(columns={'score_leadership': 'leadership'}),
            geo_df[['score_geographic']].rename(columns={'score_geographic': 'geo'}),
            bond_df[['score_bonds']].rename(columns={'score_bonds': 'bonds'}),
            stress_df[['score_stress']].rename(columns={'score_stress': 'stress'})
        ]
        if liquidity_df is not None:
            dfs.append(liquidity_df[['score_liquidity']].rename(columns={'score_liquidity': 'liquidity'}))

        # Alinear todos los DataFrames por su índice (fechas)
        combined = pd.concat(dfs, axis=1)
        # Eliminar filas con todos NaN (si las hay)
        combined = combined.dropna(how='all')
        # Rellenar NaN con el último valor válido y luego 0
        combined = combined.ffill().fillna(0)

        # --- 1. Score global ponderado ---
        score_global = pd.Series(0.0, index=combined.index)
        for module, weight in self.base_weights.items():
            if module in combined.columns:
                score_global += combined[module] * weight
            else:
                logger.debug(f"Módulo {module} no presente, se omite")

        # --- 2. Dispersión entre módulos principales (sin liquidez) ---
        # Tomamos los módulos que realmente existen
        available_disp = [m for m in self.disp_modules if m in combined.columns]
        if len(available_disp) > 1:
            dispersion = combined[available_disp].std(axis=1)
        else:
            dispersion = pd.Series(0.0, index=combined.index)

        # --- 3. Penalización por estrés (solo si stress > 0) ---
        if 'stress' in combined.columns:
            # Solo la parte positiva del estrés penaliza
            stress_positive = combined['stress'].clip(lower=0)
            penalty_stress = stress_positive * self.stress_factor
        else:
            penalty_stress = pd.Series(0.0, index=combined.index)

        # --- 4. Penalización por dispersión ---
        penalty_disp = dispersion * self.disp_factor
        # Acotar penalizaciones para que el factor de exposición no sea negativo
        # (opcional, pero podemos limitar cada penalización a un máximo, ej. 0.5)
        max_penalty = 0.8  # para que exposure_factor nunca sea menor que 0.2
        penalty_disp = penalty_disp.clip(upper=max_penalty)
        penalty_stress = penalty_stress.clip(upper=max_penalty)

        # --- 5. Factor de exposición final ---
        exposure_factor = 1.0 - penalty_disp - penalty_stress
        exposure_factor = exposure_factor.clip(lower=0.0, upper=1.0)

        # --- 6. Score suavizado para visualización (media móvil de 5 días) ---
        score_smoothed = score_global.rolling(window=5, min_periods=1).mean()

        # --- 7. Construir DataFrame de resultados ---
        resultados = pd.DataFrame({
            'score_global': score_global,
            'score_smoothed': score_smoothed,
            'exposure_factor': exposure_factor,
            'dispersion': dispersion,
            'penalty_disp': penalty_disp,
            'penalty_stress': penalty_stress
        }, index=combined.index)

        # Rellenar cualquier posible NaN residual
        resultados = resultados.ffill().fillna(0)

        return resultados


if __name__ == "__main__":
    # Ejemplo de prueba
    from src.data_layer import DataLayer
    from src.regime_engine import RegimeEngine
    from src.leadership_engine import LeadershipEngine
    from src.geographic_engine import GeographicEngine
    from src.bond_engine import BondEngine
    from src.stress_engine import StressEngine
    from src.liquidity_engine import LiquidityEngine

    logging.basicConfig(level=logging.INFO)

    dl = DataLayer()
    df = dl.load_latest()

    engines = {
        'regime': RegimeEngine(),
        'leadership': LeadershipEngine(),
        'geo': GeographicEngine(),
        'bonds': BondEngine(),
        'stress': StressEngine(),
        'liquidity': LiquidityEngine()
    }

    # Calcular scores de cada motor
    module_dfs = {}
    for name, eng in engines.items():
        result = eng.calcular_todo(df)
        module_dfs[name] = result

    # Llamar al scoring engine
    scoring = ScoringEngine()
    resultados = scoring.calcular_todo(
        regime_df=module_dfs['regime'],
        leadership_df=module_dfs['leadership'],
        geo_df=module_dfs['geo'],
        bond_df=module_dfs['bonds'],
        stress_df=module_dfs['stress'],
        liquidity_df=module_dfs['liquidity']
    )

    print("\nResultados globales (últimos 5 días):")
    print(resultados.tail())