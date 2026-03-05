"""
scoring.py - Motor de scoring global
Combina los cinco módulos (régimen, liderazgo, geográfico, bonos, estrés)
aplica pesos, penalizaciones y genera el score global final.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.normalization import normalize_module_series

logger = logging.getLogger(__name__)

class ScoringEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.weights = self.config['weights']['base']
        self.penalties = self.config.get('exposicion', {})
        self.disp_factor = self.penalties.get('dispersion_penal_factor', 0.5)
        self.stress_factor = self.penalties.get('stress_penalty_factor', 0.4)

        # Normalización de módulos (opcional)
        norm_config = self.config.get('module_normalization', {})
        self.norm_enabled = norm_config.get('enabled', False)
        self.norm_window = norm_config.get('window', 252)
        self.norm_method = norm_config.get('method', 'zscore_tanh')

        # Ponderación dinámica por VIX
        dyn_config = self.config.get('dynamic_weighting', {})
        self.dyn_enabled = dyn_config.get('enabled', False)
        self.vix_umbral = dyn_config.get('vix_umbral', 25)
        self.vix_factor = dyn_config.get('vix_factor', 0.01)

    def _align_dataframes(self, dfs):
        """Alinea múltiples DataFrames por índice."""
        combined = pd.concat(dfs, axis=1)
        combined = combined.dropna()
        return combined

    def calcular_todo(self, regime_df, leadership_df, geo_df, bond_df, stress_df, vix_series=None):
        """
        Entradas: DataFrames con una columna cada uno:
            - regime_df: 'score_regime'
            - leadership_df: 'score_leadership'
            - geo_df: 'score_geo'
            - bond_df: 'score_bonds'
            - stress_df: 'score_stress'
        vix_series: Serie con valores de VIX (para ponderación dinámica)
        """
        # Construir lista de DataFrames y renombrar columnas
        scores_list = [
            regime_df[['score_regime']].rename(columns={'score_regime': 'regime'}),
            leadership_df[['score_leadership']].rename(columns={'score_leadership': 'leadership'}),
            geo_df[['score_geo']].rename(columns={'score_geo': 'geo'}),
            bond_df[['score_bonds']].rename(columns={'score_bonds': 'bonds'}),
            stress_df[['score_stress']].rename(columns={'score_stress': 'stress'})
        ]

        # Alinear por fecha
        df = self._align_dataframes(scores_list)

        # Normalización de módulos (opcional)
        if self.norm_enabled:
            for col in ['regime', 'leadership', 'geo', 'bonds', 'stress']:
                df[col] = normalize_module_series(df[col],
                                                   window=self.norm_window,
                                                   method=self.norm_method)

        # Ponderación dinámica por VIX
        pesos = self.weights.copy()
        if self.dyn_enabled and vix_series is not None:
            # Usar el último VIX disponible
            vix = vix_series.iloc[-1] if isinstance(vix_series, pd.Series) else vix_series
            if vix > self.vix_umbral:
                factor = np.exp(-self.vix_factor * (vix - self.vix_umbral))
                for k in pesos:
                    pesos[k] *= factor
                # Renormalizar
                total = sum(pesos.values())
                for k in pesos:
                    pesos[k] /= total

        # Calcular score base ponderado
        df['score_base'] = (pesos['regime'] * df['regime'] +
                            pesos['leadership'] * df['leadership'] +
                            pesos['geo'] * df['geo'] +
                            pesos['bonds'] * df['bonds'] +
                            pesos['stress'] * df['stress'])

        # Penalización por dispersión entre módulos
        df['dispersion'] = df[['regime', 'leadership', 'geo', 'bonds', 'stress']].std(axis=1)
        df['dispersion_penalty'] = df['dispersion'] * self.disp_factor

        # Penalización por estrés (solo si stress > 0)
        df['stress_penalty'] = df['stress'].clip(lower=0) * self.stress_factor

        # Score global final
        df['score_global_raw'] = df['score_base'] - df['dispersion_penalty'] - df['stress_penalty']
        df['score_global'] = np.tanh(df['score_global_raw'])

        # Suavizado (media 10 días)
        df['score_smoothed'] = df['score_global'].rolling(10, min_periods=1).mean()

        # Clasificación de régimen
        def classify(score):
            if score > 0.6:
                return 'RISK_ON'
            if score > 0.2:
                return 'RISK_ON_MODERATE'
            if score > -0.2:
                return 'NEUTRAL'
            if score > -0.6:
                return 'RISK_OFF_MODERATE'
            return 'STRESS'

        df['regime_state'] = df['score_smoothed'].apply(classify)

        # Devolver solo las columnas principales (podemos añadir más si se desea)
        return df[['score_global', 'score_smoothed', 'dispersion', 'regime_state']]


if __name__ == "__main__":
    from data_layer import DataLayer
    from regime_engine import RegimeEngine
    from leadership_engine import LeadershipEngine
    from geographic_engine import GeographicEngine
    from bond_engine import BondEngine
    from stress_engine import StressEngine

    logging.basicConfig(level=logging.INFO)

    # Cargar datos
    dl = DataLayer()
    df = dl.load_latest()
    print("Datos cargados. Últimas fechas:", df.index[-5:])

    # Instanciar motores
    regime = RegimeEngine()
    leadership = LeadershipEngine()
    geo = GeographicEngine()
    bond = BondEngine()
    stress = StressEngine()

    # Calcular scores individuales
    regime_scores = regime.calcular_todo(df)
    leadership_scores = leadership.calcular_todo(df)
    geo_scores = geo.calcular_todo(df)
    bond_scores = bond.calcular_todo(df)
    stress_scores = stress.calcular_todo(df)

    # Obtener VIX para ponderación dinámica
    vix = df['^VIX'] if '^VIX' in df.columns else None

    # Scoring global
    scoring = ScoringEngine()
    resultado = scoring.calcular_todo(regime_scores, leadership_scores, geo_scores,
                                      bond_scores, stress_scores, vix_series=vix)

    print("\nScore global (últimos 5 días):")
    print(resultado.tail())