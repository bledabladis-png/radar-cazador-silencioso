"""
scoring.py - Scoring Engine v3.0 (alineado con documento de auditoría)
- Score provisional con pesos base (para fase)
- Determinación de fase del ciclo mediante CycleEngine
- Pesos dinámicos por fase (desde regime_weights.yaml)
- Score global con pesos dinámicos
- Ajuste por correlación (PCA) vía correlation_engine
- Score coherente = score_global * exp(-dispersion)
- Exposure factor = coherencia - penalty_stress - penalty_breadth
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

from src.cycle_engine import CycleEngine
from src.correlation_engine import adjust_weights_pca
from src.macro_engines_v7 import ciclo_institucional

logger = logging.getLogger(__name__)

class ScoringEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Pesos base (para score provisional)
        wbase = self.config.get('weights', {}).get('base', {})
        self.base_weights = {
            'regime': wbase.get('regime', 0.18),
            'leadership': wbase.get('leadership', 0.14),
            'geo': wbase.get('geo', 0.10),
            'bonds': wbase.get('bonds', 0.20),
            'stress': wbase.get('stress', 0.05),
            'liquidity': wbase.get('liquidity', 0.20),
            'breadth': wbase.get('breadth', 0.13)
        }

        # Factores de penalización
        pen = self.config.get('penalties', {})
        self.stress_factor = pen.get('stress_factor', 0.5)
        self.breadth_factor = pen.get('breadth_factor', 0.3)
        self.dispersion_max = pen.get('dispersion_max', 0.8)

        # Módulos para dispersión (sin liquidez ni breadth)
        self.disp_modules = ['regime', 'leadership', 'geo', 'bonds', 'stress']

        # Cargar pesos por fase
        weights_path = Path('config/regime_weights.yaml')
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.fase_weights = yaml.safe_load(f)

        # Inicializar CycleEngine
        self.cycle_engine = CycleEngine()

    def calcular_todo(self, regime_df, leadership_df, geo_df, bond_df, stress_df,
                      liquidity_df=None, breadth_df=None):
        """
        Entradas: DataFrames con una columna cada uno (score_regime, score_leadership, etc.)
        liquidity_df y breadth_df son opcionales.
        Retorna DataFrame con scores y métricas.
        """
        # Construir lista de DataFrames disponibles
        dfs = [
            regime_df[['score_regime']].rename(columns={'score_regime': 'regime'}),
            leadership_df[['score_leadership']].rename(columns={'score_leadership': 'leadership'}),
            geo_df[['score_geographic']].rename(columns={'score_geographic': 'geo'}),
            bond_df[['score_bonds']].rename(columns={'score_bonds': 'bonds'}),
            stress_df[['score_stress']].rename(columns={'score_stress': 'stress'})
        ]
        if liquidity_df is not None:
            dfs.append(liquidity_df[['score_liquidity']].rename(columns={'score_liquidity': 'liquidity'}))
        if breadth_df is not None:
            dfs.append(breadth_df[['score_breadth']].rename(columns={'score_breadth': 'breadth'}))

        # Alinear por índice
        combined = pd.concat(dfs, axis=1)
        combined = combined.dropna(how='all')
        combined = combined.ffill().fillna(0)

        motores_disponibles = list(combined.columns)

        # Preparar DataFrame de resultados
        resultados = pd.DataFrame(index=combined.index)
        resultados['score_global'] = np.nan
        resultados['score_coherente'] = np.nan
        resultados['exposure_factor'] = np.nan
        resultados['dispersion'] = np.nan
        resultados['penalty_stress'] = np.nan
        resultados['penalty_breadth'] = np.nan
        resultados['fase_ciclo'] = None
        resultados['ciclo_institucional'] = None
        resultados['pend_3d'] = np.nan
        resultados['pend_5d'] = np.nan
        resultados['pend_10d'] = np.nan
        resultados['aceleracion'] = np.nan
        resultados['motores_mejorando'] = np.nan

        # Para el cálculo de fases necesitamos fila anterior
        fila_anterior = None

        for idx in combined.index:
            fila_actual = combined.loc[idx]

            # --- 1. Score provisional con pesos base (para fase) ---
            score_prov = 0.0
            for mod, peso in self.base_weights.items():
                if mod in motores_disponibles:
                    score_prov += fila_actual[mod] * peso

            # --- 2. Métricas dinámicas sobre score_prov ---
            hist_hasta_idx = combined.loc[:idx]
            if len(hist_hasta_idx) >= 4:
                pend_3d = (score_prov - hist_hasta_idx.iloc[-4]['regime']) / 3 if len(hist_hasta_idx) >= 4 else 0
                pend_5d = (score_prov - hist_hasta_idx.iloc[-6]['regime']) / 5 if len(hist_hasta_idx) >= 6 else 0
                pend_10d = (score_prov - hist_hasta_idx.iloc[-11]['regime']) / 10 if len(hist_hasta_idx) >= 11 else 0
            else:
                pend_3d = pend_5d = pend_10d = 0.0

            pend_3d = pend_3d if not np.isnan(pend_3d) else 0.0
            pend_5d = pend_5d if not np.isnan(pend_5d) else 0.0
            pend_10d = pend_10d if not np.isnan(pend_10d) else 0.0
            aceleracion = pend_3d - pend_10d

            # Motores mejorando (pendiente 5d positiva)
            motores_mejorando = 0
            if len(hist_hasta_idx) >= 6:
                for mod in motores_disponibles:
                    serie = hist_hasta_idx[mod]
                    if len(serie) >= 6:
                        pend_mod = (serie.iloc[-1] - serie.iloc[-6]) / 5
                        if pend_mod > 0.01:  # umbral mínimo
                            motores_mejorando += 1

            # --- 3. Clasificar fase del ciclo (7 fases) ---
            datos_para_fase = pd.Series({
                'score_global': score_prov,
                'stress': fila_actual.get('stress', 0),
                'bonds': fila_actual.get('bonds', 0),
                'liquidity': fila_actual.get('liquidity', 0),
                'regime': fila_actual.get('regime', 0),
                'leadership': fila_actual.get('leadership', 0),
                'geo': fila_actual.get('geo', 0),
                'pend_3d': pend_3d,
                'pend_5d': pend_5d,
                'pend_10d': pend_10d,
                'aceleracion': aceleracion,
                'motores_mejorando': motores_mejorando,
                'dispersion': 0  # aún no calculada
            })
            fase, desc = self.cycle_engine.clasificar(datos_para_fase, fila_anterior)
            fila_anterior = datos_para_fase.copy()  # guardar para próxima iteración

            # --- 4. Pesos dinámicos según fase ---
            pesos_fase = self.fase_weights.get(fase, self.fase_weights.get('NEUTRAL', {}))
            # Si faltan algunos motores en pesos_fase, usar los base
            pesos_dinamicos = {}
            for mod in motores_disponibles:
                if mod in pesos_fase:
                    pesos_dinamicos[mod] = pesos_fase[mod]
                else:
                    pesos_dinamicos[mod] = self.base_weights.get(mod, 0)

            # Renormalizar a suma 1
            total = sum(pesos_dinamicos.values())
            if total > 0:
                for mod in pesos_dinamicos:
                    pesos_dinamicos[mod] /= total

            # --- 5. Ajuste por correlación (PCA) ---
            if len(hist_hasta_idx) >= 30:
                # Usar último año (252 días) para PCA
                ventana_pca = hist_hasta_idx.iloc[-252:]
                pesos_pca = adjust_weights_pca(pesos_dinamicos, ventana_pca, n_components=None)
            else:
                pesos_pca = pesos_dinamicos

            # --- 6. Score global con pesos ajustados ---
            score_global = 0.0
            for mod, peso in pesos_pca.items():
                score_global += fila_actual[mod] * peso

            # --- 7. Dispersión (desviación estándar de motores seleccionados) ---
            valores_disp = [fila_actual[mod] for mod in self.disp_modules if mod in motores_disponibles]
            dispersion = np.std(valores_disp) if len(valores_disp) > 1 else 0.0
            dispersion = min(dispersion, self.dispersion_max)

            # --- 8. Penalizaciones ---
            stress = fila_actual.get('stress', 0)
            penalty_stress = max(0, stress) * self.stress_factor if stress < 0 else 0  # stress negativo malo
            breadth = fila_actual.get('breadth', 0)
            penalty_breadth = max(0, -breadth) * self.breadth_factor if breadth < 0 else 0

            # --- 9. Score coherente y exposure factor ---
            coherencia = np.exp(-dispersion)
            score_coherente = score_global * coherencia
            exposure_factor = coherencia - penalty_stress - penalty_breadth
            exposure_factor = np.clip(exposure_factor, 0, 1)

            # --- 10. Ciclo institucional (4 fases) ---
            ciclo_inst = ciclo_institucional({
                'score_global': score_global,
                'score_breadth': breadth,
                'score_stress': stress
            })

            # Guardar resultados
            resultados.loc[idx, 'score_global'] = score_global
            resultados.loc[idx, 'score_coherente'] = score_coherente
            resultados.loc[idx, 'exposure_factor'] = exposure_factor
            resultados.loc[idx, 'dispersion'] = dispersion
            resultados.loc[idx, 'penalty_stress'] = penalty_stress
            resultados.loc[idx, 'penalty_breadth'] = penalty_breadth
            resultados.loc[idx, 'fase_ciclo'] = fase
            resultados.loc[idx, 'ciclo_institucional'] = ciclo_inst
            resultados.loc[idx, 'pend_3d'] = pend_3d
            resultados.loc[idx, 'pend_5d'] = pend_5d
            resultados.loc[idx, 'pend_10d'] = pend_10d
            resultados.loc[idx, 'aceleracion'] = aceleracion
            resultados.loc[idx, 'motores_mejorando'] = motores_mejorando

        return resultados