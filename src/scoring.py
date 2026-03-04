"""
scoring.py - Cálculo del score global y métricas de coherencia
Combina los scores de los cuatro motores con ponderación dinámica.
Calcula:
- Score global
- Dispersión ponderada
- Señal de acumulación
- Ajuste de pesos por VIX
"""

import pandas as pd
import numpy as np
import yaml
import logging

logger = logging.getLogger(__name__)

class ScoringEngine:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Pesos base de los módulos (de config.yaml)
        self.base_weights = self.config['weights']['base']
        # Parámetros de dispersión
        self.disp_weights = self.config['dispersion']['weights']
        self.disp_riesgo_alto = self.config['dispersion']['riesgo_alto']
        self.disp_acumulacion_alta = self.config['dispersion']['acumulacion_alta']
        self.disp_regimen_fuerte_disp = self.config['dispersion']['regimen_fuerte_disp']
        self.disp_regimen_fuerte_score = self.config['dispersion']['regimen_fuerte_score']
        # Señal de acumulación
        self.accum_weights = self.config['accumulation_signal']['weights']
        self.small_threshold = self.config['accumulation_signal']['small_threshold']
        # Ajuste dinámico por VIX
        self.vix_ajuste = self.config['weights']['dynamic']['vix_ajuste']
        self.vix_umbral = self.config['weights']['dynamic']['vix_umbral']
        self.vix_factor = self.config['weights']['dynamic']['vix_factor']

    def calcular_score_global(self, df, scores_dict):
        """
        scores_dict debe contener las series de los cuatro módulos con los nombres:
        'regime', 'leadership', 'geo', 'stress'
        Devuelve un DataFrame con score_global, dispersion, accumulation_signal y otros.
        """
        # Asegurar que todos los índices están alineados
        common_index = df.index
        for key in scores_dict:
            scores_dict[key] = scores_dict[key].reindex(common_index).fillna(0)

        # Obtener pesos base
        w_reg = self.base_weights['regime']
        w_lead = self.base_weights['leadership']
        w_geo = self.base_weights['geo']
        w_stress = self.base_weights['stress']

        # Ajuste dinámico por VIX (si está activado y la columna VIX existe)
        if self.vix_ajuste and '^VIX' in df.columns:
            vix = df['^VIX'].ffill().fillna(20)  # valor por defecto
            # Aplicar factor exponencial si VIX > umbral
            factor = np.where(vix > self.vix_umbral,
                              np.exp(-self.vix_factor * (vix - self.vix_umbral)),
                              1.0)
            # Los pesos se multiplican por factor y luego se renormalizan
            w_reg_adj = w_reg * factor
            w_lead_adj = w_lead * factor
            w_geo_adj = w_geo * factor
            w_stress_adj = w_stress * factor
            total = w_reg_adj + w_lead_adj + w_geo_adj + w_stress_adj
            # Evitar división por cero
            total = np.where(total == 0, 1, total)
            w_reg_final = w_reg_adj / total
            w_lead_final = w_lead_adj / total
            w_geo_final = w_geo_adj / total
            w_stress_final = w_stress_adj / total
        else:
            w_reg_final = w_reg
            w_lead_final = w_lead
            w_geo_final = w_geo
            w_stress_final = w_stress

        # Calcular score global ponderado
        score_global = (w_reg_final * scores_dict['regime'] +
                        w_lead_final * scores_dict['leadership'] +
                        w_geo_final * scores_dict['geo'] +
                        w_stress_final * scores_dict['stress'])

        # Dispersión ponderada (usando pesos fijos para los componentes principales)
        # Según documentación: [0.4, 0.3, 0.15] para regime, leadership, geo (sin stress)
        scores_array = np.column_stack([scores_dict['regime'],
                                        scores_dict['leadership'],
                                        scores_dict['geo']])
        media_pond = np.average(scores_array, weights=self.disp_weights, axis=1)
        var_pond = np.average((scores_array.T - media_pond).T ** 2,
                              weights=self.disp_weights, axis=1)
        dispersion = np.sqrt(var_pond)

        return {
            'score_global': score_global,
            'dispersion': dispersion,
            'pesos': {
                'regime': w_reg_final,
                'leadership': w_lead_final,
                'geo': w_geo_final,
                'stress': w_stress_final
            }
        }

    def calcular_todo(self, df, regime_df, leadership_df, geo_df, stress_df):
        """
        Unifica todos los scores y calcula el global y la señal de acumulación.
        """
        # Construir diccionario de scores principales
        scores_dict = {
            'regime': regime_df['score_regime'],
            'leadership': leadership_df['score_leadership'],
            'geo': geo_df['score_geo'],
            'stress': stress_df['score_stress']
        }

        # Calcular global y dispersión (devuelve dict con 'score_global', 'dispersion', 'pesos')
        resultados_dict = self.calcular_score_global(df, scores_dict)

        # Extraer los pesos para usarlos después (no se incluirán en el DataFrame)
        pesos = resultados_dict.pop('pesos')  # elimina 'pesos' del dict y lo guarda aparte

        # Añadir componentes individuales al dict
        resultados_dict['score_regime'] = regime_df['score_regime']
        resultados_dict['score_leadership'] = leadership_df['score_leadership']
        resultados_dict['score_geo'] = geo_df['score_geo']
        resultados_dict['score_stress'] = stress_df['score_stress']
        resultados_dict['score_tendencia'] = regime_df['score_tendencia']
        resultados_dict['score_credito'] = regime_df['score_credito']
        resultados_dict['score_curva'] = regime_df['score_curva']
        resultados_dict['score_small'] = leadership_df['score_small']
        resultados_dict['score_cyclical'] = leadership_df['score_cyclical']
        resultados_dict['score_em'] = geo_df['score_em']
        resultados_dict['score_dm'] = geo_df['score_dm']
        resultados_dict['alert_vix'] = stress_df['alert_vix']
        resultados_dict['compresion_flag'] = stress_df['compresion_flag']
        resultados_dict['drawdown_credit_penalty'] = stress_df['drawdown_credit_penalty']

        # Señal de acumulación
        cond1 = (regime_df['score_curva'] > 0).astype(float)
        cond2 = (leadership_df['score_small'] > self.small_threshold).astype(float)
        cond3 = stress_df['compresion_flag']
        cond4 = regime_df['flag_drawdown']
        cond5 = ((resultados_dict['dispersion'] > self.disp_acumulacion_alta) &
                 (resultados_dict['score_global'] > 0)).astype(float)

        accum_signals = np.column_stack([cond1, cond2, cond3, cond4, cond5])
        accum_score = np.dot(accum_signals, self.accum_weights)
        resultados_dict['accumulation_signal'] = accum_score

        # Crear DataFrame a partir del dict (ahora ya no contiene 'pesos')
        resultados_df = pd.DataFrame(resultados_dict)

        # Adjuntar los pesos como atributo (opcional) o devolverlos aparte
        resultados_df.attrs['pesos'] = pesos

        return resultados_df


if __name__ == "__main__":
    from data_layer import DataLayer
    from regime_engine import RegimeEngine
    from leadership_engine import LeadershipEngine
    from geographic_engine import GeographicEngine
    from stress_engine import StressEngine

    # Cargar datos
    dl = DataLayer()
    df = dl.load_latest()

    # Calcular cada motor
    regime = RegimeEngine()
    regime_df = regime.calcular_todo(df)

    leadership = LeadershipEngine()
    leadership_df = leadership.calcular_todo(df)

    geo = GeographicEngine()
    geo_df = geo.calcular_todo(df)

    stress = StressEngine()
    stress_df = stress.calcular_stress(df)

    # Combinar
    scoring = ScoringEngine()
    resultados_df = scoring.calcular_todo(df, regime_df, leadership_df, geo_df, stress_df)

    print("Últimos 5 días de scores globales:")
    print(resultados_df[['score_global', 'dispersion', 'accumulation_signal']].tail())

    # Obtener los pesos del último día desde el atributo
    pesos_ultimo_dia = resultados_df.attrs['pesos']
    # Los pesos son arrays de numpy, usamos indexación normal [-1]
    pesos_ultimo_valor = {k: v[-1] for k, v in pesos_ultimo_dia.items()}
    print("\nPesos del último día:")
    print(pesos_ultimo_valor)