"""
risk.py - Módulo de gestión de riesgo y capital
Toma el score global y aplica reglas de contingencia para determinar la exposición final.
Funciones:
- exposicion_base: asigna exposición según tramos de score.
- aplicar_cash_forzado: reduce exposición si VIX es alto.
- aplicar_dispersion: penaliza si la dispersión entre componentes es alta.
- penalizacion_liquidez: ajusta por spreads de los ETFs.
- aplicar_turnover: resta costes de comisión y rotación.
- control_volatilidad_subcartera: escala carteras para cumplir volatilidad objetivo.
- aplicar_reglas_riesgo: orquesta todas las anteriores y registra factores.
"""

import numpy as np
import pandas as pd
import yaml
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.cfg_exp = self.config['exposicion']

    def exposicion_base(self, score):
        """
        Calcula la exposición base a partir del score usando tramos lineales.
        Los tramos se definen en config.yaml como:
        base_tramos:
          - min: -1.0, exp: 0.0
          - min: -0.5, exp_min: 0.0, exp_max: 0.3
          - min: 0.0, exp_min: 0.3, exp_max: 0.7
          - min: 0.5, exp_min: 0.7, exp_max: 1.0
        Devuelve exposición base y metadatos del tramo.
        """
        tramos = sorted(self.cfg_exp['base_tramos'], key=lambda x: x['min'])
        # Si el score es menor que el mínimo del primer tramo, devolver exposición 0
        if score <= tramos[0]['min']:
            return 0.0, {'tramo': tramos[0].get('id', 'inferior'), 'min': tramos[0]['min'], 'exp': 0.0}
        # Recorrer tramos
        for i, tramo in enumerate(tramos):
            s_min = tramo['min']
            # Último tramo
            if i == len(tramos) - 1:
                # El último tramo debe tener exp fija o interpolación hasta 1.0
                exp = tramo.get('exp', tramo.get('exp_max', 1.0))
                return exp, {'tramo': tramo.get('id', 'superior'), 'min': s_min, 'exp': exp}
            s_max = tramos[i+1]['min']
            # Si el score está entre s_min y s_max
            if s_min < score <= s_max:
                exp_min = tramo.get('exp_min', tramo.get('exp', 0.0))
                exp_max = tramos[i+1].get('exp_min', tramos[i+1].get('exp', 1.0))
                # Interpolación lineal
                exp = exp_min + (score - s_min) * (exp_max - exp_min) / (s_max - s_min)
                return exp, {'tramo': tramo.get('id', i), 'min': s_min, 'max': s_max, 'exp_min': exp_min, 'exp_max': exp_max}
        # Si no encaja (score > último tramo), devolver exposición máxima del último tramo
        return tramos[-1].get('exp', 1.0), {'tramo': 'ultimo', 'exp': tramos[-1].get('exp', 1.0)}

    def aplicar_cash_forzado(self, exp, vix):
        """
        Reduce la exposición según niveles de VIX.
        Escalones definidos en config.yaml:
          - hasta: 1.2, factor: 0.8
          - hasta: 1.4, factor: 0.5
          - hasta: 1.5, factor: 0.2
          - default: 0.0
        """
        umbral = self.cfg_exp['cash_forzado']['VIX_umbral']
        if vix <= umbral:
            return exp, 1.0
        # Calcular ratio vix/umbral
        ratio = vix / umbral
        for escalon in self.cfg_exp['cash_forzado']['escalones']:
            if 'hasta' in escalon and escalon['hasta'] is not None:
                if ratio <= escalon['hasta']:
                    return exp * escalon['factor'], escalon['factor']
        # Si supera todos los escalones, factor 0.0
        return 0.0, 0.0

    def aplicar_dispersion(self, exp, dispersion):
        """
        Penaliza la exposición si la dispersión supera un umbral.
        Factor = 1 - 0.5 * min(1, dispersion / umbral)
        """
        umbral = self.cfg_exp['dispersion_reduccion']['umbral']
        max_pen = self.cfg_exp['dispersion_reduccion']['max_penalizacion']
        if dispersion > umbral:
            factor = 1 - max_pen * min(1, dispersion / umbral)
            return exp * factor, factor
        return exp, 1.0

    def penalizacion_liquidez(self, exp, spreads):
        """
        Ajusta la exposición por liquidez de los ETFs.
        spreads: diccionario con {ticker: spread_actual}
        Por cada ticker con spread > umbral, se multiplica por un factor.
        """
        umbral = self.cfg_exp['spread_alert']['umbral']
        slope = self.cfg_exp['spread_alert']['slope']
        min_factor = self.cfg_exp['spread_alert']['min_factor']
        factores = {}
        exp_ajustada = exp
        for ticker, spread in spreads.items():
            if spread > umbral:
                # factor = max(min_factor, 1 - (spread - umbral)/slope)
                factor = max(min_factor, 1 - (spread - umbral) / slope)
                exp_ajustada *= factor
                factores[ticker] = factor
            else:
                factores[ticker] = 1.0
        return exp_ajustada, factores

    def aplicar_turnover(self, exp, exp_prev, capital):
        """
        Aplica costes de comisión y rotación.
        turnover = abs(exp - exp_prev)
        coste_turnover = turnover * turnover_cost_factor
        comision_fija = commission / capital
        exp_ajustada = max(0, exp - coste_turnover - comision_fija)
        """
        turnover = abs(exp - exp_prev)
        coste_turnover = turnover * self.cfg_exp['turnover']['cost_factor']
        comision = self.cfg_exp['turnover']['commission'] / capital
        exp_ajustada = exp - coste_turnover - comision
        exp_ajustada = max(0.0, exp_ajustada)
        return exp_ajustada, turnover

    def control_volatilidad_subcartera(self, weights_dict, cov_matrices, target_vols):
        """
        Escala las carteras para cumplir con la volatilidad objetivo.
        weights_dict: dict con {nombre_cartera: vector de pesos}
        cov_matrices: dict con {nombre_cartera: matriz de covarianza}
        target_vols: dict con {nombre_cartera: volatilidad objetivo}
        Devuelve dict con pesos escalados y factores de escala.
        """
        escalas = {}
        pesos_escalados = {}
        for key, weights in weights_dict.items():
            cov = cov_matrices[key]
            # Calcular volatilidad actual
            port_vol = np.sqrt(weights.T @ cov @ weights)
            if port_vol <= target_vols[key]:
                escala = 1.0
            else:
                escala = target_vols[key] / port_vol
            pesos_escalados[key] = weights * escala
            escalas[key] = escala
        return pesos_escalados, escalas

    def aplicar_reglas_riesgo(self, score, contexto):
        # --- INICIO NUEVO: comprobar freeze ---
        if contexto.get('operations_freeze', False):
            logger.warning("Freeze activado por drift. Exposición forzada a 0.")
            return 0.0, {'freeze': True}
        # --- FIN NUEVO ---

        penalizaciones = {}

        # 1. Exposición base
        exp_base, info_tramo = self.exposicion_base(score)
        penalizaciones['tramo'] = info_tramo
        exp = exp_base

        # 2. Cash forzado por VIX
        exp, factor_vix = self.aplicar_cash_forzado(exp, contexto['vix'])
        penalizaciones['vix_factor'] = factor_vix

        # 3. Penalización por dispersión
        exp, factor_disp = self.aplicar_dispersion(exp, contexto['dispersion'])
        penalizaciones['dispersion_factor'] = factor_disp

        # 4. Penalización por liquidez
        exp, factores_spread = self.penalizacion_liquidez(exp, contexto['spreads'])
        penalizaciones['spread_factors'] = factores_spread

        # 5. Costes de turnover y comisión
        exp, turnover = self.aplicar_turnover(exp, contexto['exp_prev'], contexto['capital'])
        penalizaciones['turnover'] = turnover
        penalizaciones['coste_turnover'] = turnover * self.cfg_exp['turnover']['cost_factor']
        penalizaciones['comision'] = self.cfg_exp['turnover']['commission'] / contexto['capital']

        return exp, penalizaciones


if __name__ == "__main__":
    # Ejemplo de uso con datos simulados
    rm = RiskManager()

    # Contexto simulado (valores típicos)
    contexto = {
        'vix': 22.0,
        'dispersion': 0.35,
        'spreads': {'SPY': 0.001, 'EEM': 0.004, 'JNK': 0.006},  # algunos superan umbral 0.005
        'exp_prev': 0.5,
        'capital': 100000
    }

    score = 0.2  # score global de ejemplo

    exp_final, penalizaciones = rm.aplicar_reglas_riesgo(score, contexto)

    print("=== Resultado de gestión de riesgo ===")
    print(f"Score: {score}")
    print(f"Exposición base: {penalizaciones['tramo']}")
    print(f"Factor VIX: {penalizaciones['vix_factor']:.3f}")
    print(f"Factor dispersión: {penalizaciones['dispersion_factor']:.3f}")
    print("Factores de liquidez:")
    for ticker, factor in penalizaciones['spread_factors'].items():
        print(f"  {ticker}: {factor:.3f}")
    print(f"Turnover: {penalizaciones['turnover']:.3f}")
    print(f"Coste turnover: {penalizaciones['coste_turnover']:.4f}")
    print(f"Comisión: {penalizaciones['comision']:.6f}")
    print(f"\nExposición final: {exp_final:.4f}")