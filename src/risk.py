"""
risk.py - Módulo de gestión de riesgo y capital
Toma el score global y aplica reglas de contingencia para determinar la exposición final.
Funciones:
- exposicion_base: asigna exposición según tramos de score (interpolación lineal).
- aplicar_cash_forzado: reduce exposición si VIX es alto.
- aplicar_dispersion: penaliza si la dispersión entre módulos es alta.
- penalizacion_liquidez: ajusta por spreads de los ETFs.
- aplicar_turnover: resta costes de comisión y rotación.
- control_volatilidad_subcartera: escala carteras para cumplir volatilidad objetivo (opcional).
- aplicar_reglas_riesgo: orquesta todas las anteriores y registra factores.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import os
from datetime import datetime

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
          - min: -1.0, max: -0.5, exp_min: 0.0, exp_max: 0.0
          - min: -0.5, max: 0.0,  exp_min: 0.0, exp_max: 0.25
          - min: 0.0,  max: 0.5,  exp_min: 0.25, exp_max: 0.75
          - min: 0.5,  max: 1.0,  exp_min: 0.75, exp_max: 1.0
        Devuelve exposición base y metadatos del tramo.
        """
        tramos = sorted(self.cfg_exp['base_tramos'], key=lambda x: x['min'])
        # Si el score es menor que el mínimo del primer tramo, devolver exposición mínima
        if score <= tramos[0]['min']:
            return 0.0, {'tramo': 'inferior', 'min': tramos[0]['min'], 'exp': 0.0}
        # Recorrer tramos
        for i, tramo in enumerate(tramos):
            s_min = tramo['min']
            s_max = tramo.get('max', 1.0)
            if s_min < score <= s_max:
                exp_min = tramo.get('exp_min', tramo.get('exp', 0.0))
                exp_max = tramo.get('exp_max', exp_min)
                # Interpolación lineal
                exp = exp_min + (score - s_min) * (exp_max - exp_min) / (s_max - s_min)
                return exp, {'tramo': tramo.get('id', i), 'min': s_min, 'max': s_max}
        # Si no encaja, devolver exposición máxima del último tramo
        ultimo = tramos[-1]
        exp_max = ultimo.get('exp_max', ultimo.get('exp', 1.0))
        return exp_max, {'tramo': 'superior', 'exp': exp_max}

    def aplicar_cash_forzado(self, exp, vix):
        """
        Reduce la exposición según niveles de VIX.
        Escalones definidos en config.yaml:
          vix_umbrales: [20, 25, 30]
          vix_penalizaciones: [0.0, 0.25, 0.5]
        """
        umbrales = self.cfg_exp.get('vix_umbrales', [])
        penalizaciones = self.cfg_exp.get('vix_penalizaciones', [])
        for umbral, penal in zip(umbrales, penalizaciones):
            if vix >= umbral:
                return exp * (1 - penal), penal
        return exp, 0.0

    def aplicar_dispersion(self, exp, dispersion):
        """
        Penaliza la exposición si la dispersión supera un umbral.
        Factor = 1 - max_penal * min(1, dispersion / umbral)
        """
        umbral = self.cfg_exp.get('dispersion_max', 0.8)
        max_penal = self.cfg_exp.get('dispersion_penal_factor', 0.5)
        if dispersion > umbral:
            factor = 1 - max_penal * min(1, dispersion / umbral)
            return exp * factor, factor
        return exp, 1.0

    def penalizacion_liquidez(self, exp, spreads):
        """
        Ajusta la exposición por liquidez de los ETFs.
        spreads: diccionario con {ticker: spread_actual}
        Por cada ticker con spread > umbral, se multiplica por un factor.
        """
        umbral = self.cfg_exp.get('spread_umbral', 0.005)
        slope = self.cfg_exp.get('spread_slope', 0.01)
        min_factor = self.cfg_exp.get('spread_min_factor', 0.5)
        factores = {}
        exp_ajustada = exp
        for ticker, spread in spreads.items():
            if spread > umbral:
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
        cost_factor = self.cfg_exp.get('turnover_cost_factor', 0.001)
        comision = self.cfg_exp.get('comision_fija', 5.0)
        turnover = abs(exp - exp_prev)
        coste_turnover = turnover * cost_factor
        exp_ajustada = exp - coste_turnover - comision / capital
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
            port_vol = np.sqrt(weights.T @ cov @ weights)
            if port_vol <= target_vols[key]:
                escala = 1.0
            else:
                escala = target_vols[key] / port_vol
            pesos_escalados[key] = weights * escala
            escalas[key] = escala
        return pesos_escalados, escalas

    def aplicar_reglas_riesgo(self, score, contexto):
        """
        Función principal que aplica todas las reglas en orden y devuelve la exposición final
        junto con un diccionario de factores de penalización.
        contexto: dict con al menos:
            - vix: valor actual del VIX
            - dispersion: valor de dispersión (de scoring)
            - spreads: dict con spreads actuales de los ETFs
            - exp_prev: exposición anterior
            - capital: capital actual
        """
        penalizaciones = {}

        # --- 1. Exposición base ---
        exp_base, info_tramo = self.exposicion_base(score)
        penalizaciones['tramo'] = info_tramo
        exp = exp_base

        # --- 2. Cash forzado por VIX ---
        exp, factor_vix = self.aplicar_cash_forzado(exp, contexto['vix'])
        penalizaciones['vix_factor'] = factor_vix

        # --- 3. Penalización por dispersión ---
        exp, factor_disp = self.aplicar_dispersion(exp, contexto['dispersion'])
        penalizaciones['dispersion_factor'] = factor_disp

        # --- 4. Penalización por liquidez ---
        exp, factores_spread = self.penalizacion_liquidez(exp, contexto['spreads'])
        penalizaciones['spread_factors'] = factores_spread

        # --- 5. Costes de turnover y comisión ---
        exp, turnover = self.aplicar_turnover(exp, contexto['exp_prev'], contexto['capital'])
        penalizaciones['turnover'] = turnover
        penalizaciones['coste_turnover'] = turnover * self.cfg_exp.get('turnover_cost_factor', 0.001)
        penalizaciones['comision'] = self.cfg_exp.get('comision_fija', 5.0) / contexto['capital']

        # --- 6. (Opcional) Aquí se podría añadir control de volatilidad por subcartera ---
        # No se implementa en esta versión básica, pero está la función disponible.

        # --- 7. Logging detallado de la evolución (opcional, se puede llamar desde fuera) ---
        log_riesgo = {
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'score': score,
            'exp_base': exp_base,
            'exp_vix': exp_base * (1 - factor_vix) if factor_vix > 0 else exp_base,
            'exp_disp': exp,
            'exp_final': exp,
            'vix': contexto['vix'],
            'dispersion': contexto['dispersion'],
            'turnover': turnover,
            'capital': contexto['capital']
        }
        # Se puede añadir a una lista para luego guardar en Parquet (lo hará run_radar.py)

        return exp, penalizaciones, log_riesgo


if __name__ == "__main__":
    # Ejemplo de uso con datos simulados
    rm = RiskManager()

    # Contexto simulado (valores típicos)
    contexto = {
        'vix': 22.0,
        'dispersion': 0.35,
        'spreads': {'SPY': 0.001, 'EEM': 0.004, 'JNK': 0.006},
        'exp_prev': 0.5,
        'capital': 100000
    }

    score = 0.2

    exp_final, penalizaciones, log = rm.aplicar_reglas_riesgo(score, contexto)

    print("=== Resultado de gestión de riesgo ===")
    print(f"Score: {score}")
    print(f"Exposición base: {log['exp_base']:.4f}")
    print(f"Factor VIX: {penalizaciones['vix_factor']:.3f}")
    print(f"Factor dispersión: {penalizaciones['dispersion_factor']:.3f}")
    print("Factores de liquidez:")
    for ticker, factor in penalizaciones['spread_factors'].items():
        print(f"  {ticker}: {factor:.3f}")
    print(f"Turnover: {penalizaciones['turnover']:.3f}")
    print(f"Coste turnover: {penalizaciones['coste_turnover']:.4f}")
    print(f"Comisión: {penalizaciones['comision']:.6f}")
    print(f"\nExposición final: {exp_final:.4f}")
    print("\nLog de evolución:")
    print(log)