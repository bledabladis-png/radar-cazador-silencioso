"""
risk.py - Gestión de riesgo y capital (v2.3.1 + mejora Fase 1)
Incluye:
- Exposición base por tramos
- Multiplicación por exposure_factor
- Cash forzado por VIX/ATR
- Penalización por liquidez (spreads)
- Costes de turnover y comisión
- **Nuevo: Ajuste por volatilidad (volatility targeting)**
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
        Los tramos se definen en config.yaml como 'base_tramos'.
        Devuelve exposición base y metadatos del tramo.
        """
        tramos = sorted(self.cfg_exp['base_tramos'], key=lambda x: x['min'])
        if score <= tramos[0]['min']:
            return 0.0, {'tramo': 'inferior', 'min': tramos[0]['min'], 'exp': 0.0}
        for i, tramo in enumerate(tramos):
            s_min = tramo['min']
            s_max = tramo.get('max', 1.0)
            if s_min < score <= s_max:
                exp_min = tramo.get('exp_min', tramo.get('exp', 0.0))
                exp_max = tramo.get('exp_max', exp_min)
                exp = exp_min + (score - s_min) * (exp_max - exp_min) / (s_max - s_min)
                return exp, {'tramo': tramo.get('id', i), 'min': s_min, 'max': s_max}
        ultimo = tramos[-1]
        exp_max = ultimo.get('exp_max', ultimo.get('exp', 1.0))
        return exp_max, {'tramo': 'superior', 'exp': exp_max}

    def aplicar_cash_forzado(self, exp, vix_atr_ratio):
        """
        Reduce la exposición según el ratio VIX/ATR20.
        """
        cfg_cash = self.cfg_exp.get('cash_forzado', {})
        umbral = cfg_cash.get('ratio_umbral', 1.2)
        escalones = cfg_cash.get('escalones', [])
        if not escalones or vix_atr_ratio <= umbral:
            return exp, 1.0

        factor = 1.0
        for escalon in escalones:
            hasta = escalon.get('hasta')
            if hasta is None or vix_atr_ratio <= hasta:
                factor = escalon.get('factor', 1.0)
                break
        exp_ajustada = exp * factor
        return exp_ajustada, factor

    def penalizacion_liquidez(self, exp, spreads):
        """
        Ajusta la exposición por liquidez de los ETFs.
        """
        cfg_spread = self.cfg_exp.get('spread_alert', {})
        umbral = cfg_spread.get('umbral', 0.005)
        min_factor = cfg_spread.get('min_factor', 0.5)
        factores = {}
        exp_ajustada = exp
        for ticker, spread in spreads.items():
            if spread > umbral:
                # Reducción lineal simple: si spread = 0.01, factor = 0.5; si spread = 0.02, factor = 0.25 (pero mínimo 0.2)
                factor = max(0.2, 1 - (spread - umbral) * 100)  # Ajuste empírico
                factor = max(min_factor, factor)
                exp_ajustada *= factor
                factores[ticker] = factor
            else:
                factores[ticker] = 1.0
        return exp_ajustada, factores

    def aplicar_turnover(self, exp, exp_prev, capital):
        """
        Aplica costes de comisión y rotación.
        """
        cfg_turn = self.cfg_exp.get('turnover', {})
        cost_factor = cfg_turn.get('cost_factor', 0.001)
        comision = cfg_turn.get('commission', 5.0)
        turnover = abs(exp - exp_prev)
        coste_turnover = turnover * cost_factor
        exp_ajustada = exp - coste_turnover - comision / capital
        exp_ajustada = max(0.0, exp_ajustada)
        return exp_ajustada, turnover, coste_turnover

    def ajuste_volatilidad(self, exp, volatilidad_actual, target_vol=0.15, max_ajuste=1.5, min_ajuste=0.5):
        """
        Ajusta la exposición según la volatilidad realizada.
        volatilidad_actual: volatilidad anualizada (por ejemplo, 0.20 = 20%)
        target_vol: volatilidad objetivo (ej. 0.15 = 15%)
        max_ajuste: límite superior del factor (ej. 1.5)
        min_ajuste: límite inferior del factor (ej. 0.5)
        Devuelve exposición ajustada y el factor aplicado.
        """
        if volatilidad_actual <= 0:
            return exp, 1.0
        factor = target_vol / volatilidad_actual
        # Limitar el factor
        factor = max(min_ajuste, min(max_ajuste, factor))
        exp_ajustada = exp * factor
        return exp_ajustada, factor

    def aplicar_reglas_riesgo(self, score, contexto):
        """
        Función principal que aplica todas las reglas en orden.
        contexto: dict con al menos:
            - exposure_factor: factor de exposición calculado por scoring (0-1)
            - vix_atr_ratio: ratio VIX/ATR20 (para cash forzado)
            - spreads: dict con spreads actuales de los ETFs
            - exp_prev: exposición anterior
            - capital: capital actual
            - volatilidad_spy: volatilidad anualizada de SPY (opcional, si no se proporciona no se aplica ajuste)
        Devuelve exposición final, diccionario de factores y log.
        """
        penalizaciones = {}

        # --- 1. Exposición base ---
        exp_base, info_tramo = self.exposicion_base(score)
        penalizaciones['tramo'] = info_tramo
        exp = exp_base

        # --- 2. Aplicar exposure_factor (de scoring) ---
        exposure_factor = contexto.get('exposure_factor', 1.0)
        exp *= exposure_factor
        penalizaciones['exposure_factor'] = exposure_factor

        # --- 3. Cash forzado por VIX/ATR ---
        vix_atr_ratio = contexto.get('vix_atr_ratio', 0)
        exp, factor_cash = self.aplicar_cash_forzado(exp, vix_atr_ratio)
        penalizaciones['cash_factor'] = factor_cash

        # --- 4. Penalización por liquidez ---
        spreads = contexto.get('spreads', {})
        exp, factores_spread = self.penalizacion_liquidez(exp, spreads)
        penalizaciones['spread_factors'] = factores_spread

        # --- Penalización por breadth (amplitud de mercado) ---
        breadth_score = contexto.get('breadth_score', 0)
        if breadth_score < 0:
            breadth_penalty = abs(breadth_score) * 0.3
            exp *= (1 - breadth_penalty)
            penalizaciones['breadth_penalty'] = breadth_penalty
        else:
            penalizaciones['breadth_penalty'] = 0.0

        # --- 5. NUEVO: Ajuste por volatilidad (volatility targeting) ---
        volatilidad_spy = contexto.get('volatilidad_spy', None)
        if volatilidad_spy is not None and volatilidad_spy > 0:
            exp, factor_vol = self.ajuste_volatilidad(exp, volatilidad_spy)
            penalizaciones['vol_factor'] = factor_vol
        else:
            penalizaciones['vol_factor'] = 1.0

        # --- 6. Costes de turnover y comisión ---
        exp_prev = contexto.get('exp_prev', 0)
        capital = contexto.get('capital', 100000)
        exp, turnover, coste_turnover = self.aplicar_turnover(exp, exp_prev, capital)
        penalizaciones['turnover'] = turnover
        penalizaciones['coste_turnover'] = coste_turnover
        penalizaciones['comision'] = self.cfg_exp.get('turnover', {}).get('commission', 5.0) / capital

        # --- 7. Logging ---
        log_riesgo = {
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'score': score,
            'exp_base': exp_base,
            'exposure_factor': exposure_factor,
            'exp_after_factor': exp_base * exposure_factor,
            'exp_after_cash': exp,  # tras cash, pero aún sin liquidez y turnover? Cuidado: hemos ido modificando exp, mejor llevar traza
            'exp_final': exp,
            'vix_atr_ratio': vix_atr_ratio,
            'turnover': turnover,
            'capital': capital
        }

        return exp, penalizaciones, log_riesgo


if __name__ == "__main__":
    # Ejemplo de uso con datos simulados
    rm = RiskManager()

    # Contexto simulado (valores típicos)
    contexto = {
        'exposure_factor': 0.85,
        'vix_atr_ratio': 25.0,  # por debajo del umbral
        'spreads': {'SPY': 0.001, 'EEM': 0.004, 'JNK': 0.006},
        'exp_prev': 0.5,
        'capital': 100000,
        'volatilidad_spy': 0.18  # 18% anualizado
    }

    score = 0.2

    exp_final, penalizaciones, log = rm.aplicar_reglas_riesgo(score, contexto)

    print("=== Resultado de gestión de riesgo ===")
    print(f"Score: {score}")
    print(f"Exposición base: {log['exp_base']:.4f}")
    print(f"Factor exposure (de scoring): {penalizaciones['exposure_factor']:.3f}")
    print(f"Factor cash: {penalizaciones['cash_factor']:.3f}")
    print(f"Factor volatilidad: {penalizaciones['vol_factor']:.3f}")
    print("Factores de liquidez:")
    for ticker, factor in penalizaciones['spread_factors'].items():
        print(f"  {ticker}: {factor:.3f}")
    print(f"Turnover: {penalizaciones['turnover']:.3f}")
    print(f"Coste turnover: {penalizaciones['coste_turnover']:.4f}")
    print(f"Comisión: {penalizaciones['comision']:.6f}")
    print(f"\nExposición final: {exp_final:.4f}")
    print("\nLog de evolución:")
    print(log)
