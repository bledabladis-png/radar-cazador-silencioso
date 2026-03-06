"""
risk.py - Gestión de riesgo y capital (versión simplificada)
Toma el score global y el exposure_factor (de scoring) y aplica reglas operativas:
- Exposición base por tramos
- Multiplicación por exposure_factor
- Cash forzado por VIX/ATR
- Penalización por liquidez (spreads)
- Costes de turnover y comisión
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

    def aplicar_cash_forzado(self, exp, vix_atr_ratio):
        """
        Reduce la exposición según el ratio VIX/ATR20.
        Escalones definidos en config.yaml en 'cash_forzado':
          - VIX_umbral: nivel a partir del cual se activa (por defecto 30)
          - escalones: lista de diccionarios con 'hasta' (valor del ratio) y 'factor' (multiplicador)
        Si el ratio supera el umbral, se aplica el factor correspondiente al escalón.
        """
        cfg_cash = self.cfg_exp.get('cash_forzado', {})
        umbral = cfg_cash.get('VIX_umbral', 30)  # umbral en niveles de VIX (no ratio)
        # Nota: en tu configuración, el umbral es sobre VIX, no sobre ratio. Pero aquí usamos vix_atr_ratio.
        # Para adaptarnos, asumimos que el umbral también aplica al ratio (o debería ser otro parámetro).
        # Podríamos tener un umbral específico para el ratio, pero por ahora usamos el mismo.
        # Si no hay escalones, no se aplica penalización.
        escalones = cfg_cash.get('escalones', [])
        if not escalones or vix_atr_ratio <= umbral:
            return exp, 1.0

        # Buscar el escalón correspondiente
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
        spreads: diccionario con {ticker: spread_actual}
        Parámetros de 'spread_alert' en config:
          - umbral: spread mínimo para penalizar
          - slope: pendiente para la reducción lineal
          - min_factor: factor mínimo permitido
        Por cada ticker con spread > umbral, se multiplica por un factor decreciente.
        """
        cfg_spread = self.cfg_exp.get('spread_alert', {})
        umbral = cfg_spread.get('umbral', 0.005)
        slope = cfg_spread.get('slope', 0.01)
        min_factor = cfg_spread.get('min_factor', 0.5)
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
        Parámetros de 'turnover' en config:
          - cost_factor: coste por unidad de turnover
          - commission: comisión fija en unidades monetarias
        """
        cfg_turn = self.cfg_exp.get('turnover', {})
        cost_factor = cfg_turn.get('cost_factor', 0.001)
        comision = cfg_turn.get('commission', 5.0)
        turnover = abs(exp - exp_prev)
        coste_turnover = turnover * cost_factor
        exp_ajustada = exp - coste_turnover - comision / capital
        exp_ajustada = max(0.0, exp_ajustada)
        return exp_ajustada, turnover, coste_turnover

    def aplicar_reglas_riesgo(self, score, contexto):
        """
        Función principal que aplica todas las reglas en orden.
        contexto: dict con al menos:
            - exposure_factor: factor de exposición calculado por scoring (0-1)
            - vix_atr_ratio: ratio VIX/ATR20 (para cash forzado)
            - spreads: dict con spreads actuales de los ETFs
            - exp_prev: exposición anterior
            - capital: capital actual
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

        # --- 5. Costes de turnover y comisión ---
        exp_prev = contexto.get('exp_prev', 0)
        capital = contexto.get('capital', 100000)
        exp, turnover, coste_turnover = self.aplicar_turnover(exp, exp_prev, capital)
        penalizaciones['turnover'] = turnover
        penalizaciones['coste_turnover'] = coste_turnover
        penalizaciones['comision'] = self.cfg_exp.get('turnover', {}).get('commission', 5.0) / capital

        # --- 6. Logging ---
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
        # Nota: podríamos registrar más detalles si se desea

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
        'capital': 100000
    }

    score = 0.2

    exp_final, penalizaciones, log = rm.aplicar_reglas_riesgo(score, contexto)

    print("=== Resultado de gestión de riesgo ===")
    print(f"Score: {score}")
    print(f"Exposición base: {log['exp_base']:.4f}")
    print(f"Factor exposure (de scoring): {penalizaciones['exposure_factor']:.3f}")
    print(f"Factor cash: {penalizaciones['cash_factor']:.3f}")
    print("Factores de liquidez:")
    for ticker, factor in penalizaciones['spread_factors'].items():
        print(f"  {ticker}: {factor:.3f}")
    print(f"Turnover: {penalizaciones['turnover']:.3f}")
    print(f"Coste turnover: {penalizaciones['coste_turnover']:.4f}")
    print(f"Comisión: {penalizaciones['comision']:.6f}")
    print(f"\nExposición final: {exp_final:.4f}")
    print("\nLog de evolución:")
    print(log)