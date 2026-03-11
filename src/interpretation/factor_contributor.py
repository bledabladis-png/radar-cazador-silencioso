# src/interpretation/factor_contributor.py

import pandas as pd
import yaml
import os

class FactorContributor:
    """
    Calcula la contribución de cada motor al score global,
    utilizando los pesos de la fase actual o pesos base.
    """
    def __init__(self, config_path='config/config.yaml', weights_path='config/regime_weights.yaml'):
        # Cargar pesos base
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.base_weights = self.config.get('weights', {}).get('base', {})
        
        # Cargar pesos por fase
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.phase_weights = yaml.safe_load(f)

    def get_weights_for_phase(self, phase):
        """
        Devuelve los pesos para una fase dada.
        Si la fase no existe, usa los pesos base normalizados.
        """
        weights = self.phase_weights.get(phase, self.base_weights)
        # Normalizar a suma 1
        total = sum(weights.values())
        if total > 0:
            return {k: v/total for k, v in weights.items()}
        else:
            return self.base_weights

    def compute_contributions(self, snapshot, phase):
        """
        snapshot: dict con los scores de los motores (ej. 'regime', 'leadership', ...)
        phase: fase del ciclo actual
        Retorna un dict con:
            - contributions: dict {motor: contribución}
            - top_positive: lista de (motor, contribución) ordenada descendente (positivas)
            - top_negative: lista de (motor, contribución) ordenada ascendente (negativas)
        """
        weights = self.get_weights_for_phase(phase)
        contributions = {}
        for motor, score in snapshot.items():
            if motor in weights:
                contributions[motor] = score * weights[motor]
            else:
                contributions[motor] = 0.0

        # Separar positivas y negativas
        pos = [(m, c) for m, c in contributions.items() if c > 0]
        neg = [(m, c) for m, c in contributions.items() if c < 0]
        pos.sort(key=lambda x: x[1], reverse=True)
        neg.sort(key=lambda x: x[1])  # más negativo primero

        return {
            'contributions': contributions,
            'top_positive': pos[:3],  # top 3
            'top_negative': neg[:3]   # top 3
        }

    def format_contributions(self, contrib_result):
        """
        Devuelve un texto legible con las principales contribuciones.
        """
        lines = []
        if contrib_result['top_positive']:
            lines.append("**Impulsores positivos:**")
            for motor, val in contrib_result['top_positive']:
                lines.append(f"- {motor.capitalize()}: +{val:.3f}")
        if contrib_result['top_negative']:
            lines.append("**Factores negativos:**")
            for motor, val in contrib_result['top_negative']:
                lines.append(f"- {motor.capitalize()}: {val:.3f}")
        return "\n".join(lines)
