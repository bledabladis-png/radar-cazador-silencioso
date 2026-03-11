# src/visualization/risk_map.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_risk_map(score, stress, ax=None):
    """
    Genera un mapa de riesgo 2x2 con ejes:
    - X: score_global (flujo)
    - Y: stress (invertido: valores más negativos = más estrés)
    Cuadrantes:
        Expansión (score > 0, stress > -0.2)
        Euforia   (score > 0, stress < -0.2)
        Crisis    (score < 0, stress < -0.2)
        Defensivo (score < 0, stress > -0.2)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Dibujar ejes
    ax.axhline(y=-0.2, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Colorear cuadrantes
    ax.fill_between([-1, 0], -1, -0.2, color='red', alpha=0.1, label='Crisis')
    ax.fill_between([-1, 0], -0.2, 1, color='orange', alpha=0.1, label='Defensivo')
    ax.fill_between([0, 1], -0.2, 1, color='green', alpha=0.1, label='Expansión')
    ax.fill_between([0, 1], -1, -0.2, color='blue', alpha=0.1, label='Euforia')

    # Punto actual
    ax.scatter(score, stress, color='black', s=100, zorder=5, edgecolors='white', linewidth=2, label='Actual')

    # Etiquetas de cuadrantes
    ax.text(0.5, 0.5, 'Expansión', ha='center', va='center', fontsize=10, alpha=0.5)
    ax.text(0.5, -0.6, 'Euforia', ha='center', va='center', fontsize=10, alpha=0.5)
    ax.text(-0.5, 0.5, 'Defensivo', ha='center', va='center', fontsize=10, alpha=0.5)
    ax.text(-0.5, -0.6, 'Crisis', ha='center', va='center', fontsize=10, alpha=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Score Global (flujo)')
    ax.set_ylabel('Estrés (invertido: más negativo = más estrés)')
    ax.set_title('Mapa de Riesgo')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

    return fig

def get_risk_quadrant(score, stress):
    """Devuelve el nombre del cuadrante actual."""
    if score > 0:
        if stress > -0.2:
            return "Expansión"
        else:
            return "Euforia"
    else:
        if stress > -0.2:
            return "Defensivo"
        else:
            return "Crisis"
