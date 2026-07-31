# -*- coding: utf-8 -*-
"""
signal_agreement.py -- Signal Agreement v1.2 (con Conviction)
Calcula acuerdo direccional y conviccion de las senales alineadas.
"""
import numpy as np

def compute_signal_conviction(signals, direction):
    """Calcula la intensidad media de las senales que apuntan en la direccion dominante."""
    valid = {k: v for k, v in signals.items() if v is not None and np.isfinite(v)}
    if not valid:
        return 0.0
    if direction == 'BULLISH':
        aligned = [v for v in valid.values() if v > 0]
    elif direction == 'BEARISH':
        aligned = [v for v in valid.values() if v < 0]
    else:
        return 0.0
    if not aligned:
        return 0.0
    mean_intensity = np.mean(np.abs(aligned))
    return float(np.tanh(mean_intensity * 3))

def compute_signal_agreement(signals):
    """
    signals: dict con valores numericos (>0 alcista, <0 bajista).
    Retorna dict con:
      - agreement: proporcion de senales alineadas (0.0 a 1.0)
      - direction: 'BULLISH', 'BEARISH' o 'MIXED'
      - display: string con porcentaje, direccion y conviccion
      - conviction: valor de conviccion [-1, +1]
    """
    if not signals:
        return {'agreement': 0.5, 'direction': 'MIXED', 'display': '50% MIXED', 'conviction': 0.0}

    valid = {k: v for k, v in signals.items() if v is not None and np.isfinite(v)}
    if not valid:
        return {'agreement': 0.5, 'direction': 'MIXED', 'display': '50% MIXED', 'conviction': 0.0}

    positive = sum(1 for v in valid.values() if v > 0)
    negative = sum(1 for v in valid.values() if v < 0)
    total = len(valid)

    dominant = max(positive, negative)
    agreement = dominant / total if total > 0 else 0.5

    if positive > negative:
        direction = 'BULLISH'
    elif negative > positive:
        direction = 'BEARISH'
    else:
        direction = 'MIXED'

    # Calcular conviccion
    conviction = compute_signal_conviction(valid, direction)

    # Construir display con conviccion
    display = f"{agreement:.0%} {direction} (Conv: {conviction:+.2f})"

    return {
        'agreement': agreement,
        'direction': direction,
        'display': display,
        'conviction': conviction,
        'positive_count': positive,
        'negative_count': negative,
        'total': total
    }
