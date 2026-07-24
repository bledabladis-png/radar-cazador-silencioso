# -*- coding: utf-8 -*-
"""
signal_agreement.py -- Signal Agreement v1.1
Calcula el porcentaje de señales alineadas para un sector, con dirección.
"""
import numpy as np

def compute_signal_agreement(signals):
    """
    signals: dict con valores numéricos donde >0 es alcista, <0 es bajista, 0 es neutral.
    Retorna un dict con:
      - agreement: proporción de señales alineadas en la dirección dominante (0.0 a 1.0)
      - direction: 'BULLISH', 'BEARISH' o 'MIXED'
      - display: string formateado para el reporte
    """
    if not signals:
        return {'agreement': 0.5, 'direction': 'MIXED', 'display': '50% MIXED'}
    
    valid = {k: v for k, v in signals.items() if v is not None and np.isfinite(v)}
    if not valid:
        return {'agreement': 0.5, 'direction': 'MIXED', 'display': '50% MIXED'}
    
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
    
    display = f"{agreement:.0%} {direction}"
    
    return {
        'agreement': agreement,
        'direction': direction,
        'display': display,
        'positive_count': positive,
        'negative_count': negative,
        'total': total
    }
