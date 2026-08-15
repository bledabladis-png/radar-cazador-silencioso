# -*- coding: utf-8 -*-
"""
price_flow_divergence.py -- Price-Flow Divergence Detector v1.1
Detecta divergencias entre el retorno de precio y el Flow Proxy.
"""
import numpy as np

def detect_price_flow_divergence(price_return_20d, flow_proxy_z):
    """
    Detecta divergencias entre precio y Flow Proxy.
    
    Retorna un dict con:
      - status: 'PRICE_STRONG_FLOW_UNCONFIRMED', 'PRICE_WEAK_FLOW_SUPPORTIVE', 
                'ALIGNED', o 'UNAVAILABLE'
      - message: descripcion para el reporte
    
    Umbrales (documentados y asimetricos):
      - Precio fuerte: retorno 20d > +5%
      - Precio debil: retorno 20d < -5%
      - Flow no confirmatorio: flow_proxy_z < +0.10 (alerta sensible)
      - Flow soportivo: flow_proxy_z > +0.30 (senal mas exigente)
    
    La asimetria es intencionada: detectar falta de confirmacion requiere
    menos exigencia que detectar soporte en debilidad.
    """
    if price_return_20d is None or flow_proxy_z is None:
        return {'status': 'UNAVAILABLE', 'message': 'Datos insuficientes para evaluar divergencia.'}

    if not np.isfinite(price_return_20d) or not np.isfinite(flow_proxy_z):
        return {'status': 'UNAVAILABLE', 'message': 'Datos insuficientes para evaluar divergencia.'}

    # Precio fuerte (>5% en 20d) sin confirmacion del Flow Proxy
    if price_return_20d > 0.05 and flow_proxy_z < 0.10:
        return {
            'status': 'PRICE_STRONG_FLOW_UNCONFIRMED',
            'message': f'Precio fuerte (+{price_return_20d*100:.1f}%) sin confirmacion del Flow Proxy (z={flow_proxy_z:+.2f}). El indicador no permite inferir directamente participacion institucional.'
        }

    # Precio debil (<-5% en 20d) con Flow Proxy positivo
    if price_return_20d < -0.05 and flow_proxy_z > 0.30:
        return {
            'status': 'PRICE_WEAK_FLOW_SUPPORTIVE',
            'message': f'Precio debil ({price_return_20d*100:.1f}%) con Flow Proxy positivo (z={flow_proxy_z:+.2f}). Posible presion compradora/absorcion en debilidad; requiere confirmacion adicional.'
        }

    return {'status': 'ALIGNED', 'message': ''}
