"""Motor MTE: compute_mte (DT2 Fase 6).

Extraccion literal de mte_legacy.py. Sin cambios funcionales.
El paquete indicators.mte ya no contiene mte_legacy.py.
"""
from __future__ import annotations

import json
import os

from src.utils import robust_zscore, get_col

from . import state as _mte_state
from .scoring import (
    tanh,
    _get_last,
    sector_rotation_score,
    safe_haven_score,
    credit_stress_score,
    inflation_pressure_score,
    compute_msi,
    compute_ipi,
)
from .decision import classify_mte


def compute_mte(df_market, financial_conditions_score, credit_signal,
                volatility_signal, pcr_data=None, darkpool_data=None,
                temporal_meta=None):



    """



    Calcula el MTE completo y devuelve un diccionario con todos los resultados.



    """



    try:



        # Extraer scores existentes



        fc = _get_last(financial_conditions_score)



        cred = _get_last(credit_signal)



        vol = _get_last(volatility_signal)







        # VIX term (VIX3M - VIX)



        try:



            vix_close = get_col(df_market, '^VIX', 'Close')



            vix3m_close = get_col(df_market, '^VIX3M', 'Close')



            vix_term = _get_last(tanh(robust_zscore(vix3m_close - vix_close, 60)))



        except:



            vix_term = 0.0







        # Dark Pool Z-Score



        darkpool_z = darkpool_data.get('z_score', None) if darkpool_data else None







        # PCR Z-Score



        pcr_z = pcr_data.get('z_score', None) if pcr_data else None







        # Calcular los 4 motores



        srs = sector_rotation_score(df_market)



        shs = safe_haven_score(df_market)



        cls = credit_stress_score(fc, cred, vol, vix_term, darkpool_z, pcr_z)



        ips = inflation_pressure_score(df_market)







        # �ndices



        msi = compute_msi(srs, shs, cls)



        ipi = compute_ipi(ips)







        # Escenario



        scenario, confidence = classify_mte(srs, shs, cls, ips)







        # Guardar estado en JSON para trazabilidad



        try:



            os.makedirs(os.path.dirname(_mte_state.MTE_STATE_FILE), exist_ok=True)



            with open(_mte_state.MTE_STATE_FILE, 'w', encoding='utf-8') as f:



                json.dump({



                    'schema_version': 1,
                    'temporal_contract_version': _mte_state.CURRENT_TEMPORAL_CONTRACT_VERSION,
                    'effective_date': (temporal_meta or {}).get('by_contract', {}).get('EQUITY_EOD', {}).get('effective_date'),
                    'expected_date': (temporal_meta or {}).get('by_contract', {}).get('EQUITY_EOD', {}).get('expected_date'),
                    'coverage': (temporal_meta or {}).get('by_contract', {}).get('EQUITY_EOD', {}).get('coverage'),
                    'futures_status': 'BLOCKED',
                    'scenario': scenario,
                    'confidence': confidence,



                    'msi': msi,



                    'ipi': ipi,



                    'srs': srs,



                    'shs': shs,



                    'cls': cls,



                    'ips': ips



                }, f, indent=2, default=str)



        except Exception as e:



            print(f'  MTE: No se pudo guardar estado JSON - {e}')







        return {



            'scenario': scenario,



            'confidence': confidence,



            'msi': msi,



            'ipi': ipi,



            'srs': srs,



            'shs': shs,



            'cls': cls,



            'ips': ips



        }



    except Exception as e:



        print(f"  MTE: Error - {e}")



        return None
