# -*- coding: utf-8 -*-



"""



mte.py -- Market Transition Engine v1.0



Motor de inferencia macroecon�mica basado en flujos institucionales.



"""






import numpy as np



import json



import os



from . import state as _mte_state
from .state import load_previous_scenario, save_scenario
from .scoring import (
    tanh,
    _get_last,
    sector_rotation_score,
    safe_haven_score,
    credit_stress_score,
    inflation_pressure_score,
    compute_msi,
    compute_ipi,
    score_scenarios,
)



from src.utils import safe_mean, robust_zscore, get_col







# ============================================================



# 0. FUNCIONES AUXILIARES



# ============================================================



# tanh, _get_last, scoring movidos a scoring.py (DT2 Fase 4)

# ============================================================



# 7. MATRIZ DE TRANSICIONES



# ============================================================



NORMAL_TRANSITIONS = {



    'EXPANSION':      ['SOFT LANDING', 'MIXED'],



    'SOFT LANDING':   ['EXPANSION', 'RECESSION', 'MIXED'],



    'RECESSION':      ['SOFT LANDING', 'CRISIS', 'STAGFLATION', 'MIXED'],



    'STAGFLATION':    ['RECESSION', 'MIXED'],



    'CRISIS':         ['RECESSION', 'MIXED'],



    'MIXED':          ['EXPANSION', 'SOFT LANDING', 'RECESSION', 'STAGFLATION', 'CRISIS'],



}







EXCEPTION_TRANSITIONS = {



    ('EXPANSION', 'RECESSION'): "Salto abrupto por estr�s extremo",



    ('EXPANSION', 'CRISIS'):    "Evento de mercado excepcional (ej. COVID)",



    ('SOFT LANDING', 'CRISIS'): "Deterioro s�bito de condiciones financieras",



}







def validate_transition(previous, current, cls):



    if current in NORMAL_TRANSITIONS.get(previous, []):



        return True



    if (previous, current) in EXCEPTION_TRANSITIONS and cls > 0.85:



        return True



    return False











# ============================================================



# 8. PERSISTENCIA DE ESTADO (HIST�RESIS)



# ============================================================










# state.py: load_previous_scenario, save_scenario extraidos (DT2 Fase 3)
# ============================================================



# 9. CONFIANZA



# ============================================================



def consensus_score(srs, shs, cls, ips):



    values = np.array([srs, shs, cls, ips])



    median = np.median(values)



    distance = np.abs(values - median)



    mean_distance = distance.mean()



    return float(1 - np.clip(mean_distance / 2.0, 0, 1))







def distance_to_threshold(srs, shs, cls, ips, scenario):



    """Distancia media a los umbrales del escenario, normalizada a [0,1]."""



    if scenario == 'CRISIS':



        distances = [cls - 0.5, shs - 0.3, srs - 0.3]



    elif scenario == 'RECESSION':



        distances = [cls - 0.2, shs - 0.2, srs - 0.2]



    elif scenario == 'STAGFLATION':



        distances = [ips - 0.3, srs, 0.3 - cls]



    elif scenario == 'SOFT LANDING':



        distances = [srs - 0.1, shs - 0.1, -cls]



    elif scenario == 'EXPANSION':



        distances = [-srs - 0.1, -cls - 0.1, -shs]



    else:



        return 0.5



    return float(np.clip(safe_mean([max(0, d) for d in distances]) / 0.5, 0, 1))







def compute_confidence(srs, shs, cls, ips, scenario):



    distance_conf = distance_to_threshold(srs, shs, cls, ips, scenario)



    consensus_conf = consensus_score(srs, shs, cls, ips)



    return float(np.clip(0.6 * distance_conf + 0.4 * consensus_conf, 0, 1))











# ============================================================



# 10. CLASIFICADOR PRINCIPAL



# ============================================================



def classify_mte(srs, shs, cls, ips):



    scores = score_scenarios(srs, shs, cls, ips)



    new_scenario = max(scores, key=scores.get)







    # Desempate



    if list(scores.values()).count(scores[new_scenario]) > 1:



        priority = ['CRISIS', 'RECESSION', 'STAGFLATION', 'SOFT LANDING', 'EXPANSION', 'MIXED']



        for s in priority:



            if scores[s] == scores[new_scenario]:



                new_scenario = s



                break







    # Validar transici�n



    prev_scenario, pending = load_previous_scenario()



    if not validate_transition(prev_scenario, new_scenario, cls):



        new_scenario = prev_scenario







    # Hist�resis adaptativa



    if cls > 0.85:



        save_scenario(new_scenario)



        final_scenario = new_scenario



    elif new_scenario != prev_scenario:



        if pending is None:



            save_scenario(prev_scenario, new_scenario)



            final_scenario = prev_scenario



        elif pending == new_scenario:



            save_scenario(new_scenario)



            final_scenario = new_scenario



        else:



            save_scenario(prev_scenario, None)



            final_scenario = prev_scenario



    else:



        if pending is not None:



            save_scenario(new_scenario)



        final_scenario = new_scenario







    confidence = compute_confidence(srs, shs, cls, ips, final_scenario)



    if confidence == 0.0:



        print(f"    MTE: Confianza 0% en escenario {final_scenario}. Forzando MIXED.")



        final_scenario = 'MIXED'



    return final_scenario, confidence











# ============================================================



# 11. FUNCI�N PRINCIPAL (ORQUESTADOR)



# ============================================================



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











