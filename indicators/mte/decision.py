"""Maquina de decision MTE (DT2 Fase 5).

Extraccion literal de mte_legacy.py. Sin cambios funcionales.
"""
from __future__ import annotations

import numpy as np

from src.utils import safe_mean

from .state import load_previous_scenario, save_scenario
from .scoring import score_scenarios


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
