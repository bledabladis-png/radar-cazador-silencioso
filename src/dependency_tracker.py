# -*- coding: utf-8 -*-
"""
dependency_tracker.py -- Anti-Double-Counting Matrix v1.2
Identifica dependencias directas e indirectas entre modulos.
"""
# Dependencias directas: variable -> modulos que la usan directamente
DIRECT_DEPENDENCIES = {
    'VIX': ['financial_conditions.py', 'volatility_regime.py', 'mte.py'],
    'Credit Signal': ['financial_conditions.py', 'macro_regime.py', 'mte.py'],
    'HYG/LQD (credito)': ['financial_conditions.py', 'mte.py'],
    'SOFR': ['liquidity.py', 'fls.py'],
    'WALCL': ['liquidity.py', 'fls.py'],
    'RRP': ['liquidity.py', 'fls.py'],
    'Tactical Score': ['opportunity_map'],  # Ya no es decisorio en slpm_v12.py
    'Structural Score': ['opportunity_map', 'slpm_v12.py'],
    'Persistence': ['structural_engine.py', 'slpm_v12.py'],
    'Leader Breadth': ['slpm_v12.py'],  # ya no es input del Structural Score
    'Relative Strength (RS)': ['stock_leader.py', 'slpm_v12.py', 'tactical_engine.py', 'structural_engine.py'],
    'Flow Proxy': ['stock_leader.py', 'slpm_v12.py', 'tactical_engine.py', 'structural_engine.py'],
}

# Dependencias indirectas (documentadas para conciencia del gestor)
INDIRECT_DEPENDENCIES = {
    'Persistence -> Structural Score -> SLPM': 'La persistencia alimenta el Structural Score, que a su vez alimenta el SLPM.',
    'Tactical Score -> Opportunity Map (informativo)': 'El Tactical Score se usa en el Opportunity Map. Ya no es input decisorio del SLPM.',
    'Structural Score -> Opportunity Map + SLPM': 'El Structural Score se usa tanto en el Opportunity Map como en el SLPM.',
    'LIS -> SLPM (corregido)': 'LIS era una metrica de intensidad/calidad. Tras detectar redundancia perfecta con Breadth (Spearman=1.0), fue excluido de la State Machine. Ahora es solo diagnostico.',
}

# P1.5 (2026-10-13, auditor): clasificacion en 3 niveles. El
# nivel 'INDEPENDIENTE' anterior mezclaba dos conceptos: variables
# de origen observable y evidencia estadisticamente independiente.
# Breadth, RS y Flow Proxy comparten informacion (derivan del
# precio/volumen). Se reclasifican sin tocar los calculos.
EVIDENCE_CLASSIFICATION = {
    # Observables directos: medidas primarias de mercado.
    'Precio': 'OBSERVABLE',
    'Volumen': 'OBSERVABLE',
    'VIX': 'OBSERVABLE',
    'Crédito': 'OBSERVABLE',
    'Flujo ETF primario': 'OBSERVABLE',
    # Derivadas directas: transformaciones de un observable.
    'Retorno': 'DERIVADA-DIRECTA',
    'Breadth': 'DERIVADA-DIRECTA',
    'Flow Proxy': 'DERIVADA-DIRECTA',
    'Relative Strength (RS)': 'DERIVADA-DIRECTA',
    'Momentum': 'DERIVADA-DIRECTA',
    # Derivadas compuestas: agregan varias derivadas.
    'Persistence': 'DERIVADA-COMPUESTA',
    'WLS': 'DERIVADA-COMPUESTA',
    'Tactical Score': 'DERIVADA-COMPUESTA',
    'Structural Score': 'DERIVADA-COMPUESTA',
    'Leader Breadth': 'DERIVADA-COMPUESTA',
}

def audit_double_counting():
    """
    Analiza la matriz de dependencias y retorna un resumen de riesgos.
    Incluye dependencias directas e indirectas.
    """
    critical = []
    high = []
    medium = []
    
    for var, modules in DIRECT_DEPENDENCIES.items():
        n = len(modules)
        entry = {'variable': var, 'count': n, 'modules': modules}
        if n >= 4:
            critical.append(entry)
        elif n >= 3:
            high.append(entry)
        elif n >= 2:
            medium.append(entry)
    
    lines = []
    lines.append("### Anti-Double-Counting Audit\n")
    lines.append("*Advertencia: Algunas variables son utilizadas por multiples modulos. Esto no implica error, pero el gestor debe saber que estas senhales pueden estar correlacionadas.*\n")
    
    if critical:
        lines.append(f"\n**Critico ({len(critical)} variables compartidas por 4+ modulos):**\n")
        for e in critical:
            lines.append(f"- **{e['variable']}** ({e['count']} modulos): {', '.join(e['modules'])}\n")
    
    if high:
        lines.append(f"\n**Alto ({len(high)} variables compartidas por 3 modulos):**\n")
        for e in high:
            lines.append(f"- **{e['variable']}** ({e['count']} modulos): {', '.join(e['modules'])}\n")
    
    if medium:
        lines.append(f"\n**Medio ({len(medium)} variables compartidas por 2 modulos):**\n")
        for e in medium:
            lines.append(f"- **{e['variable']}** ({e['count']} modulos): {', '.join(e['modules'])}\n")
    
    lines.append("\n**Dependencias indirectas detectadas:**\n")
    for chain, desc in INDIRECT_DEPENDENCIES.items():
        lines.append(f"- **{chain}**: {desc}\n")
    
    lines.append("\n*Esta matriz es informativa. No modifica ningun calculo.*\n")
    

    lines.append("\n**Clasificación de evidencia:**\n")
    lines.append("| Variable | Tipo |\n")
    lines.append("|----------|------|\n")
    for var, tipo in EVIDENCE_CLASSIFICATION.items():
        lines.append(f"| {var} | {tipo} |\n")
    # P1.5b (2026-10-13, auditor): nota alineada con la clasificacion
    # en 3 niveles. Evita el termino 'independiente' para derivadas.
    lines.append("\n*Clasificacion en 3 niveles: OBSERVABLE = medida "
                 "directa (precio, volumen, VIX, credito, flujo ETF). "
                 "DERIVADA-DIRECTA = transformacion de un observable "
                 "(retorno, breadth, RS, momentum, flow proxy). "
                 "DERIVADA-COMPUESTA = agrega varias derivadas "
                 "(persistence, WLS, tactical, structural, leader "
                 "breadth).*\n")
    lines.append("*Nota: DERIVADA-DIRECTA no implica independencia "
                 "estadistica. Breadth, RS y Flow Proxy comparten "
                 "informacion del precio.*\n")
    return {
        'critical': critical,
        'high': high,
        'medium': medium,
        'indirect': INDIRECT_DEPENDENCIES,
        'summary': ''.join(lines)
    }
