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
    'Tactical Score': ['opportunity_map', 'slpm_v12.py'],
    'Structural Score': ['opportunity_map', 'slpm_v12.py'],
    'Persistence': ['structural_engine.py', 'slpm_v12.py'],
    'Leader Breadth': ['structural_engine.py', 'slpm_v12.py'],
    'Relative Strength (RS)': ['stock_leader.py', 'slpm_v12.py', 'tactical_engine.py', 'structural_engine.py'],
    'Flow Proxy': ['stock_leader.py', 'slpm_v12.py', 'tactical_engine.py', 'structural_engine.py'],
}

# Dependencias indirectas (documentadas para conciencia del gestor)
INDIRECT_DEPENDENCIES = {
    'Persistence → Structural Score → SLPM': 'La persistencia alimenta el Structural Score, que a su vez alimenta el SLPM.',
    'Tactical Score → Opportunity Map + SLPM': 'El Tactical Score se usa tanto en el Opportunity Map como en el SLPM.',
    'Structural Score → Opportunity Map + SLPM': 'El Structural Score se usa tanto en el Opportunity Map como en el SLPM.',
    'LIS → SLPM (corregido)': 'LIS era una metrica de intensidad/calidad. Tras detectar redundancia perfecta con Breadth (Spearman=1.0), fue excluido de la State Machine. Ahora es solo diagnostico.',
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
    

    # Verificación dinámica de la corrección LIS/Breadth
    try:
        import inspect
        from indicators.state_machine import classify_leadership_state
        sig = inspect.signature(classify_leadership_state)
        if 'lis' in sig.parameters:
            lines.append("\n**Verificación SLPM:** ⚠️ LIS aún en State Machine (doble conteo activo).\n")
        else:
            lines.append("\n**Verificación SLPM:** ✅ LIS excluido de la State Machine. Breadth es el factor decisorio.\n")
    except Exception:
        lines.append("\n**Verificación SLPM:** No disponible.\n")

    return {
        'critical': critical,
        'high': high,
        'medium': medium,
        'indirect': INDIRECT_DEPENDENCIES,
        'summary': ''.join(lines)
    }
