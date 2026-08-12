# -*- coding: utf-8 -*-
# scripts/generate_docs.py - Genera documentacion automatica desde el codigo fuente (v2 - completa)
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCS_DIR = 'docs'
SETTINGS_FILE = 'config/settings.py'

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_constants(filepath, prefix=None):
    content = read_file(filepath)
    constants = {}
    pattern = re.compile(r'^([A-Z][A-Z_0-9]+)\s*=\s*(.+?)(?:\s*#.*)?$', re.MULTILINE)
    for match in pattern.finditer(content):
        name = match.group(1)
        value = match.group(2).strip()
        if not name.startswith('_'):
            if prefix is None or name.startswith(prefix):
                constants[name] = value
    return constants

def extract_docstring(filepath, function_name):
    content = read_file(filepath)
    pattern = rf'def {function_name}\(.*?\):\s*\n\s*"""(.*?)"""'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def format_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |']
    lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)

def generate_readme():
    sections = [
        ('01_arquitectura.md', 'Arquitectura General', 'Flujo principal, premisas, estructura de modulos'),
        ('02_configuracion.md', 'Configuracion', 'Parametros, ventanas, umbrales, pesos'),
        ('03_fuentes_datos.md', 'Fuentes de Datos', 'Proveedores, cache, validacion'),
        ('04_regimenes.md', 'Regimenes', 'Financial Conditions, Liquidity, Volatility, Macro, Sector'),
        ('05_motores.md', 'Motores Tactico y Estructural', 'Tactical Engine, Structural Engine'),
        ('06_indicadores_momentum.md', 'Indicadores: Momentum y Flujo', 'momentum.py, trend.py, flow proxy'),
        ('06_indicadores_breadth.md', 'Indicadores: Breadth', 'breadth.py, breadth_equity.py'),
        ('06_indicadores_wyckoff.md', 'Indicadores: Wyckoff', 'wyckoff.py, fases, ATR, estabilidad'),
        ('06_indicadores_slpm.md', 'Indicadores: SLPM', 'slpm_v12.py, state_machine.py, LIS, Breadth'),
        ('06_indicadores_opciones.md', 'Indicadores: Opciones (OMS)', 'options.py, PCR, IHR'),
        ('06_indicadores_darkpool.md', 'Indicadores: Dark Pools', 'darkpool.py, FINRA ATS, Z-Scores'),
        ('06_indicadores_mte.md', 'Indicadores: MTE', 'mte.py, Market Transition Engine'),
        ('07_lideres.md', 'Lideres Sectoriales e Internacionales', 'stock_leader.py, WLS, indices'),
        ('08_reporte.md', 'Generacion del Reporte', 'report_generator.py, estructura del reporte diario'),
        ('09_auditorias.md', 'Scripts de Auditoria', 'validacion, Monte Carlo, ablacion, correlaciones'),
    ]
    lines = [
        '# Radar de Rotacion Sectorial - Documentacion v4.2',
        f'**Generado automaticamente:** {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        '',
        '## Indice',
        ''
    ]
    for filename, title, desc in sections:
        lines.append(f'- [{title}]({filename}): {desc}')
    lines.append('')
    lines.append('---')
    lines.append('*Esta documentacion se genera automaticamente desde el codigo fuente. No editar manualmente.*')
    return '\n'.join(lines)

def generate_arquitectura():
    return """# Arquitectura General

## Premisas Fundamentales
- **NO trading bot:** el sistema no genera ordenes, no sugiere timing, no automatiza rotacion de cartera.
- **NO sobreingenieria:** no se usa ML, optimizacion de parametros ni complejidad gratuita. Codigo determinista y transparente.
- **Toda decision final de inversion es humana.**

## Flujo Principal
1. Descarga de datos de mercado (Yahoo Finance, FRED, CBOE, FINRA).
2. Validacion de datos (NaN, cobertura).
3. Calculo de regimenes (Macro, Financial, Liquidity, Volatility, Sector).
4. Motores tactico y estructural para cada sector.
5. Indicadores: momentum, flujo, breadth, Wyckoff, opciones, Dark Pools, MTE.
6. SLPM (Structural Leadership) para auditar liderazgo del sector #1.
7. Generacion de rankings y reporte Markdown.

## Estructura de Modulos
- un.py: orquestador principal.
- config/: settings, tickers, weights.
- egimes/: condiciones financieras, liquidez, volatilidad, macro, sector.
- indicators/: todos los indicadores y scores.
- src/: carga de datos, generacion de reporte, utilidades.
- data/: providers (yahoo, cboe, finra, fred), datos macro manuales.
- alidation/: scripts de auditoria y backtesting.
"""

def generate_config_doc():
    constants = extract_constants(SETTINGS_FILE)
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    return f"""# Configuracion del Sistema
**Archivo:** {SETTINGS_FILE}

## Constantes Globales

{table}
"""

def generate_fuentes():
    return """# Fuentes de Datos

| Fuente | Proveedor | Archivo | Actualizacion |
|--------|-----------|---------|---------------|
| Precios (ETFs, acciones, indices) | Yahoo Finance | data/providers/yahoo.py | Diaria (< 1 dia) |
| Datos macro (WALCL, SOFR, RRP) | FRED / archivos manuales | data/providers/fred.py, data/macro_manual/ | Semanal |
| Opciones (PCR, IHR) | CBOE | data/providers/cboe.py | Diaria (1-2 dias) |
| Dark Pools (ATS) | FINRA | data/providers/finra.py | Semanal |

## Cache
- CACHE_HOURS = 23: los datos de mercado se cachean por 23 horas.
- CACHE_TTL: por proveedor (yahoo=23h, fred=168h, cboe=24h, finra=168h).
"""

def generate_regimenes():
    doc_financial = extract_docstring('regimes/financial_conditions.py', 'compute_financial_conditions') or 'Calcula el score de condiciones financieras (VIX, credito, dolar, curva).'
    doc_macro = extract_docstring('regimes/macro_regime.py', 'compute_macro_regime') or 'Clasifica el regimen macro en 11 categorias.'
    return f"""# Regimenes

## Financial Conditions
{chr(96)}{chr(96)}{chr(96)}
{doc_financial}
{chr(96)}{chr(96)}{chr(96)}

## Liquidity (FRED)
Calcula la liquidez real a partir del balance de la Fed (WALCL), SOFR, Reverse Repo y Fed Funds.

## Volatility
Basado en VIX. Z-Score robusto de la volatilidad realizada a 20 dias vs mediana de 3 anios.

## Macro Regime
{chr(96)}{chr(96)}{chr(96)}
{doc_macro}
{chr(96)}{chr(96)}{chr(96)}

## Sector Regime
Ranking sectorial combinando momentum, tendencia, volatilidad, breadth y Wyckoff.
"""

def generate_motores():
    doc_tactical = extract_docstring('regimes/tactical_engine.py', 'compute_tactical_score') or 'Score de corto plazo (RS20, Momentum20, Flow, Breadth20, Aceleracion).'
    doc_structural = extract_docstring('regimes/structural_engine.py', 'compute_structural_score') or 'Score de largo plazo (RS multi-ventana, Leader Breadth, Flow Structure, Persistence).'
    return f"""# Motores Tactico y Estructural

## Tactical Engine
{chr(96)}{chr(96)}{chr(96)}
{doc_tactical}
{chr(96)}{chr(96)}{chr(96)}

## Structural Engine
{chr(96)}{chr(96)}{chr(96)}
{doc_structural}
{chr(96)}{chr(96)}{chr(96)}
"""

def generate_momentum():
    doc_flow = extract_docstring('indicators/momentum.py', 'compute_flow_proxy') or 'Flow Proxy = 0.30*ret*vol + 0.35*OBV + 0.35*CMF.'
    constants = extract_constants(SETTINGS_FILE, 'FLOW_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows) if rows else ''
    return f"""# Indicadores: Momentum y Flujo

## Flow Proxy
{chr(96)}{chr(96)}{chr(96)}
{doc_flow}
{chr(96)}{chr(96)}{chr(96)}

## Constantes de Flujo
{table}
"""

def generate_breadth():
    doc_breadth = extract_docstring('indicators/breadth.py', 'compute_breadth') or 'Calcula % de sectores sobre EMAs y nuevos maximos/minimos.'
    constants = extract_constants(SETTINGS_FILE, 'BREADTH_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows) if rows else ''
    return f"""# Indicadores: Breadth

## Breadth de Mercado
{chr(96)}{chr(96)}{chr(96)}
{doc_breadth}
{chr(96)}{chr(96)}{chr(96)}

## Constantes de Breadth
{table}
"""

def generate_wyckoff():
    doc_structural = extract_docstring('indicators/wyckoff.py', 'wyckoff_structural_score')
    doc_tactical = extract_docstring('indicators/wyckoff.py', 'wyckoff_tactical_score')
    doc_score = extract_docstring('indicators/wyckoff.py', 'wyckoff_score')
    wyckoff_constants = extract_constants(SETTINGS_FILE, 'WYCKOFF_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(wyckoff_constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    lines = [
        '# Modulo Wyckoff (v4.2)',
        '',
        '## Proposito',
        'Proporciona un score continuo de estructura de precios para ETFs sectoriales y acciones lideres.',
        '',
        '## Arquitectura',
        '- wyckoff_structural_score(): trend + ATR (70%).',
        '- wyckoff_tactical_score(): volume + effort (30%).',
        '- wyckoff_score(): combinacion ponderada de ambos.',
        '- wyckoff_structure_core(): clasifica en MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.',
        '',
    ]
    if doc_structural:
        lines.append(f'### Score Estructural\n{chr(96)}{chr(96)}{chr(96)}\n{doc_structural}\n{chr(96)}{chr(96)}{chr(96)}\n')
    if doc_tactical:
        lines.append(f'### Score Tactico\n{chr(96)}{chr(96)}{chr(96)}\n{doc_tactical}\n{chr(96)}{chr(96)}{chr(96)}\n')
    if doc_score:
        lines.append(f'### Score Combinado\n{chr(96)}{chr(96)}{chr(96)}\n{doc_score}\n{chr(96)}{chr(96)}{chr(96)}\n')
    lines.append('## Constantes Configurables')
    lines.append(table)
    return '\n'.join(lines)

def generate_slpm():
    doc_state = extract_docstring('indicators/state_machine.py', 'classify_leadership_state')
    state_machine_content = read_file('indicators/state_machine.py')
    thresholds = {}
    for match in re.finditer(r"'(\w+)':\s*([\d.]+)", state_machine_content):
        thresholds[match.group(1)] = match.group(2)
    rows = [[f'{k}', f'{v}'] for k, v in sorted(thresholds.items())]
    table = format_table(['Parametro', 'Valor'], rows)
    lines = [
        '# Structural Leadership (SLPM v1.2)',
        '',
        '## Proposito',
        'Audita la calidad del liderazgo del sector #1 del ranking.',
        '',
        '## Componentes',
        '- **Leader Breadth v2:** amplitud del liderazgo (RS, momentum, flujo, Wyckoff).',
        '- **Leader Integrity Score (LIS):** intensidad/calidad de los lideres individuales.',
        '- **Flow Divergence 2.0:** divergencias entre flujo de lideres y sector.',
        '- **State Machine:** clasifica el estado (CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY, LOST, UNRESOLVED).',
        '',
        '## Umbrales de la State Machine',
        table,
    ]
    if doc_state:
        lines.append(f'\n## Logica de Clasificacion\n{chr(96)}{chr(96)}{chr(96)}\n{doc_state}\n{chr(96)}{chr(96)}{chr(96)}')
    return '\n'.join(lines)

def generate_opciones():
    ihr_thresholds = read_file('indicators/options_metrics.py')
    lines = [
        '# Indicadores: Opciones (OMS v2.0)',
        '',
        '## Proposito',
        'Calcula el PCR (Put/Call Ratio) y el IHR (Institutional Hedge Ratio) a partir de datos de CBOE.',
        '',
        '## Metricas',
        '- PCR Total, PCR Indices, PCR Acciones, PCR ETP, PCR VIX, PCR SPX.',
        '- IHR = PCR Indices / PCR Acciones.',
        '- Volumen en Indices (% del total).',
        '- Put Share / Call Share.',
        '',
        '## Clasificacion IHR',
        '| Rango | Clasificacion |',
        '|-------|---------------|',
        '| < 0.8 | Especulacion extrema |',
        '| 0.8 - 1.2 | Especulacion alta |',
        '| 1.2 - 1.6 | Equilibrado |',
        '| 1.6 - 2.5 | Cobertura institucional alta |',
        '| > 2.5 | Cobertura institucional extrema |',
        '',
        '## Clasificacion PCR (Z-Score)',
        '| Rango | Estado |',
        '|-------|--------|',
        '| >= 2.0 | Panico |',
        '| 1.0 - 2.0 | Miedo |',
        '| -1.0 a 1.0 | Neutral |',
        '| -2.0 a -1.0 | Optimismo |',
        '| < -2.0 | Euforia |',
    ]
    return '\n'.join(lines)

def generate_darkpool():
    constants = extract_constants(SETTINGS_FILE, 'DARKPOOL_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    return f"""# Indicadores: Dark Pools (FINRA v1.0)

## Proposito
Mide el porcentaje de volumen negociado en ATS (Alternative Trading Systems) respecto al volumen total, usando datos de FINRA.

## Z-Scores
Se calculan Z-Scores robustos para 4 ventanas: 13, 26, 52 y 104 semanas.

## Clasificacion
| Z-Score | Estado |
|---------|--------|
| >= 2.5 | Acumulacion extrema |
| 1.5 a 2.5 | Acumulacion fuerte |
| 0.5 a 1.5 | Acumulacion moderada |
| -0.5 a 0.5 | Neutral |
| -1.5 a -0.5 | Distribucion moderada |
| -2.5 a -1.5 | Distribucion fuerte |
| < -2.5 | Distribucion extrema |

## Constantes
{table}
"""

def generate_mte():
    doc_mte = extract_docstring('indicators/mte.py', 'compute_confidence') or 'Confianza del escenario MTE (distancia a umbrales + consenso).'
    return f"""# Indicadores: Market Transition Engine (MTE v1.0)

## Proposito
Infiere el escenario macro que el mercado parece estar descontando, basado en 4 motores (SRS, SHS, CLS, IPS).

## Motores
- **SRS (Sector Rotation Score):** rotacion sectorial.
- **SHS (Safe Haven Score):** demanda de activos refugio.
- **CLS (Credit/Liquidity Stress Score):** estres en credito/liquidez.
- **IPS (Inflation Pressure Score):** presion inflacionaria.

## Indices
- **MSI (Market Stress Index):** SRS + SHS + CLS (0-100).
- **IPI (Inflation Pressure Index):** basado en IPS (0-100).

## Escenarios
CRISIS, RECESSION, STAGFLATION, SOFT LANDING, EXPANSION, MIXED.

## Confianza
{chr(96)}{chr(96)}{chr(96)}
{doc_mte}
{chr(96)}{chr(96)}{chr(96)}
"""

def generate_lideres():
    return """# Lideres Sectoriales e Internacionales

## Proposito
Selecciona las mejores empresas de cada sector/indice en fase favorable (ACCUMULATION o MARKUP) usando el Wyckoff Leadership Score (WLS).

## WLS (Wyckoff Leadership Score)
Combina:
- RS (Relative Strength) normalizado: 35%
- Flujo (Flow Proxy) normalizado: 25%
- RWS (Relative Wyckoff Score) normalizado: 25%
- Estabilidad: 10%
Bonus por persistencia: +5% * min(persistence_10d/10, 1.0).

## Lideres Sectoriales
- Archivo: indicators/stock_leader.py
- Fuente de holdings: data/etf_holdings.csv (actualizacion trimestral automatica desde State Street).

## Lideres Internacionales
- Archivo: indicators/index_leaders.py
- Fuente de holdings: data/index_holdings.csv
- Indices cubiertos: S&P 500, Dow Jones, Nasdaq-100, Russell 2000, Euro Stoxx 50, Ibex 35, DAX 40, FTSE 100.
"""

def generate_reporte():
    return """# Generacion del Reporte Diario

## Archivo
src/report_generator.py

## Estructura del Reporte
1. Cabecera (fecha, version).
2. Resumen de Regimenes (Macro, Financial, Liquidity, Volatility, Sectores).
3. Data Freshness y Cobertura.
4. Divergencias Detectadas (Breadth, Price-Flow).
5. Cross-Module Conflict Detector.
6. Breadth de Mercado.
7. Momentum de Precio - Sectores (20 dias).
8. Flujo Institucional - Sectores (Proxy).
9. Tactical Leaders (momentum de corto plazo).
10. Momentum y Flujo - Otros Activos.
11. Structural Ranking (fortaleza de largo plazo).
12. Rankings Sectoriales (Score combinado original).
13. Opportunity Map (Tactical vs Structural).
14. Structural Leadership (SLPM v1.2).
15. Acciones Seleccionadas (lideres sectoriales e internacionales).
16. Sentimiento de Opciones (OMS v2.0).
17. Market Transition Engine (MTE v1.0).
18. Confirmation Data (Nivel 2).
19. Cross-Asset Ratios.
20. Dark Pools (FINRA).
21. Estado Actual - Sintesis de Senhales.
22. Anti-Double-Counting Audit.
"""

def generate_auditorias():
    scripts = [
        ('wyckoff_correlation_audit.py', 'Matriz de correlaciones entre componentes Wyckoff con bootstrap'),
        ('wyckoff_weight_sensitivity.py', 'Monte Carlo de sensibilidad de pesos (Kendall Tau)'),
        ('wyckoff_ablation_components.py', 'Ablacion por componentes del modulo Wyckoff'),
        ('wyckoff_out_of_sample.py', 'Validacion out-of-sample por periodos historicos'),
        ('montecarlo_perturbacion_ranking.py', 'Perturbacion del ranking global con ruido gaussiano'),
        ('montecarlo_ranking_global.py', 'Monte Carlo del ranking sectorial global'),
        ('backtest_pesos_historicos.py', 'Backtest historico de estabilidad temporal del ranking'),
        ('slpm_ablation.py', 'Ablacion de componentes del SLPM'),
        ('forward_test_auto.py', 'Registro semanal del forward test'),
    ]
    rows = [[f'{name}', desc] for name, desc in scripts]
    table = format_table(['Script', 'Descripcion'], rows)
    return f"""# Scripts de Auditoria

## Proposito
Scripts independientes que validan la estabilidad y robustez del sistema. No modifican el codigo productivo.

{table}
"""

# ---- MAIN ----
def main():
    os.makedirs(DOCS_DIR, exist_ok=True)
    generators = {
        'README.md': generate_readme,
        '01_arquitectura.md': generate_arquitectura,
        '02_configuracion.md': generate_config_doc,
        '03_fuentes_datos.md': generate_fuentes,
        '04_regimenes.md': generate_regimenes,
        '05_motores.md': generate_motores,
        '06_indicadores_momentum.md': generate_momentum,
        '06_indicadores_breadth.md': generate_breadth,
        '06_indicadores_wyckoff.md': generate_wyckoff,
        '06_indicadores_slpm.md': generate_slpm,
        '06_indicadores_opciones.md': generate_opciones,
        '06_indicadores_darkpool.md': generate_darkpool,
        '06_indicadores_mte.md': generate_mte,
        '07_lideres.md': generate_lideres,
        '08_reporte.md': generate_reporte,
        '09_auditorias.md': generate_auditorias,
    }
    print('Generando documentacion...')
    for filename, generator in generators.items():
        content = generator()
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'  {filename}')
    print(f'\n{len(generators)} archivos generados en docs/')

if __name__ == '__main__':
    main()
