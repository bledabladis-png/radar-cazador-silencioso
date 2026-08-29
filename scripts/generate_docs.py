# -*- coding: utf-8 -*-
# scripts/generate_docs.py - Genera documentacion automatica desde el codigo fuente (v3 - completa)
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOCS_DIR = 'docs/automatica'
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
        ('10_flujo_primario_etf.md', 'Flujo Primario ETF', 'ETF Primary Flow desde SSGA, BlackRock, Amundi'),
        ('11_cftc_position_flow.md', 'CFTC Position Flow', 'Posicionamiento semanal de futuros financieros'),
        ('12_sec_nport_positions.md', 'SEC N-PORT Position Flow', 'Flujo posicional institucional trimestral'),
        ('13_backup_providers.md', 'Proveedores de Respaldo', 'Rate limiting, circuit breaker, validación cruzada'),
        ('14_instrument_registry.md', 'Registro de Instrumentos', 'Mapeo canónico de tickers entre proveedores'),
    ]
    lines = [
        '# Radar de Rotacion Sectorial - Documentacion v4.3',
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

# ---- PLANTILLA BASE PARA CADA DOCUMENTO ----
def template(proposito, arquitectura, formulas, salidas, limitaciones=''):
    doc = f"""## Proposito
{proposito}

## Arquitectura
{arquitectura}

## Formulas
{formulas}

## Salidas
{salidas}
"""
    if limitaciones:
        doc += f"\n## Limitaciones Conocidas\n{limitaciones}\n"
    return doc

def generate_arquitectura():
    proposito = "El Radar de Rotacion Sectorial es un sistema informativo diario que analiza flujos institucionales, contexto macro, estructura de precios (Wyckoff) y estructura del mercado de opciones para producir rankings, tablas y analisis para el gestor humano."
    arquitectura = """
- 
un.py: orquestador principal.
- config/: settings, tickers, weights.
- 
egimes/: condiciones financieras, liquidez, volatilidad, macro, sector.
- indicators/: todos los indicadores y scores.
- src/: carga de datos, generacion de reporte, utilidades.
- data/: providers (yahoo, cboe, finra, fred), datos macro manuales.
- alidation/: scripts de auditoria y backtesting.
"""
    formulas = "No aplica (modulo estructural)."
    salidas = "Reporte diario en Markdown (outputs/report/reporte_diario.md)."
    return template(proposito, arquitectura, formulas, salidas)

def generate_config_doc():
    constants = extract_constants(SETTINGS_FILE)
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    proposito = "Centraliza todas las constantes del sistema: ventanas temporales, umbrales, pesos y parametros de calidad de datos."
    arquitectura = f"Archivo unico: {SETTINGS_FILE}."
    formulas = "No aplica."
    salidas = "Las constantes son importadas por todos los modulos del sistema."
    return template(proposito, arquitectura, formulas, salidas) + f"\n## Constantes Globales\n\n{table}\n"

def generate_fuentes():
    proposito = "Descripcion de los proveedores de datos utilizados por el radar."
    arquitectura = """
| Fuente | Proveedor | Archivo | Actualizacion |
|--------|-----------|---------|---------------|
| Precios | Yahoo Finance | data/providers/yahoo.py | Diaria |
| Opciones | CBOE | data/providers/cboe.py | Diaria |
| Dark Pools | FINRA | data/providers/finra.py | Semanal |
| Macro | FRED / manual | data/providers/fred.py, data/macro_manual/ | Semanal |
"""
    formulas = "No aplica."
    salidas = "DataFrames de OHLCV, datos de opciones, datos ATS y series macroeconomicas."
    return template(proposito, arquitectura, formulas, salidas)

def generate_regimenes():
    proposito = "Modulos que evaluan el contexto macroeconomico, las condiciones financieras, la liquidez real, la volatilidad y la amplitud sectorial."
    arquitectura = """
- inancial_conditions.py: score basado en VIX, credito, dolar y curva (0.40/0.30/0.15/0.15).
- liquidity.py: liquidez real a partir de WALCL, SOFR, RRP y Fed Funds.
- olatility_regime.py: regimen de volatilidad basado en VIX.
- macro_regime.py: clasificacion en 11 categorias macro.
- sector_regime.py: ranking sectorial combinando momentum, tendencia, volatilidad, breadth y Wyckoff.
"""
    formulas = """
- **Financial Score:** 0.40*VIX_norm + 0.30*Credito_norm + 0.15*Dolar_norm + 0.15*Curva_norm.
- **Liquidity Score:** media ponderada de senhales normalizadas (0.35*Fed Balance + 0.25*RRP + 0.20*SOFR + 0.20*Fed Funds).
"""
    salidas = """
- Estados: ABUNDANTE, NEUTRAL, ESTRECHA, HIGH_STRESS, EXTREME_STRESS, CRISIS, LIQUIDITY CRISIS, RECESSION, INFLATION SHOCK, STAGFLATION, GOLDILOCKS, EXPANSION, LATE EXPANSION, RECOVERY, DEFLATION, SLOWDOWN, MIXED.
- Regimen sectorial: BROAD PARTICIPATION, ROTATIONAL, NARROW RALLY, CYCLICAL LEADERSHIP, DEFENSIVE LEADERSHIP, MIXED.
"""
    return template(proposito, arquitectura, formulas, salidas)

def generate_motores():
    doc_tactical = extract_docstring('regimes/tactical_engine.py', 'compute_tactical_score') or 'Score de corto plazo (RS20 30%, Momentum20 25%, Flow 20%, Breadth20 15%, Aceleracion 10%).'
    doc_structural = extract_docstring('regimes/structural_engine.py', 'compute_structural_score') or 'Score de largo plazo (RS multi-ventana 35%, Leader Breadth 25%, Flow Structure 20%, Persistence 20%).'
    proposito = "Calculan el Tactical Score (corto plazo) y el Structural Score (largo plazo) para cada sector."
    arquitectura = """
- 	actical_engine.py: compute_tactical_score().
- structural_engine.py: compute_structural_score().
Ambos se usan en el Opportunity Map y en el SLPM.
"""
    formulas = f"""
**Tactical Score:** {doc_tactical}
**Structural Score:** {doc_structural}
"""
    salidas = """
- Tactical Score: valor entre -1 y +1.
- Structural Score: valor entre -1 y +1.
Ambos aparecen en las tablas de rankings y en el Opportunity Map.
"""
    return template(proposito, arquitectura, formulas, salidas)

def generate_momentum():
    doc_flow = extract_docstring('indicators/momentum.py', 'compute_flow_proxy') or 'Flow Proxy = 0.30*ret*vol + 0.35*OBV + 0.35*CMF.'
    constants = extract_constants(SETTINGS_FILE, 'FLOW_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows) if rows else ''
    proposito = "Calcula el Flow Proxy (senhal de flujo institucional basada en precio y volumen) y el momentum de precio."
    arquitectura = """
- compute_flow_proxy(): combinacion de retorno*volumen, OBV y CMF.
- compute_price_momentum(): retorno porcentual a 20 dias.
"""
    formulas = f"**Flow Proxy:** {doc_flow}"
    salidas = """
- Flow Proxy: z-score utilizado en rankings sectoriales y en el SLPM.
- Momentum de precio: retorno a 20 dias mostrado en tablas del reporte.
"""
    return template(proposito, arquitectura, formulas, salidas) + f"\n## Constantes de Flujo\n{table}\n"

def generate_breadth():
    doc_breadth = extract_docstring('indicators/breadth.py', 'compute_breadth') or 'Calcula % de sectores sobre EMAs y nuevos maximos/minimos.'
    constants = extract_constants(SETTINGS_FILE, 'BREADTH_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows) if rows else ''
    proposito = "Mide la amplitud del mercado sectorial (porcentaje de sectores sobre sus EMAs) y detecta divergencias."
    arquitectura = """
- compute_breadth(): porcentajes sobre EMA20, EMA50, EMA200.
- readth_equity.py: avances/descensos del mercado general.
"""
    formulas = f"**Breadth:** {doc_breadth}"
    salidas = """
- % sobre EMA20/50/200 mostrado en la seccion Breadth de Mercado.
- Divergencias breadth en la seccion de Divergencias Detectadas.
"""
    return template(proposito, arquitectura, formulas, salidas) + f"\n## Constantes de Breadth\n{table}\n"

def generate_wyckoff():
    doc_structural = extract_docstring('indicators/wyckoff.py', 'wyckoff_structural_score')
    doc_tactical = extract_docstring('indicators/wyckoff.py', 'wyckoff_tactical_score')
    doc_score = extract_docstring('indicators/wyckoff.py', 'wyckoff_score')
    wyckoff_constants = extract_constants(SETTINGS_FILE, 'WYCKOFF_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(wyckoff_constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    proposito = "Proporciona un score continuo de estructura de precios para ETFs sectoriales y acciones lideres, basado en los principios de Wyckoff (acumulacion/distribucion)."
    arquitectura = """
- wyckoff_structural_score(): trend + ATR (70%).
- wyckoff_tactical_score(): volume + effort (30%).
- wyckoff_score(): combinacion ponderada de ambos.
- wyckoff_structure_core(): clasifica en MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.
"""
    formulas = f"""
**Score Estructural:** {doc_structural or '0.60*trend_norm + 0.40*compression_norm'}
**Score Tactico:** {doc_tactical or '0.50*volume_norm + 0.50*effort_norm'}
**Score Combinado:** {doc_score or '0.70*structural + 0.30*tactical'}
"""
    salidas = """
- Fase Wyckoff (MARKUP, ACCUMULATION, RANGE, DISTRIBUTION) en rankings sectoriales.
- Wyckoff Leadership Score (WLS) en tablas de lideres.
- Confianza y dispersion de componentes en metadatos.
"""
    return template(proposito, arquitectura, formulas, salidas) + f"\n## Constantes Configurables\n{table}\n"

def generate_slpm():
    doc_state = extract_docstring('indicators/state_machine.py', 'classify_leadership_state')
    state_machine_content = read_file('indicators/state_machine.py')
    thresholds = {}
    for match in re.finditer(r"'(\w+)':\s*([\d.]+)", state_machine_content):
        thresholds[match.group(1)] = match.group(2)
    rows = [[f'{k}', f'{v}'] for k, v in sorted(thresholds.items())]
    table = format_table(['Parametro', 'Valor'], rows)
    proposito = "Audita la calidad del liderazgo del sector #1 del ranking. No es otro ranking: evalua si el lider es estructuralmente solido."
    arquitectura = """
- compute_leader_breadth_v2(): amplitud del liderazgo (RS, momentum, flujo, Wyckoff).
- compute_leader_integrity(): LIS (intensidad/calidad de los lideres individuales).
- compute_flow_divergence_v2(): divergencias entre flujo de lideres y sector.
- classify_leadership_state(): State Machine con 6 estados + UNRESOLVED.
- confirm_transition(): histeresis temporal.
"""
    formulas = f"**State Machine:** {doc_state or 'Jerarquia de umbrales documentada en state_machine.py.'}"
    salidas = """
- Estado (CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY, LOST, UNRESOLVED).
- Leader Breadth, LIS, Flow Divergence 2.0, Effective Breadth.
- LQ Dimensions (P, C, S, Cf).
- Seccion completa en el reporte bajo 'Structural Leadership (SLPM v1.2)'.
"""
    return template(proposito, arquitectura, formulas, salidas) + f"\n## Umbrales de la State Machine\n{table}\n"

def generate_opciones():
    proposito = "Calcula el PCR (Put/Call Ratio) y el IHR (Institutional Hedge Ratio) a partir de datos de CBOE."
    arquitectura = """
- compute_pcr_signals(): orquestador principal.
- options_metrics.py: funciones de calculo (IHR, PCR, Put/Call Share, etc.).
- classify_pcr(): clasifica el Z-Score del PCR en Panico, Miedo, Neutral, Optimismo, Euforia.
- classify_ihr(): clasifica el IHR en Especulacion, Equilibrado, Cobertura institucional.
"""
    formulas = """
- **IHR:** PCR Indices / PCR Acciones.
- **Volume PCR:** Put Volume / Call Volume.
- **OI PCR:** Put OI / Call OI.
"""
    salidas = """
- Seccion 'Sentimiento de Opciones (OMS v2.0)' en el reporte.
- PCR Total, PCR Indices, PCR Acciones, IHR, Volumen en Indices, Put/Call Share.
"""
    limitaciones = "El Z-Score del PCR requiere al menos 20 dias de historial. Con menos de 20 registros, no se calcula."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)

def generate_darkpool():
    constants = extract_constants(SETTINGS_FILE, 'DARKPOOL_')
    rows = [[f'{k}', f'{v}'] for k, v in sorted(constants.items())]
    table = format_table(['Constante', 'Valor'], rows)
    proposito = "Mide el porcentaje de volumen negociado en ATS (Alternative Trading Systems) respecto al volumen total, usando datos de FINRA."
    arquitectura = """
- compute_darkpool_signals(): orquestador principal.
- FinraProvider: descarga datos ATS semanales.
- _compute_z_for_window(): calcula Z-Score robusto para cada ventana.
- classify_darkpool(): clasifica en Acumulacion/Distribucion extrema, fuerte, moderada o Neutral.
"""
    formulas = """
- Z-Score robusto (mediana/MAD) para ventanas de 13, 26, 52 y 104 semanas.
- % ATS medio: media del porcentaje de volumen ATS entre todos los tickers.
"""
    salidas = """
- Seccion 'Actividad en ATS - Dark Pools (FINRA v1.0)' en el reporte.
- % Volumen en ATS medio, Z-Scores por ventana, Top 5 tickers.
- Advertencia de obsolescencia si los datos tienen >21 dias.
"""
    limitaciones = "Los datos de FINRA pueden tener un desfase de varias semanas. Si la antiguedad supera los 21 dias, los datos no se usan para clasificacion actual."
    return template(proposito, arquitectura, formulas, salidas, limitaciones) + f"\n## Constantes\n{table}\n"

def generate_mte():
    proposito = "Infiere el escenario macro que el mercado parece estar descontando, basado en 4 motores (SRS, SHS, CLS, IPS)."
    arquitectura = """
- compute_srs(): Sector Rotation Score.
- compute_shs(): Safe Haven Score.
- compute_cls(): Credit/Liquidity Stress Score.
- compute_ips(): Inflation Pressure Score.
- compute_msi(): Market Stress Index (SRS + SHS + CLS).
- compute_ipi(): Inflation Pressure Index.
- classify_mte(): clasifica en CRISIS, RECESSION, STAGFLATION, SOFT LANDING, EXPANSION, MIXED.
- compute_confidence(): confianza del escenario (distancia a umbrales + consenso).
"""
    formulas = """
- **MSI:** agregacion de SRS, SHS y CLS (0-100).
- **IPI:** basado en IPS (0-100).
- **Confianza:** 0.6 * distancia_umbrales + 0.4 * consenso_motores.
"""
    salidas = """
- Seccion 'Market Transition Engine (MTE v1.0)' en el reporte.
- Escenario candidato, MSI, IPI, scores de los 4 motores, Signal Consistency.
"""
    limitaciones = "El escenario se marca como (UNCONFIRMED) si la confianza es inferior al 50%. La confianza no esta calibrada historicamente."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)

def generate_lideres():
    proposito = "Selecciona las mejores empresas de cada sector/indice en fase favorable (ACCUMULATION o MARKUP) usando el Wyckoff Leadership Score (WLS)."
    arquitectura = """
- stock_leader.py: compute_stock_metrics(), compute_wls(), generate_leader_section().
- index_leaders.py: analogo para indices internacionales.
- Fuente de holdings: data/etf_holdings.csv (sectores) y data/index_holdings.csv (indices).
- Actualizacion trimestral automatica via GitHub Actions.
"""
    formulas = """
- **WLS:** 0.35*rs_z + 0.25*flow_proxy_z_norm + 0.25*rws_z + 0.10*stab_z, con bonus por persistencia.
- **RWS:** Relative Wyckoff Score (normalizacion intra-sector/indice).
"""
    salidas = """
- Tablas 'Acciones Seleccionadas por el Modelo de Liderazgo Sectorial' en el reporte.
- Tablas 'Indices Internacionales - Oportunidades de Acumulacion' en el reporte.
- Archivos CSV: nalisis_lideres.csv y nalisis_lideres_internacionales.csv.
"""
    limitaciones = "Solo se muestran sectores/indices en fase ACCUMULATION o MARKUP. El resto se omiten por no cumplir criterios de liderazgo estructural."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)

def generate_reporte():
    proposito = "Genera el reporte diario en formato Markdown con todas las secciones del radar."
    arquitectura = """
- generate_daily_report(): funcion principal con mas de 20 parametros.
- Secciones: Resumen de Regimenes, Data Freshness, Divergencias, Breadth, Momentum, Flujo, Rankings, Opportunity Map, SLPM, Lideres, Opciones, MTE, Dark Pools, Cross-Asset, Anti-Double-Counting.
"""
    formulas = "No aplica (formato de salida)."
    salidas = "Archivo outputs/report/reporte_diario.md generado en cada ejecucion."
    return template(proposito, arquitectura, formulas, salidas)

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
        ('sensitivity_persistence.py', 'Sensibilidad de Persistence a umbral y lookback'),
        ('sensitivity_coverage.py', 'Sensibilidad de Coverage en SLPM'),
        ('regresion_base_vs_lis.py', 'Regresión BASE vs BASE+LIS'),
        ('audit_rs_flow_channels.py', 'Auditoría de canales RS/Flow'),
        ('redundancia_mte_fc.py', 'Redundancia MTE vs Financial Conditions'),
        ('solapamiento_fls_liquidity.py', 'Solapamiento FLS vs Liquidity'),
    ]
    rows = [[name, desc] for name, desc in scripts]
    table = format_table(['Script', 'Descripcion'], rows)
    proposito = "Scripts independientes que validan la estabilidad y robustez del sistema. No modifican el codigo productivo."
    arquitectura = f"Ubicados en alidation/. Se ejecutan manualmente con py validation/<script>.py."
    formulas = "No aplica."
    salidas = "Resultados en consola y archivos CSV en outputs/."
    return template(proposito, arquitectura, formulas, salidas) + f"\n{table}\n"


def generate_etf_primary_flow():
    proposito = "Documenta los modulos de flujo primario de ETFs (ETF Primary Flow), que estiman creaciones/redenciones de participaciones usando los cambios en Shares Outstanding multiplicados por el NAV."
    arquitectura = """
- data/providers/ssga_fund_data.py: ETFs SPDR USA (11 sectores) + FEZ.
- data/providers/blackrock_fund_data.py: iShares DAXEX (DAX 40).
- data/providers/blackrock_isf_fund_data.py: iShares ISF.L (FTSE 100).
- data/providers/amundi_fund_data.py: Amundi LYXI (Ibex 35).
- Todos guardan históricos en outputs/history/ y se integran en run.py.
"""
    formulas = """
- **ETF Primary Flow:** ΔSharesOutstanding × NAV.
- **Flow % Assets:** EstimatedFlow / AUM (o TotalNetAssets), en decimal.
- **Flow Z-Score:** rolling z-score (120 días) sobre Flow % Assets.
- **Flow 5d/20d:** medias móviles del flujo estimado.
"""
    salidas = """
- outputs/history/etf_primary_flow.csv
- outputs/history/blackrock_dax_primary_flow.csv
- outputs/history/blackrock_isf_primary_flow.csv
- outputs/history/amundi_lyxi_primary_flow.csv
- Secciones en el reporte: '## Flujo Primario ETF (SPDR)', '## Flujo Primario DAXEX', '## Flujo Primario ISF.L', '## Flujo Primario LYXI'.
"""
    limitaciones = "El flujo primario estimado no es flujo institucional directo; solo refleja cambios en participaciones. Amundi requiere dos fechas efectivas para calcular el flujo."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)


def generate_cftc_position_flow():
    proposito = "Documenta el módulo de posicionamiento semanal de futuros financieros (CFTC TFF), que mide cambios en posiciones reportadas por tipo de participante."
    arquitectura = """
- data/providers/cftc_data.py: descarga CSV de CFTC TFF (Futures Only).
- Selecciona contratos: E-MINI S&P 500, Nasdaq-100, Russell, DJIA, VIX, UST 10Y.
- Calcula net position, position change y flow z-score por participante (dealer, asset_mgr, lev_money).
"""
    formulas = """
- **Net Position:** Long - Short.
- **Position Change:** NetPosition(t) - NetPosition(t-1).
- **Flow Z-Score:** rolling 52 semanas del position change.
"""
    salidas = """
- outputs/history/cftc_position_flow.csv
- Sección en el reporte: '## Posicionamiento CFTC (TFF, Semanal)'.
"""
    limitaciones = "Frecuencia semanal. No representa flujo de capital al contado, sino posicionamiento en futuros."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)


def generate_sec_nport_positions():
    proposito = "Documenta los módulos de extracción de posiciones institucionales desde SEC N-PORT, con granularidad fondo + activo + fecha de reporte."
    arquitectura = """
- data/providers/sec_nport_positioning.py: extrae posiciones de N-PORT (REGISTRANT, FUND_REPORTED_HOLDING, IDENTIFIERS).
- data/providers/sec_nport_position_change.py: calcula cambios de balance por fondo y activo.
- data/providers/sec_nport_quarters_position_change.py: compara trimestres Q1 y Q2.
- data/providers/sec_fund_flow.py: extrae flujos de fondos desde FUND_REPORTED_INFO.
- data/providers/sec_nport_international_leader_flows.py: cruza N-PORT de FEZ con líderes internacionales.
"""
    formulas = """
- **Position Change:** BALANCE(t) - BALANCE(t-1).
- **Position Change %:** PositionChange / BALANCE(t-1) × 100.
- **Net Fund Flow:** Sales + Reinvestment - Redemption.
"""
    salidas = """
- outputs/history/sec_nport_positions.csv
- outputs/history/sec_nport_position_change.csv
- outputs/history/sec_nport_position_change_quarterly.csv
- outputs/history/sec_fund_flow.csv
- outputs/report/sec_nport_international_leader_flows.csv
"""
    limitaciones = "Los datasets N-PORT se publican trimestralmente aunque los datos son mensuales. No se integra en run.py diario."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)


def generate_backup_providers():
    proposito = "Describe los proveedores de respaldo multi-API y los mecanismos de resiliencia: rate limiting, circuit breaker y validación cruzada."
    arquitectura = """
- data/providers/backup_providers.py: Alpha Vantage, Tiingo, Twelve Data, Finnhub, FMP.
- RateLimiter: límites diarios y por minuto por proveedor.
- CircuitBreaker: desactiva temporalmente tras fallos consecutivos.
- Validación cruzada contra caché local.
- data/providers/polygon.py: proveedor Polygon.io / Massive.
"""
    formulas = """
- Presupuesto global de respaldo: 20 llamadas por ejecución.
- Límites específicos configurados en RateLimiter por proveedor.
"""
    salidas = """
- DataFrames de OHLCV unificados (MultiIndex con ticker canónico).
- Mensajes de trazabilidad: [RESPALDO], [RATE], [CIRCUIT], [VALIDACIÓN].
"""
    limitaciones = "Solo se activa cuando Yahoo Finance falla. No reemplaza la fuente primaria."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)


def generate_instrument_registry():
    proposito = "Documenta el registro central de instrumentos que mapea tickers canónicos (Yahoo) a símbolos específicos de cada proveedor."
    arquitectura = """
- src/instrument_registry.py: diccionario INSTRUMENTS con equivalencias por proveedor.
- Función resolve_symbol(canonical_ticker, provider) devuelve el símbolo correcto o None.
- Cobertura explícita para BRK-B, BF-B, ^GSPC, ^STOXX50E, ^VIX3M, MOGA y otros.
- Los proveedores normalizan siempre al ticker canónico en sus salidas.
"""
    formulas = "No aplica (mapeo estático)."
    salidas = "Tickers normalizados en todos los DataFrames de proveedores, evitando duplicados."
    limitaciones = "Requiere actualización manual al añadir nuevos instrumentos o proveedores."
    return template(proposito, arquitectura, formulas, salidas, limitaciones)



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
        '10_flujo_primario_etf.md': generate_etf_primary_flow,
        '11_cftc_position_flow.md': generate_cftc_position_flow,
        '12_sec_nport_positions.md': generate_sec_nport_positions,
        '13_backup_providers.md': generate_backup_providers,
        '14_instrument_registry.md': generate_instrument_registry,
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
