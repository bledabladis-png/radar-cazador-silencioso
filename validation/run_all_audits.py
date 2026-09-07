# -*- coding: utf-8 -*-
# validation/run_all_audits.py
# Fase E v4.3: Monitorizacion continua - comprobaciones rapidas
import os
import pandas as pd
from datetime import datetime

OUT = 'outputs/audit/informe_monitorizacion.md'
report = []
def log(msg=''):
    print(msg)
    report.append(msg)

def check(cond, ok_msg, fail_msg):
    if cond:
        log(f'✅ {ok_msg}')
    else:
        log(f'❌ {fail_msg}')

log('# Informe de Monitorizacion Continua')
log(f'**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('')

# 1. Duplicados en holdings
for path in ['data/etf_holdings.csv', 'data/index_holdings.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        dups = df.duplicated(subset=['etf','ticker']).sum()
        check(dups == 0, f'{path}: sin duplicados ({len(df)} registros)', f'{path}: {dups} duplicados')
    else:
        log(f'❌ {path} no existe')

# 2. NaN en CSV de lideres
for path in ['outputs/report/analisis_lideres.csv', 'outputs/report/analisis_lideres_internacionales.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        nan_cols = df.columns[df.isna().any()].tolist()
        # Se permite NaN en rs_mom y wls? Mejor detectar NaN en columnas criticas
        critical = ['rs', 'flow_proxy_z', 'wyckoff_score', 'wls']
        nan_critical = {c: df[c].isna().sum() for c in critical if c in df.columns}
        if any(v > 0 for v in nan_critical.values()):
            log(f'❌ {path}: NaN en columnas criticas {nan_critical}')
        else:
            log(f'✅ {path}: sin NaN en columnas criticas')
        # Persistencia en rango 0-1
        if 'persistence_10d' in df.columns:
            p = df['persistence_10d'].dropna()
            check(p.between(0,1).all(), f'{path}: persistencia en rango 0-1', f'{path}: persistencia fuera de rango [{p.min():.2f}, {p.max():.2f}]')
    else:
        log(f'ℹ️ {path} no disponible (puede no haberse generado hoy)')

# 3. Frescura de fuentes
for path, col in [('outputs/history/pcr_history.csv','date'), ('outputs/history/darkpool_history.csv','week')]:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=[col])
        last = df[col].max()
        age = (pd.Timestamp.now() - pd.Timestamp(last)).days
        estado = 'CURRENT' if age <= 7 else ('RECENT' if age <= 14 else ('STALE' if age <= 21 else 'ARCHIVAL'))
        log(f'ℹ️ {path}: ultimo dato {last.date()} ({age} dias, {estado})')
    else:
        log(f'ℹ️ {path} no existe')

# 3b. Matriz de Evidencia
path_ev = 'outputs/history/evidence_matrix.csv'
if os.path.exists(path_ev):
    ev = pd.read_csv(path_ev)
    check(len(ev) == 11, 'Matriz de Evidencia: 11 sectores', f'Matriz de Evidencia: {len(ev)} sectores')
    lecturas = set(ev['alignment_reading'].dropna().unique())
    lecturas_ok = {'EVIDENCIA PREDOMINANTEMENTE FAVORABLE','EVIDENCIA PREDOMINANTEMENTE DESFAVORABLE','EVIDENCIA MIXTA','EVIDENCIA INSUFICIENTE'}
    check(lecturas.issubset(lecturas_ok), f'Matriz de Evidencia: lecturas validas {lecturas}', f'Matriz de Evidencia: lecturas invalidas {lecturas}')
    check('evidence_score' not in ev.columns, 'Matriz de Evidencia: sin evidence_score', 'Matriz de Evidencia: contiene evidence_score')
    check('credit_evidence' in ev.columns and 'volatility_evidence' in ev.columns, 'Matriz de Evidencia: columnas contexto presentes', 'Matriz de Evidencia: faltan columnas contexto')
    if 'n_sector_evidence_valid' in ev.columns:
        check(ev['n_sector_evidence_valid'].max() <= 5, 'Matriz de Evidencia: balance maximo 5 evidencias sectoriales', f"Matriz de Evidencia: balance maximo {ev['n_sector_evidence_valid'].max()}")
else:
    log('❌ No se encontro evidence_matrix.csv')

# 3c. Integridad específica para sector_concentration.csv
sc_path = 'outputs/history/sector_concentration.csv'
if os.path.exists(sc_path):
    sc = pd.read_csv(sc_path)
    check(sc['date'].notna().all(), f'{sc_path}: date sin NaN', f'{sc_path}: {sc["date"].isna().sum()} fechas NaN')
    check(len(sc) == 11, f'{sc_path}: 11 filas', f'{sc_path}: {len(sc)} filas')
    check(sc['sector'].nunique() == 11, f'{sc_path}: 11 sectores', f'{sc_path}: {sc["sector"].nunique()} sectores')
else:
    log(f'❌ {sc_path} no existe')

# 3c-bis. Cobertura informativa de medianas
sc_path = 'outputs/history/sector_concentration.csv'
if os.path.exists(sc_path):
    sc = pd.read_csv(sc_path)
    if {'flow_median','wyckoff_median'}.issubset(sc.columns):
        n_flow_valid = sc['flow_median'].notna().sum()
        n_wyck_valid = sc['wyckoff_median'].notna().sum()
        log(f'ℹ️ {sc_path}: flow_median válidos en {n_flow_valid}/11 sectores')
        log(f'ℹ️ {sc_path}: wyckoff_median válidos en {n_wyck_valid}/11 sectores')
    else:
        log(f'ℹ️ {sc_path}: no contiene columnas de medianas')
else:
    log(f'❌ {sc_path} no existe')

# 3c. Duplicados en CSVs sectoriales
csv_files = [
    ('outputs/history/sector_concentration.csv', ['date','sector']),
    ('outputs/history/sector_flow_characteristics.csv', ['date','sector']),
    ('outputs/history/sector_breadth.csv', ['date','sector']),
    ('outputs/history/sector_breadth_momentum.csv', ['date','sector']),
    ('outputs/history/sector_leader_divergence.csv', ['date','sector']),
    ('outputs/history/sector_wyckoff_distribution.csv', ['date','sector']),
    ('outputs/history/leader_representativeness.csv', ['date','sector','ticker']),
    ('outputs/history/sector_regime_matrix.csv', ['date','sector']),
    ('outputs/history/sector_dispersion.csv', ['date']),
    ('outputs/history/sector_correlation_summary.csv', ['date','window']),
    ('outputs/history/cross_asset_correlation.csv', ['date','window','sector','asset']),
    ('outputs/history/cross_asset_context.csv', ['date','window','sector','asset_class']),
    ('outputs/history/volatility_structure.csv', ['date']),
    ('outputs/history/data_quality.csv', ['date','source']),
    ('outputs/history/sector_correlation_matrix.csv', ['date','window','sector1','sector2']),
    ('outputs/history/evidence_matrix.csv', ['date','sector']),
]
for f, subset in csv_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        if set(subset).issubset(df.columns):
            dup = df.duplicated(subset=subset).sum()
            check(dup == 0, f'{f}: sin duplicados {subset}', f'{f}: {dup} duplicados')
        else:
            log(f'ℹ️ {f} no tiene columnas {subset}')
    else:
        log(f'ℹ️ {f} no existe')

# 3c-ter. Validación de regímenes precio-flujo primario
sfc_path = 'outputs/history/sector_flow_characteristics.csv'
if os.path.exists(sfc_path):
    sfc = pd.read_csv(sfc_path, encoding='utf-8')
    if {'price_ret_20d','flow_20d_sum','price_flow_regime_20d'}.issubset(sfc.columns):
        mask_valid = sfc['price_ret_20d'].notna() & sfc['flow_20d_sum'].notna()
        bad = mask_valid & sfc['price_flow_regime_20d'].isna()
        check(bad.sum() == 0, f'{sfc_path}: regímenes 20d presentes cuando datos válidos', f'{sfc_path}: {bad.sum()} regímenes faltantes')
        mask_missing = sfc['price_ret_20d'].isna() | sfc['flow_20d_sum'].isna()
        good_nd = mask_missing & (sfc['price_flow_regime_20d'] == 'N/D')
        # No es un fallo si el régimen es N/D; comprobamos que no esté vacío
        empty_nd = mask_missing & sfc['price_flow_regime_20d'].isna()
        check(empty_nd.sum() == 0, f'{sfc_path}: sin regímenes NaN cuando faltan datos', f'{sfc_path}: {empty_nd.sum()} N/D ausentes')
        log(f'ℹ️ {sfc_path}: price_flow_regime_20d válidos en {sfc["price_flow_regime_20d"].notna().sum()}/11 sectores')
    else:
        log(f'ℹ️ {sfc_path} no tiene columnas de régimen')
else:
    log(f'❌ {sfc_path} no existe')

# 4. Reporte diario
path_report = 'outputs/report/reporte_diario.md'
if os.path.exists(path_report):
    size = os.path.getsize(path_report)
    with open(path_report, 'r', encoding='utf-8') as f:
        contenido = f.read()
    secciones = ['Resumen de Regimenes', 'Breadth de Mercado', 'Rankings Sectoriales', 'Opportunity Map', 'Sentimiento de Opciones', 'Market Transition Engine']
    faltan = [s for s in secciones if s not in contenido]
    check(size > 10000, f'Reporte diario generado ({size} bytes)', f'Reporte diario demasiado pequeño ({size} bytes)')
    check(not faltan, f'Todas las secciones principales presentes', f'Secciones faltantes: {faltan}')
else:
    log('❌ No se encontro reporte_diario.md')

# 5. Verificación de selección de líderes
try:
    import subprocess
    subprocess.run(['py', 'validation/verify_leader_selection.py'], check=False)
except Exception as e:
    print(f'Error en verificación de líderes: {e}')

# 6. Estados clave
for path, key in [('outputs/state/slpm_state.json','state'), ('outputs/state/mte_state.json','scenario')]:
    if os.path.exists(path):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log(f'ℹ️ {path}: {data.get(key, "N/A")}')
    else:
        log(f'ℹ️ {path} no existe')

# Guardar informe
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f'\nInforme guardado en {OUT}')
