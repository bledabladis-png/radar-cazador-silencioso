# -*- coding: utf-8 -*-
# validation/run_all_audits.py
# Fase E v4.3: Monitorizacion continua - comprobaciones rapidas
import os
import sys
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
