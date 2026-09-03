# -*- coding: utf-8 -*-
# validation/leader_selection_audit.py
# Auditoria descriptiva de la seleccion de lideres sectoriales e internacionales
import pandas as pd
import os

output_lines = []
def log(msg=''):
    print(msg)
    output_lines.append(msg)

log('# Auditoria de Seleccion de Lideres')
log(f"Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
log('')

files = {
    'Sectorial': 'outputs/report/analisis_lideres.csv',
    'Internacional': 'outputs/report/analisis_lideres_internacionales.csv'
}

for label, path in files.items():
    if not os.path.exists(path):
        log(f'## {label}: archivo no encontrado ({path})')
        continue
    
    df = pd.read_csv(path)
    log(f'## {label}')
    log(f'Columnas: {df.columns.tolist()}')
    log(f'Registros: {len(df)}')
    log('')
    
    # Detectar columna de agrupacion
    group_col = 'sector' if 'sector' in df.columns else ('indice' if 'indice' in df.columns else None)
    if group_col is None:
        log('  No se encontro columna de agrupacion (sector/indice).')
        continue
    
    # Estadisticas por grupo
    for group_name, group in df.groupby(group_col):
        wls = group['wls'].dropna()
        wy = group['wyckoff_score'].dropna()
        if wls.empty:
            continue
        log(f'  {group_name}: n={len(group)}')
        log(f'    WLS: media={wls.mean():.3f}, mediana={wls.median():.3f}, min={wls.min():.3f}, max={wls.max():.3f}')
        if not wy.empty:
            top1_wy = group.sort_values('wls', ascending=False).iloc[0]['wyckoff_score']
            flag = '  *** POSIBLE "MEJOR DE UN GRUPO MALO" (wyckoff_score negativo) ***' if top1_wy < 0 else ''
            log(f'    Wyckoff score del lider #1: {top1_wy:+.3f}{flag}')
        log('')
    
    # Correlaciones entre componentes (si existen)
    comp_cols = ['rs_z', 'flow_proxy_z_norm', 'rws_z', 'stab_z']
    if all(c in df.columns for c in comp_cols):
        corr = df[comp_cols].corr(method='spearman')
        log('  Correlacion Spearman entre componentes WLS:')
        log(corr.round(3).to_string())
        log('')
    else:
        # Usar variables base como proxy
        alt_cols = ['rs', 'flow_proxy_z', 'wyckoff_score', 'stability']
        existing = [c for c in alt_cols if c in df.columns]
        if len(existing) >= 2:
            corr = df[existing].corr(method='spearman')
            log('  Correlacion Spearman entre variables base (proxy componentes WLS):')
            log(corr.round(3).to_string())
            log('')
    
    # Persistencia
    if 'persistence_10d' in df.columns:
        p = df['persistence_10d'].dropna()
        if not p.empty:
            log(f'  Persistencia: media={p.mean():.2f}, min={p.min():.2f}, max={p.max():.2f}')
            log('')

# Guardar informe
with open('outputs/audit/auditoria_lideres.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))
print('\nInforme guardado en outputs/audit/auditoria_lideres.md')
