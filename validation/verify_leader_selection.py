# -*- coding: utf-8 -*-
# validation/verify_leader_selection.py
# Verifica que los líderes seleccionados son los más representativos según holdings y WLS
import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = 'outputs/audit/verify_leader_selection.md'
HAS_ERRORS = False
HAS_WARNINGS = False

def log(msg=''):
    global HAS_ERRORS, HAS_WARNINGS
    print(msg)
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
    if '❌' in msg:
        HAS_ERRORS = True
    elif '⚠️' in msg:
        HAS_WARNINGS = True

# Limpiar informe anterior
if os.path.exists(OUT):
    os.remove(OUT)

log('# Verificación de Selección de Líderes')
log(f'**Fecha:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
log('')

# ------------------------------------------------------------
# 1. SECTORES: etf_holdings.csv (con weight)
# ------------------------------------------------------------
log('## 1. Líderes Sectoriales (ETF USA)')
try:
    h = pd.read_csv('data/etf_holdings.csv')
    log(f'Columnas en etf_holdings.csv: {h.columns.tolist()}')
    l = pd.read_csv('outputs/report/analisis_lideres.csv')

    if 'weight' not in h.columns:
        log('  ❌ No existe columna weight en etf_holdings.csv. No se puede validar por peso.')
    else:
        h = h.sort_values(['etf', 'weight'], ascending=[True, False])
        top20 = h.groupby('etf').head(20)

        sectores_lideres = l['sector'].unique()
        for sector in sectores_lideres:
            tickers_lideres = set(l[l['sector'] == sector]['ticker'])
            tickers_top20 = set(top20[top20['etf'] == sector]['ticker'])
            missing = tickers_lideres - tickers_top20
            if missing:
                log(f'  ⚠️ {sector}: {len(missing)} ticker(s) no están en top 20: {missing}')
            else:
                log(f'  ✅ {sector}: todos los líderes están en top 20')
except Exception as e:
    log(f'  ❌ Error: {e}')
log('')

# ------------------------------------------------------------
# 2. ÍNDICES: index_holdings.csv (puede o no tener weight)
# ------------------------------------------------------------
log('## 2. Líderes Internacionales (Índices)')
try:
    h = pd.read_csv('data/index_holdings.csv')
    log(f'Columnas en index_holdings.csv: {h.columns.tolist()}')
    l = pd.read_csv('outputs/report/analisis_lideres_internacionales.csv')

    from config.index_tickers import INDEX_CONFIG

    indices_lideres = l['indice'].unique()
    for nombre_indice in indices_lideres:
        if nombre_indice in INDEX_CONFIG:
            etf = INDEX_CONFIG[nombre_indice]['etf_ticker']
            max_comp = INDEX_CONFIG[nombre_indice]['max_companies']
        else:
            log(f'  ⚠️ {nombre_indice}: no encontrado en INDEX_CONFIG. Se omite.')
            continue

        h_idx = h[h['etf'] == etf].copy()
        if h_idx.empty:
            log(f'  ❌ {nombre_indice} ({etf}): no hay holdings en index_holdings.csv')
            continue

        if 'weight' in h_idx.columns:
            h_idx = h_idx.sort_values('weight', ascending=False)
            candidate_universe = h_idx.head(max_comp)
            log(f'  ℹ️ {nombre_indice} ({etf}): usando top {max_comp} por weight')
        else:
            candidate_universe = h_idx.head(max_comp)
            log(f'  ⚠️ {nombre_indice} ({etf}): no hay columna weight. Usando orden de archivo (primeros {max_comp}).')

        tickers_lideres = set(l[l['indice'] == nombre_indice]['ticker'])
        tickers_candidatos = set(candidate_universe['ticker'])
        missing = tickers_lideres - tickers_candidatos

        if missing:
            log(f'  ⚠️ {nombre_indice}: {len(missing)} ticker(s) no están en top {max_comp}: {missing}')
        else:
            log(f'  ✅ {nombre_indice}: todos los líderes están dentro de top {max_comp}')
except Exception as e:
    log(f'  ❌ Error: {e}')
log('')

# ------------------------------------------------------------
# 3. Coherencia de Top 5 en CSVs de salida
# ------------------------------------------------------------
log('## 3. Coherencia de Top 5 en CSVs de salida')
try:
    l_sec = pd.read_csv('outputs/report/analisis_lideres.csv')
    for sector, group in l_sec.groupby('sector'):
        if len(group) < 5:
            log(f'  ⚠️ Sector {sector}: solo {len(group)} líderes guardados (< 5)')
            continue
        top5 = group.head(5)
        if top5['wls'].is_monotonic_decreasing:
            log(f'  ✅ Sector {sector}: top 5 ordenado por WLS desc')
        else:
            log(f'  ❌ Sector {sector}: top 5 no está ordenado por WLS desc')
except Exception as e:
    log(f'  ❌ Error sectores: {e}')

try:
    l_int = pd.read_csv('outputs/report/analisis_lideres_internacionales.csv')
    for indice, group in l_int.groupby('indice'):
        if len(group) < 5:
            log(f'  ⚠️ Índice {indice}: solo {len(group)} líderes guardados (< 5)')
            continue
        top5 = group.head(5)
        if top5['wls'].is_monotonic_decreasing:
            log(f'  ✅ Índice {indice}: top 5 ordenado por WLS desc')
        else:
            log(f'  ❌ Índice {indice}: top 5 no está ordenado por WLS desc')
except Exception as e:
    log(f'  ❌ Error índices: {e}')

log('')
log('## Resumen')
log('Verificación finalizada. Revisar los checks anteriores.')
