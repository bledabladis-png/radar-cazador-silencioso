# -*- coding: utf-8 -*-
# validation/holdings_audit.py
# Fase 4: Auditoria de holdings y seleccion de lideres
import pandas as pd
import os
from datetime import datetime

OUTPUT_MD = 'outputs/audit/auditoria_holdings.md'

def log(msg=''):
    print(msg)
    with open(OUTPUT_MD, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

# Limpiar informe anterior
if os.path.exists(OUTPUT_MD):
    os.remove(OUTPUT_MD)

log('# Auditoria Fase 4 - Holdings y Lideres')
log(f'**Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('')

# ------------------------------------------------------------
# 1. Cargar archivos de holdings
# ------------------------------------------------------------
for label, path in [('Sectorial (ETF holdings)', 'data/etf_holdings.csv'),
                    ('Internacional (Index holdings)', 'data/index_holdings.csv')]:
    log(f'## {label}')
    if not os.path.exists(path):
        log(f'Archivo no encontrado: {path}')
        continue
    df = pd.read_csv(path)
    log(f'Registros totales: {len(df)}')
    log(f'Columnas: {df.columns.tolist()}')
    if 'etf' in df.columns and 'ticker' in df.columns:
        # Duplicados
        dups = df.duplicated(subset=['etf','ticker']).sum()
        # Tickers nulos o vacios
        nulls = df['ticker'].isna().sum() + (df['ticker'].astype(str).str.strip() == '').sum()
        # Tickers con espacios o minusculas
        lower_ok = (df['ticker'] == df['ticker'].str.upper()).sum() != len(df)
        log(f'Duplicados: {dups}')
        log(f'Tickers nulos/vacios: {nulls}')
        log(f'Tickers en minuscula: {lower_ok}')
        # Conteo por ETF/índice
        summary = df.groupby('etf')['ticker'].agg(['count']).reset_index()
        log('Conteo por ETF/indice:')
        log(summary.to_string(index=False))
    else:
        log('No se encontraron columnas etf/ticker esperadas.')
    log('')

# ------------------------------------------------------------
# 2. Coherencia con CSVs de lideres
# ------------------------------------------------------------
for label, holdings_path, leaders_path, group_col in [
    ('Sectorial', 'data/etf_holdings.csv', 'outputs/report/analisis_lideres.csv', 'sector'),
    ('Internacional', 'data/index_holdings.csv', 'outputs/report/analisis_lideres_internacionales.csv', 'indice')
]:
    log(f'## Coherencia {label}')
    if not os.path.exists(holdings_path) or not os.path.exists(leaders_path):
        log('Archivos necesarios no disponibles.')
        continue
    h = pd.read_csv(holdings_path)
    l = pd.read_csv(leaders_path)
    # Tickers en holdings
    tickers_holdings = set(h['ticker'].str.upper())
    # Tickers en líderes
    tickers_leaders = set(l['ticker'].str.upper()) if 'ticker' in l.columns else set()
    if not tickers_leaders:
        log('CSV de líderes sin tickers.')
        continue
    missing_in_holdings = tickers_leaders - tickers_holdings
    log(f'Tickers en líderes no presentes en holdings: {len(missing_in_holdings)}')
    if missing_in_holdings:
        log('  ' + ', '.join(sorted(missing_in_holdings)[:20]))
    # Revisar si hay líderes para todos los grupos con fase favorable
    log(f'Grupos en holdings: {sorted(h["etf"].unique())}')
    log(f'Grupos en líderes: {sorted(l[group_col].unique()) if group_col in l.columns else "N/A"}')
    log('')

# ------------------------------------------------------------
# 3. Revisar persistencia y WLS (post-fix)
# ------------------------------------------------------------
for label, path in [('Sectorial', 'outputs/report/analisis_lideres.csv'),
                    ('Internacional', 'outputs/report/analisis_lideres_internacionales.csv')]:
    log(f'## Persistencia y WLS {label}')
    if not os.path.exists(path):
        log('Archivo no disponible.')
        continue
    df = pd.read_csv(path)
    if 'persistence_10d' in df.columns:
        p = df['persistence_10d'].dropna()
        log(f'Rango persistencia: [{p.min():.2f}, {p.max():.2f}] (esperado 0-1)')
        if p.min() < 0 or p.max() > 1:
            log('  *** FUERA DE RANGO ***')
        else:
            log('  OK')
    if 'wls' in df.columns:
        w = df['wls'].dropna()
        log(f'WLS: media={w.mean():.3f}, min={w.min():.3f}, max={w.max():.3f}')
    # Posible "mejor de un grupo malo"
    group_col = 'sector' if 'sector' in df.columns else ('indice' if 'indice' in df.columns else None)
    if group_col and 'wyckoff_score' in df.columns:
        log('Lider #1 por grupo y su wyckoff_score:')
        for g, group in df.groupby(group_col):
            top = group.sort_values('wls', ascending=False).iloc[0]
            flag = ' *** POSIBLE PROBLEMA ***' if top['wyckoff_score'] < 0 else ''
            log(f'  {g}: {top["ticker"]} (wls={top["wls"]:.2f}, wyckoff={top["wyckoff_score"]:.2f}){flag}')
    log('')

log('## Conclusion')
log('Auditoria de holdings y lideres completada. Ver detalles anteriores.')
