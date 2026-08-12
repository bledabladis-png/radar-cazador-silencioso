# -*- coding: utf-8 -*-
# validation/data_freshness_audit.py
# Fase 3: Auditoria de frescura de datos y umbrales
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    FRESHNESS_CURRENT_DAYS, FRESHNESS_RECENT_DAYS, FRESHNESS_STALE_DAYS,
    MAX_NAN_RATIO, EXPECTED_SECTOR_COUNT, MIN_VALID_SECTORS, MIN_SECTOR_COVERAGE, MIN_VALID_TICKERS
)

def classify_age(days):
    if days <= FRESHNESS_CURRENT_DAYS:
        return 'CURRENT'
    elif days <= FRESHNESS_RECENT_DAYS:
        return 'RECENT'
    elif days <= FRESHNESS_STALE_DAYS:
        return 'STALE'
    else:
        return 'ARCHIVAL'

fuentes = []

# 1. CBOE (Opciones)
pcr_path = 'outputs/history/pcr_history.csv'
if os.path.exists(pcr_path):
    df = pd.read_csv(pcr_path, parse_dates=['date'])
    last = df['date'].max()
    days = (pd.Timestamp.now() - pd.Timestamp(last)).days
    fuentes.append(('CBOE (Opciones)', pcr_path, str(last.date()), days, classify_age(days)))
else:
    fuentes.append(('CBOE (Opciones)', pcr_path, 'NO EXISTE', None, 'DESCONOCIDO'))

# 2. FINRA (Dark Pools)
dp_path = 'outputs/history/darkpool_history.csv'
if os.path.exists(dp_path):
    df = pd.read_csv(dp_path, parse_dates=['week'])
    last = df['week'].max()
    days = (pd.Timestamp.now() - pd.Timestamp(last)).days
    fuentes.append(('FINRA (Dark Pools)', dp_path, str(last.date()), days, classify_age(days)))
else:
    fuentes.append(('FINRA (Dark Pools)', dp_path, 'NO EXISTE', None, 'DESCONOCIDO'))

# 3. FRED / Macro manual
macro_dir = 'data/macro_manual'
if os.path.isdir(macro_dir):
    max_date = None
    for f in os.listdir(macro_dir):
        if f.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(macro_dir, f), parse_dates=['date'])
                if 'date' in df.columns:
                    m = df['date'].max()
                    if max_date is None or m > max_date:
                        max_date = m
            except:
                pass
    if max_date is not None:
        days = (pd.Timestamp.now() - pd.Timestamp(max_date)).days
        fuentes.append(('FRED/Macro Manual', macro_dir, str(max_date.date()), days, classify_age(days)))
    else:
        fuentes.append(('FRED/Macro Manual', macro_dir, 'SIN FECHAS', None, 'DESCONOCIDO'))
else:
    fuentes.append(('FRED/Macro Manual', macro_dir, 'NO EXISTE', None, 'DESCONOCIDO'))

# 4. Yahoo Finance (Precios)
# No hay un CSV de precios cacheado local; la descarga es en vivo cada ejecución.
fuentes.append(('Yahoo Finance (Precios)', 'Descarga en vivo', 'HOY', 0, 'CURRENT'))

# Generar tabla
print('=== AUDITORIA DE FRESCURA DE DATOS ===')
print(f'Fecha actual: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
print(f'Umbrales: CURRENT<={FRESHNESS_CURRENT_DAYS}d, RECENT<={FRESHNESS_RECENT_DAYS}d, STALE<={FRESHNESS_STALE_DAYS}d, >{FRESHNESS_STALE_DAYS}=ARCHIVAL')
print()

table = '| Fuente | Archivo | Ultimo dato | Antiguedad (dias) | Estado |\n'
table += '|--------|---------|-------------|-------------------|--------|\n'
for fuente, archivo, ultimo, dias, estado in fuentes:
    dias_str = str(dias) if dias is not None else 'N/A'
    table += f'| {fuente} | {archivo} | {ultimo} | {dias_str} | {estado} |\n'

print(table)

# Configuracion de calidad
print('=== CONFIGURACION DE CALIDAD DE DATOS ===')
print(f'MAX_NAN_RATIO = {MAX_NAN_RATIO}')
print(f'EXPECTED_SECTOR_COUNT = {EXPECTED_SECTOR_COUNT}')
print(f'MIN_VALID_SECTORS = {MIN_VALID_SECTORS}')
print(f'MIN_SECTOR_COVERAGE = {MIN_SECTOR_COVERAGE}')
print(f'MIN_VALID_TICKERS = {MIN_VALID_TICKERS}')
print('(La verificacion real de cobertura se ejecuta en run.py con validate_market_data() y compute_breadth())')

# Guardar informe
with open('outputs/audit/auditoria_frescura.md', 'w', encoding='utf-8') as f:
    f.write('# Auditoria Fase 3 - Frescura y Proveedores de Datos\n\n')
    f.write(f'**Fecha:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n\n')
    f.write('## Resultados\n\n')
    f.write(table + '\n')
    f.write('## Configuracion de Calidad\n\n')
    f.write(f'- MAX_NAN_RATIO = {MAX_NAN_RATIO}\n')
    f.write(f'- EXPECTED_SECTOR_COUNT = {EXPECTED_SECTOR_COUNT}\n')
    f.write(f'- MIN_VALID_SECTORS = {MIN_VALID_SECTORS}\n')
    f.write(f'- MIN_SECTOR_COVERAGE = {MIN_SECTOR_COVERAGE}\n')
    f.write(f'- MIN_VALID_TICKERS = {MIN_VALID_TICKERS}\n')
    f.write('\n*Esta auditoria verifica la antiguedad de los datos almacenados. La validacion completa de cobertura se realiza durante la ejecucion de run.py.*\n')

print('\nInforme guardado en outputs/audit/auditoria_frescura.md')
