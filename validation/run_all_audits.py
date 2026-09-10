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
        log(f'[OK] {ok_msg}')
    else:
        log(f'[FAIL] {fail_msg}')

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
        log(f'[FAIL] {path} no existe')

# 2. NaN en CSV de lideres
for path in ['outputs/report/analisis_lideres.csv', 'outputs/report/analisis_lideres_internacionales.csv']:
    if os.path.exists(path):
        df = pd.read_csv(path)
        nan_cols = df.columns[df.isna().any()].tolist()
        # Se permite NaN en rs_mom y wls? Mejor detectar NaN en columnas criticas
        critical = ['rs', 'flow_proxy_z', 'wyckoff_score', 'wls']
        nan_critical = {c: df[c].isna().sum() for c in critical if c in df.columns}
        if any(v > 0 for v in nan_critical.values()):
            log(f'[FAIL] {path}: NaN en columnas criticas {nan_critical}')
        else:
            log(f'[OK] {path}: sin NaN en columnas criticas')
        # Persistencia en rango 0-1
        if 'persistence_10d' in df.columns:
            p = df['persistence_10d'].dropna()
            check(p.between(0,1).all(), f'{path}: persistencia en rango 0-1', f'{path}: persistencia fuera de rango [{p.min():.2f}, {p.max():.2f}]')
    else:
        log(f'[INFO] {path} no disponible (puede no haberse generado hoy)')

# 3. Frescura de fuentes
for path, col in [('outputs/history/pcr_history.csv','date'), ('outputs/history/darkpool_history.csv','week')]:
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=[col])
        last = df[col].max()
        age = (pd.Timestamp.now() - pd.Timestamp(last)).days
        estado = 'CURRENT' if age <= 7 else ('RECENT' if age <= 14 else ('STALE' if age <= 21 else 'ARCHIVAL'))
        log(f'[INFO] {path}: ultimo dato {last.date()} ({age} dias, {estado})')
    else:
        log(f'[INFO] {path} no existe')

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
    log('[FAIL] No se encontro evidence_matrix.csv')

# 3c. Integridad específica para sector_concentration.csv
sc_path = 'outputs/history/sector_concentration.csv'
if os.path.exists(sc_path):
    sc = pd.read_csv(sc_path)
    # Validación histórica: fecha sin NaN solo en la última fecha
    if sc['date'].notna().any():
        latest = pd.to_datetime(sc['date']).max()
        latest_df = sc[pd.to_datetime(sc['date']) == latest]
        check(latest_df['date'].notna().all(), f'{sc_path}: última fecha sin NaN', f'{sc_path}: NaN en última fecha')
        check(latest_df['sector'].nunique() == 11, f'{sc_path}: 11 sectores en última fecha', f'{sc_path}: {latest_df["sector"].nunique()} sectores en última fecha')
        dup = latest_df.duplicated(subset=['sector']).sum()
        check(dup == 0, f'{sc_path}: sin duplicados en última fecha', f'{sc_path}: {dup} duplicados en última fecha')
        log(f'[INFO] {sc_path}: {len(sc)} filas, fechas {sc["date"].nunique()}')
    else:
        log(f'[WARN] {sc_path}: no hay fechas válidas')
else:
    log(f'[FAIL] {sc_path} no existe')
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
            log(f'[INFO] {f} no tiene columnas {subset}')
    else:
        log(f'[INFO] {f} no existe')

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
        log(f'[INFO] {sfc_path}: price_flow_regime_20d válidos en {sfc["price_flow_regime_20d"].notna().sum()}/11 sectores')
    else:
        log(f'[INFO] {sfc_path} no tiene columnas de régimen')
else:
    log(f'[FAIL] {sfc_path} no existe')

# 3c-quater. Validación RS Interno/Absoluto
rs_path = 'outputs/history/rs_internal.csv'
if os.path.exists(rs_path):
    rdf = pd.read_csv(rs_path, encoding='utf-8')
    required_cols = ['date','sector','ticker','price_ret_20d','sector_ret_20d','benchmark_ret_20d','rs_abs_20d','rs_internal_20d','classification']
    check(set(required_cols).issubset(rdf.columns), f'{rs_path}: columnas requeridas presentes', f'{rs_path}: faltan columnas {set(required_cols)-set(rdf.columns)}')
    if set(required_cols).issubset(rdf.columns):
        dup = rdf.duplicated(subset=['date','sector','ticker']).sum()
        check(dup == 0, f'{rs_path}: sin duplicados date+sector+ticker', f'{rs_path}: {dup} duplicados')
        check(rdf['rs_abs_20d'].isna().sum() == 0, f'{rs_path}: sin NaN en rs_abs_20d', f'{rs_path}: {rdf["rs_abs_20d"].isna().sum()} NaN')
        check(rdf['rs_internal_20d'].isna().sum() == 0, f'{rs_path}: sin NaN en rs_internal_20d', f'{rs_path}: {rdf["rs_internal_20d"].isna().sum()} NaN')
        allowed = {'Liderazgo relativo doble','Fortaleza sectorial','Liderazgo interno en sector débil','Debilidad relativa doble','N/D'}
        bad_cls = set(rdf['classification'].unique()) - allowed
        check(len(bad_cls) == 0, f'{rs_path}: clasificaciones válidas', f'{rs_path}: clasificaciones inválidas {bad_cls}')
        log(f'[INFO] {rs_path}: {len(rdf)} filas, {rdf["sector"].nunique()} sectores, {rdf["ticker"].nunique()} tickers')
else:
    log(f'[FAIL] {rs_path} no existe')

# 3c-quinquies. Validación Persistencia sectorial
ppath = 'outputs/history/sector_persistence.csv'
if os.path.exists(ppath):
    pdf = pd.read_csv(ppath, encoding='utf-8')
    check({'date','sector','persistence'}.issubset(pdf.columns), f'{ppath}: columnas correctas', f'{ppath}: faltan columnas')
    check(pdf['date'].notna().all(), f'{ppath}: date sin NaN', f'{ppath}: {pdf["date"].isna().sum()} NaN en date')
    dup = pdf.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{ppath}: sin duplicados date+sector', f'{ppath}: {dup} duplicados')
    check(pdf['persistence'].between(0,1).all(), f'{ppath}: persistence en [0,1]', f'{ppath}: valores fuera de [0,1]')
    latest = pd.to_datetime(pdf['date']).max()
    latest_df = pdf[pd.to_datetime(pdf['date']) == latest]
    check(len(latest_df) == 11, f'{ppath}: 11 sectores en última fecha', f'{ppath}: {len(latest_df)} sectores')
    log(f'[INFO] {ppath}: {len(pdf)} filas, {pdf["sector"].nunique()} sectores, fechas {pdf["date"].nunique()}')
else:
    log(f'[FAIL] {ppath} no existe')

# 3c-sexies. Validación Rotación sectorial histórica
rank_hist_path = 'outputs/history/sector_rank_history.csv'
rank_deltas_path = 'outputs/history/sector_rank_deltas.csv'

if os.path.exists(rank_hist_path):
    rh = pd.read_csv(rank_hist_path, encoding='utf-8')
    check({'date','sector','score','rank'}.issubset(rh.columns), f'{rank_hist_path}: columnas correctas', f'{rank_hist_path}: faltan columnas')
    check(rh['date'].notna().all(), f'{rank_hist_path}: date sin NaN', f'{rank_hist_path}: {rh["date"].isna().sum()} NaN en date')
    check(rh['sector'].notna().all(), f'{rank_hist_path}: sector sin NaN', f'{rank_hist_path}: {rh["sector"].isna().sum()} NaN en sector')
    check(rh['rank'].between(1,11).all(), f'{rank_hist_path}: rank entre 1 y 11', f'{rank_hist_path}: rank fuera de rango')
    dup = rh.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{rank_hist_path}: sin duplicados date+sector', f'{rank_hist_path}: {dup} duplicados')
    latest = pd.to_datetime(rh['date']).max()
    latest_df = rh[pd.to_datetime(rh['date']) == latest]
    check(len(latest_df) == 11, f'{rank_hist_path}: 11 sectores en última fecha', f'{rank_hist_path}: {len(latest_df)} sectores')
    log(f'[INFO] {rank_hist_path}: {len(rh)} filas, {rh["sector"].nunique()} sectores, fechas {rh["date"].nunique()}')
else:
    log(f'[FAIL] {rank_hist_path} no existe')

if os.path.exists(rank_deltas_path):
    rd = pd.read_csv(rank_deltas_path, encoding='utf-8')
    expected_cols = ['sector','rank_actual','rank_change_5d','rank_change_10d','rank_change_20d','lectura_5d','lectura_10d','lectura_20d']
    check(set(expected_cols).issubset(rd.columns), f'{rank_deltas_path}: columnas correctas', f'{rank_deltas_path}: faltan columnas {set(expected_cols)-set(rd.columns)}')
    if set(expected_cols).issubset(rd.columns):
        check(rd['sector'].nunique() == 11, f'{rank_deltas_path}: 11 sectores', f'{rank_deltas_path}: {rd["sector"].nunique()} sectores')
        allowed_lecturas = {'Fuerte mejora','Estable','Fuerte deterioro','N/D'}
        bad = set(rd['lectura_5d'].unique()) | set(rd['lectura_10d'].unique()) | set(rd['lectura_20d'].unique())
        bad = bad - allowed_lecturas
        check(len(bad) == 0, f'{rank_deltas_path}: lecturas válidas', f'{rank_deltas_path}: lecturas inválidas {bad}')
        log(f'[INFO] {rank_deltas_path}: {len(rd)} sectores')
else:
    log(f'[FAIL] {rank_deltas_path} no existe')

# 3c-sexies. Validación Rotación sectorial histórica
rank_hist_path = 'outputs/history/sector_rank_history.csv'
rank_deltas_path = 'outputs/history/sector_rank_deltas.csv'

if os.path.exists(rank_hist_path):
    rh = pd.read_csv(rank_hist_path, encoding='utf-8')
    check({'date','sector','score','rank'}.issubset(rh.columns), f'{rank_hist_path}: columnas correctas', f'{rank_hist_path}: faltan columnas')
    check(rh['date'].notna().all(), f'{rank_hist_path}: date sin NaN', f'{rank_hist_path}: {rh["date"].isna().sum()} NaN en date')
    check(rh['sector'].notna().all(), f'{rank_hist_path}: sector sin NaN', f'{rank_hist_path}: {rh["sector"].isna().sum()} NaN en sector')
    check(rh['rank'].between(1,11).all(), f'{rank_hist_path}: rank entre 1 y 11', f'{rank_hist_path}: rank fuera de rango')
    dup = rh.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{rank_hist_path}: sin duplicados date+sector', f'{rank_hist_path}: {dup} duplicados')
    latest = pd.to_datetime(rh['date']).max()
    latest_df = rh[pd.to_datetime(rh['date']) == latest]
    check(len(latest_df) == 11, f'{rank_hist_path}: 11 sectores en última fecha', f'{rank_hist_path}: {len(latest_df)} sectores')
    log(f'[INFO] {rank_hist_path}: {len(rh)} filas, {rh["sector"].nunique()} sectores, fechas {rh["date"].nunique()}')
else:
    log(f'[FAIL] {rank_hist_path} no existe')

if os.path.exists(rank_deltas_path):
    rd = pd.read_csv(rank_deltas_path, encoding='utf-8')
    expected_cols = ['sector','rank_actual','rank_change_5d','rank_change_10d','rank_change_20d','lectura_5d','lectura_10d','lectura_20d']
    check(set(expected_cols).issubset(rd.columns), f'{rank_deltas_path}: columnas correctas', f'{rank_deltas_path}: faltan columnas {set(expected_cols)-set(rd.columns)}')
    if set(expected_cols).issubset(rd.columns):
        check(rd['sector'].nunique() == 11, f'{rank_deltas_path}: 11 sectores', f'{rank_deltas_path}: {rd["sector"].nunique()} sectores')
        allowed_lecturas = {'Fuerte mejora','Estable','Fuerte deterioro','N/D'}
        bad = set(rd['lectura_5d'].unique()) | set(rd['lectura_10d'].unique()) | set(rd['lectura_20d'].unique())
        bad = bad - allowed_lecturas
        check(len(bad) == 0, f'{rank_deltas_path}: lecturas válidas', f'{rank_deltas_path}: lecturas inválidas {bad}')
        log(f'[INFO] {rank_deltas_path}: {len(rd)} sectores')
else:
    log(f'[FAIL] {rank_deltas_path} no existe')

# 3c-septies. Validación Matriz de Régimen Sectorial
rm_path = 'outputs/history/sector_regime_matrix.csv'
if os.path.exists(rm_path):
    rm = pd.read_csv(rm_path, encoding='utf-8')
    required = ['date','sector','price_ret_20d','pct_above_ema50','flow_20d_sum','wyckoff_phase','price_positive','breadth_positive','flow_positive','structure_positive','positive_conditions','data_complete','regime_reading']
    check(set(required).issubset(rm.columns), f'{rm_path}: columnas correctas', f'{rm_path}: faltan columnas {set(required)-set(rm.columns)}')
    check(rm['date'].notna().all(), f'{rm_path}: date sin NaN', f'{rm_path}: {rm["date"].isna().sum()} NaN en date')
    check(rm['sector'].nunique() == 11, f'{rm_path}: 11 sectores', f'{rm_path}: {rm["sector"].nunique()} sectores')
    check(rm.isna().sum().sum() == 0, f'{rm_path}: sin NaN', f'{rm_path}: {rm.isna().sum().sum()} NaN')
    allowed = {'Alineación positiva','Constructivo','Mixto','Débil','Debilidad alineada','N/D'}
    bad = set(rm['regime_reading'].unique()) - allowed
    check(len(bad) == 0, f'{rm_path}: lecturas válidas', f'{rm_path}: lecturas inválidas {bad}')
    dup = rm.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{rm_path}: sin duplicados date+sector', f'{rm_path}: {dup} duplicados')
    log(f'[INFO] {rm_path}: {len(rm)} filas, {rm["sector"].nunique()} sectores, fechas {rm["date"].nunique()}')
else:
    log(f'[FAIL] {rm_path} no existe')

# 3c-octies. Validación Distribución Wyckoff sectorial
wy_path = 'outputs/history/sector_wyckoff_distribution.csv'
if os.path.exists(wy_path):
    wydf = pd.read_csv(wy_path, encoding='utf-8')
    required_wy = ['date','sector','n_total','n_valid_wyckoff','n_insufficient_wyckoff','coverage_wyckoff',
                   'count_accumulation','pct_accumulation','count_markup','pct_markup',
                   'count_range','pct_range','count_distribution','pct_distribution',
                   'count_markdown','pct_markdown']
    check(set(required_wy).issubset(wydf.columns), f'{wy_path}: columnas correctas', f'{wy_path}: faltan columnas {set(required_wy)-set(wydf.columns)}')
    check(wydf['date'].notna().all(), f'{wy_path}: date sin NaN', f'{wy_path}: {wydf["date"].isna().sum()} NaN en date')
    check(wydf['sector'].nunique() == 11, f'{wy_path}: 11 sectores', f'{wy_path}: {wydf["sector"].nunique()} sectores')
    check(wydf['coverage_wyckoff'].between(0,100).all(), f'{wy_path}: coverage entre 0 y 100', f'{wy_path}: coverage fuera de rango')
    # Suma de porcentajes solo cuando n_valid >=5
    mask_valid = wydf['n_valid_wyckoff'] >= 5
    pct_cols = ['pct_accumulation','pct_markup','pct_range','pct_distribution','pct_markdown']
    if mask_valid.any():
        sums = wydf.loc[mask_valid, pct_cols].sum(axis=1)
        ok_sums = (sums >= 99.5) & (sums <= 100.5)
        check(ok_sums.all(), f'{wy_path}: suma de pct ~100%', f'{wy_path}: sumas inválidas {sums[~ok_sums].tolist()}')
        # Para n_valid <5, los pct deben ser NaN
        mask_invalid = ~mask_valid
        if mask_invalid.any():
            check(wydf.loc[mask_invalid, pct_cols].isna().all().all(), f'{wy_path}: pct NaN cuando n_valid<5', f'{wy_path}: pct presentes con n_valid<5')
    dup = wydf.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{wy_path}: sin duplicados date+sector', f'{wy_path}: {dup} duplicados')
    log(f'[INFO] {wy_path}: {len(wydf)} filas, fechas {wydf["date"].nunique()}')
else:
    log(f'[FAIL] {wy_path} no existe')

# 3c-nonies. Validación Divergencia sector-líderes
sld_path = 'outputs/history/sector_leader_divergence.csv'
if os.path.exists(sld_path):
    sld = pd.read_csv(sld_path, encoding='utf-8')
    required_sld = ['date','sector','sector_ret_20d','n_leaders_total','n_leaders_valid','n_leaders_positive','n_leaders_negative','n_leaders_beating_sector','classification']
    check(set(required_sld).issubset(sld.columns), f'{sld_path}: columnas correctas', f'{sld_path}: faltan columnas {set(required_sld)-set(sld.columns)}')
    check(sld['date'].notna().all(), f'{sld_path}: date sin NaN', f'{sld_path}: {sld["date"].isna().sum()} NaN en date')
    check(sld['sector'].notna().all(), f'{sld_path}: sector sin NaN', f'{sld_path}: {sld["sector"].isna().sum()} NaN en sector')
    allowed_sld = {'Alineación positiva','Alineación negativa','Liderazgo relativo de líderes','Divergencia negativa','Mixto','N/D'}
    bad_sld = set(sld['classification'].unique()) - allowed_sld
    check(len(bad_sld) == 0, f'{sld_path}: clasificaciones válidas', f'{sld_path}: clasificaciones inválidas {bad_sld}')
    check(sld['n_leaders_valid'].between(0,5).all(), f'{sld_path}: n_valid entre 0 y 5', f'{sld_path}: n_valid fuera de rango')
    check((sld['n_leaders_positive'] + sld['n_leaders_negative'] <= sld['n_leaders_valid']).all(), f'{sld_path}: suma positivos+negativos <= n_valid', f'{sld_path}: inconsistencia en conteos')
    dup = sld.duplicated(subset=['date','sector']).sum()
    check(dup == 0, f'{sld_path}: sin duplicados date+sector', f'{sld_path}: {dup} duplicados')
    log(f'[INFO] {sld_path}: {len(sld)} sectores con datos')
else:
    log(f'[FAIL] {sld_path} no existe')

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
    log('[FAIL] No se encontro reporte_diario.md')

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
        log(f'[INFO] {path}: {data.get(key, "N/A")}')
    else:
        log(f'[INFO] {path} no existe')

# Guardar informe
# 3c-septies. Validación de guardas de cobertura en breadth_equity
be_path = 'indicators/breadth_equity.py'
if os.path.exists(be_path):
    with open(be_path, 'r', encoding='utf-8') as f:
        be_content = f.read()
    check('active_tickers < 20' in be_content and 'return None' in be_content,
          'breadth_equity.py: guarda de cobertura presente',
          'breadth_equity.py: falta guarda de cobertura')
else:
    log(f'âŒ {be_path} no existe')
log('')
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))
print(f'\nInforme guardado en {OUT}')
