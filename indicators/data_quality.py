# -*- coding: utf-8 -*-
"""
Calidad, frescura y cobertura de datos v1.1
Describe actualidad y completitud de las principales fuentes.
No alimenta motores, scores, pesos ni State Machine.
No crea Data Quality Score.
"""

from config.settings import (
    EXPECTED_SECTOR_COUNT, FRESHNESS_FINRA,
    FRESHNESS_SEC, FRESHNESS_CFTC,
)
from src.market_calendar import last_expected_market_date
from config.tickers import MARKET_TICKERS
import pandas as pd
import numpy as np
from pathlib import Path

from src.market_calendar import _last_market_session

def classify_freshness(age_days, frequency):
    if pd.isna(age_days):
        return 'N/D'
    if frequency == 'finra':
        # F6-1b (2026-10-07): unificar con FRESHNESS_FINRA. Antes tenia
        # los umbrales antiguos (30, 45, 60) hardcoded, divergentes de
        # settings.FRESHNESS_FINRA (20, 28, 45). El fix F6-02 del
        # 2026-09-28 actualizo helpers.py pero no este fichero.
        max_current, max_recent, max_stale = FRESHNESS_FINRA
        if age_days <= max_current: return 'CURRENT'
        elif age_days <= max_recent: return 'RECENT'
        elif age_days <= max_stale: return 'STALE'
        else: return 'ARCHIVAL'
    elif frequency == 'sec':
        # H7 (2026-10-16, auditoria): umbrales centralizados en
        # config.settings.FRESHNESS_SEC. Antes hardcoded aqui; ahora la
        # fuente unica es settings, y la nota del reporte los consume
        # desde ahi tambien.
        max_current, max_recent, max_stale = FRESHNESS_SEC
        if age_days <= max_current: return 'CURRENT'
        elif age_days <= max_recent: return 'RECENT'
        elif age_days <= max_stale: return 'STALE'
        else: return 'ARCHIVAL'
    elif frequency == 'cftc':
        # H7: umbrales centralizados en config.settings.FRESHNESS_CFTC.
        max_current, max_recent, max_stale = FRESHNESS_CFTC
        if age_days <= max_current: return 'CURRENT'
        elif age_days <= max_recent: return 'RECENT'
        elif age_days <= max_stale: return 'STALE'
        else: return 'ARCHIVAL'
    elif frequency == 'fred':
        if age_days <= 30: return 'CURRENT'
        elif age_days <= 60: return 'RECENT'
        elif age_days <= 90: return 'STALE'
        else: return 'ARCHIVAL'
    else:
        if age_days <= 3: return 'CURRENT'
        elif age_days <= 7: return 'RECENT'
        elif age_days <= 14: return 'STALE'
        else: return 'ARCHIVAL'

def _read_schema_version(csv_path):
    """Lee <path>.schema (entero). Retorna None si no existe o invalido.

    Fresh checkout sin .schema = v1 implicito = None. No se inventa.
    """
    try:
        p = Path(str(csv_path) + ".schema")
        if p.exists():
            return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    return None

def _read_date_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

# H3 (2026-10-15): la lista de fuentes se eleva a constante de
# modulo para permitir tests contractuales sin invocar compute.
SOURCES = [
    {
        'source': 'Yahoo Finance',
        'file': 'data/market_data.parquet',
        'date_cols': None,
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Precios y métricas sectoriales'
    },
    {
        'source': 'SSGA ETF Flow',
        'file': 'outputs/history/etf_primary_flow.csv',
        'date_cols': ['Date'],
        'frequency': 'daily',
        'coverage_logic': 'flow_tickers',
        'notes': 'Flujo primario'
    },
    {
        'source': 'CBOE PCR',
        'file': 'outputs/history/pcr_history.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'rows',
        'notes': 'Put/Call ratio'
    },
    {
        'source': 'FINRA Dark Pool',
        'file': 'outputs/history/darkpool_history.csv',
        'date_cols': ['week'],
        'frequency': 'finra',
        'coverage_logic': 'rows',
        'coverage_col': 'ratio',
        'walk_back': False,  # Fix G (2026-09-30): 'week' es fecha-semana, no fecha-dia. _last_market_session no aplica.
        'notes': 'Retraso regulatorio 2-4 semanas'
    },
    {
        'source': 'CFTC Position Flow',
        'file': 'outputs/history/cftc_position_flow.csv',
        'date_cols': ['date'],
        'frequency': 'cftc',
        'coverage_logic': 'rows',
        'coverage_col': 'net_position',
        'notes': 'Frescura CFTC (semanal); umbrales en config/settings.FRESHNESS_CFTC'
    },
    {
        'source': 'SEC N-PORT (posiciones)',
        'file': 'outputs/history/sec_nport_position_change_quarterly.csv',
        'date_cols': ['REPORT_DATE'],
        'frequency': 'sec',
        'coverage_logic': 'rows',
        'coverage_col': 'POSITION_CHANGE',
        # H11/H12 (2026-10-17, REP-AUDIT-20261017): declarar las 4
        # dimensiones temporales. Frecuencia origen (mensual, N-PORT
        # amendments desde 2024) != frecuencia ingesta (trimestral,
        # ZIPs N-PORT del pipeline). Fecha observacion (cierre del
        # trimestre) != fecha presentacion (dia del filing en SEC).
        'notes': 'Posiciones N-PORT. Origen: mensual (SEC); ingesta: '
                 'trimestral (ZIPs). Latencia ~60d entre cierre y '
                 'filing. Fecha observacion = cierre trimestre; fecha '
                 'presentacion = dia del filing'
    },
    # H3 (2026-10-15, auditoria externa): el pipeline consume DOS
    # productos SEC distintos. Antes solo se declaraba el primero y
    # el reporte daba impresion de que SEC entera estaba stale. El
    # segundo (flujos N-PORT-P QQQ, Item B.6) tiene su propia
    # cadencia y va mas adelantado (Q2 mientras posiciones esta en
    # Q1).
    {
        'source': 'SEC N-PORT-P QQQ (flujos)',
        'file': 'outputs/history/qqq_nport_flow.csv',
        'date_cols': ['report_date'],
        'frequency': 'sec',
        'coverage_logic': 'rows',
        'coverage_col': 'net_flow',
        'notes': 'Flujos trimestrales N-PORT-P QQQ (Item B.6)'
    },
    {
        'source': 'Macro FRED',
        'file': 'outputs/history/macro_regime.csv',
        'date_cols': ['date'],
        'frequency': 'fred',
        'coverage_logic': 'rows',
        'coverage_col': 'macro_score',
        'notes': 'Datos macro manuales'
    },
    {
        'source': 'Sector Concentration',
        'file': 'outputs/history/sector_concentration.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Concentración y dispersión'
    },
    {
        'source': 'Sector Breadth Momentum',
        'file': 'outputs/history/sector_breadth_momentum.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Momentum amplitud'
    },
    {
        'source': 'Volatility Structure',
        'file': 'outputs/history/volatility_structure.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'rows',
        'coverage_col': 'vix_level',
        'notes': 'Estructura volatilidad'
    },
    {
        'source': 'Cross Asset Context',
        'file': 'outputs/history/cross_asset_context.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial_family',
        'notes': 'Contexto cross-asset'
    },
    {
        'source': 'Evidence Matrix',
        'file': 'outputs/history/evidence_matrix.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial_evidence',
        'notes': 'Matriz evidencia'
    },
    # 2026-10-10: fuentes anadidas. Faltaban 12 CSVs que alimentan
    # secciones del reporte sin declaracion de frescura. Todas
    # bursatiles (segundo candado temporal del pipeline).
    {
        'source': 'Sector Breadth',
        'file': 'outputs/history/sector_breadth.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Amplitud y salud sectorial'
    },
    {
        'source': 'Sector Persistence',
        'file': 'outputs/history/sector_persistence.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Persistencia sectorial'
    },
    # 2026-10-10: SLPM retirado de la lista. slpm_history.csv
    # esta congelado desde 2026-07-17 (mtime agosto). Grep no
    # encuentra writer activo. Declarar su frescura sin writer
    # solo anade ruido ARCHIVAL permanente.
    # 2026-10-20 (REP-AUDIT-20261020-SLPM-FOSSIL): CSV eliminado
    # del repo. Artifact de daily_run.yml y metrica
    # structural_vs_breadth de monitor_double_counting.py retirados.
    # slpm_state.json tambien retirado (fosil sin productor desde C3.1).
    {
        'source': 'Sector Regime Matrix',
        'file': 'outputs/history/sector_regime_matrix.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Matriz de regimen sectorial'
    },
    {
        'source': 'Sector Wyckoff Distribution',
        'file': 'outputs/history/sector_wyckoff_distribution.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Distribucion Wyckoff sectorial'
    },
    {
        'source': 'RS Internal',
        'file': 'outputs/history/rs_internal.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Liderazgo relativo interno'
    },
    {
        'source': 'Sector Rank History',
        'file': 'outputs/history/sector_rank_history.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Rotacion sectorial historica'
    },
    {
        'source': 'Sector Dispersion',
        'file': 'outputs/history/sector_dispersion.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'from_row',
        'notes': 'Dispersion entre sectores'
    },
    {
        'source': 'Sector Correlation Summary',
        'file': 'outputs/history/sector_correlation_summary.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'from_row',
        'notes': 'Correlacion entre sectores'
    },
    {
        'source': 'Sector Flow Characteristics',
        'file': 'outputs/history/sector_flow_characteristics.csv',
        'date_cols': ['date'],
        'frequency': 'daily',
        'coverage_logic': 'sectorial',
        'notes': 'Flujo primario sectorial'
    },
]


def compute_data_quality(reference_date=None):
    """Calcula frescura y cobertura de las fuentes.

    F-UX-03 (2026-09-28): con reference_date, la decision no depende
    de datetime.now(). Fallback legacy cuando es None.
    """
    # REP-AUDIT-20261018-DQ-DATE (2026-10-18): date = sesion bursatil real,
    # no fecha del run. Evita filas en sabados/domingos/festivos.
    # Fase 4b P7 (2026-10-28): observacion. Raise si None.
    from src.utils import resolve_observation_date
    _ref_obs = resolve_observation_date(reference_date)
    _session_date = last_expected_market_date(pd.Timestamp(_ref_obs))
    sources = SOURCES

    rows = []
    for src in sources:
        if src['source'] == 'Macro FRED':
            try:
                import json
                liq_state_path = Path('outputs/state/liquidity_state.json')
                if liq_state_path.exists():
                    with liq_state_path.open('r', encoding='utf-8') as _f:
                        _state = json.load(_f)
                    _ld = pd.to_datetime(_state.get('date'))
                    last_date = _last_market_session(_ld) if pd.notna(_ld) else pd.NaT
                else:
                    last_date = pd.NaT
            except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
                last_date = pd.NaT
            age = (_session_date - last_date.date()).days if pd.notna(last_date) else np.nan
            freshness = classify_freshness(age, src['frequency'])
            # H2 (2026-10-17): aplicar coverage_logic tambien en la rama
            # especial de Macro FRED. Antes solo escribia NaN aunque el
            # source declarase 'rows' + coverage_col.
            _n_total = np.nan
            _n_valid = np.nan
            _coverage = np.nan
            _logic = src.get('coverage_logic')
            if _logic == 'rows':
                _csv_p = Path(src['file'])
                if _csv_p.exists():
                    try:
                        _csv_df = pd.read_csv(_csv_p)
                        _col = src.get('coverage_col') or 'total_pcr'
                        _n_total = len(_csv_df)
                        _n_valid = int(_csv_df[_col].notna().sum()) if _col in _csv_df.columns else 0
                        _coverage = _n_valid / _n_total if _n_total else np.nan
                    except (OSError, ValueError, pd.errors.ParserError):
                        pass
            rows.append({
                'date': _session_date.strftime('%Y-%m-%d'),
                'source': src['source'],
                'last_date': last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else np.nan,
                'age_calendar_days': age,
                'frequency': src['frequency'],
                'freshness': freshness,
                'n_total': _n_total,
                'n_valid': _n_valid,
                'coverage': _coverage,
                'notes': src.get('notes', ''),
            })
            continue
        path = Path(src['file'])
        if not path.exists():
            rows.append({
                'date': _session_date.strftime('%Y-%m-%d'),
                'source': src['source'],
                'last_date': np.nan,
                'age_calendar_days': np.nan,
                'frequency': src['frequency'],
                'freshness': 'N/D',
                'n_total': np.nan,
                'n_valid': np.nan,
                'coverage': np.nan,
                'notes': src.get('notes', ''),
            })
            continue

        # Rama Parquet (D3): market_data.parquet usa index como fecha.
        if str(path).endswith('.parquet'):
            try:
                df_pq = pd.read_parquet(path)
                last_date = _last_market_session(pd.Timestamp(df_pq.index[-1])) if len(df_pq) > 0 else pd.NaT
                age = (_session_date - last_date.date()).days if pd.notna(last_date) else np.nan
                freshness = classify_freshness(age, src['frequency'])

                n_total = np.nan
                n_valid = np.nan
                coverage = np.nan
                logic = src.get('coverage_logic')
                if logic == 'sectorial' and len(df_pq) > 0:
                    # H1 (2026-10-17): market_data.parquet tiene MultiIndex
                    # (Price, Ticker) e index Date. Cobertura sectorial:
                    # cuantos de los 11 ETFs sectoriales tienen Close
                    # no-NaN en la ultima sesion. Regresion del commit
                    # 1a4e7ea2 (2026-09-12) que cambio sector_breadth.csv
                    # por market_data.parquet sin reimplementar la logica.
                    last_row = df_pq.iloc[-1]
                    sectors = MARKET_TICKERS.get('sectors', [])
                    n_total = EXPECTED_SECTOR_COUNT
                    n_valid = int(sum(
                        1 for t in sectors
                        if ('Close', t) in df_pq.columns
                        and pd.notna(last_row[('Close', t)])
                    ))
                    coverage = n_valid / n_total if n_total else np.nan

                rows.append({
                    'date': _session_date.strftime('%Y-%m-%d'),
                    'source': src['source'],
                    'last_date': last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else np.nan,
                    'age_calendar_days': age,
                    'frequency': src['frequency'],
                    'freshness': freshness,
                    'n_total': n_total,
                    'n_valid': n_valid,
                    'coverage': coverage,
                    'notes': src.get('notes', ''),
                })
            except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
                rows.append({
                    'date': _session_date.strftime('%Y-%m-%d'),
                    'source': src['source'],
                    'last_date': np.nan,
                    'age_calendar_days': np.nan,
                    'frequency': src['frequency'],
                    'freshness': 'N/D',
                    'n_total': np.nan,
                    'n_valid': np.nan,
                    'coverage': np.nan,
                    'notes': f'Error Parquet: {e}',
                })
            continue

        try:
            df = pd.read_csv(path)
            date_col = _read_date_col(df, src['date_cols'])
            if date_col is None:
                last_date = pd.NaT
            else:
                _ld = pd.to_datetime(df[date_col], errors='coerce').max()
                # Fix G (2026-09-30): walk_back=False -> no aplicar
                # _last_market_session. Aplica a FINRA (fecha-semana).
                if src.get('walk_back', True) and pd.notna(_ld):
                    last_date = _last_market_session(_ld)
                else:
                    last_date = _ld
            age = (_session_date - last_date.date()).days if pd.notna(last_date) else np.nan
            freshness = classify_freshness(age, src['frequency'])

            n_total = np.nan
            n_valid = np.nan
            coverage = np.nan

            logic = src.get('coverage_logic')
            if logic == 'sectorial':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = EXPECTED_SECTOR_COUNT
                    n_valid = int(last_df['sector'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'flow_tickers':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = 12
                    n_valid = int(last_df['ticker'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'rows':
                n_total = len(df)
                # H2 (2026-10-17): coverage_col configurable. Antes
                # 'rows' estaba hardcodeado a 'total_pcr' (solo PCR).
                # Las demas fuentes con esta logica declaran su
                # coverage_col en el bloque SOURCES.
                _col = src.get('coverage_col') or 'total_pcr'
                n_valid = int(df[_col].notna().sum()) if _col in df.columns else 0
                coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'sectorial_family':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = EXPECTED_SECTOR_COUNT
                    n_valid = int(last_df[last_df['mean_corr'].notna()]['sector'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'sectorial_evidence':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = EXPECTED_SECTOR_COUNT
                    n_valid = int(last_df[last_df['alignment_reading'] != 'EVIDENCIA INSUFICIENTE']['sector'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'from_row':
                # H2 (2026-10-17): fuentes que ya calculan n_total/n_valid/
                # coverage en su writer. Leer de la ultima fila del CSV.
                # Algunos writers usan nombres propios (n_sectors,
                # n_valid_pairs). Se aceptan como aliases.
                _aliases_total = ('n_total', 'n_sectors')
                _aliases_valid = ('n_valid', 'n_valid_pairs')
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    if not last_df.empty:
                        row = last_df.iloc[-1]
                        for _fld in _aliases_total:
                            if _fld in row.index and pd.notna(row[_fld]):
                                try:
                                    n_total = float(row[_fld]); break
                                except (TypeError, ValueError):
                                    pass
                        for _fld in _aliases_valid:
                            if _fld in row.index and pd.notna(row[_fld]):
                                try:
                                    n_valid = float(row[_fld]); break
                                except (TypeError, ValueError):
                                    pass
                        if 'coverage' in row.index and pd.notna(row['coverage']):
                            try:
                                coverage = float(row['coverage'])
                            except (TypeError, ValueError):
                                pass
            # else coverage permanece NaN

            rows.append({
                'date': _session_date.strftime('%Y-%m-%d'),
                'source': src['source'],
                'last_date': last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else np.nan,
                'age_calendar_days': age,
                'frequency': src['frequency'],
                'freshness': freshness,
                'n_total': n_total,
                'n_valid': n_valid,
                'coverage': coverage,
                'notes': src.get('notes', ''),
            })
        except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
            rows.append({
                'date': _session_date.strftime('%Y-%m-%d'),
                'source': src['source'],
                'last_date': np.nan,
                'age_calendar_days': np.nan,
                'frequency': src['frequency'],
                'freshness': 'N/D',
                'n_total': np.nan,
                'n_valid': np.nan,
                'coverage': np.nan,
                'notes': f'Error lectura: {e}',
            })

    # --- Fuentes europeas: Euronext, Xetra, BME ---
    # Lee european_coverage.csv del run anterior (lag 1 dia).
    # Agrega por source: fecha mas reciente, status global.
    _eu_path = Path('outputs/history/european_coverage.csv')
    for _eu_source in ('Euronext', 'Xetra', 'BME'):
        _row = {
            'date': _session_date.strftime('%Y-%m-%d'),
            'source': _eu_source,
            'last_date': np.nan,
            'age_calendar_days': np.nan,
            'frequency': 'daily',
            'freshness': 'N/D',
            'n_total': np.nan,
            'n_valid': np.nan,
            'coverage': np.nan,
            'notes': '',
        }
        try:
            if _eu_path.exists():
                _eu_df = pd.read_csv(_eu_path)
                if 'date' in _eu_df.columns:
                    _eu_last_run = pd.to_datetime(_eu_df['date'], errors='coerce').max()
                    _eu_df = _eu_df[pd.to_datetime(_eu_df['date'], errors='coerce') == _eu_last_run]
                _eu_df = _eu_df[_eu_df['source'] == _eu_source]
                if not _eu_df.empty:
                    _ok = int((_eu_df['status'] == 'OK').sum())
                    _n_total = len(_eu_df)
                    _last_raw = pd.to_datetime(_eu_df['last_date'], errors='coerce').max()
                    _last = _last_market_session(_last_raw) if pd.notna(_last_raw) else pd.NaT
                    _age = (_session_date - _last.date()).days if pd.notna(_last) else np.nan
                    _fresh = classify_freshness(_age, 'daily')
                    _row['last_date'] = _last.strftime('%Y-%m-%d') if pd.notna(_last) else np.nan
                    _row['age_calendar_days'] = _age
                    _row['freshness'] = _fresh
                    _row['n_total'] = _n_total
                    _row['n_valid'] = _ok
                    _row['coverage'] = _ok / _n_total if _n_total else np.nan
                    _row['notes'] = f'{_eu_source} ({_ok}/{_n_total} OK)'
        except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as _e:
            _row['notes'] = f'Error lectura european_coverage.csv: {_e}'
        rows.append(_row)

    # Fase 2: anadir schema_v por fuente leyendo .schema lateral.
    # Parquet no lleva .schema (es .csv.schema solo para CSV). Fuentes
    # sin .schema: None -> v1 implicito en presentacion.
    _path_by_source = {}
    for _src in sources:
        _path_by_source[_src['source']] = _src['file']
    for _row in rows:
        _src_name = _row.get('source')
        _csv_path = _path_by_source.get(_src_name)
        if _csv_path is None:
            _row['schema_v'] = None
            continue
        _row['schema_v'] = _read_schema_version(_csv_path)
    return pd.DataFrame(rows)
