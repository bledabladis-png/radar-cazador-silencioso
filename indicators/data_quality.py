# -*- coding: utf-8 -*-
"""
Calidad, frescura y cobertura de datos v1.1
Describe actualidad y completitud de las principales fuentes.
No alimenta motores, scores, pesos ni State Machine.
No crea Data Quality Score.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.market_calendar import _last_market_session

def classify_freshness(age_days, frequency):
    if pd.isna(age_days):
        return 'N/D'
    if frequency == 'finra':
        if age_days <= 30: return 'CURRENT'
        elif age_days <= 45: return 'RECENT'
        elif age_days <= 60: return 'STALE'
        else: return 'ARCHIVAL'
    elif frequency == 'sec' or frequency == 'cftc':
        if age_days <= 45: return 'CURRENT'
        elif age_days <= 90: return 'RECENT'
        elif age_days <= 120: return 'STALE'
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

def _read_date_col(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

def compute_data_quality():
    now = datetime.now()
    sources = [
        {
            'source': 'Yahoo Finance',
            'file': 'data/market_data.parquet',
            'date_cols': None,
            'frequency': 'daily',
            'coverage_logic': None,
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
            'coverage_logic': None,
            'notes': 'Retraso regulatorio 2-4 semanas'
        },
        {
            'source': 'CFTC Position Flow',
            'file': 'outputs/history/cftc_position_flow.csv',
            'date_cols': ['date'],
            'frequency': 'cftc',
            'coverage_logic': None,
            'notes': 'Frescura 30 días'
        },
        {
            'source': 'SEC N-PORT',
            'file': 'outputs/history/sec_nport_position_change_quarterly.csv',
            'date_cols': ['date'],
            'frequency': 'sec',
            'coverage_logic': None,
            'notes': 'Trimestral'
        },
        {
            'source': 'Macro FRED',
            'file': 'outputs/history/macro_regime.csv',
            'date_cols': ['date'],
            'frequency': 'fred',
            'coverage_logic': None,
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
            'coverage_logic': None,
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
    ]

    rows = []
    for src in sources:
        if src['source'] == 'Macro FRED':
            try:
                import json
                liq_state_path = Path('outputs/state/liquidity_state.json')
                if liq_state_path.exists():
                    with liq_state_path.open('r') as _f:
                        _state = json.load(_f)
                    _ld = pd.to_datetime(_state.get('date'))
                    last_date = _last_market_session(_ld) if pd.notna(_ld) else pd.NaT
                else:
                    last_date = pd.NaT
            except Exception:
                last_date = pd.NaT
            age = (now - last_date).days if pd.notna(last_date) else np.nan
            freshness = classify_freshness(age, src['frequency'])
            rows.append({
                'date': now.strftime('%Y-%m-%d'),
                'source': src['source'],
                'last_date': last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else np.nan,
                'age_calendar_days': age,
                'frequency': src['frequency'],
                'freshness': freshness,
                'n_total': np.nan,
                'n_valid': np.nan,
                'coverage': np.nan,
                'notes': src.get('notes', ''),
            })
            continue
        path = Path(src['file'])
        if not path.exists():
            rows.append({
                'date': now.strftime('%Y-%m-%d'),
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
                age = (now - last_date).days if pd.notna(last_date) else np.nan
                freshness = classify_freshness(age, src['frequency'])
                rows.append({
                    'date': now.strftime('%Y-%m-%d'),
                    'source': src['source'],
                    'last_date': last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else np.nan,
                    'age_calendar_days': age,
                    'frequency': src['frequency'],
                    'freshness': freshness,
                    'n_total': np.nan,
                    'n_valid': np.nan,
                    'coverage': np.nan,
                    'notes': src.get('notes', ''),
                })
            except Exception as e:
                rows.append({
                    'date': now.strftime('%Y-%m-%d'),
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
                last_date = _last_market_session(_ld) if pd.notna(_ld) else pd.NaT
            age = (now - last_date).days if pd.notna(last_date) else np.nan
            freshness = classify_freshness(age, src['frequency'])

            n_total = np.nan
            n_valid = np.nan
            coverage = np.nan

            logic = src.get('coverage_logic')
            if logic == 'sectorial':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = 11
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
                n_valid = int(df['total_pcr'].notna().sum()) if 'total_pcr' in df.columns else 0
                coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'sectorial_family':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = 11
                    n_valid = int(last_df[last_df['mean_corr'].notna()]['sector'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            elif logic == 'sectorial_evidence':
                if date_col is not None:
                    last_df = df[pd.to_datetime(df[date_col], errors='coerce') == last_date]
                    n_total = 11
                    n_valid = int(last_df[last_df['alignment_reading'] != 'EVIDENCIA INSUFICIENTE']['sector'].nunique())
                    coverage = n_valid / n_total if n_total else np.nan
            # else coverage permanece NaN

            rows.append({
                'date': now.strftime('%Y-%m-%d'),
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
        except Exception as e:
            rows.append({
                'date': now.strftime('%Y-%m-%d'),
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
            'date': now.strftime('%Y-%m-%d'),
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
                    _age = (now - _last).days if pd.notna(_last) else np.nan
                    _fresh = classify_freshness(_age, 'daily')
                    _row['last_date'] = _last.strftime('%Y-%m-%d') if pd.notna(_last) else np.nan
                    _row['age_calendar_days'] = _age
                    _row['freshness'] = _fresh
                    _row['n_total'] = _n_total
                    _row['n_valid'] = _ok
                    _row['coverage'] = _ok / _n_total if _n_total else np.nan
                    _row['notes'] = f'{_eu_source} ({_ok}/{_n_total} OK)'
        except Exception as _e:
            _row['notes'] = f'Error lectura european_coverage.csv: {_e}'
        rows.append(_row)

    return pd.DataFrame(rows)
