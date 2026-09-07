# -*- coding: utf-8 -*-
"""
Calidad, frescura y cobertura de datos v1.0
Describe actualidad y completitud de las principales fuentes.
No alimenta motores, scores, pesos ni State Machine.
No crea Data Quality Score.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

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

def compute_data_quality():
    sources = [
        {'source': 'Yahoo Finance', 'file': 'outputs/history/sector_breadth.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Precios y métricas sectoriales'},
        {'source': 'CBOE PCR', 'file': 'outputs/history/pcr_history.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Put/Call ratio'},
        {'source': 'FINRA Dark Pool', 'file': 'outputs/history/darkpool_history.csv', 'date_col': 'week', 'frequency': 'finra', 'notes': 'Retraso regulatorio 2-4 semanas'},
        {'source': 'CFTC Position Flow', 'file': 'outputs/history/cftc_position_flow.csv', 'date_col': 'date', 'frequency': 'cftc', 'notes': 'Frescura 30 días'},
        {'source': 'SEC N-PORT', 'file': 'outputs/history/sec_nport_position_change_quarterly.csv', 'date_col': 'date', 'frequency': 'sec', 'notes': 'Trimestral'},
        {'source': 'Macro FRED', 'file': 'outputs/history/macro_regime.csv', 'date_col': 'date', 'frequency': 'fred', 'notes': 'Datos macro manuales'},
        {'source': 'SSGA ETF Flow', 'file': 'outputs/history/etf_primary_flow.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Flujo primario'},
        {'source': 'Sector Concentration', 'file': 'outputs/history/sector_concentration.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Concentración y dispersión'},
        {'source': 'Sector Breadth Momentum', 'file': 'outputs/history/sector_breadth_momentum.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Momentum amplitud'},
        {'source': 'Volatility Structure', 'file': 'outputs/history/volatility_structure.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Estructura volatilidad'},
        {'source': 'Cross Asset Context', 'file': 'outputs/history/cross_asset_context.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Contexto cross-asset'},
        {'source': 'Evidence Matrix', 'file': 'outputs/history/evidence_matrix.csv', 'date_col': 'date', 'frequency': 'daily', 'notes': 'Matriz evidencia'},
    ]

    rows = []
    now = datetime.now()

    for src in sources:
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

        try:
            df = pd.read_csv(path)
            if src['date_col'] in df.columns:
                dates = pd.to_datetime(df[src['date_col']], errors='coerce')
                last_date = dates.max()
            else:
                last_date = pd.NaT
            age = (now - last_date).days if pd.notna(last_date) else np.nan
            freshness = classify_freshness(age, src['frequency'])

            n_total = np.nan
            n_valid = np.nan
            coverage = np.nan

            if src['source'] in ['Yahoo Finance', 'Sector Concentration', 'Sector Breadth Momentum']:
                if 'n_total' in df.columns and 'n_valid_return20' in df.columns:
                    n_total = df['n_total'].sum()
                    n_valid = df['n_valid_return20'].sum()
                    coverage = n_valid / n_total if n_total else np.nan
            elif src['source'] == 'CBOE PCR':
                n_total = len(df)
                n_valid = df['total_pcr'].notna().sum() if 'total_pcr' in df.columns else 0
                coverage = n_valid / n_total if n_total else np.nan
            else:
                n_total = len(df)
                n_valid = int(df.notna().sum().mean()) if not df.empty else 0
                coverage = n_valid / n_total if n_total else np.nan

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

    return pd.DataFrame(rows)