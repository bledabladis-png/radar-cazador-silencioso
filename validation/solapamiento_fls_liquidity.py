# -*- coding: utf-8 -*-
"""
Solapamiento FLS vs Liquidity.

Calcula correlación entre el Funding & Liquidity Stress (FLS) y el
score de Liquidez Real (FRED) usando series históricas muestreadas.

Salida: outputs/audit/solapamiento_fls_liquidity.csv
"""

from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from data.providers.router import DataRouter
from indicators.fls import manual_robust_zscore
from src.utils import robust_zscore, tanh_normalize

OUTPUT_PATH = Path('outputs/audit/solapamiento_fls_liquidity.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FILES_FLS = {
    'sofr': Path('data/macro_manual/sofr.csv'),
    'walcl': Path('data/macro_manual/walcl.csv'),
    'rrpp': Path('data/macro_manual/rrpp.csv'),
    'cp': Path('data/macro_manual/commercial_paper.csv'),
    'discount': Path('data/macro_manual/discount_rate.csv'),
}

def load_fls_series():
    """Carga las series de FLS desde CSVs manuales."""
    series = {}
    for key, path in FILES_FLS.items():
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = df.columns[0]
        series[key] = df[col].dropna()
    return series

def compute_fls_at(series_dict, date):
    """Calcula FLS value en una fecha dada usando las series históricas."""
    components = []
    for key, s in series_dict.items():
        if date in s.index:
            # valor en fecha exacta o último anterior
            sub = s.loc[:date]
            if sub.empty:
                continue
            z = manual_robust_zscore(sub, window=252)
            components.append(float(np.tanh(z)))
    if not components:
        return np.nan
    return float(np.mean(components))

def main():
    # 1) Cargar series FLS
    fls_series = load_fls_series()
    if not fls_series:
        raise SystemExit("No se encontraron series FLS")

    # 2) Cargar datos FRED para Liquidity
    router = DataRouter()
    fed = router.get_fed_data()
    if fed is None or fed.empty:
        raise SystemExit("No se pudieron obtener datos FRED")

    # 3) Construir fechas comunes (intersección de índices)
    common_dates = fed.index
    for s in fls_series.values():
        common_dates = common_dates.intersection(s.index)
    if len(common_dates) < 30:
        print(f"ADVERTENCIA: solo {len(common_dates)} fechas comunes. Se usarán todas.")
    # Muestrear cada 5 días para ligereza
    sample_dates = common_dates[::5]
    if len(sample_dates) < 10:
        sample_dates = common_dates

    rows = []
    for date in sample_dates:
        try:
            # FLS en fecha
            fls_val = compute_fls_at(fls_series, date)

            # Liquidity en fecha usando lógica de compute_liquidity_score
            fed_until = fed.loc[:date].ffill()
            signals = {}
            if 'fed_balance' in fed_until.columns and not fed_until['fed_balance'].isna().all():
                z = robust_zscore(fed_until['fed_balance'], window=60)
                if pd.notna(z.iloc[-1]):
                    signals['fed_balance'] = tanh_normalize(z).iloc[-1]
            if 'reverse_repo' in fed_until.columns and not fed_until['reverse_repo'].isna().all():
                z = robust_zscore(fed_until['reverse_repo'], window=60)
                val = -tanh_normalize(z).iloc[-1]
                if pd.notna(val):
                    signals['reverse_repo'] = val
            if 'sofr' in fed_until.columns and not fed_until['sofr'].isna().all():
                z = robust_zscore(fed_until['sofr'], window=60)
                val = -tanh_normalize(z).iloc[-1]
                if pd.notna(val):
                    signals['sofr'] = val
            if 'fed_funds' in fed_until.columns and not fed_until['fed_funds'].isna().all():
                z = robust_zscore(fed_until['fed_funds'], window=60)
                val = -tanh_normalize(z).iloc[-1]
                if pd.notna(val):
                    signals['fed_funds'] = val

            if not signals:
                liq_score = np.nan
            else:
                weights = {'fed_balance': 0.35, 'reverse_repo': 0.25, 'sofr': 0.20, 'fed_funds': 0.20}
                available = [k for k in weights if k in signals]
                w_sum = sum(weights[k] for k in available)
                liq_score = sum(signals[k] * weights[k] / w_sum for k in available)

            if pd.notna(fls_val) and pd.notna(liq_score):
                rows.append({'date': date, 'fls': fls_val, 'liquidity': liq_score})
        except Exception as e:
            print(f"Error en {date}: {e}")
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        print("No se pudieron calcular pares de datos.")
        return

    print(f"Registros analizados: {len(df)}")
    rho, pval = spearmanr(df['fls'], df['liquidity'], nan_policy='omit')
    summary = pd.DataFrame([{
        'n_registros': len(df),
        'spearman_rho': rho,
        'p_value': pval,
        'fls_mean': df['fls'].mean(),
        'liquidity_mean': df['liquidity'].mean(),
    }])
    print("\nResumen de solapamiento:")
    print(summary.to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT_PATH.with_name('solapamiento_fls_liquidity_summary.csv'), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
