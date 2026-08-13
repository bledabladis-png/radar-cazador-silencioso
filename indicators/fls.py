# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def manual_robust_zscore(series, window=252):
    if len(series.dropna()) < window:
        return 0.0
    last_values = series.dropna().iloc[-window:]
    median = last_values.median()
    mad = np.median(np.abs(last_values - median))
    if mad == 0:
        return 0.0
    return (series.iloc[-1] - median) / (1.4826 * mad)

def compute_fls():
    """
    Calcula el Funding & Liquidity Stress (FLS).
    Retorna el valor normalizado y el desglose por componente.
    """
    files = {
        'sofr': 'data/macro_manual/sofr.csv',
        'walcl': 'data/macro_manual/walcl.csv',
        'rrpp': 'data/macro_manual/rrpp.csv',
        'cp': 'data/macro_manual/commercial_paper.csv',
        'discount': 'data/macro_manual/discount_rate.csv',
    }
    
    detail = {}
    stresses = []

    # 1. SOFR
    try:
        sofr = pd.read_csv(files['sofr'], index_col=0, parse_dates=True)['SOFR']
        sofr_z = manual_robust_zscore(sofr)
        sofr_stress = float(np.tanh(sofr_z))
        stresses.append(sofr_stress)
        detail['SOFR'] = {'value': sofr_stress, 'stressed': sofr_stress > 0.3}
    except:
        detail['SOFR'] = {'value': None, 'stressed': False}

    # 2. WALCL
    try:
        walcl = pd.read_csv(files['walcl'], index_col=0, parse_dates=True)['WALCL']
        walcl_chg = walcl.pct_change(252, fill_method=None)
        walcl_z = -manual_robust_zscore(walcl_chg)
        walcl_stress = float(np.tanh(walcl_z))
        stresses.append(walcl_stress)
        detail['WALCL'] = {'value': walcl_stress, 'stressed': walcl_stress > 0.3}
    except:
        detail['WALCL'] = {'value': None, 'stressed': False}

    # 3. Reverse Repo
    try:
        rrpp = pd.read_csv(files['rrpp'], index_col=0, parse_dates=True)['RRPONTSYD']
        rrpp_chg = rrpp.pct_change(252, fill_method=None)
        rrpp_z = -manual_robust_zscore(rrpp_chg)
        rrpp_stress = float(np.tanh(rrpp_z))
        stresses.append(rrpp_stress)
        detail['RRP'] = {'value': rrpp_stress, 'stressed': rrpp_stress > 0.3}
    except:
        detail['RRP'] = {'value': None, 'stressed': False}

    # 4. Commercial Paper
    try:
        cp = pd.read_csv(files['cp'], index_col=0, parse_dates=True)['COMPOUT']
        cp_z = manual_robust_zscore(cp)
        cp_stress = float(np.tanh(cp_z))
        stresses.append(cp_stress)
        detail['CP'] = {'value': cp_stress, 'stressed': cp_stress > 0.3}
    except:
        detail['CP'] = {'value': None, 'stressed': False}

    # 5. Discount Rate
    try:
        disc = pd.read_csv(files['discount'], index_col=0, parse_dates=True)['DPRIME']
        disc_z = manual_robust_zscore(disc)
        disc_stress = float(np.tanh(disc_z))
        stresses.append(disc_stress)
        detail['Discount'] = {'value': disc_stress, 'stressed': disc_stress > 0.3}
    except:
        detail['Discount'] = {'value': None, 'stressed': False}

    if stresses:
        fls_value = float(np.mean(stresses))
        fls_normalized = float(np.clip((fls_value + 1) / 2, 0, 1))
        stressed_count = sum(1 for d in detail.values() if d.get('stressed', False))
    else:
        fls_value = 0.0
        fls_normalized = 0.5
        stressed_count = 0

    return {
        'fls_value': fls_value,
        'fls_normalized': fls_normalized,
        'components': len(stresses),
        'total_components': 5,
        'stressed_components': stressed_count,
        'detail': detail
    }
