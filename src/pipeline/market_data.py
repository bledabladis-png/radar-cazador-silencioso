# -*- coding: utf-8 -*-
"""Fase 9b del pipeline: PCR + Dark Pools + Volatilidad estructural
+ Calidad de datos.

Extraido de run.py (refactor C2, fase C2-9b).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import append_dedup


def _compute_pcr():
    print("Calculando sentimiento de opciones (PCR)...")
    pcr_data = None
    try:
        from indicators.options import compute_pcr_signals
        pcr_data = compute_pcr_signals()
        if pcr_data and pcr_data.get('status') == 'OK':
            print(f"  PCR Total: {pcr_data['total_pcr']:.2f} (Z: {pcr_data['z_score']:.2f}, Estado: {pcr_data['state']})")
        elif pcr_data:
            print(f"  OMS STATUS: {pcr_data['status']}")
    except Exception as e:
        print(f"  Modulo PCR omitido: {e}")
    return pcr_data


def _compute_darkpool():
    print("Calculando Dark Pools (FINRA ATS)...")
    darkpool_data = None
    try:
        from indicators.darkpool import compute_darkpool_signals
        darkpool_data = compute_darkpool_signals()
        if darkpool_data:
            print(f"  Dark Pool medio: {darkpool_data['media_dark_pool']:.2f}% "
                  f"({darkpool_data['n_tickers_ats']}/{darkpool_data['n_tickers_total']} tickers)")
        else:
            print("  Dark Pools: no disponible")
    except Exception as e:
        print(f"  Modulo Dark Pools omitido: {e}")
    return darkpool_data


def _compute_vol_structure(df_market, pcr_data):
    try:
        from indicators.volatility_structure import compute_volatility_structure
        pcr_vol = {}
        if pcr_data:
            pcr_vol['zscore'] = pcr_data.get('z_score', np.nan)
            # No recalcular percentil; si no existe, se deja NaN
            if 'percentile_20d' in pcr_data:
                pcr_vol['percentile_20d'] = pcr_data['percentile_20d']
        vol_structure_df = compute_volatility_structure(df_market, pcr_data=pcr_vol if pcr_vol else None)
        vs_path = Path('outputs/history/volatility_structure.csv')
        vs_path.parent.mkdir(parents=True, exist_ok=True)
        if not vol_structure_df.empty:
            if vs_path.exists():
                hist_vs = pd.read_csv(vs_path)
                vol_structure_df = append_dedup(hist_vs, vol_structure_df, ['date'])
            vol_structure_df.to_csv(vs_path, index=False)
            print("  Volatilidad estructural calculada.")
        else:
            vol_structure_df = None
    except Exception as e:
        print(f"  Volatilidad estructural omitida: {e}")
        vol_structure_df = None
    return vol_structure_df


def _compute_data_quality():
    try:
        from indicators.data_quality import compute_data_quality
        data_quality_df = compute_data_quality()
        dq_path = Path('outputs/history/data_quality.csv')
        dq_path.parent.mkdir(parents=True, exist_ok=True)
        if not data_quality_df.empty:
            if dq_path.exists():
                hist_dq = pd.read_csv(dq_path)
                data_quality_df = append_dedup(hist_dq, data_quality_df, ['date','source'])
            data_quality_df.to_csv(dq_path, index=False)
            print("  Calidad de datos calculada.")
        else:
            data_quality_df = None
    except Exception as e:
        print(f"  Calidad de datos omitida: {e}")
        data_quality_df = None
    return data_quality_df


def compute_market_data(df_market):
    """Ejecuta PCR + Dark Pools + Volatilidad estructural + Calidad datos.

    Returns:
        dict con keys:
            pcr_data, darkpool_data, vol_structure_df, data_quality_df
    """
    pcr_data = _compute_pcr()
    darkpool_data = _compute_darkpool()
    vol_structure_df = _compute_vol_structure(df_market, pcr_data)
    data_quality_df = _compute_data_quality()
    return {
        'pcr_data': pcr_data,
        'darkpool_data': darkpool_data,
        'vol_structure_df': vol_structure_df,
        'data_quality_df': data_quality_df,
    }
