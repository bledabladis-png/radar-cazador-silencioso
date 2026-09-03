# -*- coding: utf-8 -*-
"""
Redundancia MTE vs Financial Conditions.

Evalúa la correlación entre el score de condiciones financieras y
los componentes principales del Market Transition Engine (MSI, IPI,
SRS, SHS, CLS, IPS). No modifica el sistema productivo.

Salida: outputs/audit/redundancia_mte_fc.csv
"""

from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from data.providers.router import DataRouter
from src.utils import get_col
from regimes.financial_conditions import compute_financial_conditions
from indicators.mte import (
    sector_rotation_score,
    safe_haven_score,
    credit_stress_score,
    inflation_pressure_score,
    compute_msi,
    compute_ipi,
)

OUTPUT_PATH = Path('outputs/audit/redundancia_mte_fc.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_data():
    cache = Path('data/market_data.csv')
    if cache.exists():
        df = pd.read_csv(cache, header=[0,1], index_col=0, parse_dates=True)
        # Verificar que contenga los tickers básicos
        if any(('Close', s) in df.columns for s in ['^GSPC','^VIX']):
            return df
    print("Descargando datos de mercado...")
    router = DataRouter()
    tickers = ['^GSPC','^VIX','HYG','LQD','DX-Y.NYB','^TNX','^FVX','XLE','^SPGSCI','TIP']
    data = router.get_market_data(tickers, period="5y")
    if data is None or data.empty:
        raise RuntimeError("No se pudieron descargar datos")
    data.to_csv(cache)
    return data

def main():
    data = load_data()

    # Calcular Financial Conditions
    fc_score, fc_regime, fc_conf = compute_financial_conditions(data)
    if not isinstance(fc_score, pd.Series):
        fc_series = pd.Series(fc_score, index=data.index)
    else:
        fc_series = fc_score

    # Calcular componentes MTE
    try:
        srs = sector_rotation_score(data)
    except Exception as e:
        print(f"SRS error: {e}"); srs = np.nan
    try:
        shs = safe_haven_score(data)
    except Exception as e:
        print(f"SHS error: {e}"); shs = np.nan

    try:
        # Necesitamos credit_signal y volatility_signal para CLS
        credit_signal = (get_col(data, 'HYG', 'Close') / get_col(data, 'LQD', 'Close')).pct_change(20)
        volatility_signal = get_col(data, '^VIX', 'Close')
        cls = credit_stress_score(fc_series.iloc[-1], credit_signal.iloc[-1],
                                  volatility_signal.iloc[-1], vix_term=0.0,
                                  darkpool_z=None, pcr_z=None)
    except Exception as e:
        print(f"CLS error: {e}"); cls = np.nan

    try:
        ips = inflation_pressure_score(data)
    except Exception as e:
        print(f"IPS error: {e}"); ips = np.nan

    msi = compute_msi(srs, shs, cls) if all(pd.notna(v) for v in [srs, shs, cls]) else np.nan
    ipi = compute_ipi(ips) if pd.notna(ips) else np.nan

    # Construir DataFrame con valores actuales y una serie histórica de FC
    # Para correlación, usamos fc_series y calculamos proxies históricos simplificados
    # No recalculamos MTE histórico completo para no sobrecargar
    # En su lugar, reportamos correlación instantánea transversal no posible
    # Mostramos valores actuales y un análisis de correlación entre FC y componentes
    # usando los últimos 60 días de FC vs proxies simples de componentes (opcional)

    # Para simplicidad, reportamos valores actuales y un resumen
    results = {
        'fc_score': float(fc_series.iloc[-1]) if not fc_series.empty else np.nan,
        'fc_regime': fc_regime,
        'fc_conf': float(fc_conf),
        'srs': srs,
        'shs': shs,
        'cls': cls,
        'ips': ips,
        'msi': msi,
        'ipi': ipi,
    }
    df = pd.DataFrame([results])
    print("Valores actuales:")
    print(df.to_string(index=False))

    # Matriz de correlación con series temporales aproximadas
    # Calculamos FC histórico y componentes MTE en cada fecha (muestreado semanal)
    # para no saturar el computo usamos ultimas 30 fechas
    fechas = data.index[-30:]
    rows_corr = []
    for date in fechas:
        try:
            df_until = data.loc[:date]
            fc_val = fc_series.loc[:date].iloc[-1] if date in fc_series.index else fc_series.iloc[-1]
            srs_val = sector_rotation_score(df_until)
            shs_val = safe_haven_score(df_until)
            cls_val = credit_stress_score(
                fc_val,
                (get_col(df_until, 'HYG', 'Close') / get_col(df_until, 'LQD', 'Close')).iloc[-1],
                get_col(df_until, '^VIX', 'Close').iloc[-1],
                vix_term=0.0,
                darkpool_z=None,
                pcr_z=None
            )
            ips_val = inflation_pressure_score(df_until)
            msi_val = compute_msi(srs_val, shs_val, cls_val)
            ipi_val = compute_ipi(ips_val)
            rows_corr.append({
                'date': date,
                'fc': fc_val,
                'srs': srs_val,
                'shs': shs_val,
                'cls': cls_val,
                'ips': ips_val,
                'msi': msi_val,
                'ipi': ipi_val,
            })
        except Exception:
            continue

    corr_df = pd.DataFrame(rows_corr)
    if not corr_df.empty:
        corr_results = []
        for comp in ['srs','shs','cls','ips','msi','ipi']:
            if corr_df[comp].notna().sum() >= 5:
                rho, p = spearmanr(corr_df['fc'], corr_df[comp], nan_policy='omit')
                corr_results.append({'par': f'fc vs {comp}', 'spearman_rho': rho, 'p_value': p})
        corr_out = pd.DataFrame(corr_results)
        print("\nCorrelación Spearman histórica (30 días):")
        print(corr_out.to_string(index=False))
        corr_out.to_csv(OUTPUT_PATH.with_name('redundancia_mte_fc_corr.csv'), index=False, encoding='utf-8-sig')

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nGuardado en {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
