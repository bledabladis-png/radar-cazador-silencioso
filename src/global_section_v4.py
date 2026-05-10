# global_section_v4.py – Generador de sección del reporte para Radar Global v4.0

import pandas as pd
import numpy as np
from global_config_v4 import *
from global_flow_v4 import compute_flow_pressure, compute_capital_rotation, flow_participation_ratio
from global_risk_v4 import compute_risk_direction, compute_volatility_regime, compute_risk_breadth
from global_cross_region_v4 import compute_region_alignment, compute_coherence_v4, compute_dominance_flag
from global_stability_v4 import signal_stability, stability_flags, regime_shift_detector

# ------------------------------------------------------------
# VALIDACIÓN DE DATOS (heredada de v3.19)
# ------------------------------------------------------------
def validate_global_data(df_global):
    issues = {}
    valid_cols = []
    for col in df_global.columns:
        if not col.endswith('_close'):
            continue
        ticker = col.replace('_close', '')
        close = df_global[col]
        if close.isna().sum() / len(close) > 0.1:
            issues[ticker] = 'DATA ISSUE (NaNs >10%)'
            continue
        if (close <= 0).any():
            issues[ticker] = 'DATA ISSUE (close <=0)'
            continue
        weekly_ret = close.pct_change(periods=5, fill_method=None).dropna()
        if (abs(weekly_ret) > 0.40).any():
            issues[ticker] = 'DATA ISSUE (extreme weekly return >40%)'
            continue
        elif (abs(weekly_ret) > 0.25).any():
            issues[ticker] = 'EXTREME EVENT (ret >25%, señal válida pero inusual)'
        valid_cols.append(ticker)
    return valid_cols, issues

def compute_weekly_data(df_global, valid_tickers):
    closes = pd.DataFrame()
    volumes = pd.DataFrame()
    for t in valid_tickers:
        close_col = f"{t}_close"
        if close_col in df_global.columns:
            closes[t] = df_global[close_col]
        vol_col = f"{t}_volume"
        if vol_col in df_global.columns:
            volumes[t] = df_global[vol_col]
    fridays = closes.index[closes.index.weekday == 4]
    closes_weekly = closes.loc[fridays]
    volumes_weekly = volumes.loc[fridays]
    closes_weekly = closes_weekly.dropna(how='all')
    volumes_weekly = volumes_weekly.dropna(how='all')
    
    weekly_returns = closes_weekly.pct_change(fill_method=None).dropna()
    return weekly_returns, volumes_weekly, closes_weekly

# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PARA EL GLOBAL RISK SCORE
# ------------------------------------------------------------
def compute_global_risk_score(flow_spy, flow_ezu, flow_ewj, flow_eem, hyg_flow):
    sigmoid = lambda x: (np.tanh(x) + 1) / 2

    geo_continuous = np.mean([
        sigmoid(flow_spy), sigmoid(flow_ezu), sigmoid(flow_ewj), sigmoid(flow_eem)
    ])
    
    # Flow alignment: fracción de bloques geográficos con flujo positivo
    n_positive = sum(1 for v in [flow_spy, flow_ezu, flow_ewj, flow_eem] if v > 0)
    flow_align_norm = n_positive / 4.0   # [0, 0.25, 0.50, 0.75, 1.0]
    
    hyg_confirm = (np.tanh(hyg_flow) * np.tanh(flow_spy) + 1) / 2
    
    score = 0.50 * geo_continuous + 0.45 * flow_align_norm + 0.05 * hyg_confirm
    return np.clip(score, 0, 1)

# ------------------------------------------------------------
# GENERADOR PRINCIPAL
# ------------------------------------------------------------
def generate_global_section_v4(df_global):
    lines = ["\n## Radar Global v4.0 – Contexto Semanal\n"]

    valid_tickers, issues = validate_global_data(df_global)
    if not valid_tickers:
        lines.append("*Radar Global no disponible: datos insuficientes.*\n")
        return lines

    weekly_returns, volumes_weekly, closes_weekly = compute_weekly_data(df_global, valid_tickers)
    if weekly_returns.empty:
        lines.append("*Datos semanales insuficientes.*\n")
        return lines

    dollar_vol = pd.DataFrame()
    for t in weekly_returns.columns:
        if t in closes_weekly.columns and t in volumes_weekly.columns:
            dollar_vol[t] = closes_weekly[t] * volumes_weekly[t].reindex(closes_weekly.index)

    # -------------------------------------------
    # FLOW MODULE
    # -------------------------------------------
    flow_df = compute_flow_pressure(weekly_returns, dollar_vol)

    # -------------------------------------------
    # RISK MODULE
    # -------------------------------------------
    direction_df = compute_risk_direction(weekly_returns)

    # -------------------------------------------
    # LOOK-AHEAD MITIGATION
    # -------------------------------------------
    flow_df = flow_df.shift(1).iloc[1:]           # elimina solo la primera fila NaN
    direction_df = direction_df.shift(1).iloc[1:]
    weekly_returns_shifted = weekly_returns.shift(1).iloc[1:]

    if flow_df.empty or direction_df.empty or weekly_returns_shifted.empty:
        lines.append("*Datos semanales insuficientes tras ajuste de look-ahead.*\n")
        return lines

    # -------------------------------------------
    # MÉTRICAS COMPUESTAS
    # -------------------------------------------
    capital_rot = compute_capital_rotation(
        flow_df, 
        FLOW_ASSETS['equity'], 
        FLOW_ASSETS['fixed_income'], 
        FLOW_ASSETS['commodities']
    )
    participation = flow_participation_ratio(flow_df, FLOW_ASSETS['equity'])

    risk_assets_available = [t for t in RISK_ASSETS['equity'] if t in weekly_returns_shifted.columns]
    vol_regime = compute_volatility_regime(weekly_returns_shifted, risk_assets_available) if risk_assets_available else 1.0
    risk_breadth = compute_risk_breadth(direction_df, RISK_ASSETS['equity'])

    alignment = compute_region_alignment(weekly_returns_shifted, CROSS_REGION)
    
    spy_flow = flow_df['SPY'] if 'SPY' in flow_df.columns else pd.Series()
    global_flow = flow_df[['EZU', 'EWJ', 'EEM']].median(axis=1) if all(t in flow_df.columns for t in ['EZU', 'EWJ', 'EEM']) else pd.Series()
    spy_dir = direction_df['SPY'] if 'SPY' in direction_df.columns else pd.Series()
    global_dir = direction_df[['EZU', 'EWJ', 'EEM']].median(axis=1) if all(t in direction_df.columns for t in ['EZU', 'EWJ', 'EEM']) else pd.Series()
    
    coherence = compute_coherence_v4(spy_flow, global_flow, spy_dir, global_dir)
    
    spy_val = flow_df['SPY'].iloc[-1] if 'SPY' in flow_df.columns else 0
    others_median = np.median([flow_df[t].iloc[-1] for t in ['EZU', 'EWJ', 'EEM'] if t in flow_df.columns])
    dominance = compute_dominance_flag(spy_val, others_median)

    uup_close = closes_weekly['UUP'] if 'UUP' in closes_weekly.columns else None
    if uup_close is not None and len(uup_close) >= 10:
        trend = uup_close.iloc[-1] / uup_close.iloc[-10] - 1
        usd = 'STRONG' if trend > 0.02 else 'WEAK' if trend < -0.02 else 'NEUTRAL'
    else:
        usd = 'NEUTRAL'

    # -------------------------------------------
    # GLOBAL RISK SCORE & REGIME STATE
    # -------------------------------------------
    score = compute_global_risk_score(
        flow_df['SPY'].iloc[-1] if 'SPY' in flow_df.columns else 0,
        flow_df['EZU'].iloc[-1] if 'EZU' in flow_df.columns else 0,
        flow_df['EWJ'].iloc[-1] if 'EWJ' in flow_df.columns else 0,
        flow_df['EEM'].iloc[-1] if 'EEM' in flow_df.columns else 0,
        flow_df['HYG'].iloc[-1] if 'HYG' in flow_df.columns else 0
    )

    regime_state_file = 'data/global_regime_history.csv'
    previous_label = None
    persistence_counter = 0
    try:
        regime_hist = pd.read_csv(regime_state_file, index_col=0, parse_dates=True)
        if not regime_hist.empty:
            previous_label = regime_hist['regime'].iloc[-1]
            if 'persistence' in regime_hist.columns:
                persistence_counter = int(regime_hist['persistence'].iloc[-1])
    except FileNotFoundError:
        pass

    # Asignar etiqueta según score (sin histéresis en esta versión)
    if score > 0.60:
        label = 'RISK-ON'
    elif score > 0.50:
        label = 'WEAK RISK-ON'
    elif score > 0.40:
        label = 'NEUTRAL'
    elif score > 0.30:
        label = 'WEAK RISK-OFF'
    else:
        label = 'RISK-OFF'
    persistence_counter = persistence_counter + 1 if label == previous_label else 1

    new_regime_df = pd.DataFrame({
        'regime': [label],
        'score': [score],
        'persistence': [persistence_counter]
    }, index=[pd.Timestamp.now()])
    try:
        old_regime = pd.read_csv(regime_state_file, index_col=0, parse_dates=True)
        regime_combined = pd.concat([old_regime, new_regime_df])
    except FileNotFoundError:
        regime_combined = new_regime_df
    regime_combined.to_csv(regime_state_file)

    # -------------------------------------------
    # STABILITY LAYER & REGIME SHIFT
    # -------------------------------------------
    capital_rot_series = pd.Series(dtype=float)
    risk_breadth_series = pd.Series(dtype=float)
    alignment_series = pd.Series(dtype=float)
    vol_regime_series = pd.Series(dtype=float)

    try:
        hist = pd.read_csv('data/global_metrics_history.csv', index_col=0, parse_dates=True)
        if 'capital_rotation' in hist.columns:
            capital_rot_series = hist['capital_rotation'].dropna()
        if 'risk_breadth' in hist.columns:
            risk_breadth_series = hist['risk_breadth'].dropna()
        if 'alignment' in hist.columns:
            alignment_series = hist['alignment'].dropna()
        if 'volatility_regime' in hist.columns:
            vol_regime_series = hist['volatility_regime'].dropna()
    except FileNotFoundError:
        pass

    capital_rot_series = pd.concat([capital_rot_series, pd.Series([capital_rot])])
    risk_breadth_series = pd.concat([risk_breadth_series, pd.Series([risk_breadth])])
    alignment_series = pd.concat([alignment_series, pd.Series([alignment])])
    vol_regime_series = pd.concat([vol_regime_series, pd.Series([vol_regime])])

    hist_df = pd.DataFrame({
        'capital_rotation': capital_rot_series,
        'risk_breadth': risk_breadth_series,
        'alignment': alignment_series,
        'volatility_regime': vol_regime_series
    })
    hist_df.to_csv('data/global_metrics_history.csv')

    rot_stab = signal_stability(capital_rot_series)
    shift_detected = regime_shift_detector(vol_regime_series)
    stab_flags = stability_flags(capital_rot_series, alignment_series)

    # -------------------------------------------
    # REPORTE
    # -------------------------------------------
    if shift_detected:
        lines.append("⚠️ **Regime Shift Detectado** – la volatilidad actual es anormalmente alta.\n")
    for flag in stab_flags:
        lines.append(flag + "  \n")
    if rot_stab is not None and not np.isnan(rot_stab):
        lines.append(f"- **Estabilidad de Capital Rotation:** {rot_stab:.2f}  \n")

    lines.append(f"**Global Risk Score:** {score:.2f} → {label}  \n\n")
    lines.append("### Flow Module\n")
    lines.append(f"- **Capital Rotation:** {capital_rot:.2f}  \n")
    lines.append(f"- **Flow Participation (RV):** {participation:.0%}  \n\n")

    lines.append("### Risk Module\n")
    lines.append(f"- **Risk Breadth:** {risk_breadth:.0%}  \n")
    lines.append(f"- **Volatility Regime (z-score):** {vol_regime:.2f}  \n\n")

    lines.append("### Cross-Region Module\n")
    lines.append(f"- **Region Alignment (precios):** {alignment:.2f}  \n")
    lines.append(f"- **Coherence US-Global (señales):** {coherence:.2f}  \n")
    if coherence < -0.5:
        lines.append("⚠️ Divergencia US-Global.\n")
    if dominance:
        lines.append("⚠️ **SPY Dominance Flag:** SPY domina el flujo (|flow_SPY| > 2× mediana global).\n")
    lines.append(f"- **DM USD Trend:** {usd}  \n\n")

    if issues:
        lines.append("### Data Issues\n")
        for t, msg in issues.items():
            lines.append(f"- `{t}`: {msg}  \n")

    return lines