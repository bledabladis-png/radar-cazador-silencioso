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
            # no se excluye, solo se marca
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
    # Filtrar solo viernes (weekday=4). Si un viernes falta, se excluye esa semana.
    fridays = closes.index[closes.index.weekday == 4]
    closes_weekly = closes.loc[fridays]
    volumes_weekly = volumes.loc[fridays]
    # Eliminar semanas sin dato en ningún activo
    closes_weekly = closes_weekly.dropna(how='all')
    volumes_weekly = volumes_weekly.dropna(how='all')
    
    # Retornos semanales (viernes a viernes)
    weekly_returns = closes_weekly.pct_change(fill_method=None).dropna()
    return weekly_returns, volumes_weekly, closes_weekly

# ------------------------------------------------------------
# FUNCIÓN AUXILIAR PARA EL GLOBAL RISK SCORE
# ------------------------------------------------------------
def compute_global_risk_score(pwm_spy, pwm_ezu, pwm_ewj, pwm_eem, hyg_pwm):
    """Global Risk Score v4.0.1 – todas las componentes normalizadas a [0,1]."""
    sigmoid = lambda x: (np.tanh(x) + 1) / 2

    # Breadth geográfico continuo
    geo_continuous = np.mean([
        sigmoid(pwm_spy), sigmoid(pwm_ezu), sigmoid(pwm_ewj), sigmoid(pwm_eem)
    ])
    
    # PWM alignment: fracción de bloques que coinciden en signo con la mediana
    signs = np.sign([pwm_spy, pwm_ezu, pwm_ewj, pwm_eem])
    median_sign = np.sign(np.median([pwm_spy, pwm_ezu, pwm_ewj, pwm_eem]))
    pwm_align_norm = (signs == median_sign).mean()   # [0,1], independiente de geo_continuous
    
    # HYG confirmación continua (sin discontinuidad en cero)
    hyg_confirm = (np.tanh(hyg_pwm) * np.tanh(pwm_spy) + 1) / 2
    
    score = 0.50 * geo_continuous + 0.45 * pwm_align_norm + 0.05 * hyg_confirm
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

    # Dollar volume
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
    # LOOK-AHEAD MITIGATION: desplazar 1 semana (t-1)
    # -------------------------------------------
    flow_df = flow_df.shift(1).dropna()
    direction_df = direction_df.shift(1).dropna()
    weekly_returns_shifted = weekly_returns.shift(1).dropna()

    if flow_df.empty or direction_df.empty or weekly_returns_shifted.empty:
        lines.append("*Datos semanales insuficientes tras ajuste de look-ahead.*\n")
        return lines

    # -------------------------------------------
    # MÉTRICAS COMPUESTAS (usando datos desplazados)
    # -------------------------------------------
    # Flow metrics
    capital_rot = compute_capital_rotation(
        flow_df, 
        FLOW_ASSETS['equity'], 
        FLOW_ASSETS['fixed_income'], 
        FLOW_ASSETS['commodities']
    )
    participation = flow_participation_ratio(flow_df, FLOW_ASSETS['equity'])

    # Risk metrics
    risk_assets_available = [t for t in RISK_ASSETS['equity'] if t in weekly_returns_shifted.columns]
    vol_regime = compute_volatility_regime(weekly_returns_shifted, risk_assets_available) if risk_assets_available else 1.0
    risk_breadth = compute_risk_breadth(direction_df, RISK_ASSETS['equity'])

    # Cross-Region metrics
    alignment = compute_region_alignment(weekly_returns_shifted, CROSS_REGION)
    
    # Coherence v4
    spy_flow = flow_df['SPY'] if 'SPY' in flow_df.columns else pd.Series()
    global_flow = flow_df[['EZU', 'EWJ', 'EEM']].median(axis=1) if all(t in flow_df.columns for t in ['EZU', 'EWJ', 'EEM']) else pd.Series()
    spy_dir = direction_df['SPY'] if 'SPY' in direction_df.columns else pd.Series()
    global_dir = direction_df[['EZU', 'EWJ', 'EEM']].median(axis=1) if all(t in direction_df.columns for t in ['EZU', 'EWJ', 'EEM']) else pd.Series()
    
    coherence = compute_coherence_v4(spy_flow, global_flow, spy_dir, global_dir)
    
    # Dominance flag
    spy_val = flow_df['SPY'].iloc[-1] if 'SPY' in flow_df.columns else 0
    others_median = np.median([flow_df[t].iloc[-1] for t in ['EZU', 'EWJ', 'EEM'] if t in flow_df.columns])
    dominance = compute_dominance_flag(spy_val, others_median)

    # USD Regime
    uup_close = closes_weekly['UUP'] if 'UUP' in closes_weekly.columns else None
    if uup_close is not None and len(uup_close) >= 10:
        trend = uup_close.iloc[-1] / uup_close.iloc[-10] - 1
        usd = 'STRONG' if trend > 0.02 else 'WEAK' if trend < -0.02 else 'NEUTRAL'
    else:
        usd = 'NEUTRAL'

    # -------------------------------------------
    # STABILITY LAYER & REGIME SHIFT
    # -------------------------------------------
    # Construir series históricas para stability
    capital_rot_series = pd.Series(dtype=float)  # Placeholder
    risk_breadth_series = pd.Series(dtype=float) # Placeholder
    alignment_series = pd.Series(dtype=float)    # Placeholder
    vol_regime_series = pd.Series(dtype=float)   # Placeholder

    # Intentar cargar históricos si existen
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

    # Añadir el valor actual a las series para el cálculo de estabilidad
    capital_rot_series = pd.concat([capital_rot_series, pd.Series([capital_rot])])
    risk_breadth_series = pd.concat([risk_breadth_series, pd.Series([risk_breadth])])
    alignment_series = pd.concat([alignment_series, pd.Series([alignment])])
    vol_regime_series = pd.concat([vol_regime_series, pd.Series([vol_regime])])

    # Guardar el histórico actualizado
    hist_df = pd.DataFrame({
        'capital_rotation': capital_rot_series,
        'risk_breadth': risk_breadth_series,
        'alignment': alignment_series,
        'volatility_regime': vol_regime_series
    })
    hist_df.to_csv('data/global_metrics_history.csv')

    # Calcular estabilidad y regime shift
    rot_stab = signal_stability(capital_rot_series)
    shift_detected = regime_shift_detector(vol_regime_series)
    stab_flags = stability_flags(capital_rot_series, risk_breadth_series, alignment_series)

    # -------------------------------------------
    # REPORTE
    # -------------------------------------------
    if shift_detected:
        lines.append("⚠️ **Regime Shift Detectado** – la volatilidad actual es anormalmente alta.\n")
    for flag in stab_flags:
        lines.append(flag + "  \n")
    if rot_stab is not None and not np.isnan(rot_stab):
        lines.append(f"- **Estabilidad de Capital Rotation:** {rot_stab:.2f}  \n")

    lines.append("### Flow Module\n")
    lines.append(f"- **Capital Rotation:** {capital_rot:.2f}  \n")
    lines.append(f"- **Flow Participation (RV):** {participation:.0%}  \n\n")

    lines.append("### Risk Module\n")
    lines.append(f"- **Risk Breadth:** {risk_breadth:.0%}  \n")
    lines.append(f"- **Volatility Regime (z-score):** {vol_regime:.2f}  \n\n")

    lines.append("### Cross-Region Module\n")
    lines.append(f"- **Region Alignment (corr):** {alignment:.2f}  \n")
    lines.append(f"- **Coherence US-Global:** {coherence:.2f}  \n")
    if coherence < -0.5:
        lines.append("⚠️ Divergencia US-Global.\n")
    if dominance:
        lines.append("⚠️ **Dominancia US:** SPY concentra la señal de flujo.\n")
    lines.append(f"- **DM USD Trend:** {usd}  \n\n")

    if issues:
        lines.append("### Data Issues\n")
        for t, msg in issues.items():
            lines.append(f"- `{t}`: {msg}  \n")

    return lines