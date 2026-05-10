"""
global_radar.py – Radar Global v3.19 (contexto semanal, no contaminante).
Calcula PWM, breadths, regímenes, correlation stress, USD trend,
risk driver, coherence US-Global y genera la sección del reporte.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from features import robust_zscore

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------
WEEKLY_ANCHOR_DAY = 3  # Thursday (0=Mon, 3=Thu)
MIN_RISK_ASSETS_FRACTION = 0.70

# Calidad base por activo (flujo proxy)
FLOW_QUALITY = {
    'SPY': 1.0, 'EZU': 0.8, 'EWJ': 0.75, 'EEM': 0.7,
    'VGK': 0.8, 'IWM': 0.85, 'FXI': 0.6,
    'TLT': 0.9, 'HYG': 0.6, 'LQD': 0.7,
    'GLD': 0.6, 'DBC': 0.6, 'XOP': 0.5,
    'UUP': 0.7
}

# ------------------------------------------------------------
# VALIDACIÓN DE DATOS
# ------------------------------------------------------------
def validate_global_data(df_global):
    """Retorna DataFrame filtrado y diccionario de issues."""
    issues = {}
    valid_cols = []

    for col in df_global.columns:
        if not col.endswith('_close'):
            continue
        ticker = col.replace('_close', '')
        close = df_global[col]
        vol_col = f"{ticker}_volume"
        volume = df_global[vol_col] if vol_col in df_global.columns else pd.Series(dtype=float)

        # Chequeos básicos
        if close.isna().sum() / len(close) > 0.1:
            issues[ticker] = 'DATA ISSUE (NaNs >10%)'
            continue
        if (close <= 0).any():
            issues[ticker] = 'DATA ISSUE (close <=0)'
            continue
        if vol_col in df_global.columns and (volume < 0).any():
            issues[ticker] = 'DATA ISSUE (negative volume)'
            continue
        # Retornos semanales extremos
        weekly_ret = close.pct_change(5).dropna()
        if (abs(weekly_ret) > 0.25).any():
            issues[ticker] = 'DATA ISSUE (extreme weekly return)'
            continue

        valid_cols.append(ticker)

    return valid_cols, issues

# ------------------------------------------------------------
# MÉTRICAS PRINCIPALES
# ------------------------------------------------------------
def compute_weekly_data(df_global, valid_tickers):
    """
    Calcula retornos semanales alineados al jueves (Thursday close).
    Retorna DataFrame de retornos y de volúmenes, reindexados al jueves.
    """
    closes = pd.DataFrame()
    volumes = pd.DataFrame()

    for t in valid_tickers:
        close_col = f"{t}_close"
        if close_col in df_global.columns:
            closes[t] = df_global[close_col]
        vol_col = f"{t}_volume"
        if vol_col in df_global.columns:
            volumes[t] = df_global[vol_col]

    # Filtrar solo jueves (weekday=3) – si no hay jueves, usar el último día de la semana
    thursdays = closes.index[closes.index.weekday == WEEKLY_ANCHOR_DAY]
    if len(thursdays) < 5:
        # fallback: último día de cada semana
        closes_weekly = closes.resample('W').last()
        volumes_weekly = volumes.resample('W').last()
    else:
        closes_weekly = closes.loc[thursdays]
        volumes_weekly = volumes.loc[thursdays]

    # Retornos semanales (jueves a jueves)
    weekly_returns = closes_weekly.pct_change().dropna()
    return weekly_returns, volumes_weekly, closes_weekly

def compute_pwm(weekly_returns, dollar_vol, window=60):
    """
    Participation-Weighted Momentum (fórmula final v3.19).
    """
    pwm_df = pd.DataFrame(index=weekly_returns.index)
    for t in weekly_returns.columns:
        ret = weekly_returns[t]
        dv = dollar_vol[t].reindex(ret.index)

        # Volatilidad 20 semanas
        vol_20 = ret.rolling(20).std()
        vol_floor = vol_20.rolling(252, min_periods=20).quantile(0.1)
        # Floor absoluto + relativo para evitar explosiones
        vol_floor_abs = 0.01
        adj_vol = vol_20.combine(vol_floor, max).clip(lower=vol_floor_abs)

        # Participation z-score
        participation = robust_zscore(dv, window=window)

        # Direction con floor de volatilidad y winsorización
        ret_clipped = ret.clip(lower=ret.quantile(0.01), upper=ret.quantile(0.99))
        direction = ret_clipped / adj_vol

        # Combinación convexa (0.6 participation + 0.4 direction)
        pwm_raw = 0.6 * participation + 0.4 * direction

        # Filtro de shocks mecánicos (volumen extremo + retorno pequeño)
        vol_pct = dv.rolling(252, min_periods=20).rank(pct=True)
        mask_shock = (vol_pct > 0.99) & (abs(ret) < 0.5 * vol_20)
        pwm_raw[mask_shock] = pwm_raw[mask_shock] * 0.5

        # Normalización final
        pwm_z = robust_zscore(pwm_raw, window=window)
        pwm_mom = pwm_z.ewm(halflife=2, min_periods=10).mean()
        pwm_df[t] = pwm_mom

    return pwm_df

def compute_breadths(pwm_latest, valid_tickers, geo_equities, growth_risk, inflation, defensive):
    """Calcula los distintos breadths."""
    geo = [t for t in geo_equities if t in valid_tickers]
    gr = [t for t in growth_risk if t in valid_tickers]
    inf = [t for t in inflation if t in valid_tickers]
    defs = [t for t in defensive if t in valid_tickers]

    geo_breadth = sum(pwm_latest.get(t, 0) > 0 for t in geo) / max(len(geo), 1)
    growth_breadth = sum(pwm_latest.get(t, 0) > 0 for t in gr) / max(len(gr), 1)
    infl_breadth = sum(pwm_latest.get(t, 0) > 0 for t in inf) / max(len(inf), 1)
    def_breadth = sum(pwm_latest.get(t, 0) > 0 for t in defs) / max(len(defs), 1)

    return {
        'geo': geo_breadth,
        'growth_risk': growth_breadth,
        'inflation': infl_breadth,
        'defensive': def_breadth
    }

def correlation_stress(weekly_returns, risk_assets, window_ewma=60, lookback_years=3):
    """
    Calcula el percentil de la correlación media entre risk assets usando EWMA.
    Alerta si corr_pct > 0.9.
    """
    risk_cols = [t for t in risk_assets if t in weekly_returns.columns]
    if len(risk_cols) < 4:
        return False, 0.0

    # Correlación media EWMA
    ret_risk = weekly_returns[risk_cols].dropna()
    corr_ewma = ret_risk.ewm(span=window_ewma).corr().droplevel(0)
    mean_corr = corr_ewma.groupby(level=0).apply(lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean())
    if len(mean_corr) >= lookback_years*52:
        # Z-score basado en MAD (robusto a cambios de régimen)
        rolling_median = mean_corr.rolling(lookback_years*52, min_periods=52).median()
        rolling_mad = (mean_corr - rolling_median).abs().rolling(lookback_years*52, min_periods=52).median()
        z_corr = (mean_corr.iloc[-1] - rolling_median.iloc[-1]) / (1.4826 * rolling_mad.iloc[-1] + 1e-9)
        stress_flag = z_corr > 2.0
        return stress_flag, z_corr if pd.notna(z_corr) else 0.0
    return False, 0.0

def usd_regime(uup_weekly_close, window=10):
    """Régimen USD basado en tendencia de UUP en 10 semanas."""
    if uup_weekly_close is None or len(uup_weekly_close) < window:
        return 'NEUTRAL'
    trend = uup_weekly_close.iloc[-1] / uup_weekly_close.iloc[-window] - 1
    if trend > 0.02:
        return 'STRONG'
    elif trend < -0.02:
        return 'WEAK'
    return 'NEUTRAL'

def risk_driver(pwm_latest, previous_drivers=None):
    """
    Dominant Cross-Asset Pattern con confianza basada en persistencia.
    previous_drivers: lista de los últimos 4 drivers (opcional).
    Retorna (driver, confidence_level).
    """
    spy = pwm_latest.get('SPY', 0)
    tlt = pwm_latest.get('TLT', 0)
    dbc = pwm_latest.get('DBC', 0)
    hyg = pwm_latest.get('HYG', 0)
    gld = pwm_latest.get('GLD', 0)

    if spy > 0 and tlt > 0:
        driver = 'LIQUIDITY'
    elif dbc > 0 and hyg > 0:
        driver = 'REFLATION'
    elif gld > 0 and tlt > 0:
        driver = 'DEFENSIVE'
    else:
        driver = 'MIXED'

    # Confianza basada en persistencia (últimas 4 semanas)
    if previous_drivers is None:
        confidence = 'MEDIUM'
    else:
        same_count = sum(1 for d in previous_drivers[-4:] if d == driver)
        if same_count >= 4:
            confidence = 'HIGH'
        elif same_count >= 2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

    return driver, confidence

def compute_global_risk_score(pwm_spy, pwm_ezu, pwm_ewj, pwm_eem, hyg_pwm):
    """Global Risk Score v3.19.1 – todas las componentes normalizadas a [0,1]."""
    # Breadth geográfico continuo con sigmoid (tanh+1)/2
    sigmoid = lambda x: (np.tanh(x) + 1) / 2
    geo_continuous = np.mean([
        sigmoid(pwm_spy), sigmoid(pwm_ezu), sigmoid(pwm_ewj), sigmoid(pwm_eem)
    ])
    
    # PWM alignment normalizado
    pwm_alignment = np.median([pwm_spy, pwm_ezu, pwm_ewj, pwm_eem])
    pwm_align_norm = sigmoid(pwm_alignment)
    
    # HYG confirmación binaria (1 si coincide signo con SPY)
    hyg_confirm = 1.0 if np.sign(hyg_pwm) == np.sign(pwm_spy) else 0.0
    
    score = 0.50 * geo_continuous + 0.45 * pwm_align_norm + 0.05 * hyg_confirm
    return np.clip(score, 0, 1)

def coherence_score(pwm_spy, pwm_global_rv):
    """Coherence continuo US-Global."""
    return np.tanh(pwm_spy) * np.tanh(pwm_global_rv)

def global_regime_label(score, previous_label, persistence_counter):
    """Etiqueta cualitativa con histéresis asimétrica."""
    new_label = None
    if score > 0.60:
        new_label = 'RISK-ON'
    elif score > 0.50:
        new_label = 'WEAK RISK-ON'
    elif score > 0.40:
        new_label = 'NEUTRAL'
    elif score > 0.30:
        new_label = 'WEAK RISK-OFF'
    else:
        new_label = 'RISK-OFF'

    # Histéresis
    if previous_label is None:
        return new_label, 1 if new_label else 0

    if new_label == previous_label:
        return new_label, persistence_counter + 1
    else:
        # Upgrade requiere 3 semanas, downgrade 1 semana
        is_upgrade = (new_label in ['RISK-ON', 'WEAK RISK-ON'] and
                      previous_label in ['NEUTRAL', 'WEAK RISK-OFF', 'RISK-OFF'])
        if is_upgrade:
            if persistence_counter >= 3:
                return new_label, 1
            else:
                return previous_label, persistence_counter + 1
        else:  # downgrade o cambio lateral: aceptar inmediatamente
            return new_label, 1

# ------------------------------------------------------------
# SECCIÓN DEL REPORTE
# ------------------------------------------------------------
def generate_global_section(df_global, global_state=None):
    """
    Genera las líneas Markdown para la sección Radar Global.
    Se llama desde run.py después del radar US.
    """
    lines = ["\n## Radar Global (v3.19) – Contexto Semanal\n"]

    # Validar datos
    valid_tickers, issues = validate_global_data(df_global)

    if not valid_tickers:
        lines.append("*Radar Global no disponible: datos insuficientes.*\n")
        return lines

    # Definir grupos
    geo_equities = ['SPY', 'EZU', 'EWJ', 'EEM']
    growth_risk = ['SPY', 'EZU', 'EWJ', 'EEM', 'HYG']
    inflation = ['DBC']
    defensive = ['TLT', 'GLD']
    risk_assets = ['SPY', 'EZU', 'EWJ', 'EEM', 'HYG', 'DBC', 'XOP']

    # Datos semanales
    weekly_returns, volumes, closes_weekly = compute_weekly_data(df_global, valid_tickers)

    if weekly_returns.empty:
        lines.append("*Datos semanales insuficientes para calcular métricas.*\n")
        return lines

    # Dollar vol medio
    dollar_vol = pd.DataFrame()
    for t in weekly_returns.columns:
        if t in closes_weekly.columns and t in volumes.columns:
            dollar_vol[t] = closes_weekly[t] * volumes[t].reindex(closes_weekly.index)

    # PWM
    pwm_df = compute_pwm(weekly_returns, dollar_vol)
    pwm_latest = pwm_df.iloc[-1].to_dict()

    # Breadths
    breadths = compute_breadths(pwm_latest, valid_tickers, geo_equities, growth_risk, inflation, defensive)

    # Correlación stress
    corr_flag, corr_pct = correlation_stress(weekly_returns, risk_assets)

    # USD Régimen
    uup_close = closes_weekly['UUP'] if 'UUP' in closes_weekly.columns else None
    usd = usd_regime(uup_close)

    # Risk Driver
    driver, driver_conf = risk_driver(pwm_latest)
    lines.append(f"**Dominant Pattern:** {driver} (confidence: {driver_conf})  \n")

    # Global Risk Score
    geo_pwm_vals = [pwm_latest.get(t, 0) for t in geo_equities if t in pwm_latest]
    pwm_alignment = np.median(geo_pwm_vals) if geo_pwm_vals else 0
    hyg_pwm = pwm_latest.get('HYG', 0)
    score = compute_global_risk_score(
        pwm_latest.get('SPY', 0),
        pwm_latest.get('EZU', 0),
        pwm_latest.get('EWJ', 0),
        pwm_latest.get('EEM', 0),
        hyg_pwm
    )

    # Cargar estado previo (simplificado: sin estado, se asigna directamente)
    # En producción se podría guardar en un archivo JSON de estado semanal
    label, _ = global_regime_label(score, None, 0)

    # Coherence
    pwm_spy = pwm_latest.get('SPY', 0)
    global_rv = np.median([pwm_latest.get(t, 0) for t in ['EZU', 'EWJ', 'EEM'] if t in pwm_latest])
    coherence = coherence_score(pwm_spy, global_rv)

    # Reporte
    lines.append(f"**Global Risk Score:** {score:.2f}  \n")
    lines.append(f"**Régimen:** {label}  \n")
    lines.append(f"**DM USD Trend:** {usd}  \n")
    lines.append(f"**Coherence US-Global:** {coherence:.2f}  \n")
    if coherence < -0.5:
        lines.append("⚠️ Divergencia US-Global: reducir convicción en señales domésticas.\n")
    # Breadth geográfico continuo (el mismo que usa el Global Risk Score)
    sigmoid = lambda x: (np.tanh(x) + 1) / 2
    geo_cont = np.mean([sigmoid(pwm_latest.get(t, 0)) for t in geo_equities])
    lines.append(f"**Breadth Geográfico (continuo):** {geo_cont:.2f}  \n")
    lines.append(f"**Breadth Growth Risk:** {breadths['growth_risk']:.0%}  \n")
    lines.append(f"**Breadth Inflation/Reflation:** {breadths['inflation']:.0%}  \n")
    lines.append(f"**Breadth Defensive:** {breadths['defensive']:.0%}  \n")
    if corr_flag:
        lines.append(f"⚠️ **Correlation Stress entre risk assets (pct {corr_pct:.2f}).**\n")
    if issues:
        for t, msg in issues.items():
            lines.append(f"- `{t}`: {msg}  \n")
    if len(valid_tickers) < len(geo_equities) * MIN_RISK_ASSETS_FRACTION:
        lines.append("⚠️ Baja cobertura de activos de RV. Régimen con confianza reducida.\n")

    return lines