import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import download_market_data
from rotation_radar import run_radar, run_flow_radar
from utils import save_flow_history, plot_flow_dispersion, save_markdown_report, interpret_flow_intensity
from features import compute_volume_zscore, compute_flow_acceleration, compute_price_zscore, compute_acceleration_zscore, compute_features

def supervision(*args, **kwargs):
    print("\nSUPERVISION: Función no disponible (placeholder)")

# CFTC opcional
try:
    from cftc_loader import load_cftc_manual, parse_cftc_financials, compute_cftc_signal
    CFTC_AVAILABLE = True
except ImportError:
    CFTC_AVAILABLE = False

# -------------------------------
# Funciones de distribucion v3.15 (añade persistencia informativa)
# -------------------------------
def distribution_score_v33(price_mom, flow_mom, flow_acc, vol_z):
    p = np.tanh(price_mom)
    f = np.tanh(flow_mom)
    a = np.tanh(flow_acc)
    v = np.tanh(vol_z)

    div = p * (-f)
    if abs(div) < 0.02:
        div = 0.0

    flow_residual = (-f) - div
    if abs(f) < 0.2:
        flow_residual *= 0.5

    if a < 0:
        acc_component = -a
    else:
        acc_component = -1.5 * a

    volume_boost = (0.8 * v) if flow_mom < 0 else 0.0

    x = 2.0 * div + 0.8 * flow_residual + acc_component + volume_boost
    return x

def distribution_prob_continuous(price_mom, flow_mom, flow_acc, vol_z, temperature=1.5):
    x = distribution_score_v33(price_mom, flow_mom, flow_acc, vol_z)
    prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def classify_distribution(risk_score):
    if risk_score > 1.2:
        return "FUERTE"
    elif risk_score > 0.6:
        return "MODERADA"
    elif risk_score > 0.3:
        return "DEBIL"
    else:
        return "NINGUNA"

def compute_confidence(div_cont, vol_z, accel, risk_score):
    raw = abs(div_cont) * (0.5 * abs(vol_z) + 0.5 * abs(accel))
    if risk_score > 1.2:
        return "ALTA"
    elif raw > 0.6:
        return "ALTA"
    elif raw > 0.3:
        return "MEDIA"
    else:
        return "BAJA"

def unified_score(prob_bin, prob_cont, conviction, vix_z):
    conv_norm = (np.tanh(conviction) + 1) / 2
    score = 0.4 * prob_bin + 0.3 * prob_cont + 0.3 * conv_norm
    vol_factor = 1 / (1 + max(0, vix_z - 0.5))
    score = score * vol_factor
    return score

# Score de oportunidad (bonifica acumulación)
def opportunity_score(price_mom, flow_mom, flow_acc, vol_z, phase):
    base = 0.4 * flow_mom + 0.3 * price_mom + 0.2 * flow_acc + 0.1 * vol_z
    if phase in ["ACUMULACION", "ACUMULACION FUERTE"]:
        base += 0.2
    elif phase in ["DISTRIBUCION CONFIRMADA", "DISTRIBUCION TEMPRANA"]:
        base -= 0.2
    return base

# Operabilidad escalonada (sin cambios)
def operability_level(score):
    if score > 0.4:
        return "OPORTUNIDAD CLARA"
    elif score > 0.2:
        return "OPORTUNIDAD MODERADA"
    elif score > 0.05:
        return "SEGUIR CON PRUDENCIA"
    else:
        return "NO OPERAR"

# Clasificación de fase (existente, con z-scores)
def classify_phase(price_z, flow_z, acc_z, vol_z):
    if price_z > 1.0 and flow_z < -0.5:
        return "DISTRIBUCION CONFIRMADA"
    if price_z > 0.5 and flow_z < 0:
        return "DISTRIBUCION TEMPRANA"
    if price_z < -0.5 and flow_z > 0.5:
        return "ACUMULACION FUERTE"
    if price_z < 0 and flow_z > 0:
        return "ACUMULACION"
    if flow_z > 0 and acc_z > 0:
        return "CONFIRMACION ALCISTA"
    if flow_z < 0 and acc_z < 0:
        return "CONFIRMACION BAJISTA"
    if abs(price_z) < 0.2 and flow_z < -0.5:
        return "AGOTAMIENTO"
    return "NEUTRAL"

def classify_direction(flow_z):
    if flow_z > 0.2:
        return "ALCISTA"
    elif flow_z < -0.2:
        return "BAJISTA"
    else:
        return "TRANSICION"

def macro_adjustment(conviction, cftc_spy_z):
    if cftc_spy_z is None:
        return conviction
    return conviction * (1 + 0.3 * cftc_spy_z)

def volatility_adjustment(conviction, vix_z):
    factor = 1 / (1 + max(0, vix_z - 0.5))
    return conviction * factor

def divergence_score(price_mom, flow_mom):
    p = np.tanh(price_mom)
    f = np.tanh(flow_mom)
    return p * (-f)

def distribution_score_binary(price_mom, flow_mom, flow_acc, vol_z, weights=None):
    if weights is None:
        weights = {'divergence': 0.4, 'flow_neg': 0.2, 'flow_acc': 0.3, 'volume': 0.1}
    score = 0.0
    if price_mom > 0 and flow_mom < 0:
        score += weights['divergence']
    if flow_mom < -0.1:
        score += weights['flow_neg']
    if flow_acc < 0:
        score += weights['flow_acc']
    if vol_z > 1:
        score += weights['volume']
    return score

def prob_distribution_binary(score, k=5):
    return 1 / (1 + np.exp(-k * (score - 0.5)))

def system_confidence(vix_z, breadth, flow_dispersion):
    conf = (0.4 * max(0, breadth) + 
            0.3 * (1 - min(vix_z, 2) / 2) + 
            0.3 * (1 - flow_dispersion))
    return max(0, min(1, conf))

def signal_quality(price_mom, flow_mom, vol_z):
    intensity = abs(price_mom) + abs(flow_mom)
    penalty = abs(vol_z) * 0.1
    return max(0, intensity - penalty)

def adjust_score_by_confidence(score, confidence):
    return score * (0.5 + 0.5 * confidence)

# ---------- NUEVAS FUNCIONES DE PERSISTENCIA (solo informativa) ----------
def calculate_persistence(flow_mom_series, window=3):
    """
    Calcula persistencia del flujo en los últimos N días.
    Retorna: "PERSISTENCIA_ALCISTA", "PERSISTENCIA_BAJISTA", "TRANSICION" o "SIN_DATOS"
    """
    if len(flow_mom_series) < window:
        return "SIN_DATOS"
    recent = flow_mom_series.tail(window)
    positive = (recent > 0).sum()
    negative = (recent < 0).sum()
    if positive >= 2:
        return "PERSISTENCIA_ALCISTA"
    elif negative >= 2:
        return "PERSISTENCIA_BAJISTA"
    else:
        return "TRANSICION"

def signal_state(persistence, phase):
    """
    Deriva un estado de señal legible (opcional).
    """
    if persistence == "TRANSICION" and phase in ["ACUMULACION", "DISTRIBUCION_TEMPRANA"]:
        return "SEÑAL TEMPRANA"
    elif persistence in ["PERSISTENCIA_ALCISTA", "PERSISTENCIA_BAJISTA"]:
        return "SEÑAL CONSOLIDADA"
    else:
        return "NEUTRAL"

# ---------- Funciones de persistencia histórica (cargar/guardar) ----------
def load_flow_history(file="flow_history.csv"):
    if os.path.exists(file):
        return pd.read_csv(file, index_col=0)
    else:
        return pd.DataFrame()

def save_flow_history_df(df, file="flow_history.csv"):
    df.to_csv(file)

# ------------------------------------------------------------

def compute_macro_context(df, features):
    """Calcula contexto macro (no intrusivo)"""
    spy_ret = df['SPY'].pct_change(20).iloc[-1]
    tlt_ret = df['TLT'].pct_change(20).iloc[-1]
    vix_z = features['vix_z'].iloc[-1]
    regime = 0.5 * spy_ret - 0.3 * tlt_ret - 0.2 * vix_z

    # Growth vs Value (QQQ/SPY) - usar df['QQQ'] ya descargado
    ratio_gv = df['QQQ'] / df['SPY']
    growth_ratio = ratio_gv.iloc[-1] / ratio_gv.rolling(20).mean().iloc[-1] - 1

    # Global vs US (ACWI/SPY)
    ratio_global = df['ACWI'] / df['SPY']
    global_ratio = ratio_global.iloc[-1] / ratio_global.rolling(20).mean().iloc[-1] - 1

    return {
        'regime': regime,
        'growth_vs_value': growth_ratio,
        'global_strength': global_ratio
    }

def interpret_macro(context):
    regime = context['regime']
    if regime > 0.05:
        regime_label = "RISK-ON"
    elif regime < -0.05:
        regime_label = "RISK-OFF"
    else:
        regime_label = "NEUTRAL"
    return regime_label

def adjust_operability(base_level, vix_z):
    if vix_z > 1.5:
        downgrade = {
            "OPORTUNIDAD CLARA": "OPORTUNIDAD MODERADA",
            "OPORTUNIDAD MODERADA": "SEGUIR CON PRUDENCIA",
            "SEGUIR CON PRUDENCIA": "RIESGO ALTO",
            "NO OPERAR": "NO OPERAR"
        }
        return downgrade.get(base_level, base_level)
    return base_level

def compute_persistence(history_df, sector, window=3):
    """Calcula persistencia de flujo para un sector usando histórico"""
    if sector not in history_df.columns:
        return "SIN_DATOS"
    recent = history_df[sector].dropna().tail(window)
    if len(recent) < window:
        return "SIN_DATOS"
    positives = (recent > 0).sum()
    negatives = (recent < 0).sum()
    if positives >= 2:
        return "ESTABLE ALCISTA"
    elif negatives >= 2:
        return "ESTABLE BAJISTA"
    else:
        return "TRANSICION"


    print("\n" + "=" * 60)
    print("SUPERVISION DEL MODELO (calidad y estabilidad)")
    print("=" * 60)
    flow_recent = flow_mom.iloc[-lookback:] if len(flow_mom) > lookback else flow_mom
    if len(flow_recent) == 0:
        print("No hay datos suficientes para supervision.")
        return
    if flow_recent.isnull().any().any():
        print("ADVERTENCIA: Existen valores NaN en los ultimos datos.")
    else:
        print("No hay NaN en los ultimos datos.")
    latest_flow = flow_mom.iloc[-1]
    print(f"Rango de flujos (ultimo dia): min={latest_flow.min():.2f}, max={latest_flow.max():.2f}")
    if abs(latest_flow.max()) > 5 or abs(latest_flow.min()) > 5:
        print("ADVERTENCIA: Flujos extremos (>|5|).")
    disp_recent = flow_mom.std(axis=1).iloc[-5:]
    print(f"Dispersion ultimos 5 dias: {disp_recent.values.round(3)}")
    if len(disp_recent) > 1 and disp_recent.std() > 0.2:
        print("ADVERTENCIA: Dispersion muy variable.")
    top_sectors_last5 = flow_mom.iloc[-5:].idxmax(axis=1)
    unique_top = top_sectors_last5.nunique()
    if unique_top == 1:
        print(f"Liderazgo estable: {top_sectors_last5.iloc[-1]} lider ultimos 5 dias.")
    else:
        print(f"Rotacion de liderazgo: {unique_top} sectores distintos.")
    print(f"Dias totales en flow_momentum: {len(flow_mom)}")
    top_price_3 = [sec for sec, _ in ranking_price[:3]]
    bottom_flow_3 = list(ranking_flow.keys())[-3:]
    divergence_warning = set(top_price_3) & set(bottom_flow_3)
    if divergence_warning:
        print(f"ALERTA EXTREMA: {divergence_warning} top3 precio pero bottom3 flujo -> distribucion.")
    
    try:
        from scipy.stats import spearmanr
        future_returns = df['SPY'].pct_change().shift(-5)
        signal_series = pd.Series(index=flow_mom.index, dtype=float)
        for sec in sectors:
            signal_series = signal_series.fillna(0) + distribution_prob_cont.get(sec, 0)
        common = signal_series.dropna().index.intersection(future_returns.dropna().index)
        if len(common) > 60:
            ic_values = []
            for i in range(60, len(common)):
                idx = common[i-60:i]
                s = signal_series.loc[idx].values
                r = future_returns.loc[idx].values
                if np.std(s) > 1e-9 and np.std(r) > 1e-9:
                    ic = spearmanr(s, r)[0]
                    ic_values.append(ic)
                else:
                    ic_values.append(np.nan)
            if ic_values:
                latest_ic = ic_values[-1]
                if not np.isnan(latest_ic):
                    print(f"IC rolling (60d) de la senal: {latest_ic:.3f}")
                else:
                    print("IC rolling: varianza cero en la ventana, no se pudo calcular")
            else:
                print("IC rolling: no se generaron valori")
        else:
            print("IC rolling: se necesitan al menos 60 observaciones")
    except Exception as e:
        print(f"IC rolling no disponible: {e}")

def enrich_with_cftc_sector(report_lines, cftc_raw):
    if cftc_raw is None:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("No se encontro archivo CFTC. Para activar, descargue manualmente desde\n")
        report_lines.append("https://www.cftc.gov/dea/newcot/FinFutWk.txt y guardelo como data/cftc_raw.txt\n")
        return
    parsed = parse_cftc_financials(cftc_raw)
    if parsed is None or parsed.empty:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("No se pudieron parsear los datos CFTC.\n")
        return
    with_signal = compute_cftc_signal(parsed)
    if with_signal.empty:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("Error al calcular senal CFTC.\n")
        return
    
    latest_date = with_signal['date'].max()
    latest_data = with_signal[with_signal['date'] == latest_date].copy()
    
    sector_mapping = {
        "E-MINI S&P 500": "SPY",
        "NASDAQ-100": "XLK",
        "RUSSELL": "XLI",
        "UST 10Y": "XLF",
        "FED FUNDS": "XLF",
        "SOFR": "XLF",
        "CRUDE OIL": "XLE",
        "BBG COMMODITY": "XLE",
        "USD INDEX": "MACRO",
        "EURO FX": "MACRO",
        "JAPANESE YEN": "MACRO",
        "VIX": "RISK"
    }
    
    sector_signals = {}
    for _, row in latest_data.iterrows():
        market = row['market']
        z = row['cftc_z']
        if pd.isna(z):
            continue
        for key, sector in sector_mapping.items():
            if key in market:
                if sector not in sector_signals:
                    sector_signals[sector] = []
                sector_signals[sector].append(z)
                break
    if not sector_signals:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("No se encontraron mercados relevantes en los datos.\n")
        return
    
    report_lines.append("\n## Confirmacion CFTC (conviccion institucional por sector)\n")
    report_lines.append(f"*Datos semanales al {latest_date.strftime('%Y-%m-%d')} (retraso de 3 dias)*\n\n")
    report_lines.append("| Sector | Z-score (mediana) | Instrumentos | Interpretacion |\n")
    report_lines.append("|--------|-------------------|--------------|----------------|\n")
    for sector, z_list in sorted(sector_signals.items(), key=lambda x: np.median(x[1]), reverse=True):
        median_z = np.median(z_list)
        n = len(z_list)
        if median_z > 1.5:
            interp = "Conviccion FUERTEMENTE alcista"
        elif median_z > 0.8:
            interp = "Conviccion alcista moderada"
        elif median_z < -1.5:
            interp = "Conviccion FUERTEMENTE bajista"
        elif median_z < -0.8:
            interp = "Conviccion bajista moderada"
        else:
            interp = "Neutral"
        report_lines.append(f"| {sector} | {median_z:.2f} | {n} | {interp} |\n")
    report_lines.append("\n*Nota: CFTC semanal con retraso (martes a viernes). Solo para contexto macro.*\n")

def main():
    print("=== RADAR DE ROTACION SECTORIAL v3.15 (con persistencia informativa) ===\n")
    df = download_market_data()
    features = compute_features(df)
    
    ranking_price, dispersion_price, breadth_price, vix_z, stress, regime_price, accion_price = run_radar(df)
    ranking_flow, flow_dispersion, flow_breadth, regime_flow, flow_mom = run_flow_radar(df)
    
    alertas = []
    top_price_2 = [sec for sec, _ in ranking_price[:2]]
    bottom_flow_2 = list(ranking_flow.keys())[-2:]
    for sec in top_price_2:
        if sec in bottom_flow_2:
            alertas.append(f"ALERTA ROJA: {sec} esta entre los 2 mejores en precio pero entre los 2 peores en flujo -> senal de distribucion (posible techo).")
    bottom_price_2 = [sec for sec, _ in ranking_price[-2:]]
    top_flow_2 = list(ranking_flow.keys())[:2]
    for sec in bottom_price_2:
        if sec in top_flow_2:
            alertas.append(f"ALERTA VERDE: {sec} esta entre los 2 peores en precio pero entre los 2 mejores en flujo -> posible acumulacion (oportunidad).")
    
    # ========== Confirmacion de distribucion v3.15 ==========
    sectors = list(ranking_flow.keys())
    flow_acc_df = compute_flow_acceleration(flow_mom, window=5)
    vol_z_df = compute_volume_zscore(df, sectors, window=20)
    price_mom_dict = {sec: mom for sec, mom in ranking_price}
    
    price_z_df = compute_price_zscore(df, sectors, window=60)
    acc_z_df = compute_acceleration_zscore(flow_acc_df, window=20)
    latest_price_z = price_z_df.iloc[-1]
    latest_acc_z = acc_z_df.iloc[-1]
    
    latest_flow_mom = flow_mom.iloc[-1]
    latest_flow_acc = flow_acc_df.iloc[-1]
    latest_vol_z = vol_z_df.iloc[-1]
    
    distribution_scores = {}
    distribution_probs_bin = {}
    distribution_div_cont = {}
    distribution_prob_cont = {}
    risk_score_dict = {}
    
    # Cargar histórico de flujos para persistencia (usamos flow_mom diario)
    flow_history = load_flow_history("flow_history_persistencia.csv")
    # Guardaremos los valores de flow_mom (no el score) para cada sector
    for sec in sectors:
        if sec not in flow_history.columns:
            flow_history[sec] = np.nan
    # Actualizar con el valor más reciente
    for sec in sectors:
        flow_history.loc[pd.Timestamp.now().normalize(), sec] = latest_flow_mom.get(sec, np.nan)
    save_flow_history_df(flow_history, "flow_history_persistencia.csv")
    
    for sec in sectors:
        pm = price_mom_dict.get(sec, 0)
        fm = latest_flow_mom.get(sec, 0)
        fa = latest_flow_acc.get(sec, 0)
        vz = latest_vol_z.get(sec, 0)
        
        score_bin = distribution_score_binary(pm, fm, fa, vz)
        distribution_scores[sec] = score_bin
        distribution_probs_bin[sec] = prob_distribution_binary(score_bin)
        distribution_div_cont[sec] = divergence_score(pm, fm)
        
        risk_score = distribution_score_v33(pm, fm, fa, vz)
        prob_cont = distribution_prob_continuous(pm, fm, fa, vz, temperature=1.5)
        
        distribution_prob_cont[sec] = prob_cont
        risk_score_dict[sec] = risk_score
    
    # Calcular fase y dirección
    fase_dict = {}
    direccion_dict = {}
    for sec in sectors:
        pm = price_mom_dict.get(sec, 0)
        fm = latest_flow_mom.get(sec, 0)
        fa = latest_flow_acc.get(sec, 0)
        vz = latest_vol_z.get(sec, 0)
        fase_dict[sec] = classify_phase(latest_price_z.get(sec, 0), fm, latest_acc_z.get(sec, 0), vz)
        direccion_dict[sec] = classify_direction(fm)
    
    # Calcular oportunidad y operabilidad
    oportunidad_dict = {}
    operabilidad_dict = {}
    for sec in sectors:
        pm = price_mom_dict.get(sec, 0)
        fm = latest_flow_mom.get(sec, 0)
        fa = latest_flow_acc.get(sec, 0)
        vz = latest_vol_z.get(sec, 0)
        fase = fase_dict[sec]
        oportunidad_dict[sec] = opportunity_score(pm, fm, fa, vz, fase)
        operabilidad_dict[sec] = operability_level(oportunidad_dict[sec])
    
    # Calcular persistencia (usando histórico de flow_mom)
    persistencia_dict = {}
    estado_senal_dict = {}
    for sec in sectors:
        if sec in flow_history.columns:
            flow_series = flow_history[sec].dropna()
            persistencia_dict[sec] = calculate_persistence(flow_series, window=3)
            estado_senal_dict[sec] = signal_state(persistencia_dict[sec], fase_dict[sec])
        else:
            persistencia_dict[sec] = "SIN_DATOS"
            estado_senal_dict[sec] = "NEUTRAL"
    
    # Ordenar sectores por oportunidad (mayor a menor)
    sorted_sectors = sorted(sectors, key=lambda x: oportunidad_dict.get(x, 0), reverse=True)
    
    # Obtener CFTC del SPY si está disponible
    cftc_spy_z = None
    cftc_raw = None
    if CFTC_AVAILABLE:
        cftc_raw = load_cftc_manual(path="data/cftc_raw.txt")
        if cftc_raw is not None:
            try:
                parsed = parse_cftc_financials(cftc_raw)
                if parsed is not None:
                    with_signal = compute_cftc_signal(parsed)
                    if not with_signal.empty:
                        spy_data = with_signal[with_signal['market'].str.contains("S&P 500", case=False)]
                        if not spy_data.empty:
                            cftc_spy_z = spy_data['cftc_z'].iloc[-1]
            except:
                pass
    
    # Construir tabla de distribucion
    dist_lines = ["\n## Confirmacion de Distribucion (dinero inteligente saliendo)\n"]
    rank_history = pd.DataFrame()
    for i in range(1, 6):
        if i <= len(flow_mom):
            rank = flow_mom.iloc[-i].rank(ascending=False)
            rank_history[f'day_{i}'] = rank
    if not rank_history.empty:
        n_sectors = rank_history.shape[0]
        rank_change = rank_history.diff(axis=1).abs().mean().mean() / n_sectors
        dist_lines.append(f"\n**Volatilidad del ranking (cambio medio en ultimos 5 dias):** {rank_change:.3f}\n")
        if rank_change > 0.15:
            dist_lines.append("*Interpretación: Rotación extrema / cambio de régimen*\n")
        elif rank_change > 0.05:
            dist_lines.append("*Interpretación: Rotación moderada*\n")
        else:
            dist_lines.append("*Interpretación: Mercado estable*\n")
    else:
        dist_lines.append("\n**Volatilidad del ranking:** No hay datos suficientes\n")
    
    dist_lines.append("| Sector | Price Mom | Flow Mom | Aceleracion | Volumen Z | Divergencia | Score (bin) | Prob (bin) | Prob (continua) | Risk Score | Intensidad | Clasificacion | Fase | Direccion | Conviccion Ajustada | Score Unificado | Confianza | Calidad | Oportunidad | Operabilidad | Persistencia | Estado Señal | Alerta |\n")
    dist_lines.append("|--------|-----------|----------|-------------|-----------|-------------|-------------|------------|-----------------|------------|------------|--------------|------|----------|--------------------|-----------------|----------|---------|-------------|--------------|--------------|--------------|--------|\n")
    
    for sec in sorted_sectors:
        pm = price_mom_dict.get(sec, 0)
        fm = latest_flow_mom.get(sec, 0)
        fa = latest_flow_acc.get(sec, 0)
        vz = latest_vol_z.get(sec, 0)
        score_bin = distribution_scores[sec]
        prob_bin = distribution_probs_bin[sec]
        prob_cont = distribution_prob_cont[sec]
        div_cont = distribution_div_cont[sec]
        risk_score = risk_score_dict[sec]
        intensidad = compute_confidence(div_cont, vz, fa, risk_score)
        clasif = classify_distribution(risk_score)
        fase = fase_dict[sec]
        direccion = direccion_dict[sec]
        conv_ajustada = macro_adjustment(risk_score, cftc_spy_z)
        conv_ajustada = volatility_adjustment(conv_ajustada, vix_z)
        score_unif = unified_score(prob_bin, prob_cont, conv_ajustada, vix_z)
        
        conf_sistema = system_confidence(vix_z, breadth_price, flow_dispersion)
        calidad = signal_quality(pm, fm, vz)
        oportunidad = oportunidad_dict[sec]
        operabilidad = operabilidad_dict[sec]
        persistencia = persistencia_dict[sec]
        estado_senal = estado_senal_dict[sec]
        
        if score_bin >= 0.7:
            alerta = "DISTRIBUCION FUERTE"
        elif score_bin >= 0.4:
            alerta = "POSIBLE DISTRIBUCION"
        else:
            alerta = "SIN SENAL"
        
        dist_lines.append(f"| {sec} | {pm:.3f} | {fm:.2f} | {fa:.2f} | {vz:.2f} | {div_cont:.2f} | {score_bin:.2f} | {prob_bin:.2%} | {prob_cont:.2%} | {risk_score:.2f} | {intensidad} | {clasif} | {fase} | {direccion} | {conv_ajustada:.2f} | {score_unif:.2f} | {conf_sistema:.2f} | {calidad:.2f} | {oportunidad:.2f} | {operabilidad} | {persistencia} | {estado_senal} | {alerta} |\n")
    
    # Guardar flujos y gráfico
    save_flow_history(flow_mom)
    plot_flow_dispersion(flow_mom)
    
    # Generar reporte Markdown
    from utils import save_markdown_report
    temp_md = "outputs/reporte_diario_temp.md"
    save_markdown_report(ranking_price, ranking_flow, flow_dispersion, flow_breadth, regime_flow,
                         dispersion_price, breadth_price, vix_z, regime_price, accion_price,
                         alertas, output_path=temp_md)
    with open(temp_md, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## Conclusión"):
            insert_pos = i
            break
    
    # Contexto macro
    macro_context = compute_macro_context(df, features)
    regime_label = interpret_macro(macro_context)
    macro_lines = [
        "\n## Contexto Macro\n",
        f"- **Régimen de mercado:** {regime_label} ({macro_context['regime']:.2f})\n",
        f"- **Growth vs Value (QQQ/SPY):** {macro_context['growth_vs_value']:.2%}\n",
        f"- **Fortaleza global (ACWI/SPY):** {macro_context['global_strength']:.2%}\n"
    ]
    lines[insert_pos:insert_pos] = macro_lines + dist_lines
    cftc_lines = []
    enrich_with_cftc_sector(cftc_lines, cftc_raw)
    for i, line in enumerate(lines):
        if line.startswith("## Conclusión"):
            insert_pos = i
            break
    lines[insert_pos:insert_pos] = cftc_lines
    
    with open("outputs/reporte_diario.md", 'w', encoding='utf-8') as f:
        f.writelines(lines)
    os.remove(temp_md)
    
    print("\nHistorico de flujos guardado en outputs/flow_history.csv")
    print("Grafico de dispersion guardado en outputs/flow_dispersion.png")
    print("Reporte diario (con CFTC y distribucion) guardado en outputs/reporte_diario.md")
    
    # Validacion cuantitativa
    try:
        from validation import evaluate_signal
        future_returns = df['SPY'].pct_change().shift(-5)
        common = future_returns.dropna().index.intersection(flow_mom.index)
        signal_df = pd.DataFrame(index=flow_mom.index)
        for sec in sectors:
            signal_df[sec] = distribution_prob_cont.get(sec, 0)
        signal_series = signal_df.mean(axis=1)
        val = evaluate_signal(signal_series, future_returns, threshold=0.7)
        print("\n=== VALIDACION CUANTITATIVA (probabilidad continua) ===")
        if val['n_signals'] == 0:
            print(val['message'])
        else:
            print(f"Alpha (retorno exceso): {val['alpha']:.4f}")
            print(f"Hit ratio (acierto bajista): {val['hit_ratio']:.2%}")
            print(f"Numero de senales: {val['n_signals']}")
    except Exception as e:
        print(f"Validacion no disponible: {e}")
    
    supervision(flow_mom, ranking_flow, ranking_price, df, sectors, distribution_prob_cont)
    print("\nEjecucion completada.")

if __name__ == '__main__':
    main()






