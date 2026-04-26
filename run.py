import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import download_market_data
from rotation_radar import run_radar, run_flow_radar
from utils import plot_flow_dispersion, save_markdown_report
from features import compute_volume_zscore, compute_flow_acceleration, compute_price_zscore, compute_acceleration_zscore, compute_features
from macro_confirm import compute_macro_score
from stock_leader import prepare_multi_index, generate_leader_section
from wyckoff_detector import wyckoff_structure_core, wyckoff_score, range_compression
from flow_attribution import FlowAttributionEngine
from oms import compute_oms, classify_oms, oms_modifier
from options_data_loader import get_historical_options_data 
from hkex_pcr import get_hkex_pcr_series
from stock_data_loader import fetch_stock_prices

# ------------------------------------------------------------
# FUNCIONES AUXILIARES (sin cambios)
# ------------------------------------------------------------
def adaptive_window(volatility, base_window=60):
    vol = np.clip(volatility, -2, 2)
    scale = 1 - (vol * 0.25)
    return int(np.clip(base_window * scale, 30, 90))

def detect_regime_transition(macro_series, flow_series, threshold=0.2):
    macro_delta = macro_series.diff(5)
    flow_delta = flow_series.diff(5)
    if macro_delta.iloc[-1] > threshold and flow_delta.iloc[-1] > threshold:
        return 1
    elif macro_delta.iloc[-1] < -threshold and flow_delta.iloc[-1] < -threshold:
        return -1
    return 0

def detect_wyckoff_transition(structure_series, score_series, lookback=5):
    if len(structure_series) < lookback + 1:
        return "SIN DATOS SUFICIENTES"
    fase_actual = structure_series.iloc[-1]
    fase_anterior = structure_series.iloc[-lookback-1]
    score_actual = score_series.iloc[-1]
    score_anterior = score_series.iloc[-lookback-1]
    if pd.isna(fase_actual) or pd.isna(fase_anterior):
        return "DATOS INSUFICIENTES"
    if fase_actual == fase_anterior:
        return f"ESTABLE ({fase_actual})"
    if (fase_anterior == "RANGE" and fase_actual == "MARKUP") or \
       (fase_anterior == "ACCUMULATION" and fase_actual == "MARKUP"):
        intensidad = "FUERTE" if (score_actual - score_anterior) > 0.2 else "DÉBIL"
        return f"TRANSICIÓN ALCISTA: {fase_anterior} → {fase_actual} ({intensidad})"
    if (fase_anterior == "MARKUP" and fase_actual == "DISTRIBUTION") or \
       (fase_anterior == "DISTRIBUTION" and fase_actual == "RANGE"):
        intensidad = "FUERTE" if (score_anterior - score_actual) > 0.2 else "DÉBIL"
        return f"TRANSICIÓN BAJISTA: {fase_anterior} → {fase_actual} ({intensidad})"
    return f"CAMBIO DE FASE: {fase_anterior} → {fase_actual}"

def compute_flow_momentum(flow_series):
    delta1 = flow_series.diff(3)
    delta2 = delta1.diff(3)
    return delta2.ewm(span=5).mean()

def is_tradeable(edge, truth, structure):
    if structure == 0:
        return False
    if truth < 0.5:
        return False
    if abs(edge) < 0.3:
        return False
    return True

def compute_alignment_score(macro, flow, structure):
    signals = np.array([np.sign(macro), np.sign(flow), np.sign(structure)])
    agreement = np.sum(signals == signals[0]) / len(signals)
    return agreement

def rolling_regression_residual(x, y, window=60):
    resid = pd.Series(index=y.index, dtype=float)
    for i in range(window, len(y)):
        xw = x.iloc[i-window:i].values
        yw = y.iloc[i-window:i].values
        mask = np.isfinite(xw) & np.isfinite(yw)
        xw = xw[mask]; yw = yw[mask]
        if len(xw) < 2:
            resid.iloc[i] = np.nan
            continue
        if np.std(xw) < 1e-9:
            resid.iloc[i] = y.iloc[i] - np.mean(yw)
            continue
        try:
            weights = 1 / (1 + np.abs(yw - np.median(yw)))
            X = np.vstack([np.ones(len(xw)), xw]).T
            W = np.diag(weights)
            beta = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ yw, rcond=None)[0]
            y_hat = beta[0] + beta[1] * x.iloc[i]
            resid.iloc[i] = y.iloc[i] - y_hat
        except:
            resid.iloc[i] = np.nan
    return resid

def rolling_robust_zscore(series, window=60):
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    return (series - median) / (1.4826 * mad + 1e-9)

def compute_macro_series(df, features, window=60):
    from macro_confirm import compute_macro_score
    macro_scores = []
    dates = df.index
    breadth_series = features['breadth_signal']
    for i, date in enumerate(dates):
        if i < window:
            macro_scores.append(np.nan)
            continue
        sub_df = df.iloc[:i+1]
        breadth_val = breadth_series.loc[date] if date in breadth_series.index else 0.0
        macro_scores.append(compute_macro_score(sub_df, breadth_signal=breadth_val))
    return pd.Series(macro_scores, index=dates)

def apply_decay(series, halflife=5):
    series_clean = series.dropna()
    if len(series_clean) < halflife:
        return series
    ewm = series_clean.ewm(halflife=halflife, adjust=False).mean()
    return ewm.reindex(series.index).ffill()

# CFTC (opcional)
try:
    from cftc_loader import load_cftc_manual, parse_cftc_financials, compute_cftc_signal, update_cftc_history, compute_cftc_zscore_from_history
    CFTC_AVAILABLE = True
except ImportError:
    CFTC_AVAILABLE = False

def compute_regime_score(df, features):
    spy_ma50 = df['SPY'].rolling(50).mean().iloc[-1]
    spy_ma200 = df['SPY'].rolling(200).mean().iloc[-1]
    trend_raw = (spy_ma50 / spy_ma200 - 1) * 5
    trend = np.tanh(trend_raw)
    trend_norm = (trend + 1) / 2
    breadth_signal = features['breadth_signal'].iloc[-1]
    breadth_norm = np.clip((breadth_signal + 0.5) / 1.5, 0, 1)
    vix_z = features['vix_z'].iloc[-1]
    vix_series = df['^VIX'].dropna()
    if len(vix_series) >= 5:
        recent = vix_series.tail(5)
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        vix_trend = slope
    else:
        vix_trend = 0.0
    vix_component = np.exp(-vix_z) * (1 - np.tanh(vix_trend))
    vix_component = np.clip(vix_component, 0, 1)
    from features import robust_zscore
    credit_ratio = df['HYG'] / df['LQD']
    credit_z = robust_zscore(credit_ratio, window=60).iloc[-1]
    credit_norm = 1 - np.clip(credit_z, 0, 2) / 2
    regime_score = (0.4 * trend_norm + 0.2 * breadth_norm + 0.2 * vix_component + 0.2 * credit_norm)
    regime_score = np.clip(regime_score, 0, 1)
    if regime_score > 0.6:
        label = "EXPANSION"
    elif regime_score < 0.4:
        label = "CONTRACTION"
    else:
        label = "TRANSITION"
    return regime_score, label

def get_wls_weights(regime_score):
    w_flow = 0.20 + 0.20 * regime_score
    w_rs = 0.20 + 0.15 * regime_score
    w_persist = 0.30 - 0.10 * regime_score
    w_struct = 0.30 - 0.15 * regime_score
    w_stability = 0.10
    total = w_flow + w_rs + w_persist + w_struct + w_stability
    return {
        "flow": w_flow / total,
        "rs": w_rs / total,
        "persist": w_persist / total,
        "structure": w_struct / total,
        "stability": w_stability / total
    }

def compute_price_structure_advanced(df):
    return wyckoff_structure_core(df)

def compute_wyckoff_structure_series(df):
    phases = []
    for i in range(len(df)):
        if i < 200:
            phases.append(np.nan)
            continue
        sub_df = df.iloc[:i+1]
        phases.append(wyckoff_structure_core(sub_df))
    return pd.Series(phases, index=df.index)

def compute_wyckoff_score_series(df):
    scores = []
    for i in range(len(df)):
        if i < 60:
            scores.append(np.nan)
            continue
        sub_df = df.iloc[:i+1]
        scores.append(wyckoff_score(sub_df).iloc[-1])
    return pd.Series(scores, index=df.index)

def compute_edge_hierarchical(macro, flow, structure):
    w_flow = 0.5
    w_macro = 0.3
    w_structure = 0.2
    flow_s = np.tanh(flow)
    macro_s = np.tanh(macro)
    struct_s = structure
    weighted = w_flow * flow_s + w_macro * macro_s + w_structure * struct_s
    consistency = 1 - np.std([flow_s, macro_s, struct_s])
    return np.clip(weighted * consistency, -1, 1)

def compute_truth(macro, flow, structure):
    macro_strength = abs(macro)
    flow_strength = abs(flow)
    agreement = 1 if np.sign(macro) == np.sign(flow) else 0
    penalty = 0.5 if structure == 0 else 1.0
    return np.clip((0.5 * macro_strength + 0.5 * flow_strength) * agreement * penalty, 0, 1)

# ------------------------------------------------------------
# FUNCIONES DE DISTRIBUCIÓN (resumidas, se mantienen igual)
# ------------------------------------------------------------
def distribution_score_v33(price_mom, flow_mom, flow_acc, vol_z):
    p = np.tanh(price_mom); f = np.tanh(flow_mom); a = np.tanh(flow_acc); v = np.tanh(vol_z)
    div = p * (-f)
    if abs(div) < 0.02: div = 0.0
    flow_residual = (-f) - div
    if abs(f) < 0.2: flow_residual *= 0.5
    acc_component = -a if a < 0 else -1.5 * a
    volume_boost = (0.8 * v) if flow_mom < 0 else 0.0
    return 2.0 * div + 0.8 * flow_residual + acc_component + volume_boost

def distribution_prob_continuous(price_mom, flow_mom, flow_acc, vol_z, temperature=1.5):
    x = distribution_score_v33(price_mom, flow_mom, flow_acc, vol_z)
    return 1 / (1 + np.exp(-x / temperature))

def classify_distribution(risk_score):
    if risk_score > 1.2: return "FUERTE"
    elif risk_score > 0.6: return "MODERADA"
    elif risk_score > 0.3: return "DEBIL"
    else: return "NINGUNA"

def compute_confidence(div_cont, vol_z, accel, risk_score):
    raw = abs(div_cont) * (0.5 * abs(vol_z) + 0.5 * abs(accel))
    if risk_score > 1.2: return "ALTA"
    elif raw > 0.6: return "ALTA"
    elif raw > 0.3: return "MEDIA"
    else: return "BAJA"

def unified_score(prob_bin, prob_cont, conviction, vix_z):
    conv_norm = (np.tanh(conviction) + 1) / 2
    score = 0.4 * prob_bin + 0.3 * prob_cont + 0.3 * conv_norm
    vol_factor = 1 / (1 + max(0, vix_z - 0.5))
    return score * vol_factor

def opportunity_score(price_mom, flow_mom, flow_acc, vol_z, phase):
    base = 0.4 * flow_mom + 0.3 * price_mom + 0.2 * flow_acc + 0.1 * vol_z
    if phase in ["ACUMULACION", "ACUMULACION FUERTE"]: base += 0.2
    elif phase in ["DISTRIBUCION CONFIRMADA", "DISTRIBUCION TEMPRANA"]: base -= 0.2
    return base

def operability_level(score):
    if score > 0.4: return "OPORTUNIDAD CLARA"
    elif score > 0.2: return "OPORTUNIDAD MODERADA"
    elif score > 0.05: return "SEGUIR CON PRUDENCIA"
    else: return "NO OPERAR"

def classify_phase(price_z, flow_z, acc_z, vol_z):
    if price_z > 1.0 and flow_z < -0.5: return "DISTRIBUCION CONFIRMADA"
    if price_z > 0.5 and flow_z < 0: return "DISTRIBUCION TEMPRANA"
    if price_z < -0.5 and flow_z > 0.5: return "ACUMULACION FUERTE"
    if price_z < 0 and flow_z > 0: return "ACUMULACION"
    if flow_z > 0 and acc_z > 0: return "CONFIRMACION ALCISTA"
    if flow_z < 0 and acc_z < 0: return "CONFIRMACION BAJISTA"
    if abs(price_z) < 0.2 and flow_z < -0.5: return "AGOTAMIENTO"
    return "NEUTRAL"

def classify_direction(flow_z):
    if flow_z > 0.2: return "ALCISTA"
    elif flow_z < -0.2: return "BAJISTA"
    else: return "TRANSICION"

def macro_adjustment(conviction, cftc_spy_z):
    if cftc_spy_z is None: return conviction
    return conviction * (1 + 0.3 * cftc_spy_z)

def volatility_adjustment(conviction, vix_z):
    factor = 1 / (1 + max(0, vix_z - 0.5))
    return conviction * factor

def divergence_score(price_mom, flow_mom):
    return np.tanh(price_mom) * (-np.tanh(flow_mom))

def distribution_score_binary(price_mom, flow_mom, flow_acc, vol_z, weights=None):
    if weights is None:
        weights = {'divergence': 0.4, 'flow_neg': 0.2, 'flow_acc': 0.3, 'volume': 0.1}
    score = 0.0
    if price_mom > 0 and flow_mom < 0: score += weights['divergence']
    if flow_mom < -0.1: score += weights['flow_neg']
    if flow_acc < 0: score += weights['flow_acc']
    if vol_z > 1: score += weights['volume']
    return score

def prob_distribution_binary(score, k=5):
    return 1 / (1 + np.exp(-k * (score - 0.5)))

def system_confidence(vix_z, breadth, flow_dispersion):
    return max(0, min(1, 0.4 * max(0, breadth) + 0.3 * (1 - min(vix_z, 2)/2) + 0.3 * (1 - flow_dispersion)))

def signal_quality(price_mom, flow_mom, vol_z):
    return max(0, abs(price_mom) + abs(flow_mom) - abs(vol_z) * 0.1)

def calculate_persistence(flow_mom_series, window=3):
    if len(flow_mom_series) < window: return "SIN_DATOS"
    recent = flow_mom_series.tail(window)
    pos = (recent > 0).sum()
    neg = (recent < 0).sum()
    if pos >= 2: return "PERSISTENCIA_ALCISTA"
    elif neg >= 2: return "PERSISTENCIA_BAJISTA"
    else: return "TRANSICION"

def signal_state(persistence, phase):
    if persistence == "TRANSICION" and phase in ["ACUMULACION", "DISTRIBUCION_TEMPRANA"]:
        return "SEÑAL TEMPRANA"
    elif persistence in ["PERSISTENCIA_ALCISTA", "PERSISTENCIA_BAJISTA"]:
        return "SEÑAL CONSOLIDADA"
    else: return "NEUTRAL"

def load_flow_history(file="flow_history.csv"):
    if os.path.exists(file):
        return pd.read_csv(file, index_col=0)
    else:
        return pd.DataFrame()

def save_flow_history_df(df, file="flow_history.csv"):
    df.to_csv(file)

def enrich_with_cftc_sector(report_lines, cftc_with_z, raw_file=None):
    """
    Genera la tabla CFTC en el reporte usando el DataFrame con z-scores (histórico).
    Si no hay cftc_with_z, intenta usar raw_file como respaldo.
    """
    if cftc_with_z is not None and not cftc_with_z.empty:
        # Usar el histórico con z-scores
        if 'cftc_z' not in cftc_with_z.columns:
            report_lines.append("\n## Confirmacion CFTC\n")
            report_lines.append("No se encontró la columna 'cftc_z' en los datos históricos.\n")
            return
        
        latest_date = cftc_with_z['date'].max()
        latest_data = cftc_with_z[cftc_with_z['date'] == latest_date].copy()
        
        from config import CFTC_MARKETS
        
        sector_z = {}
        for sector, substring in CFTC_MARKETS.items():
            mask = latest_data['market'].str.contains(substring, case=False, na=False)
            mask &= ~latest_data['market'].str.contains("MICRO|NANO", case=False, na=False)
            if mask.any():
                z_vals = latest_data.loc[mask, 'cftc_z']
                if not z_vals.empty:
                    sector_z[sector] = np.median(z_vals)
        
        if not sector_z:
            report_lines.append("\n## Confirmacion CFTC\n")
            report_lines.append("No se encontraron mercados relevantes en el histórico.\n")
            return
        
        report_lines.append("\n## Confirmacion CFTC (posicionamiento institucional)\n")
        report_lines.append(f"*Datos semanales al {latest_date.strftime('%Y-%m-%d')} (retraso de 3 dias)*\n\n")
        report_lines.append("| Sector | Z-score (mediana) | Instrumentos | Interpretacion |\n")
        report_lines.append("|--------|-------------------|--------------|----------------|\n")
        for sector, z in sorted(sector_z.items(), key=lambda x: x[1], reverse=True):
            if z > 1.5:
                interp = "FUERTEMENTE ALCISTA"
            elif z > 0.5:
                interp = "ALCISTA"
            elif z < -1.5:
                interp = "FUERTEMENTE BAJISTA"
            elif z < -0.5:
                interp = "BAJISTA"
            else:
                interp = "NEUTRAL"
            report_lines.append(f"| {sector} | {z:.2f} | - | {interp} |\n")
        report_lines.append("\n*Nota: CFTC semanal con retraso (martes a viernes). Solo para contexto macro.*\n")
        return
    
    # Fallback: usar raw_file si existe (comportamiento anterior)
    if raw_file is None:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("No se encontró archivo CFTC. Para activar, descargue manualmente desde\n")
        report_lines.append("https://www.cftc.gov/dea/newcot/FinFutWk.txt y guardelo como data/cftc_raw.txt\n")
        return
    
    parsed = parse_cftc_financials(raw_file)
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

    from config import CFTC_MARKETS

    sector_signals = {}
    for sector, substring in CFTC_MARKETS.items():
        mask = latest_data['market'].str.contains(substring, case=False, na=False)
        mask &= ~latest_data['market'].str.contains("MICRO|NANO", case=False, na=False)
        if mask.any():
            net_vals = latest_data.loc[mask, 'cftc_z']
            if not net_vals.empty:
                sector_signals[sector] = net_vals.tolist()

    if not sector_signals:
        report_lines.append("\n## Confirmacion CFTC\n")
        report_lines.append("No se encontraron mercados relevantes en los datos.\n")
        return

    report_lines.append("\n## Confirmacion CFTC (posicionamiento institucional)\n")
    report_lines.append(f"*Datos semanales al {latest_date.strftime('%Y-%m-%d')} (retraso de 3 dias)*\n\n")
    report_lines.append("| Sector | Z-score (mediana) | Instrumentos | Interpretacion |\n")
    report_lines.append("|--------|-------------------|--------------|----------------|\n")
    for sector, net_list in sorted(sector_signals.items(), key=lambda x: np.median(x[1]), reverse=True):
        median_z = np.median(net_list)
        n = len(net_list)
        if median_z > 1.5:
            interp = "FUERTEMENTE ALCISTA"
        elif median_z > 0.5:
            interp = "ALCISTA"
        elif median_z < -1.5:
            interp = "FUERTEMENTE BAJISTA"
        elif median_z < -0.5:
            interp = "BAJISTA"
        else:
            interp = "NEUTRAL"
        report_lines.append(f"| {sector} | {median_z:.2f} | {n} | {interp} |\n")
    report_lines.append("\n*Nota: CFTC semanal con retraso (martes a viernes). Solo para contexto macro.*\n")

def compute_synthetic_factors(cftc_sector_z):
    risk_sectors = ['SPY', 'XLK', 'XLF', 'XLI']
    risk_vals = [cftc_sector_z.get(s, 0) for s in risk_sectors]
    risk = np.median(risk_vals) if risk_vals else 0.0

    def_sectors = ['XLP', 'XLV', 'XLU']
    def_vals = [cftc_sector_z.get(s, 0) for s in def_sectors]
    defensive = np.median(def_vals) if def_vals else 0.0

    rotation = risk - defensive

    spy_z = cftc_sector_z.get('SPY', 0)
    if rotation > 0.2 and spy_z > 0:
        xly_inferred = "ACUMULACION FUERTE"
    elif rotation > 0:
        xly_inferred = "ACUMULACION DEBIL"
    else:
        xly_inferred = "NEUTRAL/DEBIL"

    short_vals = [
        cftc_sector_z.get('RATES_2Y', 0) * 0.5,
        cftc_sector_z.get('FED', 0) * 0.3,
        cftc_sector_z.get('SOFR', 0) * 0.2
    ]
    short_avg = np.mean(short_vals)

    long_vals = [
        cftc_sector_z.get('RATES_10Y', 0) * 0.6,
        cftc_sector_z.get('RATES_LONG', 0) * 0.4
    ]
    long_avg = np.mean(long_vals)

    curve_spread = long_avg - short_avg

    return {
        'risk': risk,
        'defensive': defensive,
        'rotation': rotation,
        'xly_inferred': xly_inferred,
        'short_rates': short_avg,
        'long_rates': long_avg,
        'curve_spread': curve_spread
    }

def main():
    print("=== RADAR DE ROTACION SECTORIAL v3.15 (con persistencia informativa) ===\n")
    df = download_market_data()

    # ---------------------------------------------------------
    # CARGA DE DATOS DE ACCIONES PARA EL ANÁLISIS DE LÍDERES
    # ---------------------------------------------------------
    stock_prices_df = None
    holdings_df = None
    tickers_para_lideres = []
    try:
        holdings_df = pd.read_csv("data/etf_holdings.csv")
        if 'ticker' not in holdings_df.columns:
            print("[Líderes] El archivo de holdings no tiene columna 'ticker'. No se podrá generar análisis.")
        else:
            tickers_para_lideres = holdings_df['ticker'].unique().tolist()
            print(f"[Líderes] Se descargarán/cachearán precios para {len(tickers_para_lideres)} tickers.")
            stock_prices_df = fetch_stock_prices(tickers_para_lideres, force=False)
            if not stock_prices_df.empty:
                # Normalizar índices para alinear con df
                if not isinstance(stock_prices_df.index, pd.DatetimeIndex):
                    stock_prices_df.index = pd.to_datetime(stock_prices_df.index)
                stock_prices_df = stock_prices_df.reindex(df.index)
                df = pd.concat([df, stock_prices_df], axis=1)
                print("[Líderes] Datos de acciones añadidos correctamente.")
            else:
                print("[Líderes] Advertencia: stock_prices_df está vacío. No se añadirán datos de acciones.")
    except Exception as e:
        print(f"[Líderes] Error al cargar datos de acciones: {e}")
        stock_prices_df = None

    features = compute_features(df)
    
    vix_z = features['vix_z'].iloc[-1]
    win = adaptive_window(vix_z, base_window=60)
    print(f"Ventana adaptativa para ortogonalización: {win} días")

    regime_score, regime_label = compute_regime_score(df, features)
    wls_weights = get_wls_weights(regime_score)
    print(f"Régimen cuantitativo: {regime_label} (score: {regime_score:.2f})")

    ranking_price, dispersion_price, breadth_price, vix_z_radar, stress, regime_price, accion_price = run_radar(df)
    ranking_flow, flow_dispersion, flow_breadth, regime_flow, flow_mom = run_flow_radar(df)
    sectors = list(ranking_flow.keys())

    # Macro causal
    breadth_signal = features['breadth_signal'].iloc[-1]
    macro_score_new = compute_macro_score(df, breadth_signal=breadth_signal)
    print("\n=== NUEVA MACRO (CAUSAL) ===")
    print(f"Macro Score (continuo): {macro_score_new:.2f}")

    # Flow attribution SPY
    flow_engine = FlowAttributionEngine(window=20)
    spy_df = pd.DataFrame({
        'open': df['SPY_open'],
        'high': df['SPY_high'],
        'low': df['SPY_low'],
        'close': df['SPY'],
        'volume': df['SPY_volume']
    }).dropna()
    ret_spy = spy_df['close'].pct_change().dropna()
    dv_spy = spy_df['close'] * spy_df['volume']
    spy_flow_metrics = flow_engine.compute(ret_spy, dv_spy)
    flow_momentum = compute_flow_momentum(
        0.5 * spy_flow_metrics['persistence_orth'] +
        0.3 * spy_flow_metrics['intensity_orth'] -
        0.2 * spy_flow_metrics['irregularity_orth']
    ).iloc[-1] if len(spy_flow_metrics) > 0 else 0

    # Wyckoff series
    spy_structure_series = compute_wyckoff_structure_series(spy_df)
    wyckoff_score_series = compute_wyckoff_score_series(spy_df)
    wyckoff_transition = detect_wyckoff_transition(spy_structure_series, wyckoff_score_series, lookback=5)
    spy_structure = compute_price_structure_advanced(spy_df)
    spy_structure_value = {"MARKUP":1, "RANGE":0, "DISTRIBUTION":-1}.get(spy_structure, 0)
    print(f"SPY Price Structure: {spy_structure}")
    spy_flow_state = flow_engine.classify_last(spy_flow_metrics)
    
    macro_series = compute_macro_series(df, features, window=60)
    flow_signal_series = (0.5 * spy_flow_metrics['persistence_orth'] +
                          0.3 * spy_flow_metrics['intensity_orth'] -
                          0.2 * spy_flow_metrics['irregularity_orth'])

    if macro_series.dropna().empty or flow_signal_series.dropna().empty:
        transition = 0
    else:
        transition = detect_regime_transition(macro_series, flow_signal_series)

    # Estabilidad de flujo
    if len(flow_signal_series) >= 5:
        stability = 1 - flow_signal_series.rolling(5).std().iloc[-1]
        stability = np.clip(stability, 0, 1)
    else:
        stability = np.nan

    # Ortogonalización macro-flujo
    macro_aligned = macro_series.reindex(flow_signal_series.index).dropna()
    flow_aligned = flow_signal_series.dropna()
    common_idx = macro_aligned.index.intersection(flow_aligned.index)
    macro_clean = macro_aligned[common_idx]
    flow_clean = flow_aligned[common_idx]
    macro_z = rolling_robust_zscore(macro_clean, window=win)
    flow_z = rolling_robust_zscore(flow_clean, window=win)
    macro_z = macro_z.replace([np.inf, -np.inf], np.nan)
    flow_z = flow_z.replace([np.inf, -np.inf], np.nan)
    valid = macro_z.notna() & flow_z.notna()
    macro_z = macro_z[valid]
    flow_z = flow_z[valid]

    if len(macro_z) <= win or len(flow_z) <= win:
        print(f"Advertencia: pocos datos para ortogonalización (len={len(macro_z)}). Usando flujo sin ortogonalizar.")
        flow_orthogonal_series = flow_z
    else:
        flow_orthogonal_series = rolling_regression_residual(macro_z, flow_z, window=win)

    macro_smoothed = apply_decay(macro_series, halflife=5)
    if len(flow_orthogonal_series) > 0:
        flow_smoothed = apply_decay(flow_orthogonal_series, halflife=5)
    else:
        flow_smoothed = pd.Series(dtype=float)

    macro_clean = macro_smoothed.dropna()
    flow_clean = flow_smoothed.dropna()
    macro_score_new = macro_clean.iloc[-1] if not macro_clean.empty else macro_score_new
    flow_orthogonal = flow_clean.iloc[-1] if not flow_clean.empty else 0.0

    edge_new = compute_edge_hierarchical(macro_score_new, flow_orthogonal, spy_structure_value)
    truth_score = compute_truth(macro_score_new, flow_orthogonal, spy_structure_value)
    alignment = compute_alignment_score(macro_score_new, flow_orthogonal, spy_structure_value)
    tradeable = is_tradeable(edge_new, truth_score, spy_structure_value)
    print(f"Edge Score: {edge_new:.2f}")
    print(f"Truth Score: {truth_score:.2f}")

    # ETFs sectoriales (flow state)
    etf_flow_states = {}
    for sec in sectors:
        etf_df = pd.DataFrame({'close': df[sec], 'volume': df[f"{sec}_volume"]}).dropna()
        if len(etf_df) >= 60:
            ret_etf = etf_df['close'].pct_change().dropna()
            dv_etf = etf_df['close'] * etf_df['volume']
            metrics = flow_engine.compute(ret_etf, dv_etf)
            etf_flow_states[sec] = flow_engine.classify_last(metrics)
        else:
            etf_flow_states[sec] = "INSUFICIENT_DATA"

    # =========================================================
    # OPTIONS MARKET STRUCTURE (OMS v1.1)
    # =========================================================
    try:
        options_df = get_historical_options_data(years=[2024, 2025, 2026])
        if not options_df.empty:
            total_volume = options_df.groupby('Trade Date')['Volume'].sum()
            total_volume = total_volume.sort_index()
            common_idx = spy_df.index.intersection(total_volume.index)
            total_volume = total_volume.reindex(common_idx).ffill()
        else:
            total_volume = pd.Series(0, index=spy_df.index)
    except Exception as e:
        print(f"Error cargando datos para OMS: {e}")
        total_volume = pd.Series(0, index=spy_df.index)
        options_df = pd.DataFrame()

    try:
        pcr_series = get_hkex_pcr_series(days_back=730)
        pcr_series = pcr_series.reindex(spy_df.index).ffill().fillna(1.0)
    except Exception as e:
        print(f"Error cargando PCR de HKEX: {e}")
        pcr_series = pd.Series(1.0, index=spy_df.index)

    try:
        oms_df = compute_oms(pcr_series, options_df)
        latest_oms = oms_df['oms'].iloc[-1]
        oms_regime = classify_oms(latest_oms)
        oms_mod = oms_modifier(latest_oms)
        sentiment_val = oms_df['sentiment'].iloc[-1]
        activity_heat_val = oms_df['activity'].iloc[-1]
        fragmentation_val = oms_df['fragmentation'].iloc[-1]
    except Exception as e:
        print(f"Error en OMS: {e}")
        latest_oms = np.nan
        oms_regime = "NO DISPONIBLE"
        oms_mod = {"risk_bias": "NEUTRAL", "confidence_boost": 1.0}
        sentiment_val = activity_heat_val = fragmentation_val = np.nan

    # =========================================================
    # WYCKOFF POR SECTOR
    # =========================================================
    sector_wyckoff_phase = {}
    sector_wyckoff_score = {}
    sector_wyckoff_transition = {}
    for sec in sectors:
        sector_df = pd.DataFrame({
            'open': df[f"{sec}_open"],
            'high': df[f"{sec}_high"],
            'low': df[f"{sec}_low"],
            'close': df[sec],
            'volume': df[f"{sec}_volume"]
        }).dropna()
        if len(sector_df) >= 200:
            phase = wyckoff_structure_core(sector_df)
            score = wyckoff_score(sector_df).iloc[-1]
            struct_series = compute_wyckoff_structure_series(sector_df)
            score_series = compute_wyckoff_score_series(sector_df)
            trans = detect_wyckoff_transition(struct_series, score_series, lookback=5)
        else:
            phase = "INSUFICIENT_DATA"
            score = np.nan
            trans = "SIN DATOS SUFICIENTES"
        sector_wyckoff_phase[sec] = phase
        sector_wyckoff_score[sec] = score
        sector_wyckoff_transition[sec] = trans

    # =========================================================
    # ALERTAS, DISTRIBUCIÓN, CFTC Y REPORTE
    # =========================================================
    alertas = []
    top_price_2 = [sec for sec, _ in ranking_price[:2]]
    bottom_flow_2 = list(ranking_flow.keys())[-2:]
    for sec in top_price_2:
        if sec in bottom_flow_2:
            alertas.append(f"ALERTA ROJA: {sec} esta entre los 2 mejores en precio pero entre los 2 peores en flujo -> posible techo.")
    bottom_price_2 = [sec for sec, _ in ranking_price[-2:]]
    top_flow_2 = list(ranking_flow.keys())[:2]
    for sec in bottom_price_2:
        if sec in top_flow_2:
            alertas.append(f"ALERTA VERDE: {sec} esta entre los 2 peores en precio pero entre los 2 mejores en flujo -> posible acumulación.")

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

    # Distribución
    distribution_scores = {}
    distribution_probs_bin = {}
    distribution_div_cont = {}
    distribution_prob_cont = {}
    risk_score_dict = {}

    flow_history = load_flow_history("outputs/flow_history.csv")
    for sec in sectors:
        if sec not in flow_history.columns:
            flow_history[sec] = np.nan
    hoy = pd.Timestamp.now().normalize()
    for sec in sectors:
        flow_history.loc[hoy, sec] = latest_flow_mom.get(sec, np.nan)
    save_flow_history_df(flow_history, "outputs/flow_history.csv")

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

    fase_dict = {}
    direccion_dict = {}
    for sec in sectors:
        fm = latest_flow_mom.get(sec, 0)
        fase_dict[sec] = classify_phase(latest_price_z.get(sec, 0), fm, latest_acc_z.get(sec, 0), latest_vol_z.get(sec, 0))
        direccion_dict[sec] = classify_direction(fm)

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

    sorted_sectors = sorted(sectors, key=lambda x: oportunidad_dict.get(x, 0), reverse=True)

    # --- CFTC con histórico ---
    cftc_spy_z = None
    cftc_raw = None
    cftc_with_z = None
    if CFTC_AVAILABLE:
        cftc_history_df = update_cftc_history()
        if cftc_history_df is not None and not cftc_history_df.empty:
            cftc_with_z = compute_cftc_zscore_from_history()
            if cftc_with_z is not None and not cftc_with_z.empty:
                latest = cftc_with_z.loc[cftc_with_z.groupby('market')['date'].idxmax()]
                spy_row = latest[latest['market'].str.contains("S&P 500", case=False)]
                if not spy_row.empty:
                    cftc_spy_z = spy_row['cftc_z'].iloc[-1]
            else:
                cftc_raw = load_cftc_manual(path="data/cftc_raw.txt")
                if cftc_raw is not None:
                    parsed = parse_cftc_financials(cftc_raw)
                    if parsed is not None:
                        with_signal = compute_cftc_signal(parsed)
                        if with_signal is not None and not with_signal.empty:
                            spy_data = with_signal[with_signal['market'].str.contains("S&P 500", case=False)]
                            if not spy_data.empty:
                                cftc_spy_z = spy_data['cftc_z'].iloc[-1]
                            cftc_with_z = with_signal
        else:
            cftc_raw = load_cftc_manual(path="data/cftc_raw.txt")
            if cftc_raw is not None:
                parsed = parse_cftc_financials(cftc_raw)
                if parsed is not None:
                    cftc_with_z = compute_cftc_signal(parsed)
                    if cftc_with_z is not None and not cftc_with_z.empty:
                        spy_data = cftc_with_z[cftc_with_z['market'].str.contains("S&P 500", case=False)]
                        if not spy_data.empty:
                            cftc_spy_z = spy_data['cftc_z'].iloc[-1]

    cftc_sector_z = {}
    if cftc_with_z is not None and not cftc_with_z.empty:
        if 'cftc_z' in cftc_with_z.columns:
            latest_data = cftc_with_z.loc[cftc_with_z.groupby('market')['date'].idxmax()]
            from config import CFTC_MARKETS
            for sector, substring in CFTC_MARKETS.items():
                mask = latest_data['market'].str.contains(substring, case=False, na=False)
                mask &= ~latest_data['market'].str.contains("MICRO|NANO", case=False, na=False)
                if mask.any():
                    z_vals = latest_data.loc[mask, 'cftc_z']
                    if not z_vals.empty:
                        cftc_sector_z[sector] = np.median(z_vals)
        else:
            print("[CFTC] Advertencia: 'cftc_z' no encontrada en los datos. Los factores sintéticos no estarán disponibles.")

    synthetic = compute_synthetic_factors(cftc_sector_z)
    synth_lines = [
        "\n## Factores Sintéticos (Agregados)\n",
        f"- **Riesgo (cíclicos):** {synthetic.get('risk', 0):.2f}\n",
        f"- **Defensivo:** {synthetic.get('defensive', 0):.2f}\n",
        f"- **Rotación:** {synthetic.get('rotation', 0):.2f}\n",
        f"- **XLY (inferido):** {synthetic.get('xly_inferred', 'N/A')}\n",
        f"- **Tipos cortos:** {synthetic.get('short_rates', 0):.2f}\n",
        f"- **Tipos largos:** {synthetic.get('long_rates', 0):.2f}\n",
        f"- **Curva:** {synthetic.get('curve_spread', 0):.2f}\n"
    ]

    dist_lines = ["\n## Confirmacion de Distribucion\n"]
    rank_history = pd.DataFrame()
    for i in range(1, 6):
        if i <= len(flow_mom):
            rank = flow_mom.iloc[-i].rank(ascending=False)
            rank_history[f'day_{i}'] = rank
    if not rank_history.empty:
        n_sectors = rank_history.shape[0]
        rank_change = rank_history.diff(axis=1).abs().mean().mean() / n_sectors
        dist_lines.append(f"\n**Volatilidad del ranking (5d):** {rank_change:.3f}\n")
        if rank_change > 0.15:
            dist_lines.append("*Rotación extrema*\n")
        elif rank_change > 0.05:
            dist_lines.append("*Rotación moderada*\n")
        else:
            dist_lines.append("*Mercado estable*\n")
    else:
        dist_lines.append("\n**Volatilidad del ranking:** No hay datos\n")

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
        alerta = "DISTRIBUCION FUERTE" if score_bin >= 0.7 else "POSIBLE DISTRIBUCION" if score_bin >= 0.4 else "SIN SENAL"
        dist_lines.append(f"| {sec} | {pm:.3f} | {fm:.2f} | {fa:.2f} | {vz:.2f} | {div_cont:.2f} | {score_bin:.2f} | {prob_bin:.2%} | {prob_cont:.2%} | {risk_score:.2f} | {intensidad} | {clasif} | {fase} | {direccion} | {conv_ajustada:.2f} | {score_unif:.2f} | {conf_sistema:.2f} | {calidad:.2f} | {oportunidad:.2f} | {operabilidad} | {persistencia} | {estado_senal} | {alerta} |\n")

    plot_flow_dispersion(flow_mom)

    temp_md = "outputs/reporte_diario_temp.md"
    save_markdown_report(ranking_price, ranking_flow, flow_dispersion, flow_breadth, regime_flow,
                         dispersion_price, breadth_price, vix_z, regime_price, accion_price,
                         alertas, output_path=temp_md)
    with open(temp_md, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Insertar factores y distribución
    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## Conclusion"):
            insert_pos = i
            break
    lines[insert_pos:insert_pos] = synth_lines + dist_lines

    # Insertar contexto macro y modelo causal
    for i, line in enumerate(lines):
        if line.startswith("## Conclusion"):
            insert_pos = i
            break

    macro_label = "RISK-ON" if macro_score_new > 0.5 else "RISK-OFF" if macro_score_new < -0.5 else "NEUTRAL"
    macro_new_lines = [
        "\n## Contexto Macro (Unificado)\n",
        f"- **Macro Score (causal):** {macro_score_new:.2f}\n",
        f"- **Régimen cualitativo:** {macro_label}\n"
    ]
    causal_lines = []
    causal_lines.append(f"- **Transición régimen:** {transition}\n")
    causal_lines.append(f"- **Flow momentum:** {flow_momentum:.3f}\n")
    causal_lines.append(f"- **Estabilidad flujo (5d):** {stability:.2f}\n")
    causal_lines.append(f"- **Alineación global:** {alignment:.2f}\n")
    causal_lines.append(f"- **Operable:** {'Sí' if tradeable else 'No'}\n")
    causal_lines.append(f"- **Transición Wyckoff:** {wyckoff_transition}\n")
    causal_lines.append("\n## Modelo Causal (Macro → Flow → Price)\n")
    causal_lines.append(f"- **Macro Score:** {macro_score_new:.2f}\n")
    causal_lines.append(f"- **SPY Flow State:** {spy_flow_state}\n")
    causal_lines.append(f"- **SPY Price Structure:** {spy_structure}\n")
    causal_lines.append(f"- **Edge Score:** {edge_new:.2f}\n")
    causal_lines.append(f"- **Truth Score:** {truth_score:.2f}\n")
    causal_lines.append("\n**Flow Attribution por Sector:**\n")
    causal_lines.append("| Sector | Flow State |\n")
    causal_lines.append("|--------|------------|\n")
    for sec, state in etf_flow_states.items():
        causal_lines.append(f"| {sec} | {state} |\n")
    causal_lines.append("\n")

    # Tablas Wyckoff
    causal_lines.append("**Fase Wyckoff por Sector:**\n")
    causal_lines.append("| Sector | Fase Wyckoff | Score |\n")
    causal_lines.append("|--------|--------------|-------|\n")
    for sec in sectors:
        phase = sector_wyckoff_phase.get(sec, "N/A")
        score = sector_wyckoff_score.get(sec, np.nan)
        causal_lines.append(f"| {sec} | {phase} | {score:.2f} |\n")
    causal_lines.append("\n")

    causal_lines.append("**Transición Wyckoff por Sector (5 días):**\n")
    causal_lines.append("| Sector | Transición |\n")
    causal_lines.append("|--------|------------|\n")
    for sec in sectors:
        trans = sector_wyckoff_transition.get(sec, "N/A")
        causal_lines.append(f"| {sec} | {trans} |\n")
    causal_lines.append("\n")

    # OMS
    causal_lines.append("\n## Options Market Structure (OMS v1.1)\n")
    causal_lines.append(f"- **Sentimiento (PCR Hong Kong):** {sentiment_val:.2f}\n")
    causal_lines.append(f"- **Activity Heat:** {activity_heat_val:.2f}\n")
    causal_lines.append(f"- **Fragmentación (HHI):** {fragmentation_val:.2f}\n")
    causal_lines.append(f"- **OMS Score:** {latest_oms:.2f}\n")
    causal_lines.append(f"- **Régimen:** {oms_regime}\n")
    causal_lines.append("\n*Interpretación: ESTABLE → mercado amortigua movimientos; FRÁGIL → riesgo de amplificación.*\n")
    causal_lines.append("\n*Nota: El Put/Call ratio utilizado en el sentimiento proviene de la Bolsa de Hong Kong (HKEX).*\n")
    if oms_mod["risk_bias"] != "NEUTRAL":
        causal_lines.append(f"*Recomendación de tamaño: {oms_mod['risk_bias']} (confianza {oms_mod['confidence_boost']:.2f})*\n")

    lines[insert_pos:insert_pos] = macro_new_lines + causal_lines

    # CFTC (llamada real, debes tener la función)
    cftc_lines = []
    enrich_with_cftc_sector(cftc_lines, cftc_with_z, raw_file=cftc_raw)
    for i, line in enumerate(lines):
        if line.startswith("## Conclusion"):
            insert_pos = i
            break
    lines[insert_pos:insert_pos] = cftc_lines

    with open("outputs/reporte_diario.md", 'w', encoding='utf-8') as f:
        f.writelines(lines)
    os.remove(temp_md)

    # Líderes
    if stock_prices_df is None:
        print("[Líderes] No se generará análisis porque no se pudieron cargar datos de acciones.")
    else:
        try:
            holdings_df = pd.read_csv("data/etf_holdings.csv")
            if 'ticker' not in holdings_df.columns:
                print("[Líderes] El archivo de holdings no tiene columna 'ticker'.")
            else:
                tickers_para_lideres = holdings_df['ticker'].unique().tolist()
                # Construir df_multi con columnas de acciones y de ETFs sectoriales
                cols_to_keep = []
                # 1. Añadir columnas de acciones (close, volume, open, high, low)
                for t in tickers_para_lideres:
                    close_col = f"{t}_close"
                    vol_col = f"{t}_volume"
                    if close_col in df.columns:
                        cols_to_keep.append(close_col)
                    if vol_col in df.columns:
                        cols_to_keep.append(vol_col)
                    for suf in ['_open', '_high', '_low']:
                        col = f"{t}{suf}"
                        if col in df.columns:
                            cols_to_keep.append(col)
                # 2. Añadir columnas de los ETFs sectoriales (precio, volumen, open, high, low)
                for sec in sectors:
                    if sec in df.columns:
                        cols_to_keep.append(sec)  # precio close
                    vol_etf = f"{sec}_volume"
                    if vol_etf in df.columns:
                        cols_to_keep.append(vol_etf)
                    for suf in ['_open', '_high', '_low']:
                        col = f"{sec}{suf}"
                        if col in df.columns:
                            cols_to_keep.append(col)
                # Seleccionar columnas y construir df_multi
                if cols_to_keep:
                    df_multi = df[cols_to_keep]
                    print(f"[Líderes] Construido df_multi con {len(cols_to_keep)} columnas.")
                else:
                    df_multi = pd.DataFrame()
                    print("[Líderes] No se encontraron columnas de acciones o ETFs.")
                
                # Verificar tickers sin datos
                missing_in_df = [t for t in tickers_para_lideres if f"{t}_close" not in df.columns]
                if missing_in_df:
                    print(f"[Líderes] Advertencia: los siguientes tickers no tienen datos y se omitirán: {missing_in_df}")
                
                leader_lines = generate_leader_section(
                    fase_dict,
                    operabilidad_dict,
                    sectors,
                    df_multi,
                    holdings_df,
                    wls_weights=wls_weights
                )
                if leader_lines:
                    with open("outputs/analisis_lideres.md", "w", encoding="utf-8") as f:
                        f.writelines(leader_lines)
                    print("Análisis de líderes generado.")
                else:
                    print("No hay sectores favorables para líderes.")
        except Exception as e:
            print(f"Advertencia en líderes: {e}")

    print("\nHistorico de flujos guardado en outputs/flow_history.csv")
    print("Grafico de dispersion guardado en outputs/flow_dispersion.png")
    print("Reporte diario guardado en outputs/reporte_diario.md")

    try:
        from validation import evaluate_signal
        future_returns = df['SPY'].pct_change().shift(-5)
        signal_df = pd.DataFrame(index=flow_mom.index)
        for sec in sectors:
            signal_df[sec] = distribution_prob_cont.get(sec, 0)
        signal_series = signal_df.mean(axis=1)
        val = evaluate_signal(signal_series, future_returns, threshold=0.7)
        print("\n=== VALIDACION CUANTITATIVA ===")
        if val['n_signals'] == 0:
            print(val['message'])
        else:
            print(f"Alpha: {val['alpha']:.4f}, Hit ratio: {val['hit_ratio']:.2%}, Señales: {val['n_signals']}")
    except Exception as e:
        print(f"Validacion no disponible: {e}")

    print("\nEjecucion completada.")

if __name__ == '__main__':
    main()
