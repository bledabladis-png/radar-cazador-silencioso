"""
stock_leader.py – Módulo de análisis de líderes sectoriales.
No genera señales de trading, solo información cuantitativa.
"""

import pandas as pd
import numpy as np

from wyckoff_detector import wyckoff_score, classify_wyckoff_phase

# =========================================================
# WYCKOFF LEADERSHIP ENGINE (WLE) – VERSIÓN INSTITUCIONAL
# =========================================================

def compute_wyckoff_leadership(df, weights=None):
    """
    Calcula el Wyckoff Leadership Score (WLS) usando normalizaciones robustas (MAD).
    Si weights es None, usa pesos fijos (25/25/20/20/10).
    """
    df = df.copy()
    
    # 1. RS momentum robust z-score (MAD)
    median_rs = df["rs_mom"].median()
    mad_rs = np.median(np.abs(df["rs_mom"] - median_rs))
    df["rs_z"] = (df["rs_mom"] - median_rs) / (mad_rs + 1e-9)
    
    # 2. Flow robust z-score
    median_flow = df["flow_z"].median()
    mad_flow = np.median(np.abs(df["flow_z"] - median_flow))
    df["flow_z_norm"] = (df["flow_z"] - median_flow) / (mad_flow + 1e-9)
    
    # 3. Persistence normalizada
    df["persistence_norm"] = df["wyckoff_persistence"] / 5.0
    
    # 4. RWS con percentil 70 del sector
    baseline = df.groupby("sector")["wyckoff_score"].transform(
        lambda x: np.percentile(x, 70)
    )
    df["rws"] = df["wyckoff_score"] / (baseline + 1e-9)
    
    # 5. Stability (si no existe, asignar 1)
    if "stability" not in df.columns:
        df["stability"] = 1.0
    
    # 6. Aplicar pesos
    if weights is None:
        w_rs = 0.25
        w_flow = 0.25
        w_persist = 0.20
        w_structure = 0.20
        w_stab = 0.10
    else:
        w_rs = weights.get("rs", 0.25)
        w_flow = weights.get("flow", 0.25)
        w_persist = weights.get("persist", 0.20)
        w_structure = weights.get("structure", 0.20)
        w_stab = weights.get("stability", 0.10)
    
    df["wls"] = (
        w_rs * df["rs_z"] +
        w_flow * df["flow_z_norm"] +
        w_persist * df["persistence_norm"] +
        w_structure * df["rws"] +
        w_stab * df["stability"]
    )
    return df

# =========================================================
# WYCKOFF MEJORAS ROBUSTAS
# =========================================================

def wyckoff_persistence_robust(flow_signal_series, wyckoff_score_series, window=5):
    signal = (flow_signal_series * wyckoff_score_series).clip(lower=0)
    binary = (signal > 0.25).astype(int)
    return binary.rolling(window).sum()

def wyckoff_cycle_series(df):
    from wyckoff_detector import detect_spring, detect_sos, wyckoff_score
    spring = detect_spring(df)
    sos = detect_sos(df)
    score = wyckoff_score(df)

    cycle = np.zeros(len(df))
    spring_flag = False

    for i in range(len(df)):
        if spring.iloc[i] == 1:
            spring_flag = True
        accumulation = score.iloc[i] > 0.6
        sos_now = sos.iloc[i] == 1
        if spring_flag and accumulation and sos_now:
            cycle[i] = 1
            spring_flag = False
    return pd.Series(cycle, index=df.index)

# =========================================================
# UTILIDADES
# =========================================================

def validate_series(series, min_length=60):
    if series is None:
        return False
    if series.isna().sum() > len(series) * 0.2:
        return False
    if len(series.dropna()) < min_length:
        return False
    return True

def robust_zscore(series, window=60):
    rolling_median = series.rolling(window).median()
    mad = (series - rolling_median).abs().rolling(window).median()
    z = (series - rolling_median) / (mad * 1.4826 + 1e-9)
    return z

def compute_rsi(price, window=14):
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_multi_index(df):
    """
    Convierte el DataFrame plano del radar a un DataFrame con columnas planas
    de la forma 'TICKER_close', 'TICKER_volume', etc.
    """
    tickers = set()
    for col in df.columns:
        if "_" in col:
            continue
        tickers.add(col)
    result = pd.DataFrame(index=df.index)
    for t in tickers:
        if t not in df.columns:
            continue
        result[f"{t}_close"] = df[t]
        vol_col = f"{t}_volume"
        if vol_col in df.columns:
            result[f"{t}_volume"] = df[vol_col]
        for suf in ['_open', '_high', '_low']:
            col = f"{t}{suf}"
            if col in df.columns:
                result[f"{t}{suf}"] = df[col]
    return result

# =========================================================
# CORE: MÉTRICAS POR ACCIÓN
# =========================================================

def compute_stock_metrics(df, etf_ticker, stock_list):
    """
    Calcula métricas clave para cada acción del ETF.
    df debe tener columnas planas: 'TICKER_close', 'TICKER_volume', etc.
    """
    results = []

    close_col = f"{etf_ticker}_close"
    if close_col not in df.columns:
        return pd.DataFrame()

    price_etf = df[close_col]
    if not validate_series(price_etf):
        return pd.DataFrame()

    for ticker in stock_list:
        close_col_ticker = f"{ticker}_close"
        volume_col_ticker = f"{ticker}_volume"
        if close_col_ticker not in df.columns or volume_col_ticker not in df.columns:
            continue

        price = df[close_col_ticker]
        volume = df[volume_col_ticker]

        if not validate_series(price) or not validate_series(volume):
            continue

        try:
            # Relative Strength
            rs = price / price_etf
            rs_mom = np.log(rs).diff(20).iloc[-1]
            rs_trend = rs.rolling(20).mean().diff().iloc[-1]

            # Flujo institucional
            ret = price.pct_change()
            dollar_vol = price * volume
            flow_raw = ret * dollar_vol
            flow_z = robust_zscore(flow_raw, window=60)
            flow_signal = flow_z.ewm(span=5).mean().iloc[-1]

            # Wyckoff microstructural analysis
            ticker_df = pd.DataFrame({
                "open": df[f"{ticker}_open"] if f"{ticker}_open" in df.columns else price,
                "high": df[f"{ticker}_high"] if f"{ticker}_high" in df.columns else price,
                "low": df[f"{ticker}_low"] if f"{ticker}_low" in df.columns else price,
                "close": price,
                "volume": volume
            }).dropna()
            if len(ticker_df) >= 60:
                wyckoff_sc = wyckoff_score(ticker_df).iloc[-1]
                wyckoff_ph = classify_wyckoff_phase(ticker_df)
                wyckoff_score_series = wyckoff_score(ticker_df)
                flow_signal_series = flow_z
                persistence_series = wyckoff_persistence_robust(flow_signal_series, wyckoff_score_series, window=5)
                wyckoff_persistence_val = persistence_series.iloc[-1] if not persistence_series.empty else 0
                if pd.isna(wyckoff_persistence_val):
                    wyckoff_persistence_val = 0
                cycle_series = wyckoff_cycle_series(ticker_df)
                wyckoff_cycle_val = cycle_series.iloc[-1] if not cycle_series.empty else 0
                
                # MICROSTRUCTURE QUALITY
                from wyckoff_detector import range_compression, absorption_score
                comp_raw = range_compression(ticker_df).iloc[-1]
                compression = 1 - np.clip(comp_raw, 0, 1)
                absorption = absorption_score(ticker_df).iloc[-1]
                persistence_norm = wyckoff_persistence_val / 5.0
                structure_quality = (compression + absorption + persistence_norm) / 3.0
                structure_quality = np.clip(structure_quality, 0, 1)
                
                # STABILITY
                mean_10 = wyckoff_score_series.rolling(10).mean().iloc[-1]
                std_10 = wyckoff_score_series.rolling(10).std().iloc[-1]
                stability = (mean_10 / (std_10 + 1e-9)) * np.tanh(mean_10)
                stability = np.clip(stability, 0, 2)
            else:
                wyckoff_sc = np.nan
                wyckoff_ph = "INSUFICIENTE"
                wyckoff_persistence_val = 0
                wyckoff_cycle_val = 0
                structure_quality = 0.0
                stability = 0.0

            # Persistence filter (últimos 3 días)
            flow_positive = (flow_z > 0).astype(int)
            persistence = flow_positive.rolling(3, min_periods=3).sum().iloc[-1]
            if pd.isna(persistence):
                persistence = 0

            # Divergencia inteligente
            divergence = (flow_signal > 0.5) and (rs_mom < 0)

            # RSI y warnings
            rsi = compute_rsi(price).iloc[-1]
            ma20 = price.rolling(20).mean().iloc[-1]
            extended = price.iloc[-1] > 1.15 * ma20
            climax = volume.iloc[-1] > 2 * volume.rolling(20).mean().iloc[-1]

            warnings = []
            if rsi > 70:
                warnings.append("Sobrecompra")
            if extended:
                warnings.append("Extendido")
            if climax:
                warnings.append("Climax volumen")

            if np.isnan(rs_mom) or np.isnan(flow_signal):
                continue

            results.append({
                "ticker": ticker,
                "rs": rs.iloc[-1],
                "rs_mom": rs_mom,
                "rs_trend": rs_trend,
                "flow_z": flow_signal,
                "divergence": divergence,
                "rsi": rsi,
                "warnings": ", ".join(warnings) if warnings else "-",
                "persistence": persistence,
                "wyckoff_score": wyckoff_sc,
                "wyckoff_phase": wyckoff_ph,
                "wyckoff_persistence": wyckoff_persistence_val,
                "wyckoff_cycle": wyckoff_cycle_val,
                "structure_quality": structure_quality,
                "stability": stability
            })
        except Exception as e:
            # print(f"Error en {ticker}: {e}")  # opcional para depurar
            continue

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out

    df_out = df_out.sort_values(by=["rs_mom", "flow_z"], ascending=False)
    return df_out.reset_index(drop=True)

# =========================================================
# GENERACIÓN DE INFORME (MARKDOWN + CSV)
# =========================================================

def generate_leader_section(
    fase_dict,
    operabilidad_dict,
    sectors,
    df,
    holdings_df,
    output_csv_path="outputs/analisis_lideres.csv",
    wls_weights=None
):
    lines = []
    all_data = []

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    lines.append("# Análisis de Líderes Sectoriales\n")
    lines.append(f"**Fecha:** {timestamp}\n\n")
    lines.append("> Informe informativo. No constituye recomendación de inversión.\n\n")

    VALID_FASES = {"ACUMULACION", "ACUMULACION FUERTE", "CONFIRMACION ALCISTA"}
    VALID_OPER = {"OPORTUNIDAD CLARA", "OPORTUNIDAD MODERADA"}

    for sector in sectors:
        fase = fase_dict.get(sector, "NEUTRAL")
        oper = operabilidad_dict.get(sector, "NO OPERAR")
        if fase not in VALID_FASES or oper not in VALID_OPER:
            continue

        etf_ticker = sector
        stocks = holdings_df[holdings_df['etf'] == etf_ticker]['ticker'].tolist()
        if not stocks:
            continue

        metrics_df = compute_stock_metrics(df, etf_ticker, stocks)
        if metrics_df.empty:
            continue
        metrics_df["sector"] = sector
        all_data.append(metrics_df)

        lines.append(f"## Sector: {sector}\n")
        lines.append(f"- **Fase:** {fase}\n")
        lines.append(f"- **Operabilidad:** {oper}\n\n")
        sector_df = compute_wyckoff_leadership(metrics_df, weights=wls_weights)
        sector_df = sector_df.sort_values("wls", ascending=False)
        top_n = 3
        top_leaders = sector_df.head(top_n)
        
        lines.append(f"**Top {top_n} líderes por WLS:**\n\n")
        lines.append("| Ticker | RS | RS Mom | Flow (z) | RSI | WLS |\n")
        lines.append("|--------|----|--------|-----------|-----|-----|\n")
        for _, row in top_leaders.iterrows():
            lines.append(
                f"| {row['ticker']} | {row['rs']:.3f} | {row['rs_mom']:.2%} | "
                f"{row['flow_z']:.2f} | {row['rsi']:.1f} | {row['wls']:.2f} |\n"
            )
        lines.append("\n")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = compute_wyckoff_leadership(final_df, weights=wls_weights)
        final_df = final_df.sort_values(["sector", "wls"], ascending=[True, False])
        
        def compute_edge_score(row):
            F = np.tanh(row['flow_z'])
            P = np.tanh(row['rs_mom'])
            S = np.tanh(row['wls'])
            signals = np.array([F, P, S])
            alignment = np.mean(signals)
            consistency = 1 - np.std(signals)
            return alignment * consistency
        
        final_df['edge_score'] = final_df.apply(compute_edge_score, axis=1)

        def compute_trend_readiness(row):
            alignment = row['edge_score']
            structure = np.tanh(row['wls'])
            persistence = row['wyckoff_persistence'] / 5.0
            stability = np.tanh(row['stability'])
            score = (0.35 * alignment + 0.25 * structure +
                     0.20 * persistence + 0.20 * stability)
            return np.clip(score, -1, 1)

        final_df['trend_readiness'] = final_df.apply(compute_trend_readiness, axis=1)
        
        final_df.to_csv(output_csv_path, index=False)