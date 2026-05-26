"""
stock_leader.py – Módulo de análisis de líderes sectoriales.
No genera señales de trading, solo información cuantitativa.
Versión v3.18.1: WLS con escalas compatibles, stability z-score,
eliminación persistence 3d, avisos interpretativos.
"""

import pandas as pd
import numpy as np
from wyckoff_detector import wyckoff_score, classify_wyckoff_phase

# =========================================================
# WYCKOFF LEADERSHIP ENGINE (WLE) – VERSIÓN INSTITUCIONAL v2.0
# =========================================================

def compute_wyckoff_leadership(df, weights=None):
    """
    Calcula el Wyckoff Leadership Score (WLS) usando normalizaciones robustas (MAD).
    Versión v2.0: cuatro factores (RS, Flow, Estructura, Estabilidad).
    Pesos por defecto: 0.35 RS, 0.30 Flow, 0.25 Estructura, 0.10 Estabilidad.
    Incluye penalización por colinealidad (Spearman > 0.7 entre rs_mom y flow_z).
    """
    df = df.copy()
    
    # 1. RS momentum robust z-score (MAD) con suelo para evitar degeneración
    median_rs = df["rs_mom"].median()
    mad_rs = max(np.median(np.abs(df["rs_mom"] - median_rs)), 0.001)
    df["rs_z"] = (df["rs_mom"] - median_rs) / (mad_rs + 1e-9)
    df["rs_z"] = df["rs_z"].clip(-3, 3)

    # 2. Flow robust z-score con suelo y acotado
    median_flow = df["flow_z"].median()
    mad_flow = max(np.median(np.abs(df["flow_z"] - median_flow)), 0.001)
    df["flow_z_norm"] = (df["flow_z"] - median_flow) / (mad_flow + 1e-9)
    df["flow_z_norm"] = df["flow_z_norm"].clip(-3, 3)
    
    # 3. RWS con percentil 70 del sector (estructura relativa) → normalizar a z-score simétrico
    baseline = df.groupby("sector")["wyckoff_score"].transform(
        lambda x: np.percentile(x, 70)
    )
    df["rws"] = df["wyckoff_score"] / (baseline + 1e-9)

    def robust_zscore_intra(s):
        median = s.median()
        mad = (s - median).abs().median()
        return (s - median) / (1.4826 * mad + 1e-9)

    # RWS simétrico: puede sumar o restar al WLS
    df["rws_z"] = df.groupby("sector")["rws"].transform(robust_zscore_intra)
    df["rws_z"] = df["rws_z"].clip(-3, 3)
    
    # 4. Stability → normalizar con z-score robusto intra-sector
    if "stability" not in df.columns:
        df["stability"] = 1.0

    def robust_zscore_intra(s):
        median = s.median()
        mad = (s - median).abs().median()
        return (s - median) / (1.4826 * mad + 1e-9)

    df["stab_z"] = df.groupby("sector")["stability"].transform(robust_zscore_intra)
    df["stab_z"] = df["stab_z"].clip(-3, 3)
    
    # ============================================
    # PENALIZACIÓN POR COLINEALIDAD (Spearman)
    # ============================================
    penalized_rs_z = df["rs_z"].copy()
    penalized_flow_z_norm = df["flow_z_norm"].copy()
    
    for sector in df["sector"].unique():
        mask = df["sector"] == sector
        if mask.sum() < 5:
            continue
        rs = df.loc[mask, "rs_mom"]
        flow = df.loc[mask, "flow_z"]
        rho = rs.corr(flow, method="spearman")
        if pd.notna(rho) and rho > 0.7:
            factor = 1 - min(1.0, (rho - 0.7) / 0.3)
            penalized_flow_z_norm.loc[mask] = factor * df.loc[mask, "flow_z_norm"]
    
    # ============================================
    # PESOS Y CÁLCULO DEL WLS
    # ============================================
    if weights is None:
        w_rs = 0.35
        w_flow = 0.30
        w_structure = 0.25
        w_stab = 0.10
    else:
        w_rs = weights.get("rs", 0.35)
        w_flow = weights.get("flow", 0.30)
        w_structure = weights.get("structure", 0.25)
        w_stab = weights.get("stability", 0.10)
    
    df["wls"] = (
        w_rs * penalized_rs_z +
        w_flow * penalized_flow_z_norm +
        w_structure * df["rws_z"] +
        w_stab * df["stab_z"]
    )
    return df

# =========================================================
# WYCKOFF MEJORAS ROBUSTAS
# =========================================================

def wyckoff_persistence_robust(flow_signal_series, wyckoff_score_series, window=5):
    signal = (flow_signal_series * wyckoff_score_series).clip(lower=0)
    binary = (signal > 0.25).astype(int)
    return binary.rolling(window).sum()

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

def prepare_multi_index(df, tickers_list=None):
    """
    Convierte el DataFrame plano del radar a un DataFrame con columnas planas
    de la forma 'TICKER_close', 'TICKER_volume', etc.
    """
    if tickers_list is None:
        tickers = set()
        for col in df.columns:
            if "_" in col:
                continue
            tickers.add(col)
    else:
        tickers = tickers_list

    result = pd.DataFrame(index=df.index)
    for t in tickers:
        if t not in df.columns:
            continue

        def to_series(col_data):
            if isinstance(col_data, pd.DataFrame):
                return col_data.iloc[:, 0]
            return col_data

        result[f"{t}_close"] = to_series(df[t])

        vol_col = f"{t}_volume"
        if vol_col in df.columns:
            result[f"{t}_volume"] = to_series(df[vol_col])

        for suf in ['_open', '_high', '_low']:
            col = f"{t}{suf}"
            if col in df.columns:
                result[f"{t}{suf}"] = to_series(df[col])

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

    close_col = etf_ticker
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
            ret = price.pct_change(fill_method=None)
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
                
                # MICROSTRUCTURE QUALITY (interna, no se exporta)
                from wyckoff_detector import range_compression, absorption_score
                comp_raw = range_compression(ticker_df).iloc[-1]
                compression = 1 - np.clip(comp_raw, 0, 1)
                absorption = absorption_score(ticker_df).iloc[-1]
                persistence_norm = wyckoff_persistence_val / 5.0
                structure_quality = (compression + absorption + persistence_norm) / 3.0
                structure_quality = np.clip(structure_quality, 0, 1)
                
                # STABILITY (raw, sin clip; se normalizará intra-sector)
                mean_10 = wyckoff_score_series.rolling(10).mean().iloc[-1]
                std_10 = wyckoff_score_series.rolling(10).std().iloc[-1]
                stability = (mean_10 / (std_10 + 1e-9)) * np.tanh(mean_10)
            else:
                wyckoff_sc = np.nan
                wyckoff_ph = "INSUFICIENTE"
                stability = 0.0

            # Persistencia a 10 días (única métrica de persistencia exportada)
            flow_positive = (flow_z > 0).astype(int)
            persistence_10d = flow_positive.rolling(10, min_periods=10).sum().iloc[-1]
            if pd.isna(persistence_10d):
                persistence_10d = 0

            # Indicadores complementarios (no se exportan, solo se calculan)
            rsi = compute_rsi(price).iloc[-1]
            ma20 = price.rolling(20).mean().iloc[-1]
            extended = price.iloc[-1] > 1.15 * ma20
            climax = volume.iloc[-1] > 2 * volume.rolling(20).mean().iloc[-1]

            warnings_list = []
            if rsi > 70:
                warnings_list.append("Sobrecompra")
            if extended:
                warnings_list.append("Extendido")
            if climax:
                warnings_list.append("Climax volumen")

            if np.isnan(rs_mom) or np.isnan(flow_signal):
                continue

            results.append({
                "ticker": ticker,
                "rs": rs.iloc[-1],
                "rs_mom": rs_mom,
                "flow_z": flow_signal,
                "wyckoff_score": wyckoff_sc,
                "wyckoff_phase": wyckoff_ph,
                "persistence_10d": persistence_10d,
                "stability": stability,
            })
        except Exception:
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
    lines.append("⚠️ **Importante:** La fase Wyckoff indicada para cada acción (SPRING, MARKUP INIT, etc.) ")
    lines.append("es una clasificación **microestructural** distinta de la fase del ETF en el reporte diario ")
    lines.append("(MARKUP, DISTRIBUTION, ACCUMULATION, RANGE). No comparar directamente ambas taxonomías.\n\n")
    lines.append("⚠️ **El ranking WLS solo es válido dentro del mismo sector.** ")
    lines.append("No compare valores de `wls` entre sectores distintos.\n\n")

    VALID_FASES = {"ACUMULACION", "ACUMULACION FUERTE", "CONFIRMACION ALCISTA"}
    VALID_OPER = {"OPORTUNIDAD CLARA", "OPORTUNIDAD MODERADA"}

    for sector in sectors:
        fase = fase_dict.get(sector, "NEUTRAL")
        oper = operabilidad_dict.get(sector, "NO OPERAR")
        if fase not in VALID_FASES or oper not in VALID_OPER:
            continue
        stocks = holdings_df[holdings_df['etf'] == sector]['ticker'].tolist()
        if not stocks:
            continue
        metrics_df = compute_stock_metrics(df, sector, stocks)
        if metrics_df.empty:
            continue
        metrics_df["sector"] = sector

        # Calcular WLS una sola vez por sector
        sector_df = compute_wyckoff_leadership(metrics_df, weights=wls_weights)
        sector_df = sector_df.sort_values("wls", ascending=False)
        all_data.append(sector_df)

        lines.append(f"## Sector: {sector}\n")
        lines.append(f"- **Fase:** {fase}\n")
        lines.append(f"- **Operabilidad:** {oper}\n\n")
        top_n = 3
        top_leaders = sector_df.head(top_n)
        
        lines.append(f"**Top {top_n} líderes por WLS:**\n\n")
        lines.append("| Ticker | RS | RS Mom | Flow (z) | WLS |\n")
        lines.append("|--------|----|--------|-----------|-----|\n")
        for _, row in top_leaders.iterrows():
            lines.append(
                f"| {row['ticker']} | {row['rs']:.3f} | {row['rs_mom']:.2%} | "
                f"{row['flow_z']:.2f} | {row['wls']:.2f} |\n"
            )
        lines.append("\n")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Ya contiene los WLS calculados; solo añadimos el percentil
        final_df['sector_rank_pct'] = final_df.groupby('sector')['wls'].rank(pct=True)

        columnas_finales = [
            "ticker", "sector", "rs", "rs_mom", "flow_z",
            "wyckoff_score", "wyckoff_phase", "persistence_10d",
            "stability", "wls", "sector_rank_pct"
        ]
        columnas_existentes = [c for c in columnas_finales if c in final_df.columns]
        final_df = final_df[columnas_existentes]
        final_df = final_df.sort_values(["sector", "wls"], ascending=[True, False])

        final_df.to_csv(output_csv_path, index=False)
        print(f"Análisis de líderes CSV generado: {output_csv_path}")
        return lines
    else:
        print("No se pudo generar análisis de líderes: all_data vacío")
        return None