"""
stock_leader.py – Módulo de análisis de líderes sectoriales.
No genera señales de trading, solo información cuantitativa.
"""

import pandas as pd
import numpy as np

from wyckoff_detector import wyckoff_score, classify_wyckoff_phase

# =========================================================
# UTILIDADES
# =========================================================

def validate_series(series, min_length=60):
    """Valida que la serie tenga datos suficientes y no esté corrupta."""
    if series is None:
        return False
    if series.isna().sum() > len(series) * 0.2:
        return False
    if len(series.dropna()) < min_length:
        return False
    return True

def robust_zscore(series, window=60):
    """Z-score robusto usando mediana y MAD (misma función que el radar)."""
    rolling_median = series.rolling(window).median()
    mad = (series - rolling_median).abs().rolling(window).median()
    z = (series - rolling_median) / (mad * 1.4826)
    return z

def compute_rsi(price, window=14):
    """RSI clásico (Wilder smoothing) sin dependencias externas."""
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_multi_index(df):
    """
    Convierte el DataFrame plano del radar a MultiIndex con niveles:
    (ticker, "close"), (ticker, "volume"), (ticker, "open"), (ticker, "high"), (ticker, "low")
    """
    tickers = set()
    for col in df.columns:
        if "_" in col:
            continue
        tickers.add(col)
    multi = {}
    for t in tickers:
        if t not in df.columns:
            continue
        multi[(t, "close")] = df[t]
        vol_col = f"{t}_volume"
        if vol_col in df.columns:
            multi[(t, "volume")] = df[vol_col]
        # Añadir open, high, low
        open_col = f"{t}_open"
        high_col = f"{t}_high"
        low_col = f"{t}_low"
        if open_col in df.columns:
            multi[(t, "open")] = df[open_col]
        if high_col in df.columns:
            multi[(t, "high")] = df[high_col]
        if low_col in df.columns:
            multi[(t, "low")] = df[low_col]
    return pd.DataFrame(multi)

# =========================================================
# CORE: MÉTRICAS POR ACCIÓN
# =========================================================

def compute_stock_metrics(df, etf_ticker, stock_list):
    """
    Calcula métricas clave para cada acción del ETF:
    - RS, RS momentum (log), RS trend
    - Flow z-score (suavizado con EWM)
    - Divergencia (flow > 0.5 y rs_mom < 0)
    - RSI, warnings (sobrecompra, extendido, clímax)
    """
    results = []

    if (etf_ticker, "close") not in df.columns:
        return pd.DataFrame()

    price_etf = df[(etf_ticker, "close")]
    if not validate_series(price_etf):
        return pd.DataFrame()

    for ticker in stock_list:
        if (ticker, "close") not in df.columns or (ticker, "volume") not in df.columns:
            continue

        price = df[(ticker, "close")]
        volume = df[(ticker, "volume")]

        if not validate_series(price) or not validate_series(volume):
            continue

        try:
            # Relative Strength
            rs = price / price_etf
            # RS momentum con log-diferencias (estable cross-sector)
            rs_mom = np.log(rs).diff(20).iloc[-1]
            rs_trend = rs.rolling(20).mean().diff().iloc[-1]

            # Flujo institucional
            ret = price.pct_change()
            dollar_vol = price * volume
            flow_raw = ret * dollar_vol
            flow_z = robust_zscore(flow_raw, window=60)
            # Suavizado con EWM para menor lag
            flow_signal = flow_z.ewm(span=5).mean().iloc[-1]
            # Wyckoff microstructural analysis
            # Construir DataFrame con OHLCV para este ticker
            ticker_df = pd.DataFrame({
                "open": df[(ticker, "open")],
                "high": df[(ticker, "high")],
                "low": df[(ticker, "low")],
                "close": price,
                "volume": volume
            }).dropna()
            if len(ticker_df) >= 60:
                wyckoff_sc = wyckoff_score(ticker_df).iloc[-1]
                wyckoff_ph = classify_wyckoff_phase(ticker_df)
            else:
                wyckoff_sc = np.nan
                wyckoff_ph = "INSUFICIENTE"
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
                "wyckoff_phase": wyckoff_ph
            })
        except Exception:
            continue

    df_out = pd.DataFrame(results)
    if df_out.empty:
        return df_out

    # Ordenación implícita (NO visible como score)
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
    output_csv_path="outputs/analisis_lideres.csv"
):
    """
    Genera líneas Markdown y guarda CSV con los datos de líderes sectoriales.
    """
    lines = []
    all_data = []

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # HEADER
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

        # Acumular para CSV global
        metrics_df["sector"] = sector
        all_data.append(metrics_df)

        # Markdown sección
        lines.append(f"## Sector: {sector}\n")
        lines.append(f"- **Fase:** {fase}\n")
        lines.append(f"- **Operabilidad:** {oper}\n\n")
        lines.append("| Ticker | RS | RS Mom | Flow (z) | RSI | Divergencia | Persistencia | Wyckoff | Warnings |\n")
        lines.append("|--------|----|--------|-----------|-----|-------------|--------------|---------|----------|\n")
        for _, row in metrics_df.iterrows():
            div = "Sí" if row['divergence'] else "No"
            lines.append(
                f"| {row['ticker']} | {row['rs']:.3f} | {row['rs_mom']:.2%} | "
                f"{row['flow_z']:.2f} | {row['rsi']:.1f} | {div} | {row['persistence']}/3 | "
                f"{row['wyckoff_phase']} | {row['warnings']} |\n"
            )

        lines.append("\n")

    # CSV global
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)

    return lines