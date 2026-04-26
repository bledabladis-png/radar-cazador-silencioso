"""
backtest_wls.py – Validación estadística completa del Wyckoff Leadership Score (WLS).
Métricas: IC, rank IC, turnover, alpha decay, horizon decay layer, stability surface POR RÉGIMEN.
No genera señales de trading; solo diagnóstico.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CARGA DE DATOS
# =========================================================

def load_historical_leaders(data_folder="outputs", pattern="analisis_lideres_*.csv"):
    """
    Carga todos los archivos CSV de líderes sectoriales.
    Si no encuentra archivos con patrón de fecha, intenta con el archivo único.
    Asegura que exista columna 'fecha'.
    """
    files = sorted(Path(data_folder).glob(pattern))
    if not files:
        single = Path(data_folder) / "analisis_lideres.csv"
        if single.exists():
            files = [single]
        else:
            raise FileNotFoundError("No se encontraron archivos de líderes sectoriales.")
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        if 'fecha' not in df.columns:
            # Extraer fecha del nombre del archivo: analisis_lideres_20250412.csv
            try:
                date_str = f.stem.replace("analisis_lideres_", "")
                df['fecha'] = pd.to_datetime(date_str)
            except:
                continue
        df_list.append(df)
    if not df_list:
        raise ValueError("No se pudieron cargar datos con fechas válidas.")
    return pd.concat(df_list, ignore_index=True)

def load_price_data(price_csv="data/market_data.csv"):
    """
    Carga precios de cierre de todos los tickers desde market_data.csv.
    Selecciona columnas sin guion bajo (tickers limpios).
    """
    df_price = pd.read_csv(price_csv, index_col=0, parse_dates=True)
    ticker_cols = [c for c in df_price.columns if '_' not in c]
    return df_price[ticker_cols]

def compute_regime_for_date(df, date):
    """
    Calcula el régimen cuantitativo para una fecha específica usando el DataFrame de mercado.
    Retorna: "EXPANSION", "TRANSITION", "CONTRACTION"
    """
    # Tomar datos hasta la fecha (sin look-ahead)
    sub_df = df.loc[:date].copy()
    if len(sub_df) < 200:
        return "TRANSITION"  # insuficiente histórico
    spy_close = sub_df['SPY']
    spy_ma50 = spy_close.rolling(50).mean().iloc[-1]
    spy_ma200 = spy_close.rolling(200).mean().iloc[-1]
    trend_raw = (spy_ma50 / spy_ma200 - 1) * 5
    trend_norm = (np.tanh(trend_raw) + 1) / 2
    
    # Breadth simplificado (puede mejorarse, pero para régimen es suficiente)
    sectors = ['XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLV', 'XLU', 'XLRE']
    breadth = 0
    for sec in sectors:
        if sec in sub_df.columns:
            above = int(sub_df[sec].iloc[-1] > sub_df[sec].rolling(100).mean().iloc[-1])
            breadth += above
    breadth_signal = - (breadth / len(sectors) - 0.5) * 2  # convertir a valor en [-1,1]
    breadth_norm = np.clip((breadth_signal + 0.5) / 1.5, 0, 1)
    
    # VIX
    vix = sub_df['^VIX'].dropna()
    if len(vix) >= 60:
        vix_z = (vix.iloc[-1] - vix.rolling(60).mean().iloc[-1]) / vix.rolling(60).std().iloc[-1]
    else:
        vix_z = 0
    if len(vix) >= 5:
        slope = np.polyfit(range(5), vix.tail(5).values, 1)[0]
        vix_trend = np.tanh(slope)
    else:
        vix_trend = 0
    vix_component = np.exp(-vix_z) * (1 - vix_trend)
    vix_component = np.clip(vix_component, 0, 1)
    
    # Crédito
    if 'HYG' in sub_df.columns and 'LQD' in sub_df.columns:
        credit_ratio = sub_df['HYG'] / sub_df['LQD']
        credit_z = (credit_ratio.iloc[-1] - credit_ratio.rolling(60).mean().iloc[-1]) / (credit_ratio.rolling(60).std().iloc[-1] + 1e-9)
        credit_norm = 1 - np.clip(credit_z, 0, 2) / 2
    else:
        credit_norm = 0.5
    
    regime_score = 0.4 * trend_norm + 0.2 * breadth_norm + 0.2 * vix_component + 0.2 * credit_norm
    if regime_score > 0.6:
        return "EXPANSION"
    elif regime_score < 0.4:
        return "CONTRACTION"
    else:
        return "TRANSITION"

def add_regime_column(df_leaders, market_df):
    """
    Añade la columna 'regime' al DataFrame de líderes basándose en la fecha.
    """
    regimes = []
    for fecha in df_leaders['fecha']:
        regime = compute_regime_for_date(market_df, fecha)
        regimes.append(regime)
    df_leaders['regime'] = regimes
    return df_leaders

# =========================================================
# CÁLCULO DE RETORNOS FORWARD
# =========================================================

def compute_forward_return(prices, ticker, date, horizon):
    """Retorna retorno simple de un ticker desde date hacia adelante 'horizon' días."""
    if ticker not in prices.columns:
        return np.nan
    price_series = prices[ticker].dropna()
    if date not in price_series.index:
        return np.nan
    idx = price_series.index.get_loc(date)
    if idx + horizon >= len(price_series):
        return np.nan
    price_t0 = price_series.iloc[idx]
    price_tn = price_series.iloc[idx + horizon]
    return (price_tn - price_t0) / price_t0

# =========================================================
# HORIZON DECAY LAYER (HDL)
# =========================================================

def horizon_decay_weights(horizons, lam=0.08):
    """Pesos exponenciales decrecientes normalizados."""
    weights = np.exp(-lam * np.array(horizons))
    return weights / weights.sum()

def weighted_horizon_score(results_by_horizon, lam=0.08):
    """Combina spreads medios con decaimiento exponencial."""
    horizons = sorted(results_by_horizon.keys())
    weights = horizon_decay_weights(horizons, lam)
    weighted_alpha = sum(weights[i] * results_by_horizon[h]['mean_spread'] 
                         for i, h in enumerate(horizons))
    return weighted_alpha, dict(zip(horizons, weights))

# =========================================================
# WLS STABILITY SURFACE (IC por régimen y horizonte)
# =========================================================

def add_wls_rank(df):
    """Añade rango percentil del WLS por día."""
    df = df.copy()
    df["wls_rank"] = df.groupby("fecha")["wls"].rank(pct=True)
    return df

def compute_wls_stability_surface(df, horizons={"5d":5, "10d":10, "20d":20}):
    """
    Calcula matriz de IC (Spearman) por régimen y horizonte.
    Requiere columnas: fecha, wls, forward_{horizon}d, regime.
    """
    df = add_wls_rank(df)
    regimes = df["regime"].dropna().unique()
    surface = {}
    for regime in regimes:
        surface[regime] = {}
        df_reg = df[df["regime"] == regime]
        for h_name, h_days in horizons.items():
            col_ret = f"forward_{h_days}d"
            if col_ret not in df_reg.columns:
                surface[regime][h_name] = np.nan
                continue
            temp = df_reg[["wls_rank", col_ret]].dropna()
            if len(temp) < 50:
                surface[regime][h_name] = np.nan
            else:
                ic = temp["wls_rank"].corr(temp[col_ret], method="spearman")
                surface[regime][h_name] = ic
    # Retornar DataFrame con régimen como columnas y horizonte como índice
    return pd.DataFrame(surface).T

def edge_strength_score(surface_df, horizon_weights={"5d":0.5, "10d":0.3, "20d":0.2}):
    """Agrega Edge Strength Score (ESS) a partir de la surface."""
    ess = 0.0
    for regime in surface_df.columns:
        for horizon, weight in horizon_weights.items():
            if horizon in surface_df.index:
                val = surface_df.loc[horizon, regime]
                if pd.notna(val):
                    ess += abs(val) * weight
    return ess

# =========================================================
# BACKTEST PRINCIPAL
# =========================================================

def backtest_wls(df, prices, horizons=[5,10,20], top_pct=0.1, bottom_pct=0.1):
    """
    Calcula para cada horizonte:
        - Spread medio (top decile - bottom decile)
        - Sharpe del spread (anualizado)
        - Hit rate
        - Mean IC (Pearson)
        - Mean rank IC (Spearman)
        - Top decile turnover
    """
    df = df.sort_values(['fecha', 'wls'], ascending=[True, False])
    results = {}
    for h in horizons:
        spreads = []
        ics = []
        rank_ics = []
        turnovers = []
        prev_top = set()
        dates = df['fecha'].unique()
        for i, date in enumerate(dates):
            day_df = df[df['fecha'] == date].copy()
            if len(day_df) < 20:
                continue
            day_df['ret_fwd'] = day_df.apply(
                lambda r: compute_forward_return(prices, r['ticker'], date, h),
                axis=1
            )
            day_df = day_df.dropna(subset=['ret_fwd'])
            if len(day_df) < 20:
                continue
            n_top = max(1, int(len(day_df) * top_pct))
            n_bottom = max(1, int(len(day_df) * bottom_pct))
            top = day_df.nlargest(n_top, 'wls')
            bottom = day_df.nsmallest(n_bottom, 'wls')
            spread = top['ret_fwd'].mean() - bottom['ret_fwd'].mean()
            spreads.append(spread)
            ic = day_df['wls'].corr(day_df['ret_fwd'])
            rank_ic = spearmanr(day_df['wls'], day_df['ret_fwd'])[0]
            ics.append(ic)
            rank_ics.append(rank_ic)
            if i > 0:
                current_top = set(top['ticker'])
                if prev_top:
                    turnover = 1 - len(current_top & prev_top) / len(current_top)
                else:
                    turnover = 0.0
                turnovers.append(turnover)
            prev_top = set(top['ticker'])
        if len(spreads) == 0:
            results[h] = None
            continue
        mean_spread = np.mean(spreads)
        std_spread = np.std(spreads)
        results[h] = {
            'mean_spread': mean_spread,
            'std_spread': std_spread,
            'sharpe_spread': mean_spread / (std_spread + 1e-9) * np.sqrt(252),
            'hit_rate': np.mean(np.array(spreads) > 0),
            'mean_ic': np.mean(ics),
            'mean_rank_ic': np.mean(rank_ics),
            'mean_turnover': np.mean(turnovers) if turnovers else np.nan,
            'num_days': len(spreads)
        }
    return results

# =========================================================
# INTERPRETACIÓN SIMPLIFICADA
# =========================================================

def interpret_weighted_alpha(weighted_alpha):
    if weighted_alpha > 0.002:
        return "✅ Edge estructural (lento) – posible líder sostenido"
    elif weighted_alpha > 0.0005:
        return "⚠️ Edge táctico (rápido) – útil para corto plazo"
    else:
        return "❌ Sin edge significativo (ruido)"

# =========================================================
# EJECUCIÓN PRINCIPAL
# =========================================================

if __name__ == "__main__":
    print("=== Backtest completo del Wyckoff Leadership Score (WLS) con régimen ===\n")
    try:
        # 1. Cargar datos de líderes
        df = load_historical_leaders()
        print(f"Días únicos cargados: {df['fecha'].nunique()}")
        print(f"Registros totales: {len(df)}")

        # 2. Cargar precios
        prices = load_price_data()
        print(f"Datos de precios: {prices.shape[0]} días, {prices.shape[1]} tickers\n")

        # 3. Cargar mercado para régimen
        market_df = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
        print("Añadiendo columna 'regime' a los datos de líderes...")
        df = add_regime_column(df, market_df)
        print(f"Regímenes encontrados: {df['regime'].unique()}")

        # 4. Precalcular retornos forward para horizontes 5,10,20 días
        print("Calculando retornos forward...")
        for h in [5,10,20]:
            col = f"forward_{h}d"
            df[col] = df.apply(
                lambda r: compute_forward_return(prices, r['ticker'], r['fecha'], h),
                axis=1
            )

        # 5. Backtest por horizonte (global)
        res = backtest_wls(df, prices)
        print("\n=== BACKTEST GLOBAL ===")
        for h, metrics in res.items():
            if metrics is None:
                print(f"Horizonte {h}d: Sin datos suficientes.")
                continue
            print(f"\n--- Horizonte {h} días ---")
            print(f"Spread medio (top-bottom): {metrics['mean_spread']:.5f}")
            print(f"Sharpe del spread (anualizado): {metrics['sharpe_spread']:.2f}")
            print(f"Hit rate: {metrics['hit_rate']:.2%}")
            print(f"Mean IC (Pearson): {metrics['mean_ic']:.3f}")
            print(f"Mean rank IC (Spearman): {metrics['mean_rank_ic']:.3f}")
            print(f"Top decile turnover: {metrics['mean_turnover']:.2%}")
            print(f"Días válidos: {metrics['num_days']}")

        # 6. Horizon Decay Layer
        weighted_alpha, decay_weights = weighted_horizon_score(res)
        print("\n=== HORIZON DECAY LAYER (HDL) ===")
        print(f"Pesos por horizonte (λ=0.08): {decay_weights}")
        print(f"Alpha ponderado por decaimiento: {weighted_alpha:.5f}")
        print(f"Interpretación: {interpret_weighted_alpha(weighted_alpha)}")

        # 7. WLS Stability Surface por régimen
        surface = compute_wls_stability_surface(df)
        print("\n=== WLS STABILITY SURFACE (IC por régimen) ===")
        print(surface)
        ess = edge_strength_score(surface)
        print(f"\nEdge Strength Score (ESS): {ess:.3f}")

        # 8. IC medio por régimen (adicional)
        print("\n=== IC MEDIO POR RÉGIMEN (Spearman, horizonte 10d) ===")
        for regime in df['regime'].unique():
            df_reg = df[df['regime'] == regime].dropna(subset=['forward_10d', 'wls'])
            if len(df_reg) < 30:
                print(f"{regime}: datos insuficientes")
                continue
            ic = spearmanr(df_reg['wls'], df_reg['forward_10d'])[0]
            print(f"{regime}: IC = {ic:.3f} (n={len(df_reg)})")

    except Exception as e:
        print(f"Error durante el backtest: {e}")
        print("Asegúrate de tener al menos 30 días de histórico y precios en market_data.csv")