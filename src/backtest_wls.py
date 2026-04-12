"""
backtest_wls.py – Validación estadística completa del Wyckoff Leadership Score (WLS).
Métricas: IC, rank IC, turnover, alpha decay, horizon decay layer, stability surface.
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
    print("=== Backtest completo del Wyckoff Leadership Score (WLS) ===\n")
    try:
        # 1. Cargar datos de líderes
        df = load_historical_leaders()
        print(f"Días únicos cargados: {df['fecha'].nunique()}")
        print(f"Registros totales: {len(df)}")

        # 2. Cargar precios
        prices = load_price_data()
        print(f"Datos de precios: {prices.shape[0]} días, {prices.shape[1]} tickers\n")

        # 3. Backtest por horizonte
        res = backtest_wls(df, prices)
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

        # 4. Horizon Decay Layer
        weighted_alpha, decay_weights = weighted_horizon_score(res)
        print("\n=== HORIZON DECAY LAYER (HDL) ===")
        print(f"Pesos por horizonte (λ=0.08): {decay_weights}")
        print(f"Alpha ponderado por decaimiento: {weighted_alpha:.5f}")
        print(f"Interpretación: {interpret_weighted_alpha(weighted_alpha)}")

        # 5. WLS Stability Surface (si existe columna 'regime')
        # Si no existe, se puede generar a partir del regime_score histórico.
        # Por simplicidad, se omite si no está disponible.
        if 'regime' in df.columns:
            # Asegurar columnas forward_{h}d para cada horizonte
            for h in [5,10,20]:
                col = f'forward_{h}d'
                if col not in df.columns:
                    # Calcular sobre la marcha (costoso, se hace por cada ticker)
                    # Se recomienda precalcular en el script de backtest con los precios.
                    # Aquí asumimos que ya existen.
                    pass
            surface = compute_wls_stability_surface(df)
            print("\n=== WLS STABILITY SURFACE (IC por régimen) ===")
            print(surface)
            ess = edge_strength_score(surface)
            print(f"Edge Strength Score (ESS): {ess:.3f}")
        else:
            print("\n=== WLS STABILITY SURFACE ===")
            print("No se encontró columna 'regime' en los datos. Omitiendo superficie.")

    except Exception as e:
        print(f"Error durante el backtest: {e}")
        print("Asegúrate de tener al menos 30 días de histórico y precios en market_data.csv")