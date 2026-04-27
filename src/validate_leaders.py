"""
validate_leaders.py – Validación forward del Wyckoff Leadership Score (WLS)
No forma parte del radar diario. Ejecutar manualmente cuando se disponga de histórico.
Mide: hit rate, alpha, turnover, persistencia, y desglose por capitalización.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN
# =========================================================
DATA_FOLDER = "outputs"
PRICE_CSV = "data/market_data.csv"
HOLDINGS_CSV = "data/etf_holdings.csv"
HORIZONS = [5, 20, 60]          # días
TOP_N = 3                       # tomar top N del WLS por sector
OUTPUT_DIR = Path("outputs/leader_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# CARGA DE DATOS
# =========================================================

def load_historical_leaders():
    """Carga todos los archivos analisis_lideres_*.csv desde:
       - outputs/ (los más recientes, si existen)
       - data/historical_leaders/ (los históricos guardados)
    """
    files = []
    # Buscar en outputs/
    files.extend(sorted(Path("outputs").glob("analisis_lideres_*.csv")))
    # Buscar en data/historical_leaders/
    files.extend(sorted(Path("data/historical_leaders").glob("analisis_lideres_*.csv")))
    # Si no hay archivos con patrón, intentar con el único analisis_lideres.csv
    if not files:
        single = Path("outputs") / "analisis_lideres.csv"
        if single.exists():
            files = [single]
        else:
            raise FileNotFoundError("No se encontraron archivos de líderes.")
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        if 'fecha' not in df.columns:
            try:
                date_str = f.stem.replace("analisis_lideres_", "")
                df['fecha'] = pd.to_datetime(date_str)
            except:
                continue
        df_list.append(df)
    if not df_list:
        raise ValueError("No se pudieron cargar datos con fechas válidas.")
    return pd.concat(df_list, ignore_index=True)

def load_price_data():
    """Carga precios de cierre de market_data.csv (tickers limpios)."""
    df_price = pd.read_csv(PRICE_CSV, index_col=0, parse_dates=True)
    # Seleccionar columnas sin guion bajo (tickers de acciones y ETFs)
    ticker_cols = [c for c in df_price.columns if '_' not in c]
    return df_price[ticker_cols]

def load_holdings():
    """Carga etf_holdings.csv para mapear sector de cada acción (opcional para contexto)."""
    try:
        holdings = pd.read_csv(HOLDINGS_CSV)
        return holdings[['ticker', 'etf']].drop_duplicates()
    except:
        return pd.DataFrame()

# =========================================================
# CAPITALIZACIÓN (para desglose por buckets)
# =========================================================

def get_market_cap(ticker):
    """Obtiene la capitalización bursátil (en millones de USD) usando yfinance."""
    import yfinance as yf
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        cap = info.get('marketCap', None)
        if cap is None or cap <= 0:
            return None
        return cap / 1e6   # en millones para mejor legibilidad
    except Exception:
        return None

def classify_by_market_cap(tickers_list, cap_cache={}):
    """
    Clasifica una lista de tickers en Large, Mid, Small según su capitalización.
    Umbrales: Large > 50B, Mid 10B-50B, Small < 10B (en millones: >50000, 10000-50000, <10000)
    Retorna un diccionario {ticker: 'Large'|'Mid'|'Small'|'Unknown'}.
    """
    caps = {}
    for t in tickers_list:
        if t in cap_cache:
            caps[t] = cap_cache[t]
        else:
            cap = get_market_cap(t)
            if cap is None:
                caps[t] = 'Unknown'
            elif cap > 50000:
                caps[t] = 'Large'
            elif cap > 10000:
                caps[t] = 'Mid'
            else:
                caps[t] = 'Small'
            cap_cache[t] = caps[t]
    return caps

# =========================================================
# CÁLCULO DE RETORNOS FORWARD
# =========================================================

def get_forward_return(prices, ticker, date, horizon):
    """Retorna retorno simple (price_future/price_today - 1)."""
    if ticker not in prices.columns:
        return np.nan
    series = prices[ticker].dropna()
    if date not in series.index:
        return np.nan
    idx = series.index.get_loc(date)
    if idx + horizon >= len(series):
        return np.nan
    p0 = series.iloc[idx]
    pf = series.iloc[idx + horizon]
    return (pf - p0) / p0 if p0 != 0 else np.nan

def etf_for_ticker(ticker, holdings):
    """Retorna el ETF sectorial correspondiente (si existe)."""
    if holdings.empty:
        return None
    match = holdings[holdings['ticker'] == ticker]
    if not match.empty:
        return match.iloc[0]['etf']
    return None

# =========================================================
# MÉTRICAS POR HORIZONTE (global y por bucket)
# =========================================================

def evaluate_horizon(df_leaders, prices, holdings, horizon):
    """
    Para cada fecha, toma top N por sector (solo sectores favorables),
    calcula retorno forward, hit rate y alpha vs ETF sectorial.
    Retorna DataFrame con resultados y métricas agregadas.
    """
    records = []
    dates = df_leaders['fecha'].unique()
    for date in dates:
        day_df = df_leaders[df_leaders['fecha'] == date].copy()
        sectors = day_df['sector'].unique()
        for sector in sectors:
            sector_df = day_df[day_df['sector'] == sector].sort_values('wls', ascending=False)
            top = sector_df.head(TOP_N)
            for _, row in top.iterrows():
                ticker = row['ticker']
                ret_fwd = get_forward_return(prices, ticker, date, horizon)
                if pd.isna(ret_fwd):
                    continue
                etf_ret = get_forward_return(prices, sector, date, horizon) if sector in prices.columns else np.nan
                rel_ret = ret_fwd - etf_ret if not pd.isna(etf_ret) else np.nan
                records.append({
                    'fecha': date,
                    'sector': sector,
                    'ticker': ticker,
                    'wls': row['wls'],
                    'horizon_dias': horizon,
                    'ret_absoluto': ret_fwd,
                    'ret_etf': etf_ret,
                    'ret_relativo': rel_ret
                })
    if not records:
        return pd.DataFrame(), {}

    df_res = pd.DataFrame(records)
    hit_rate_abs = (df_res['ret_absoluto'] > 0).mean()
    hit_rate_rel = (df_res['ret_relativo'] > 0).mean() if not df_res['ret_relativo'].isna().all() else np.nan
    alpha_medio = df_res['ret_relativo'].mean() if not df_res['ret_relativo'].isna().all() else np.nan
    num_obs = len(df_res)
    ret_medio_abs = df_res['ret_absoluto'].mean()
    ret_medio_etf = df_res['ret_etf'].mean() if not df_res['ret_etf'].isna().all() else np.nan

    metrics = {
        'num_observaciones': num_obs,
        'hit_rate_absoluto': hit_rate_abs,
        'hit_rate_relativo': hit_rate_rel,
        'alpha_medio': alpha_medio,
        'retorno_medio_absoluto': ret_medio_abs,
        'retorno_medio_ETF': ret_medio_etf
    }
    return df_res, metrics

def evaluate_horizon_by_bucket(df_leaders, prices, holdings, horizon, bucket_dict):
    """
    Evalúa por separado para cada bucket de capitalización.
    Retorna dict {bucket: (df_res, metrics)}.
    """
    results = {}
    for bucket in ['Large', 'Mid', 'Small']:
        tickers_in_bucket = [t for t, b in bucket_dict.items() if b == bucket]
        df_bucket = df_leaders[df_leaders['ticker'].isin(tickers_in_bucket)].copy()
        if df_bucket.empty:
            results[bucket] = (pd.DataFrame(), {})
            continue
        df_res, metrics = evaluate_horizon(df_bucket, prices, holdings, horizon)
        results[bucket] = (df_res, metrics)
    return results

# =========================================================
# MÉTRICAS GLOBALES (turnover, persistencia)
# =========================================================

def turnover_analysis(df_leaders):
    """Calcula el turnover mensual del top 3 por sector."""
    df = df_leaders.sort_values('fecha')
    turn_data = []
    sectors = df['sector'].unique()
    for sector in sectors:
        sector_df = df[df['sector'] == sector].copy()
        dates = sector_df['fecha'].unique()
        prev_top = set()
        for date in dates:
            day_df = sector_df[sector_df['fecha'] == date].sort_values('wls', ascending=False)
            top_tickers = set(day_df.head(TOP_N)['ticker'])
            if prev_top:
                turnover = 1 - len(top_tickers & prev_top) / len(top_tickers)
                turn_data.append({'sector': sector, 'fecha': date, 'turnover': turnover})
            prev_top = top_tickers
    if not turn_data:
        return np.nan
    return pd.DataFrame(turn_data)['turnover'].mean()

def rank_persistence(df_leaders):
    """Persistencia del ranking: correlación de Spearman entre rankings consecutivos."""
    df = df_leaders.sort_values('fecha')
    sectors = df['sector'].unique()
    corrs = []
    for sector in sectors:
        sector_df = df[df['sector'] == sector].copy()
        dates = sector_df['fecha'].unique()
        for i in range(len(dates)-1):
            date1 = dates[i]
            date2 = dates[i+1]
            df1 = sector_df[sector_df['fecha'] == date1][['ticker', 'wls']].set_index('ticker')
            df2 = sector_df[sector_df['fecha'] == date2][['ticker', 'wls']].set_index('ticker')
            common = df1.index.intersection(df2.index)
            if len(common) < 3:
                continue
            rho = df1.loc[common, 'wls'].corr(df2.loc[common, 'wls'], method='spearman')
            corrs.append(rho)
    return np.nanmean(corrs) if corrs else np.nan

# =========================================================
# GENERACIÓN DE INFORME (TEXTO)
# =========================================================

def generar_informe(resultados_globales, resultados_por_bucket, turnover, persistence, output_file):
    """Escribe un archivo de texto con resultados globales y por capitalización."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== VALIDACIÓN FORWARD DEL WYCKOFF LEADERSHIP SCORE (WLS) ===\n")
        f.write(f"Fecha del informe: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Sección global
        f.write("=== RESULTADOS GLOBALES ===\n")
        for h, (_, metrics) in resultados_globales.items():
            f.write(f"\n--- HORIZONTE {h} DÍAS ---\n")
            f.write(f"Número de observaciones: {metrics['num_observaciones']}\n")
            f.write(f"Hit rate absoluto (retorno > 0): {metrics['hit_rate_absoluto']:.2%}\n")
            if not pd.isna(metrics['hit_rate_relativo']):
                f.write(f"Hit rate relativo (vs ETF): {metrics['hit_rate_relativo']:.2%}\n")
            if not pd.isna(metrics['alpha_medio']):
                f.write(f"Alpha medio (vs ETF): {metrics['alpha_medio']:.5f}\n")
            f.write(f"Retorno medio absoluto: {metrics['retorno_medio_absoluto']:.5f}\n")
            if not pd.isna(metrics['retorno_medio_ETF']):
                f.write(f"Retorno medio ETF: {metrics['retorno_medio_ETF']:.5f}\n")
        
        # Sección por capitalización
        f.write("\n=== RESULTADOS POR CAPITALIZACIÓN ===\n")
        for bucket in ['Large', 'Mid', 'Small']:
            f.write(f"\n--- BUCKET: {bucket} ---\n")
            for h in HORIZONS:
                metrics = resultados_por_bucket.get(bucket, {}).get(h, ({}, {}))[1]
                if not metrics or metrics.get('num_observaciones', 0) == 0:
                    f.write(f"Horizonte {h}: sin datos suficientes.\n")
                    continue
                f.write(f"Horizonte {h} días:\n")
                f.write(f"  Observaciones: {metrics['num_observaciones']}\n")
                f.write(f"  Hit rate absoluto: {metrics['hit_rate_absoluto']:.2%}\n")
                if not pd.isna(metrics.get('hit_rate_relativo', np.nan)):
                    f.write(f"  Hit rate relativo: {metrics['hit_rate_relativo']:.2%}\n")
                if not pd.isna(metrics.get('alpha_medio', np.nan)):
                    f.write(f"  Alpha medio: {metrics['alpha_medio']:.5f}\n")
        
        # Métricas adicionales
        f.write("\n=== MÉTRICAS ADICIONALES ===\n")
        f.write(f"Turnover mensual (top3): {turnover:.2%}\n")
        f.write(f"Persistencia del ranking (Spearman media): {persistence:.3f}\n")
        f.write("\nInterpretación:\n")
        f.write("- Hit rate > 50% → poder predictivo.\n")
        f.write("- Alpha medio positivo → top3 supera al ETF.\n")
        f.write("- Turnover < 30% → ranking estable.\n")
        f.write("- Persistencia > 0.5 → consistencia temporal.\n")
    
    print(f"Informe guardado en {output_file}")

# =========================================================
# MAIN
# =========================================================

def main():
    print("=== Validación forward del módulo de líderes (con desglose por capitalización) ===\n")
    try:
        # Cargar datos
        print("Cargando líderes históricos...")
        df_leaders = load_historical_leaders()
        print(f"Registros: {len(df_leaders)}, fechas: {df_leaders['fecha'].nunique()}")
        if df_leaders.empty:
            raise ValueError("No hay datos suficientes.")
        
        print("Cargando precios de mercado...")
        prices = load_price_data()
        print(f"Precios: {prices.shape[0]} días, {prices.shape[1]} tickers")
        
        holdings = load_holdings()
        
        # Clasificar por capitalización
        unique_tickers = df_leaders['ticker'].unique()
        print("Obteniendo capitalizaciones (puede tardar unos segundos)...")
        bucket_dict = classify_by_market_cap(unique_tickers)
        from collections import Counter
        bucket_counts = Counter(bucket_dict.values())
        print("Distribución por capitalización:", dict(bucket_counts))
        
        # Evaluar global y por bucket
        resultados_globales = {}
        resultados_por_bucket = {bucket: {} for bucket in ['Large', 'Mid', 'Small']}
        
        for h in HORIZONS:
            print(f"\nProcesando horizonte {h} días...")
            # Global
            df_res, metrics = evaluate_horizon(df_leaders, prices, holdings, h)
            if not df_res.empty:
                resultados_globales[h] = (df_res, metrics)
                print(f"  Global: obs={metrics['num_observaciones']}, hit_abs={metrics['hit_rate_absoluto']:.2%}")
            else:
                print(f"  Global: sin datos")
            
            # Por bucket
            res_bucket = evaluate_horizon_by_bucket(df_leaders, prices, holdings, h, bucket_dict)
            for bucket in ['Large', 'Mid', 'Small']:
                df_b, met_b = res_bucket[bucket]
                resultados_por_bucket[bucket][h] = (df_b, met_b)
                if met_b:
                    print(f"  {bucket}: obs={met_b.get('num_observaciones',0)}, hit_abs={met_b.get('hit_rate_absoluto',0):.2%}")
        
        if not resultados_globales:
            print("\nNo se obtuvieron resultados. ¿Faltan datos históricos?")
            return
        
        # Métricas adicionales globales
        turnover = turnover_analysis(df_leaders)
        persistence = rank_persistence(df_leaders)
        
        # Generar informe
        output_file = OUTPUT_DIR / f"validacion_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        generar_informe(resultados_globales, resultados_por_bucket, turnover, persistence, output_file)
        
        print(f"\nValidación completada.\nInforme en {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()