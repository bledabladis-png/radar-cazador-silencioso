# Auditoría Maestra v3.15 - Capa 3D CORREGIDA
# Solo procesa fechas con datos disponibles en ambos datasets.
import pandas as pd, numpy as np, os, sys, warnings
warnings.filterwarnings("ignore")
from config.tickers import MARKET_TICKERS, SECTOR_NAMES
from src.utils import get_col
from indicators.slpm_v12 import evaluate_slpm_v12
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from indicators.persistence import compute_persistence

sectores = MARKET_TICKERS["sectors"]
bench_ticker = "^GSPC"
holdings_df = pd.read_csv("data/etf_holdings.csv")

print("=" * 70)
print("CAPA 3D CORREGIDA: SLPM v1.2 con líderes reales")
print("=" * 70)

# 1. Cargar datos
print("\n[1/5] Cargando datos ...")
df_market = pd.read_csv("data/market_data.csv", header=[0,1], index_col=0, parse_dates=True)
df_market = df_market.asfreq("B").ffill().dropna(how="all")
print(f"  market_data: {len(df_market)} días ({df_market.index[0].date()} a {df_market.index[-1].date()})")

df_stocks = pd.read_csv("data/stock_prices_historical.csv", header=[0,1], index_col=0, parse_dates=True)
df_stocks = df_stocks.asfreq("B").ffill().dropna(how="all")
print(f"  stock_prices: {len(df_stocks)} días ({df_stocks.index[0].date()} a {df_stocks.index[-1].date()})")

# Encontrar rango común
start = max(df_market.index[0], df_stocks.index[0])
end = min(df_market.index[-1], df_stocks.index[-1])
print(f"  Rango común: {start.date()} a {end.date()}")

# Filtrar ambos al rango común
df_market = df_market.loc[start:end]
df_stocks = df_stocks.loc[start:end]
common_days = len(df_market)
print(f"  Días comunes: {common_days}")

# 2. Fechas de muestreo (cada 10 días para reducir carga)
fechas = df_market.index[::10]
fechas = [f for f in fechas if f >= start and f <= end]
print(f"\n[2/5] Procesando {len(fechas)} fechas (cada 10 días) ...")

resultados = []
errores_count = {"sin_lideres": 0, "slpm_fallo": 0, "otros": 0}

for i, fecha in enumerate(fechas):
    try:
        mkt = df_market.loc[:fecha]
        stk = df_stocks.loc[:fecha]
        if len(mkt) < 63 or len(stk) < 63:
            continue

        # Ranking sectorial
        sector_scores = {}
        for s in sectores:
            try:
                close_s = get_col(mkt, s, "Close")
                bench = get_col(mkt, bench_ticker, "Close")
                rs = close_s / bench
                sector_scores[s] = rs.pct_change(20).iloc[-1]
            except: sector_scores[s] = 0.0
        ranking = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        sector_results = {"ranking": [(s, SECTOR_NAMES.get(s,s), v, "") for s,v in ranking]}
        top_sector_etf = ranking[0][0]

        # Tactical/Structural scores
        tactical_scores, structural_scores = {}, {}
        for s in sectores:
            try:
                tactical_scores[s] = compute_tactical_score(mkt, s)
                structural_scores[s] = compute_structural_score(mkt, s)
            except: tactical_scores[s]=0.0; structural_scores[s]=0.0

        # Persistence
        sector_persistence = {}
        for s in sectores:
            try:
                close_s = get_col(mkt, s, "Close"); bench_s = get_col(mkt, bench_ticker, "Close")
                rs_s = close_s / bench_s
                pers = compute_persistence(rs_s.pct_change(20), threshold=0.0, lookback=12)
                sector_persistence[s] = pers if pers is not None else 0.5
            except: sector_persistence[s] = 0.5

        # Líderes reales del top sector
        top_holdings = holdings_df[holdings_df["etf"] == top_sector_etf]["ticker"].tolist()
        leader_metrics = []
        for ticker in top_holdings:
            try:
                close_stk = get_col(stk, ticker, "Close")
                close_etf = get_col(mkt, top_sector_etf, "Close")
                if len(close_stk.dropna()) < 60: continue
                rs_stk = close_stk / close_etf
                rs_mom = np.log(rs_stk).diff(20).iloc[-1]
                ret_stk = close_stk.pct_change(fill_method=None)
                vol_stk = get_col(stk, ticker, "Volume")
                dollar_vol = close_stk * vol_stk
                flow_raw = ret_stk * dollar_vol
                flow_z = (flow_raw - flow_raw.rolling(60).median()) / (flow_raw.rolling(60).mad() + 1e-9)
                flow_signal = flow_z.ewm(span=5).mean().iloc[-1]
                leader_metrics.append({
                    "ticker": ticker,
                    "rs": rs_stk.iloc[-1] if pd.notna(rs_stk.iloc[-1]) else 1.0,
                    "rs_momentum": rs_mom if pd.notna(rs_mom) else 0.0,
                    "flow_z": flow_signal if pd.notna(flow_signal) else 0.0,
                    "wyckoff_phase": "RANGE"
                })
            except: pass

        if not leader_metrics:
            errores_count["sin_lideres"] += 1
            continue

        # SLPM v1.2
        slpm = evaluate_slpm_v12(
            mkt, sector_results, leader_metrics, 0.0,
            tactical_scores=tactical_scores,
            structural_scores=structural_scores,
            sector_persistence=sector_persistence
        )
        if slpm:
            lb = slpm.get("leader_breadth_v2", {})
            li = slpm.get("leader_integrity", {})
            fd = slpm.get("flow_divergence_v2", {})
            ins = slpm.get("input_scores", {})
            resultados.append({
                "fecha": fecha,
                "sector": slpm.get("sector",""),
                "effective_breadth": lb.get("effective_composite", 0.5),
                "lis": li.get("lis", 0.0),
                "flow_divergence_composite": fd.get("composite", 0.0),
                "tactical_score": ins.get("tactical", 0.0),
                "structural_score": ins.get("structural", 0.0),
                "persistence": ins.get("persistence", 0.5),
            })
        else:
            errores_count["slpm_fallo"] += 1

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(fechas)} fechas (válidas: {len(resultados)})")
    except Exception as e:
        errores_count["otros"] += 1

print(f"\n  Final: {len(resultados)} semanas válidas, errores: {errores_count}")

if len(resultados) < 20:
    print("ERROR: Muy pocos datos para calcular correlaciones fiables.")
    print("Posible causa: stock_prices.csv tiene pocas acciones o fechas limitadas.")
    sys.exit(1)

# 3. Matriz de correlación
print(f"[3/5] Calculando matriz Spearman ...")
df_out = pd.DataFrame(resultados).dropna()
print(f"  Registros válidos: {len(df_out)}")
cols = ["effective_breadth","lis","flow_divergence_composite","tactical_score","structural_score","persistence"]
data = df_out[cols]
corr = data.corr(method="spearman")
os.makedirs("outputs", exist_ok=True)
corr.to_csv("outputs/corr_slpm_v12_full.csv")
print("  Guardada en outputs/corr_slpm_v12_full.csv")

print("\n" + "=" * 70)
print("MATRIZ SPEARMAN - SLPM v1.2 con líderes reales")
print("=" * 70)
print(corr.round(3).to_string())

print("\nALERTAS (>0.70):")
alertas = []
for i,c1 in enumerate(corr.columns):
    for j,c2 in enumerate(corr.columns):
        if i<j:
            r = corr.loc[c1,c2]
            if abs(r) > 0.70:
                alertas.append((c1,c2,r))
if alertas:
    for c1,c2,r in alertas: print(f"  {c1} <-> {c2}: {r:+.3f}")
else:
    print("  Ninguna. SLPM v1.2 diversifica correctamente sus componentes.")
print("=" * 70)
