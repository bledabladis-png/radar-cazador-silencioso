# Auditoría Maestra v3.15 - Capa 3D COMPLETA
# Recalcula SLPM v1.2 con líderes reales históricos.
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
print("CAPA 3D COMPLETA: SLPM v1.2 con líderes reales históricos")
print("=" * 70)

# 1. Cargar datos de mercado y acciones
print("\n[1/5] Cargando datos ...")
df_market = pd.read_csv("data/market_data.csv", header=[0,1], index_col=0, parse_dates=True)
df_market = df_market.asfreq("B").ffill().dropna(how="all")
print(f"  market_data: {len(df_market)} días")

df_stocks = pd.read_csv("data/stock_prices.csv", header=[0,1], index_col=0, parse_dates=True)
df_stocks = df_stocks.asfreq("B").ffill().dropna(how="all")
print(f"  stock_prices: {len(df_stocks)} días")

# 2. Preparar fechas semanales
fechas = df_market.index[::5]
print(f"\n[2/5] Procesando {len(fechas)} semanas ...")
resultados = []

for i, fecha in enumerate(fechas):
    try:
        # Filtrar hasta la fecha
        mkt = df_market.loc[:fecha]
        stk = df_stocks.loc[:fecha]
        if len(mkt) < 63: continue

        # Ranking sectorial simple basado en RS20
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
            except:
                tactical_scores[s] = 0.0; structural_scores[s] = 0.0

        # Persistence
        sector_persistence = {}
        for s in sectores:
            try:
                close_s = get_col(mkt, s, "Close"); bench_s = get_col(mkt, bench_ticker, "Close")
                rs_s = close_s / bench_s
                pers = compute_persistence(rs_s.pct_change(20), threshold=0.0, lookback=12)
                sector_persistence[s] = pers if pers is not None else 0.5
            except: sector_persistence[s] = 0.5

        # Líderes reales del sector top (usando stock_leader lógica simplificada)
        top_holdings = holdings_df[holdings_df["etf"] == top_sector_etf]["ticker"].tolist()
        leader_metrics = []
        for ticker in top_holdings:
            try:
                close_stk = get_col(stk, ticker, "Close")
                close_etf = get_col(mkt, top_sector_etf, "Close")
                if len(close_stk.dropna()) < 60: continue
                rs_stk = close_stk / close_etf
                rs_mom = np.log(rs_stk).diff(20).iloc[-1]
                # Flow proxy simplificado
                ret_stk = close_stk.pct_change(fill_method=None)
                vol_stk = get_col(stk, ticker, "Volume")
                dollar_vol = close_stk * vol_stk
                flow_raw = ret_stk * dollar_vol
                flow_z = (flow_raw - flow_raw.rolling(60).median()) / (flow_raw.rolling(60).mad() + 1e-9)
                flow_signal = flow_z.ewm(span=5).mean().iloc[-1]
                # Wyckoff simplificado (sin fase real, asumimos RANGE)
                wyckoff_phase = "RANGE"
                leader_metrics.append({
                    "ticker": ticker,
                    "rs": rs_stk.iloc[-1] if pd.notna(rs_stk.iloc[-1]) else 1.0,
                    "rs_momentum": rs_mom if pd.notna(rs_mom) else 0.0,
                    "flow_z": flow_signal if pd.notna(flow_signal) else 0.0,
                    "wyckoff_phase": wyckoff_phase
                })
            except: pass

        if not leader_metrics: continue

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
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(fechas)} semanas ...")
    except Exception as e:
        pass

if not resultados:
    print("ERROR: Sin resultados."); sys.exit(1)

# 3. Matriz de correlación
print(f"\n[3/5] Construyendo matriz con {len(resultados)} semanas ...")
df_out = pd.DataFrame(resultados).dropna()
print(f"  Semanas válidas: {len(df_out)}")
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
alertas = [(c1,c2,corr.loc[c1,c2]) for i,c1 in enumerate(corr.columns) for j,c2 in enumerate(corr.columns) if i<j and abs(corr.loc[c1,c2])>0.70]
if alertas:
    for c1,c2,r in alertas: print(f"  {c1} <-> {c2}: {r:+.3f}")
else:
    print("  Ninguna. SLPM v1.2 diversifica correctamente sus componentes.")
print("=" * 70)
