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
print("CAPA 3D FINAL v3: SLPM v1.2 con líderes reales (corregido)")
print("=" * 70)

df_market = pd.read_csv("data/market_data.csv", header=[0,1], index_col=0, parse_dates=True)
df_stocks = pd.read_csv("data/stock_prices_historical.csv", header=[0,1], index_col=0, parse_dates=True)

start = max(df_market.index[0], df_stocks.index[0])
end = min(df_market.index[-1], df_stocks.index[-1])
df_market = df_market.loc[start:end]
df_stocks = df_stocks.loc[start:end]
print(f"Días comunes: {len(df_market)}")

fechas = df_market.index[::10]
resultados = []
omitidas_sin_lideres = 0
advertencias_metricas = 0

for i, fecha in enumerate(fechas):
    try:
        mkt = df_market.loc[:fecha]
        stk = df_stocks.loc[:fecha]
        if len(mkt) < 63:
            continue

        bench = get_col(mkt, bench_ticker, "Close")
        sector_scores = {}
        for s in sectores:
            try:
                close_s = get_col(mkt, s, "Close")
                rs = close_s / bench
                sector_scores[s] = rs.pct_change(20).iloc[-1]
            except:
                sector_scores[s] = 0.0
        ranking = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        top_etf = ranking[0][0]

        top_holdings = holdings_df[holdings_df["etf"] == top_etf]["ticker"].tolist()
        leader_metrics = []
        for ticker in top_holdings:
            try:
                close_stk = get_col(stk, ticker, "Close")
                close_etf = get_col(mkt, top_etf, "Close")
                if len(close_stk.dropna()) < 60:
                    continue

                # Métricas con fallback a 0.0 si fallan
                try:
                    rs_stk = close_stk / close_etf
                    rs_val = rs_stk.iloc[-1] if pd.notna(rs_stk.iloc[-1]) else 1.0
                    rs_mom = np.log(rs_stk).diff(20).iloc[-1]
                    if pd.isna(rs_mom): rs_mom = 0.0
                except:
                    rs_val = 1.0
                    rs_mom = 0.0
                    advertencias_metricas += 1

                try:
                    ret_stk = close_stk.pct_change(fill_method=None)
                    vol_stk = get_col(stk, ticker, "Volume")
                    dollar_vol = close_stk * vol_stk
                    flow_raw = ret_stk * dollar_vol
                    flow_z = (flow_raw - flow_raw.rolling(60).median()) / (flow_raw.rolling(60).mad() + 1e-9)
                    flow_signal = flow_z.ewm(span=5).mean().iloc[-1]
                    if pd.isna(flow_signal): flow_signal = 0.0
                except:
                    flow_signal = 0.0
                    advertencias_metricas += 1

                leader_metrics.append({
                    "ticker": ticker,
                    "rs": rs_val,
                    "rs_momentum": rs_mom,
                    "flow_z": flow_signal,
                    "wyckoff_phase": "RANGE"
                })
            except:
                pass

        if len(leader_metrics) < 3:
            omitidas_sin_lideres += 1
            continue

        sector_results = {"ranking": [(s, SECTOR_NAMES.get(s, s), v, "") for s, v in ranking]}
        tactical_scores = {}
        structural_scores = {}
        for s in sectores:
            try:
                tactical_scores[s] = compute_tactical_score(mkt, s)
                structural_scores[s] = compute_structural_score(mkt, s)
            except:
                tactical_scores[s] = 0.0
                structural_scores[s] = 0.0

        sector_persistence = {}
        for s in sectores:
            try:
                close_s = get_col(mkt, s, "Close")
                bench_s = get_col(mkt, bench_ticker, "Close")
                rs_s = close_s / bench_s
                pers = compute_persistence(rs_s.pct_change(20), threshold=0.0, lookback=12)
                sector_persistence[s] = pers if pers is not None else 0.5
            except:
                sector_persistence[s] = 0.5

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
                "sector": slpm.get("sector", ""),
                "effective_breadth": lb.get("effective_composite", 0.5),
                "lis": li.get("lis", 0.0),
                "flow_divergence_composite": fd.get("composite", 0.0),
                "tactical_score": ins.get("tactical", 0.0),
                "structural_score": ins.get("structural", 0.0),
                "persistence": ins.get("persistence", 0.5),
            })

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(fechas)} fechas (válidas: {len(resultados)})")
    except Exception as e:
        pass

print(f"\nSemanas válidas: {len(resultados)}")
print(f"Omitidas por falta de líderes: {omitidas_sin_lideres}")
print(f"Advertencias de métricas: {advertencias_metricas}")

if len(resultados) < 20:
    print("ERROR: Muy pocos datos para correlaciones fiables.")
    sys.exit(1)

df_out = pd.DataFrame(resultados).dropna()
cols = ["effective_breadth", "lis", "flow_divergence_composite", "tactical_score", "structural_score", "persistence"]
data = df_out[cols]
corr = data.corr(method="spearman")
corr.to_csv("outputs/corr_slpm_v12_full.csv")
print("Matriz guardada en outputs/corr_slpm_v12_full.csv")
print("\nMATRIZ SPEARMAN - SLPM v1.2 con líderes reales")
print(corr.round(3).to_string())
print("\nALERTAS (>0.70):")
alertas = [(c1, c2, corr.loc[c1, c2]) for i, c1 in enumerate(corr.columns) for j, c2 in enumerate(corr.columns) if i < j and abs(corr.loc[c1, c2]) > 0.70]
if alertas:
    for c1, c2, r in alertas:
        print(f"  {c1} <-> {c2}: {r:+.3f}")
else:
    print("  Ninguna. SLPM v1.2 diversifica correctamente sus componentes.")
print("=" * 70)
