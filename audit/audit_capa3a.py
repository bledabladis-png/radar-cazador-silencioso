# Auditoría Maestra v3.15 - Capa 3A: Correlación Pearson entre Señales Primarias
# Mide dependencia lineal entre RS20, Flow Proxy, Momentum y Trend.
# No prueba independencia ni causalidad. Solo diagnóstico de asociación.

import pandas as pd
import numpy as np
import os
from indicators.momentum import compute_flow_proxy
from indicators.trend import trend_position
from src.utils import get_col
from config.tickers import MARKET_TICKERS

sectores = MARKET_TICKERS["sectors"]
bench_ticker = "^GSPC"
min_periodos = 63  # mínimo para cálculos fiables

print("=" * 70)
print("CAPA 3A: Correlación Pearson entre Señales Primarias")
print("=" * 70)

# 1. Cargar datos
print("\n[1/4] Cargando market_data.csv ...")
df = pd.read_csv("data/market_data.csv", header=[0,1], index_col=0, parse_dates=True)
bench = get_col(df, bench_ticker, "Close")

# 2. Calcular señales para cada sector
print("[2/4] Calculando RS20, Flow, Momentum y Trend para cada sector ...")
rs20_data, flow_data, mom_data, trend_data = {}, {}, {}, {}
errores = {}

for s in sectores:
    try:
        close = get_col(df, s, "Close")
        ret = close.pct_change(fill_method=None)
        rs = close / bench

        rs20_data[s] = rs.pct_change(20)
        flow_data[s] = compute_flow_proxy(df, s)
        mom_data[s] = ret.rolling(63).mean() / (ret.rolling(63).std() + 1e-9)
        trend_data[s] = trend_position(close)
    except Exception as e:
        errores[s] = str(e)

if errores:
    print(f"  Aviso: {len(errores)} sector(es) con errores: {list(errores.keys())}")

# 3. Construir DataFrame combinado
print("[3/4] Construyendo matriz combinada ...")
combined = pd.DataFrame()
for s in sectores:
    if s in rs20_data: combined[f"RS20_{s}"] = rs20_data[s]
    if s in flow_data: combined[f"Flow_{s}"] = flow_data[s]
    if s in mom_data:  combined[f"Momentum_{s}"] = mom_data[s]
    if s in trend_data: combined[f"Trend_{s}"] = trend_data[s]

combined.dropna(how="all", inplace=True)
print(f"  Dimensiones: {combined.shape[0]} filas x {combined.shape[1]} columnas")
print(f"  Rango de fechas: {combined.index[0].date()} a {combined.index[-1].date()}")

# 4. Matriz de correlación Pearson
print("[4/4] Calculando matriz de correlación Pearson ...")
corr = combined.corr()
os.makedirs("outputs", exist_ok=True)
corr.to_csv("outputs/corr_pearson_señales.csv")
print("  Guardada en outputs/corr_pearson_señales.csv")

# --- RESUMEN INTERPRETATIVO ---
print("\n" + "=" * 70)
print("RESUMEN DE DEPENDENCIA LINEAL")
print("=" * 70)

cols = list(corr.columns)
n_cols = len(cols)

# 5. Correlaciones altas entre tipos distintos de señal
print("\nCorrelaciones ELEVADAS (>0.80) entre tipos distintos de señal:")
elevadas = []
for i in range(n_cols):
    for j in range(i+1, n_cols):
        tipo_i = cols[i].split("_")[0]
        tipo_j = cols[j].split("_")[0]
        if tipo_i != tipo_j:
            val = corr.iloc[i, j]
            if abs(val) > 0.80:
                elevadas.append((cols[i], cols[j], val))
if elevadas:
    for a, b, v in elevadas:
        print(f"  {a}  vs  {b}: {v:+.3f}")
else:
    print("  Ninguna correlación >0.80 entre tipos distintos.")

# 6. Correlación promedio intra-tipo
print("\nCorrelación promedio entre sectores para cada tipo de señal:")
for tipo in ["RS20", "Flow", "Momentum", "Trend"]:
    sub = [c for c in cols if c.startswith(tipo + "_")]
    if len(sub) > 1:
        vals = corr.loc[sub, sub].values
        avg = vals[np.triu_indices(len(sub), k=1)].mean()
        print(f"  {tipo}: {avg:+.3f} ({len(sub)} sectores)")

# 7. Interpretación
print("\nINTERPRETACIÓN (no concluyente):")
print("  - Correlaciones >0.80 entre tipos distintos: posible redundancia.")
print("  - Correlaciones intra-tipo altas: los sectores se mueven sincronizados en esa señal.")
print("  - Esto NO prueba dependencia causal ni doble conteo. Requiere Capas 3B y 3C.")
print("\nCapacidad informativa incremental: NO evaluada aún (pendiente Spearman y outputs).")
print("=" * 70)
