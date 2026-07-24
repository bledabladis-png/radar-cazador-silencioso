# Auditoría Maestra v3.15 - Capa 3B: Correlación Spearman entre Señales y Scores
# Mide dependencia monótona entre señales primarias y scores agregados.
# Si un score tiene correlación >0.70 con una sola señal, está dominado por ella.

import pandas as pd
import numpy as np
import os
from indicators.momentum import compute_flow_proxy
from indicators.trend import trend_position
from src.utils import get_col
from config.tickers import MARKET_TICKERS
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score

sectores = MARKET_TICKERS["sectors"]
bench_ticker = "^GSPC"
min_periodos = 63

print("=" * 70)
print("CAPA 3B: Correlación Spearman entre Señales y Scores Agregados")
print("=" * 70)

# 1. Cargar datos
print("\n[1/3] Cargando market_data.csv ...")
df = pd.read_csv("data/market_data.csv", header=[0,1], index_col=0, parse_dates=True)

# 2. Calcular señales primarias y scores para cada sector
print("[2/3] Calculando señales y scores (puede tardar unos segundos) ...")
resultados = []
for s in sectores:
    try:
        close = get_col(df, s, "Close")
        ret = close.pct_change(fill_method=None)
        bench = get_col(df, bench_ticker, "Close")
        rs = close / bench

        rs20 = rs.pct_change(20)
        flow = compute_flow_proxy(df, s)
        momentum = ret.rolling(63).mean() / (ret.rolling(63).std() + 1e-9)
        trend = trend_position(close)

        tactical = compute_tactical_score(df, s)
        structural = compute_structural_score(df, s)

        temp = pd.DataFrame({
            "RS20": rs20,
            "Flow": flow,
            "Momentum": momentum,
            "Trend": trend,
            "Tactical": pd.Series(tactical, index=close.index),
            "Structural": pd.Series(structural, index=close.index),
            "Sector": s
        })
        resultados.append(temp)
    except Exception as e:
        print(f"  Error en {s}: {e}")

if not resultados:
    print("  ERROR: No se pudo calcular ningún sector.")
    exit()

combined = pd.concat(resultados)
combined.dropna(inplace=True)
print(f"  Muestra: {len(combined)} filas, {combined['Sector'].nunique()} sectores")

# 3. Matriz de correlación Spearman
print("[3/3] Calculando matriz Spearman ...")

señales = ["RS20", "Flow", "Momentum", "Trend"]
scores = ["Tactical", "Structural"]

print("\n" + "=" * 70)
print("CORRELACIÓN SPEARMAN: Señal Primaria vs Score Agregado")
print("=" * 70)
print(f"{'Señal':<12} {'Score':<12} {'Spearman':>8} {'Interpretación'}")
print("-" * 60)

alertas = []
for señal in señales:
    for score in scores:
        r = combined[señal].corr(combined[score], method="spearman")
        if abs(r) > 0.70:
            nivel = "DOMINANCIA"
            alertas.append((señal, score, r))
        elif abs(r) > 0.50:
            nivel = "influencia moderada"
        elif abs(r) > 0.30:
            nivel = "influencia baja"
        else:
            nivel = "independiente"
        print(f"{señal:<12} {score:<12} {r:+.3f}    {nivel}")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN DE DOMINANCIA")
print("=" * 70)
if alertas:
    print("Señales que DOMINAN un score agregado (Spearman >0.70):")
    for señal, score, r in alertas:
        print(f"  {señal} -> {score}: {r:+.3f}")
    print("RIESGO: El score no está diversificando; casi replica la señal dominante.")
else:
    print("Ninguna señal individual domina los scores agregados.")
    print("Los scores compuestos sí están diversificando sus fuentes de información.")

print("\nNota: Spearman mide dependencia monótona, no lineal.")
print("Una correlación baja aquí no garantiza independencia total, pero sí sugiere")
print("que el score no es un simple reflejo de una única señal subyacente.")
print("=" * 70)
