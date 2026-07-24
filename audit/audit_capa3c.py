# Auditoría Maestra v3.15 - Capa 3C: Correlación entre Outputs Finales del SLPM
# Mide dependencia monótona entre los componentes internos del SLPM
# (Breadth, Flow Divergence) y los scores externos (Tactical, Structural).
# Evalúa si el SLPM añade información diferenciada o replica señales existentes.

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("CAPA 3C: Correlación Spearman entre Outputs del SLPM")
print("=" * 70)

# 1. Cargar historial
print("\n[1/3] Cargando slpm_history.csv ...")
hist = pd.read_csv("outputs/slpm_history.csv", parse_dates=["date"])
print(f"  Registros: {len(hist)}")
print(f"  Rango: {hist['date'].min().date()} a {hist['date'].max().date()}")
print(f"  Columnas: {list(hist.columns)}")

# 2. Seleccionar columnas numéricas relevantes
cols_interes = ["tactical_score", "structural_score", "leader_breadth", "flow_divergence"]
cols_disponibles = [c for c in cols_interes if c in hist.columns]
if len(cols_disponibles) < 2:
    print("ERROR: Faltan columnas necesarias en slpm_history.csv")
    exit()

print(f"\n[2/3] Analizando {len(cols_disponibles)} variables: {cols_disponibles}")
data = hist[cols_disponibles].dropna()
print(f"  Muestra con datos completos: {len(data)} filas")

# 3. Matriz de correlación Spearman
print("[3/3] Calculando matriz Spearman ...")
corr = data.corr(method="spearman")

print("\n" + "=" * 70)
print("MATRIZ DE CORRELACIÓN SPEARMAN ENTRE OUTPUTS DEL SLPM")
print("=" * 70)
print(corr.round(3).to_string())

# --- Interpretación ---
print("\n" + "=" * 70)
print("INTERPRETACIÓN DE DEPENDENCIA")
print("=" * 70)

alertas = []
pares = []
for i, c1 in enumerate(corr.columns):
    for j, c2 in enumerate(corr.columns):
        if i < j:
            r = corr.loc[c1, c2]
            pares.append((c1, c2, r))
            if abs(r) > 0.70:
                alertas.append((c1, c2, r))

if alertas:
    print("\nALERTAS: Correlaciones ALTAS (>0.70) entre componentes del SLPM:")
    for c1, c2, r in alertas:
        print(f"  {c1}  <->  {c2}: {r:+.3f}")
    print("RIESGO: Estos componentes miden esencialmente lo mismo.")
    print("Posible redundancia en el SLPM.")
else:
    print("\nNo se detectaron correlaciones >0.70 entre componentes del SLPM.")
    print("Los componentes parecen aportar información diferenciada.")

# Comparación específica Breadth vs Flow Divergence
if "leader_breadth" in corr.columns and "flow_divergence" in corr.columns:
    r_bf = corr.loc["leader_breadth", "flow_divergence"]
    print(f"\nBreadth vs Flow Divergence: {r_bf:+.3f}")
    if abs(r_bf) < 0.30:
        print("Son señales claramente independientes. Buen diseño del SLPM.")
    elif abs(r_bf) < 0.50:
        print("Tienen cierta relación, pero aún aportan información diferenciada.")
    else:
        print("Alta dependencia: podrían estar midiendo aspectos similares del liderazgo.")

# Comparación Tactical vs Structural
if "tactical_score" in corr.columns and "structural_score" in corr.columns:
    r_ts = corr.loc["tactical_score", "structural_score"]
    print(f"\nTactical vs Structural: {r_ts:+.3f}")
    if abs(r_ts) < 0.30:
        print("Corto y largo plazo claramente diferenciados. Arquitectura sólida.")
    elif abs(r_ts) < 0.50:
        print("Cierta relación, pero los motores táctico y estructural son razonablemente distintos.")
    else:
        print("Alta dependencia: el motor estructural podría estar replicando señales tácticas.")

print("\nNota: Correlación no implica causalidad ni redundancia absoluta.")
print("Una correlación moderada (0.40-0.60) es aceptable en un sistema multivariable.")
print("Solo correlaciones >0.70 sugieren posible doble conteo.")
print("=" * 70)
