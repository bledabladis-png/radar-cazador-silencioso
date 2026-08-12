# -*- coding: utf-8 -*-
"""Backtest de umbrales del SLPM v1.2"""
import pandas as pd
import numpy as np

hist = pd.read_csv("outputs/history/slpm_history.csv", parse_dates=["date"])
print(f"Registros: {len(hist)}")

# Distribucion de estados
state_counts = hist["state"].value_counts()
for state, count in state_counts.items():
    print(f"  {state}: {count} semanas ({count/len(hist)*100:.1f}%)")

# Duracion media por estado
hist["state_block"] = (hist["state"] != hist["state"].shift(1)).cumsum()
for state in state_counts.index:
    durations = hist[hist["state"] == state].groupby("state_block").size()
    print(f"  {state}: media={durations.mean():.1f} semanas")

# Matriz de transicion
hist["prev_state"] = hist["state"].shift(1)
transitions = hist.dropna(subset=["prev_state"])
matrix = pd.crosstab(transitions["prev_state"], transitions["state"])
print(matrix.to_string())
