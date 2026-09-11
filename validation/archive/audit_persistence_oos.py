
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.persistence import compute_persistence
from indicators.momentum import compute_flow_proxy
from regimes.structural_engine import compute_structural_score
from indicators.state_machine import classify_leadership_state
from src.utils import get_col


def flatten_tickers(mapping):
    tickers = set()
    for value in mapping.values():
        if isinstance(value, str):
            tickers.add(value)
        elif isinstance(value, list):
            tickers.update(value)
        elif isinstance(value, dict):
            tickers.update(flatten_tickers(value))
    return sorted(tickers)


router = DataRouter()
all_tickers = flatten_tickers(MARKET_TICKERS)
data = router.get_market_data(all_tickers, period="5y")

close_spy = get_col(data, "^GSPC", "Close")
close_vix = get_col(data, "^VIX", "Close")
fechas = data.index.sort_values()

# Definir períodos OOS
n = len(fechas)
periodos = {
    "P1": fechas[int(n*0.55):int(n*0.70)],
    "P2": fechas[int(n*0.70):int(n*0.85)],
    "P3": fechas[int(n*0.85):],
}

resultados = []

for nombre, fechas_periodo in periodos.items():
    filas = []
    for i, fecha in enumerate(fechas_periodo):
        # Necesitamos futuro de 20 días
        pos = fechas.get_loc(fecha)
        if pos + 20 >= len(fechas):
            continue

        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue

        for sector in MARKET_TICKERS["sectors"]:
            try:
                close = get_col(df_hasta, sector, "Close")
                close_spy_local = get_col(df_hasta, "^GSPC", "Close")
                rs = close / close_spy_local
                rs20 = rs.pct_change(20, fill_method=None)

                persistence = compute_persistence(rs20, threshold=0.0, lookback=12)
                if persistence is None:
                    continue

                flow = compute_flow_proxy(df_hasta, sector).iloc[-1]
                structural = compute_structural_score(
                    df_hasta, sector,
                    leader_breadth=0.5,
                    flow_structure=flow,
                    persistence=persistence,
                )

                breadth = 0.5  # proxy fijo, documentado

                # Evento externo futuro
                spy_current = close_spy.iloc[pos]
                vix_current = close_vix.iloc[pos]
                spy_future = close_spy.iloc[pos + 20]
                vix_future = close_vix.iloc[pos + 20]

                spy_return = (spy_future / spy_current) - 1
                vix_change = (vix_future / vix_current) - 1
                target = 1 if (spy_return <= -0.10 or vix_change >= 0.20) else 0

                # Estado con y sin persistence directa
                state_con = classify_leadership_state(structural, breadth, persistence, coverage=1.0)["state"]
                state_sin = classify_leadership_state(structural, breadth, 0.0, coverage=1.0)["state"]

                filas.append({
                    "structural": structural,
                    "breadth": breadth,
                    "persistence": persistence,
                    "target": target,
                    "state_con": state_con,
                    "state_sin": state_sin,
                })
            except Exception:
                continue

    df = pd.DataFrame(filas).dropna()
    if df.empty:
        continue

    # Ablación: cambios de estado
    change_pct = (df["state_con"] != df["state_sin"]).mean()

    # Información incremental sobre evento externo
    y = df["target"].values
    if df["target"].nunique() > 1:
        X_base = df[["structural", "breadth"]].values
        X_full = df[["structural", "breadth", "persistence"]].values

        model_base = LogisticRegression(max_iter=1000).fit(X_base, y)
        model_full = LogisticRegression(max_iter=1000).fit(X_full, y)

        auc_base = roc_auc_score(y, model_base.predict_proba(X_base)[:, 1])
        auc_full = roc_auc_score(y, model_full.predict_proba(X_full)[:, 1])
        delta_auc = auc_full - auc_base
    else:
        auc_base = np.nan
        auc_full = np.nan
        delta_auc = np.nan

    resultados.append({
        "periodo": nombre,
        "n_registros": len(df),
        "cambio_estado_pct": change_pct,
        "auc_base": auc_base,
        "auc_con_persistence": auc_full,
        "delta_auc": delta_auc,
    })

    print(f"\n===== {nombre} =====")
    print(f"Registros: {len(df)}")
    print(f"Cambio de estado al quitar persistence directa: {change_pct:.1%}")
    print(f"AUC base (Structural+Breadth): {auc_base:.4f}")
    print(f"AUC con Persistence: {auc_full:.4f}")
    print(f"ΔAUC Persistence: {delta_auc:+.4f}")

# Guardar CSV resumen
resumen = pd.DataFrame(resultados)
out_dir = Path("outputs/audit")
out_dir.mkdir(parents=True, exist_ok=True)
resumen.to_csv(out_dir / "persistence_oos_analysis.csv", index=False)
print(f"\nInforme guardado en {out_dir / 'persistence_oos_analysis.csv'}")
