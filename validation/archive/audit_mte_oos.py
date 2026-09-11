
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from regimes.financial_conditions import compute_financial_conditions
from regimes.volatility_regime import compute_volatility_regime
from indicators.credit import credit_risk_signal
from indicators.mte import compute_mte
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
print("Shape datos:", data.shape)

close_spy = get_col(data, "^GSPC", "Close")
close_vix = get_col(data, "^VIX", "Close")
fechas = data.index.sort_values()

n = len(fechas)
periodos = {
    "P1": fechas[int(n*0.55):int(n*0.70)],
    "P2": fechas[int(n*0.70):int(n*0.85)],
    "P3": fechas[int(n*0.85):],
}

resultados = []

for nombre, fechas_periodo in periodos.items():
    filas = []
    for fecha in fechas_periodo:
        pos = fechas.get_loc(fecha)
        if pos + 20 >= len(fechas):
            continue

        df_hasta = data.loc[:fecha]
        if len(df_hasta) < 200:
            continue

        try:
            financial_score, _, _ = compute_financial_conditions(df_hasta)
            fc_val = float(financial_score.iloc[-1]) if len(financial_score) > 0 else 0.0

            try:
                vix = get_col(df_hasta, "^VIX", "Close")
                vix_ret = vix.pct_change(fill_method=None)
                vol_z, _, _ = compute_volatility_regime(vix_ret)
                vol_val = float(vol_z.iloc[-1]) if len(vol_z) > 0 else 0.0
            except Exception:
                vol_val = 0.0

            credit_series = credit_risk_signal(df_hasta)
            credit_val = float(credit_series.iloc[-1]) if len(credit_series) > 0 else 0.0

            mte_result = compute_mte(
                df_hasta,
                financial_score,
                credit_series,
                vol_z if 'vol_z' in locals() else pd.Series([0.0]),
                pcr_data=None,
                darkpool_data=None,
            )
            if mte_result is None:
                continue

            spy_current = close_spy.iloc[pos]
            vix_current = close_vix.iloc[pos]
            spy_future = close_spy.iloc[pos + 20]
            vix_future = close_vix.iloc[pos + 20]

            spy_return = (spy_future / spy_current) - 1
            vix_change = (vix_future / vix_current) - 1
            target = 1 if (spy_return <= -0.10 or vix_change >= 0.20) else 0

            filas.append({
                "financial_score": fc_val,
                "credit": credit_val,
                "volatility": vol_val,
                "msi": mte_result.get("msi", 0),
                "ipi": mte_result.get("ipi", 0),
                "srs": mte_result.get("srs", 0),
                "shs": mte_result.get("shs", 0),
                "cls": mte_result.get("cls", 0),
                "ips": mte_result.get("ips", 0),
                "target": target,
            })
        except Exception:
            continue

    df = pd.DataFrame(filas).dropna()
    if df.empty or df["target"].nunique() < 2:
        continue

    base_cols = ["financial_score", "credit", "volatility"]
    mte_cols = ["msi", "ipi", "srs", "shs", "cls", "ips"]
    X_base = df[base_cols].values
    X_full = df[base_cols + mte_cols].values
    X_mte = df[mte_cols].values
    y = df["target"].values

    model_base = LogisticRegression(max_iter=1000).fit(X_base, y)
    model_full = LogisticRegression(max_iter=1000).fit(X_full, y)
    model_mte = LogisticRegression(max_iter=1000).fit(X_mte, y)

    auc_base = roc_auc_score(y, model_base.predict_proba(X_base)[:, 1])
    auc_full = roc_auc_score(y, model_full.predict_proba(X_full)[:, 1])
    auc_mte = roc_auc_score(y, model_mte.predict_proba(X_mte)[:, 1])
    delta_auc = auc_full - auc_base

    resultados.append({
        "periodo": nombre,
        "n_registros": len(df),
        "auc_base": auc_base,
        "auc_mte": auc_mte,
        "auc_full": auc_full,
        "delta_auc": delta_auc,
    })

    print(f"\n===== {nombre} =====")
    print(f"Registros: {len(df)}")
    print(f"AUC base: {auc_base:.4f}")
    print(f"AUC solo MTE: {auc_mte:.4f}")
    print(f"AUC base+MTE: {auc_full:.4f}")
    print(f"ΔAUC MTE: {delta_auc:+.4f}")

resumen = pd.DataFrame(resultados)
out_dir = Path("outputs/audit")
out_dir.mkdir(parents=True, exist_ok=True)
resumen.to_csv(out_dir / "mte_oos_analysis.csv", index=False)
print(f"\nInforme guardado en {out_dir / 'mte_oos_analysis.csv'}")
