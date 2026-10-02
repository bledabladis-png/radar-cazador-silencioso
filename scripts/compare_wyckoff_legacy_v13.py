"""Comparativa legacy (indicators/wyckoff.py) vs v1.3 (indicators/wyckoff_v1.py).

Fase 5c: diagnostico. No migra consumidores.
Genera:
  - Distribucion de fases por ticker: legacy vs v1.3.
  - Por sector (etf_holdings.csv): agrupacion.
  - Por indice (index_holdings.csv): agrupacion.
  - Distribucion de wyckoff_score (componente de rws_z).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(".").resolve()
import sys
sys.path.insert(0, str(ROOT))

from indicators import wyckoff as legacy
from indicators import wyckoff_v1 as v13


def load_prices():
    df = pd.read_parquet("data/stock_prices.parquet")
    tickers = sorted(set(c[1] for c in df.columns if c[0] == "Close"))
    return df, tickers


def load_holdings():
    etf = pd.read_csv("data/etf_holdings.csv")
    idx = pd.read_csv("data/index_holdings.csv")
    return etf, idx


def classify_one(df, ticker, module, build_df_func):
    """Clasifica una fase usando el modulo dado."""
    try:
        tdf = build_df_func(df, ticker)
        if tdf is None or tdf.empty:
            return None, None
        phase = module.classify_wyckoff_phase(tdf, ticker)
        try:
            combined, _, _, _, _, _, _ = module.wyckoff_score(tdf, ticker)
            score_last = float(combined.dropna().iloc[-1]) if not combined.dropna().empty else None
        except Exception:
            score_last = None
        return phase, score_last
    except Exception as e:
        return f"ERROR:{type(e).__name__}", None


def main():
    print("=" * 100)
    print("COMPARATIVA LEGACY vs v1.3 (Fase 5c)")
    print("=" * 100)

    df, tickers = load_prices()
    print(f"\nTickers en parquet: {len(tickers)}")

    etf_h, idx_h = load_holdings()

    # --- Comparativa por ticker ---
    rows = []
    for i, tk in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  procesando {i+1}/{len(tickers)}...")
        # legacy: build_ticker_df + classify
        try:
            tdf_leg = legacy.build_ticker_df(df, tk)
            phase_leg = legacy.classify_wyckoff_phase(tdf_leg, tk) if not tdf_leg.empty else None
            combined_leg, *_ = legacy.wyckoff_score(tdf_leg, tk) if not tdf_leg.empty else (None,)
            score_leg = float(combined_leg.dropna().iloc[-1]) if combined_leg is not None and not combined_leg.dropna().empty else None
        except Exception:
            phase_leg = None
            score_leg = None

        # v1.3
        try:
            tdf_v13 = v13.build_ticker_df(df, tk)
            phase_v13 = v13.classify_wyckoff_phase(tdf_v13, tk) if not tdf_v13.empty else None
            combined_v13, *_ = v13.wyckoff_score(tdf_v13, tk) if not tdf_v13.empty else (None,)
            score_v13 = float(combined_v13.dropna().iloc[-1]) if combined_v13 is not None and not combined_v13.dropna().empty else None
        except Exception:
            phase_v13 = None
            score_v13 = None

        rows.append({
            "ticker": tk,
            "phase_legacy": phase_leg,
            "phase_v13": phase_v13,
            "score_legacy": score_leg,
            "score_v13": score_v13,
            "phase_change": (phase_leg != phase_v13) if (phase_leg and phase_v13) else None,
            "score_delta": (score_v13 - score_leg) if (score_leg is not None and score_v13 is not None) else None,
        })

    res = pd.DataFrame(rows)

    # --- Distribucion de fases ---
    print("\n" + "=" * 100)
    print("DISTRIBUCION DE FASES - UNIVERSO COMPLETO")
    print("=" * 100)
    print("\nLegacy:")
    print(res["phase_legacy"].value_counts(dropna=False).to_string())
    print("\nv1.3:")
    print(res["phase_v13"].value_counts(dropna=False).to_string())

    # --- Cambios de fase ---
    cambios = res[res["phase_change"] == True]
    print(f"\nTickers con cambio de fase: {len(cambios)}/{len(res)} ({100*len(cambios)/len(res):.1f}%)")

    if len(cambios) > 0:
        print("\nTransiciones observadas (legacy -> v1.3), top 15:")
        trans = cambios.groupby(["phase_legacy", "phase_v13"]).size().sort_values(ascending=False).head(15)
        print(trans.to_string())

    # --- Score delta ---
    valid_scores = res.dropna(subset=["score_delta"])
    if not valid_scores.empty:
        print(f"\nScore delta (v1.3 - legacy) sobre {len(valid_scores)} tickers:")
        print(f"  media   = {valid_scores['score_delta'].mean():+.4f}")
        print(f"  mediana = {valid_scores['score_delta'].median():+.4f}")
        print(f"  std     = {valid_scores['score_delta'].std():.4f}")
        print(f"  min     = {valid_scores['score_delta'].min():+.4f}")
        print(f"  max     = {valid_scores['score_delta'].max():+.4f}")
        print(f"  |delta| > 0.30: {len(valid_scores[valid_scores['score_delta'].abs() > 0.30])}")

    # --- Por sector SPDR ---
    print("\n" + "=" * 100)
    print("DISTRIBUCION POR SECTOR SPDR (usa etf_holdings.csv)")
    print("=" * 100)
    etf_h_only = etf_h[etf_h["ticker"].isin(res["ticker"])]
    for etf, group in etf_h_only.groupby("etf"):
        tk_list = list(group["ticker"])
        sub = res[res["ticker"].isin(tk_list)]
        if sub.empty:
            continue
        n = len(sub)
        leg_accum = int((sub["phase_legacy"] == "ACCUMULATION").sum())
        v13_accum = int((sub["phase_v13"] == "ACCUMULATION").sum())
        leg_markup = int((sub["phase_legacy"] == "MARKUP").sum())
        v13_markup = int((sub["phase_v13"] == "MARKUP").sum())
        changes = int((sub["phase_change"] == True).sum())
        print(f"  {etf:6} ({n:3}): legacy AC={leg_accum:2} MK={leg_markup:2} | v1.3 AC={v13_accum:2} MK={v13_markup:2} | cambios={changes}")

    # --- Guardar CSV para inspeccion ---
    out_csv = "outputs/audit/wyckoff_comparativa_5c.csv"
    Path("outputs/audit").mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"\nCSV guardado: {out_csv}")


if __name__ == "__main__":
    main()
