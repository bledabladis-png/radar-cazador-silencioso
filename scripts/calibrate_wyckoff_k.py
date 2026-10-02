"""Calibracion Fase 5b.2 - K del t_norm Wyckoff v1.3.

Protocolo: docs/auditoria/wyckoff/09_protocolo_fase_5b.md (v3).
Criterios corregidos:
  - H3/H4 intra-ticker (distribucion temporal por ticker, no cross-sectional).
  - Agregacion ticker-balanced (mediana entre tickers).
  - Validacion OOS: V1-V4 (bounds, saturacion, resolucion, no-patologico).
  - p05/p95/skew OOS pasan a diagnostico.
No toca codigo productivo.
"""
import numpy as np
import pandas as pd

GRID_1 = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SPLIT = 0.70

# Hard criteria calibracion
H1_MAX = 0.15
H2_MAX = 0.05
H3_MAX = -0.40
H4_MIN = 0.40

# Criterios validacion OOS
V2_MAX_90 = 0.15
V2_MAX_95 = 0.05
V3_MIN_IQR = 0.05


def load_prices():
    df = pd.read_parquet("data/stock_prices.parquet")
    tickers = sorted(set(c[1] for c in df.columns if c[0] == "Close"))
    return df, tickers


def compute_trend(df, ticker):
    close = df[("Close", ticker)].dropna()
    if len(close) < 200:
        return None
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    trend = (ma50 / (ma200 + 1e-9) - 1).dropna()
    return trend if len(trend) >= 200 else None


def per_ticker_metrics(trend, K):
    """Metricas temporales de t_norm para UN ticker."""
    t = np.tanh(trend / K)
    n = len(t)
    abs_t = t.abs()
    return {
        "pct_90": float((abs_t > 0.90).sum()) / n,
        "pct_95": float((abs_t > 0.95).sum()) / n,
        "p05": float(t.quantile(0.05)),
        "p95": float(t.quantile(0.95)),
        "iqr": float(t.quantile(0.75) - t.quantile(0.25)),
        "skew": float(t.skew()),
        "n": n,
    }


def ticker_balanced_aggregate(trends_by_ticker, K):
    """Agregacion ticker-balanced: mediana entre tickers."""
    per_ticker = []
    for trend in trends_by_ticker.values():
        m = per_ticker_metrics(trend, K)
        per_ticker.append(m)
    if not per_ticker:
        return None
    keys = ["pct_90", "pct_95", "p05", "p95", "iqr", "skew"]
    out = {k: float(np.median([m[k] for m in per_ticker])) for k in keys}
    out["n_tickers"] = len(per_ticker)
    return out


def check_hard_cal(m):
    fails = []
    if m["pct_90"] >= H1_MAX: fails.append("H1")
    if m["pct_95"] >= H2_MAX: fails.append("H2")
    if m["p05"] > H3_MAX:    fails.append("H3")
    if m["p95"] < H4_MIN:    fails.append("H4")
    return fails


def check_oos_v3(m):
    """Validacion OOS: V1-V4. V1 (bounds) implicito en tanh."""
    fails = []
    if m["pct_90"] >= V2_MAX_90: fails.append("V2_90")
    if m["pct_95"] >= V2_MAX_95: fails.append("V2_95")
    if m["iqr"] < V3_MIN_IQR:    fails.append("V3_iqr")
    # V4 no-patologico: n_tickers > 0 y valores finitos
    if m["n_tickers"] < 10:      fails.append("V4_ntk")
    for k in ["pct_90", "p05", "p95", "iqr"]:
        if not np.isfinite(m[k]):  fails.append("V4_nan")
    return fails


def main():
    print("=" * 100)
    print("FASE 5b.2 - CALIBRACION K (protocolo v3)")
    print("=" * 100)

    df, tickers = load_prices()
    print(f"\nTickers en parquet: {len(tickers)}")

    last_dates = []
    for tk in tickers:
        try:
            close = df[("Close", tk)].dropna()
            if len(close) > 0:
                last_dates.append(close.index[-1])
        except (KeyError, ValueError):
            pass
    last_common = min(last_dates) if last_dates else None
    print(f"Ultima sesion cerrada (min): {last_common}")

    trends_by_ticker = {}
    for tk in tickers:
        trend = compute_trend(df, tk)
        if trend is not None:
            trends_by_ticker[tk] = trend
    print(f"Tickers con trend valido (>=200 obs): {len(trends_by_ticker)}")

    all_dates = sorted(set().union(*[set(t.index) for t in trends_by_ticker.values()]))
    cutoff_idx = int(len(all_dates) * SPLIT)
    cutoff_date = all_dates[cutoff_idx]
    print(f"Split temporal 70/30 en: {cutoff_date}")

    cal_trends = {tk: t[t.index <= cutoff_date] for tk, t in trends_by_ticker.items()}
    val_trends = {tk: t[t.index > cutoff_date] for tk, t in trends_by_ticker.items()}
    cal_trends = {tk: t for tk, t in cal_trends.items() if len(t) >= 100}
    val_trends = {tk: t for tk, t in val_trends.items() if len(t) >= 50}
    print(f"Tickers en calibracion: {len(cal_trends)}, en validacion: {len(val_trends)}")

    # --- Calibracion ---
    print("\n" + "=" * 100)
    print("CALIBRACION (70% inicial) - H3/H4 intra-ticker, balanced")
    print("=" * 100)
    print(f"\n{'K':>6} | {'pct_90':>8} {'pct_95':>8} {'p05':>8} {'p95':>8} | {'H1-H4':>10} | {'iqr':>6} {'skew':>7}")
    print("-" * 100)
    cal_results = {}
    for K in GRID_1:
        m = ticker_balanced_aggregate(cal_trends, K)
        if m is None: continue
        fails = check_hard_cal(m)
        status = "PASA" if not fails else ",".join(fails)
        cal_results[K] = (m, fails)
        print(f"{K:>6.2f} | {m['pct_90']:>8.4f} {m['pct_95']:>8.4f} {m['p05']:>+8.4f} {m['p95']:>+8.4f} | {status:>10} | {m['iqr']:>6.3f} {m['skew']:>+7.3f}")

    passing = [K for K, (_, f) in cal_results.items() if not f]
    if not passing:
        print("\nNINGUN K pasa H1-H4 en calibracion. 5b.2 FALLA. Grid 5b.3.")
        return
    K_sel = max(passing)
    print(f"\nK seleccionado (mayor de los que pasan H1-H4): {K_sel}")

    # --- Validacion OOS V1-V4 ---
    print("\n" + "=" * 100)
    print(f"VALIDACION OOS (30% final) con K={K_sel}")
    print("=" * 100)
    m_val = ticker_balanced_aggregate(val_trends, K_sel)
    if m_val is None:
        print("Sin datos de validacion.")
        return
    fails_val = check_oos_v3(m_val)
    print(f"\n{'criterio':>12} | {'valor':>10} | {'umbral':>14} | {'estado':>8}")
    print("-" * 100)
    print(f"{'V2_pct_90':>12} | {m_val['pct_90']:>10.4f} | {'< 15%':>14} | {'OK' if m_val['pct_90'] < V2_MAX_90 else 'FALLA':>8}")
    print(f"{'V2_pct_95':>12} | {m_val['pct_95']:>10.4f} | {'< 5%':>14} | {'OK' if m_val['pct_95'] < V2_MAX_95 else 'FALLA':>8}")
    print(f"{'V3_iqr_cs':>12} | {m_val['iqr']:>10.4f} | {'>= 0.05':>14} | {'OK' if m_val['iqr'] >= V3_MIN_IQR else 'FALLA':>8}")
    print(f"{'V4_n_tk':>12} | {m_val['n_tickers']:>10} | {'>= 10':>14} | {'OK' if m_val['n_tickers'] >= 10 else 'FALLA':>8}")

    print(f"\nDiagnosticos OOS (no fallo):")
    print(f"  p05    = {m_val['p05']:+.4f}")
    print(f"  p95    = {m_val['p95']:+.4f}")
    print(f"  skew   = {m_val['skew']:+.4f}")

    # --- Sensibilidad pooled vs balanced ---
    print("\n" + "=" * 100)
    print(f"SENSIBILIDAD balanced vs pooled (K={K_sel})")
    print("=" * 100)
    pooled_t = pd.concat([np.tanh(t / K_sel) for t in cal_trends.values()])
    print(f"\n{'metrica':>10} | {'balanced':>10} | {'pooled':>10}")
    print("-" * 100)
    print(f"{'pct_90':>10} | {cal_results[K_sel][0]['pct_90']:>10.4f} | {float((pooled_t.abs()>0.90).mean()):>10.4f}")
    print(f"{'pct_95':>10} | {cal_results[K_sel][0]['pct_95']:>10.4f} | {float((pooled_t.abs()>0.95).mean()):>10.4f}")
    print(f"{'p05':>10} | {cal_results[K_sel][0]['p05']:>+10.4f} | {float(pooled_t.quantile(0.05)):>+10.4f}")
    print(f"{'p95':>10} | {cal_results[K_sel][0]['p95']:>+10.4f} | {float(pooled_t.quantile(0.95)):>+10.4f}")
    print(f"{'iqr':>10} | {cal_results[K_sel][0]['iqr']:>10.4f} | {float(pooled_t.quantile(0.75)-pooled_t.quantile(0.25)):>10.4f}")

    # --- Veredicto ---
    print("\n" + "=" * 100)
    print("VEREDICTO 5b.2")
    print("=" * 100)
    if not fails_val:
        print(f"Fase 5b.2 EXITOSA. K congelable = {K_sel}")
        print(f"  H1-H4 pasan en calibracion (intra-ticker, balanced).")
        print(f"  V1-V4 pasan en validacion OOS.")
        print(f"  Proximo paso: congelar K en config + proveniencia, desbloquear 5c.")
    else:
        print(f"Fase 5b.2 FALLA en validacion OOS: {fails_val}")
        print(f"  K={K_sel} no generaliza a los criterios OOS V1-V4.")


if __name__ == "__main__":
    main()
