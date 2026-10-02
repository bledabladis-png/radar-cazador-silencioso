"""Calibracion Fase 5b - K del t_norm Wyckoff v1.3.

Protocolo: docs/auditoria/wyckoff/09_protocolo_fase_5b.md (v2).
No toca codigo productivo. Solo lee stock_prices.parquet, calcula y
reporta. El resultado se pega y se analiza antes de congelar K.
"""
import numpy as np
import pandas as pd
from pathlib import Path

GRID_1 = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
H1_MAX = 0.15       # pct |t_norm| > 0.90
H2_MAX = 0.05       # pct |t_norm| > 0.95
H3_MAX = -0.40      # p05 <= -0.40
H4_MIN = 0.40       # p95 >= +0.40
SPLIT = 0.70        # calibracion = 70% inicial

def load_prices():
    df = pd.read_parquet("data/stock_prices.parquet")
    if not isinstance(df.columns, pd.MultiIndex):
        raise SystemExit("Parquet sin MultiIndex")
    tickers = sorted(set(c[1] for c in df.columns if c[0] == "Close"))
    return df, tickers

def compute_trend(df, ticker):
    close = df[("Close", ticker)].dropna()
    if len(close) < 200:
        return None
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    trend = (ma50 / (ma200 + 1e-9) - 1).dropna()
    return trend

def metrics_one_ticker(trend, K):
    """Metricas de t_norm para un ticker."""
    t_norm = np.tanh(trend / K)
    abs_t = t_norm.abs()
    n = len(t_norm)
    if n == 0:
        return None
    return {
        "pct_90": float((abs_t > 0.90).sum()) / n,
        "pct_95": float((abs_t > 0.95).sum()) / n,
        "max_abs": float(abs_t.max()),
        "p05": float(t_norm.quantile(0.05)),
        "p25": float(t_norm.quantile(0.25)),
        "p50": float(t_norm.quantile(0.50)),
        "p75": float(t_norm.quantile(0.75)),
        "p95": float(t_norm.quantile(0.95)),
        "iqr": float(t_norm.quantile(0.75) - t_norm.quantile(0.25)),
        "std": float(t_norm.std()),
        "skew": float(t_norm.skew()),
        "pct_pos": float((t_norm > 0).mean()),
        "n": n,
    }

def aggregate_balanced(trends_by_ticker, K):
    """Vista balanceada: agregar por ticker, luego agregar tickers."""
    per_ticker = []
    for tk, trend in trends_by_ticker.items():
        m = metrics_one_ticker(trend, K)
        if m is not None:
            per_ticker.append(m)
    if not per_ticker:
        return None
    keys = ["pct_90", "pct_95", "p05", "p25", "p50", "p75", "p95", "iqr",
            "std", "skew", "pct_pos"]
    out = {}
    for k in keys:
        out[k] = float(np.median([m[k] for m in per_ticker]))
    out["n_tickers"] = len(per_ticker)
    # max_abs es maximo de maximos
    out["max_abs"] = float(max(m["max_abs"] for m in per_ticker))
    return out

def aggregate_pooled(trends_by_ticker, K):
    """Vista pooled: concatenar todos los t_norm."""
    all_t = []
    for trend in trends_by_ticker.values():
        t_norm = np.tanh(trend / K)
        all_t.append(t_norm)
    pooled = pd.concat(all_t)
    n = len(pooled)
    abs_t = pooled.abs()
    return {
        "pct_90": float((abs_t > 0.90).sum()) / n,
        "pct_95": float((abs_t > 0.95).sum()) / n,
        "max_abs": float(abs_t.max()),
        "p05": float(pooled.quantile(0.05)),
        "p25": float(pooled.quantile(0.25)),
        "p50": float(pooled.quantile(0.50)),
        "p75": float(pooled.quantile(0.75)),
        "p95": float(pooled.quantile(0.95)),
        "iqr": float(pooled.quantile(0.75) - pooled.quantile(0.25)),
        "std": float(pooled.std()),
        "skew": float(pooled.skew()),
        "pct_pos": float((pooled > 0).mean()),
        "n": n,
    }

def check_hard(m):
    """H1-H4. Devuelve lista de los que fallan."""
    fails = []
    if m["pct_90"] >= H1_MAX: fails.append("H1")
    if m["pct_95"] >= H2_MAX: fails.append("H2")
    if m["p05"] > H3_MAX:    fails.append("H3")
    if m["p95"] < H4_MIN:    fails.append("H4")
    return fails

def check_diag(m):
    """D1-D3. Devuelve lista de los que fallan."""
    fails = []
    if abs(m["skew"]) >= 1.0: fails.append("D1")
    # D2 requiere rango anual; no calculado aqui
    return fails

def annual_range(trends_by_ticker, K):
    """Rango de medianas anuales de t_norm."""
    yearly_medians = {}
    for trend in trends_by_ticker.values():
        t_norm = np.tanh(trend / K)
        for year, group in t_norm.groupby(t_norm.index.year):
            yearly_medians.setdefault(year, []).append(float(group.median()))
    if not yearly_medians:
        return float("nan")
    ymed = [float(np.median(v)) for v in yearly_medians.values()]
    return float(max(ymed) - min(ymed))

def iqr_cross_sectional(trends_by_ticker, K):
    """IQR cross-sectional por fecha, mediana agregada."""
    per_date = {}
    for trend in trends_by_ticker.values():
        t_norm = np.tanh(trend / K)
        for dt, val in t_norm.items():
            per_date.setdefault(dt, []).append(float(val))
    iqrs = []
    for dt, vals in per_date.items():
        if len(vals) >= 5:
            s = pd.Series(vals)
            iqrs.append(float(s.quantile(0.75) - s.quantile(0.25)))
    if not iqrs:
        return float("nan"), float("nan")
    med = float(np.median(iqrs))
    pct_alto = float(sum(1 for v in iqrs if v > 0.10)) / len(iqrs)
    return med, pct_alto

def main():
    print("=" * 100)
    print("FASE 5b - CALIBRACION K")
    print("=" * 100)

    df, tickers = load_prices()
    print(f"\nTickers en parquet: {len(tickers)}")

    # Determinar ultima sesion cerrada comun
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

    # Calcular trend por ticker
    trends_by_ticker = {}
    for tk in tickers:
        trend = compute_trend(df, tk)
        if trend is not None and len(trend) >= 200:
            trends_by_ticker[tk] = trend
    print(f"Tickers con trend valido (>=200 obs): {len(trends_by_ticker)}")

    # Split temporal (por indice de cada ticker)
    # NOTA: split por posicion temporal, no por ticker. Cada ticker se corta
    # en su 70% de observaciones. Alternativa: split global por fecha.
    # El protocolo dice "temporal holdout 70/30" - uso corte por fecha global.
    all_dates = sorted(set().union(*[set(t.index) for t in trends_by_ticker.values()]))
    if len(all_dates) < 100:
        raise SystemExit("Serie demasiado corta")
    cutoff_idx = int(len(all_dates) * SPLIT)
    cutoff_date = all_dates[cutoff_idx]
    print(f"Split temporal 70/30 en fecha: {cutoff_date}")

    cal_trends = {tk: t[t.index <= cutoff_date] for tk, t in trends_by_ticker.items()}
    val_trends = {tk: t[t.index > cutoff_date] for tk, t in trends_by_ticker.items()}
    cal_trends = {tk: t for tk, t in cal_trends.items() if len(t) >= 100}
    val_trends = {tk: t for tk, t in val_trends.items() if len(t) >= 50}
    print(f"Tickers en calibracion: {len(cal_trends)}, en validacion: {len(val_trends)}")

    # --- Grid sobre calibracion ---
    print("\n" + "=" * 100)
    print("CALIBRACION (70% inicial)")
    print("=" * 100)
    print(f"\n{'K':>6} | {'pct_90':>8} {'pct_95':>8} {'p05':>8} {'p95':>8} | {'H1-H4':>10} | {'skew':>6} {'n_tk':>5}")
    print("-" * 100)
    results = {}
    for K in GRID_1:
        m = aggregate_balanced(cal_trends, K)
        if m is None:
            continue
        fails = check_hard(m)
        status = "PASA" if not fails else ",".join(fails)
        results[K] = (m, fails)
        print(f"{K:>6.2f} | {m['pct_90']:>8.4f} {m['pct_95']:>8.4f} {m['p05']:>+8.4f} {m['p95']:>+8.4f} | {status:>10} | {m['skew']:>+6.3f} {m['n_tickers']:>5}")

    # --- Seleccion: mayor K que cumpla H1-H4 ---
    passing = [K for K, (_, f) in results.items() if not f]
    print()
    if not passing:
        print("NINGUN K cumple H1-H4. Fase 5b FALLA. Requiere grid 5b.2.")
        return
    K_sel = max(passing)
    print(f"K seleccionado (mayor de los que pasan): {K_sel}")

    # --- Diagnostico sobre el K elegido ---
    m_sel = results[K_sel][0]
    print(f"\nDiagnosticos sobre K={K_sel}:")
    yrange = annual_range(cal_trends, K_sel)
    iqr_med, pct_alto = iqr_cross_sectional(cal_trends, K_sel)
    print(f"  D1 skewness      = {m_sel['skew']:+.3f}  ({'OK' if abs(m_sel['skew']) < 1.0 else 'FAIL'})")
    print(f"  D2 rango anual   = {yrange:.3f}      ({'OK' if yrange < 0.40 else 'FAIL'})")
    print(f"  D3 IQR_cs med    = {iqr_med:.4f}   ({'OK' if iqr_med >= 0.05 else 'FAIL'}), pct_fechas>0.10 = {pct_alto:.2%}")

    # --- Validacion 30% final ---
    print("\n" + "=" * 100)
    print(f"VALIDACION (30% final) con K={K_sel}")
    print("=" * 100)
    m_val = aggregate_balanced(val_trends, K_sel)
    if m_val is None:
        print("Sin datos de validacion suficientes.")
        return
    fails_val = check_hard(m_val)
    print(f"\n{'metrica':>10} | {'valor':>10} | {'criterio':>15} | {'estado':>8}")
    print("-" * 100)
    print(f"{'pct_90':>10} | {m_val['pct_90']:>10.4f} | {'< 15%':>15} | {'OK' if 'H1' not in fails_val else 'FALLA':>8}")
    print(f"{'pct_95':>10} | {m_val['pct_95']:>10.4f} | {'< 5%':>15} | {'OK' if 'H2' not in fails_val else 'FALLA':>8}")
    print(f"{'p05':>10} | {m_val['p05']:>+10.4f} | {'<= -0.40':>15} | {'OK' if 'H3' not in fails_val else 'FALLA':>8}")
    print(f"{'p95':>10} | {m_val['p95']:>+10.4f} | {'>= +0.40':>15} | {'OK' if 'H4' not in fails_val else 'FALLA':>8}")
    print(f"{'skew':>10} | {m_val['skew']:>+10.3f} | {'D1 |s|<1':>15} | {'OK' if abs(m_val['skew'])<1 else 'WARN':>8}")
    iqr_med_v, pct_alto_v = iqr_cross_sectional(val_trends, K_sel)
    print(f"{'IQR_cs':>10} | {iqr_med_v:>10.4f} | {'D3 >= 0.05':>15} | {'OK' if iqr_med_v>=0.05 else 'WARN':>8}")

    # --- Comparativa pooled vs balanced para el K elegido ---
    print("\n" + "=" * 100)
    print(f"SENSIBILIDAD: balanced vs pooled con K={K_sel}")
    print("=" * 100)
    m_pool = aggregate_pooled(cal_trends, K_sel)
    print(f"\n{'metrica':>10} | {'balanced':>10} | {'pooled':>10}")
    print("-" * 100)
    for k in ["pct_90", "pct_95", "p05", "p50", "p95", "iqr", "skew"]:
        print(f"{k:>10} | {m_sel[k]:>+10.4f} | {m_pool[k]:>+10.4f}")

    # --- Veredicto ---
    print("\n" + "=" * 100)
    print("VEREDICTO")
    print("=" * 100)
    if not fails_val:
        print(f"Fase 5b EXITOSA. K congelable = {K_sel}")
        print(f"  H1-H4 pasan en calibracion (70%) y en validacion (30%).")
        print(f"  Proximo paso: actualizar config + proveniencia y desbloquear 5c.")
    else:
        print(f"Fase 5b FALLA en validacion: {fails_val}")
        print(f"  Aunque K={K_sel} pasaba H1-H4 en calibracion, no generaliza al 30%.")
        print(f"  Segun protocolo §7, requiere revisar. Grid 5b.2 o revision formula.")

if __name__ == "__main__":
    main()
