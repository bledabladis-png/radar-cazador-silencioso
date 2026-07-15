"""
oms_darkpool_validation.py -- Validación institucional de OMS v2.0 y Dark Pools v1.0.
Aplica el mismo estándar que Breadth y Momentum/Flujo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN INSTITUCIONAL DE OMS v2.0 Y DARK POOLS v1.0")
print("=" * 70)


def validate_module(name, df, col, range_check, csv_path):
    """Aplica batería de validación a un módulo informativo."""
    print("\n" + "=" * 70)
    print(f"VALIDACIÓN: {name}")
    print(f"Fuente: {csv_path}  |  Columna: {col}")
    print("=" * 70)

    if df is None or col not in df.columns:
        print(f"  ERROR: datos no disponibles para {name}")
        return 0, 0

    print(f"  Registros: {len(df)}  |  Fechas: {df.index[0].date()} → {df.index[-1].date()}")

    # ---- 0. COBERTURA + RANGO LÓGICO ----
    print("\n  ── 0. COBERTURA Y RANGO ──")
    nan_pct = df[col].isna().mean() * 100
    inf_pct = np.isinf(df[col]).mean() * 100 if df[col].dtype in [np.float64, np.float32] else 0
    coverage_ok = nan_pct < 1 and inf_pct == 0
    print(f"    NaN={nan_pct:.2f}%  Inf={inf_pct:.2f}%  {'✓' if coverage_ok else '⚠️'}")

    range_ok = True
    if range_check == "positive":
        range_ok = (df[col].dropna() > 0).all()
    elif range_check == "zero_one":
        range_ok = df[col].dropna().between(0, 1).all()
    print(f"    Rango lógico: {'✓' if range_ok else '⚠️'}")

    # ---- 1. OUTLIERS (IQR) ----
    print("\n  ── 1. OUTLIERS (IQR) ──")
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    pct_out = n_out / len(df) * 100
    outliers_ok = pct_out < 10
    print(f"    Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
    print(f"    Outliers: {n_out}/{len(df)} ({pct_out:.1f}%)  {'✓' if outliers_ok else '⚠️'}")

    # ---- 2. ESTACIONARIEDAD (ADF) ----
    print("\n  ── 2. ESTACIONARIEDAD (ADF) ──")
    adf_ok = False
    if len(df) > 30:
        stat, p, *_ = adfuller(df[col].dropna())
        adf_ok = p < 0.05
        print(f"    p={p:.4f}  {'✓ Estacionaria' if adf_ok else '⚠️ No estacionaria'}")
    else:
        print(f"    Datos insuficientes ({len(df)} < 30)")

    # ---- 3. AUTOCORRELACIÓN + EFFECTIVE SAMPLE SIZE ----
    print("\n  ── 3. AUTOCORRELACIÓN Y N_eff ──")
    ac = df[col].autocorr()
    if pd.notna(ac):
        N = len(df[col].dropna())
        Neff = N * (1 - ac) / (1 + ac) if ac != -1 else N
        ac_ok = ac < 0.95
        print(f"    Autocorr lag1 = {ac:.3f}  {'✓' if ac_ok else '⚠️ Muy alta'}")
        print(f"    N = {N}  |  N_eff = {Neff:.0f}")
    else:
        ac_ok = True
        print("    No evaluable")

    # ---- 4. BOOTSTRAP DE LA MEDIA ----
    print("\n  ── 4. BOOTSTRAP (500 remuestreos) ──")
    means = []
    for _ in range(500):
        sample = df[col].sample(frac=1, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    bias = means.mean() - df[col].mean()
    bootstrap_ok = abs(bias) < 0.01
    print(f"    Media original: {df[col].mean():.4f}")
    print(f"    Bootstrap mean: {means.mean():.4f} ± {means.std():.4f}")
    print(f"    IC 95%: [{np.percentile(means, 2.5):.4f}, {np.percentile(means, 97.5):.4f}]")
    print(f"    Sesgo: {bias:.6f}  {'✓' if bootstrap_ok else '⚠️'}")

    # ---- 5. MONTE CARLO CON RUIDO ----
    print("\n  ── 5. MONTE CARLO (500 simulaciones) ──")
    corrs_mc = []
    std_col = df[col].std()
    for _ in range(500):
        noise = np.random.normal(0, std_col * 0.05, len(df))
        noise = np.clip(noise, -3 * std_col * 0.05, 3 * std_col * 0.05)
        pert = df[col].dropna() + noise[:len(df[col].dropna())]
        corrs_mc.append(pert.corr(df[col].dropna()))
    corrs_mc = np.array(corrs_mc)
    mc_ok = corrs_mc.mean() > 0.95
    print(f"    Corr media: {corrs_mc.mean():.4f}  IC95=[{np.percentile(corrs_mc,2.5):.4f}, {np.percentile(corrs_mc,97.5):.4f}]")
    print(f"    {'✓ Robusto' if mc_ok else '⚠️ Sensible'}")

    # ---- 6. ESTABILIDAD ANUAL ----
    print("\n  ── 6. ESTABILIDAD ANUAL ──")
    if 'year' not in df.columns:
        df = df.copy()
        df['year'] = df.index.year
    years = sorted(df['year'].unique())
    if len(years) > 1:
        for y in years:
            s = df[df['year'] == y][col]
            print(f"    {y}: mediana={s.median():.4f}  std={s.std():.4f}  n={len(s)}")
    else:
        print("    Solo un año de datos")

    # ---- VEREDICTO ----
    print("\n  ── VEREDICTO ──")
    checks = [
        ("Cobertura", coverage_ok),
        ("Rango lógico", range_ok),
        ("Outliers < 10%", outliers_ok),
        ("Estacionariedad (ADF)", adf_ok),
        ("Autocorrelación < 0.95", ac_ok),
        ("Bootstrap estable", bootstrap_ok),
        ("Monte Carlo robusto", mc_ok),
    ]
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    for name, ok in checks:
        print(f"    {'✓' if ok else '✗'} {name}")
    print(f"    Pruebas: {passed}/{total}")
    if passed == total:
        print(f"    VEREDICTO: ✓✓ {name} VALIDADO (NIVEL INSTITUCIONAL)")
    elif passed >= total - 1:
        print(f"    VEREDICTO: ✓ {name} ACEPTABLE CON OBSERVACIONES")
    else:
        print(f"    VEREDICTO: ⚠️ {name} REVISAR")
    return passed, total


# ============================================================
# EJECUCIÓN
# ============================================================

# OMS v2.0
try:
    df_pcr = pd.read_csv('outputs/pcr_history.csv', parse_dates=['date'], index_col='date')
    p1, t1 = validate_module("OMS v2.0", df_pcr, 'total_pcr', 'positive', 'outputs/pcr_history.csv')
except Exception as e:
    print(f"\n  OMS v2.0: ERROR - {e}")
    p1, t1 = 0, 7

# Dark Pools v1.0
try:
    df_dp = pd.read_csv('outputs/darkpool_history.csv', parse_dates=['week'], index_col='week')
    p2, t2 = validate_module("Dark Pools v1.0", df_dp, 'ratio', 'zero_one', 'outputs/darkpool_history.csv')
except Exception as e:
    print(f"\n  Dark Pools v1.0: ERROR - {e}")
    p2, t2 = 0, 7

print("\n" + "=" * 70)
print("RESUMEN FINAL")
print("=" * 70)
print(f"  OMS v2.0:        {p1}/{t1} pruebas superadas")
print(f"  Dark Pools v1.0: {p2}/{t2} pruebas superadas")
print("=" * 70)
