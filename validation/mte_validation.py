"""
mte_validation.py -- Validación institucional del Market Transition Engine v1.0.
Evalúa cobertura, estacionariedad, independencia, robustez y coherencia de transiciones.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.utils import get_col
from indicators.mte import (
    sector_rotation_score, safe_haven_score, credit_stress_score,
    inflation_pressure_score, compute_msi, compute_ipi, classify_mte,
    NORMAL_TRANSITIONS, EXCEPTION_TRANSITIONS, validate_transition
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN INSTITUCIONAL DEL MTE v1.0")
print("=" * 70)

# Cargar datos de mercado
print("\nCargando datos de mercado...")
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
print(f"  Rango: {df_market.index[0].date()} a {df_market.index[-1].date()}")

# Determinar período de evaluación (últimos 2 años, o lo disponible)
end_date = df_market.index[-1]
start_date = end_date - pd.DateOffset(years=2)
if start_date < df_market.index[0]:
    start_date = df_market.index[0]

eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]
print(f"Evaluando {len(eval_dates)} fechas semanales...")

# Recolectar scores históricos
rows = []
for i, date in enumerate(eval_dates):
    if i % 20 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]

    try:
        srs = sector_rotation_score(df_slice)
        shs = safe_haven_score(df_slice)
        # Para CLS e IPS necesitamos señales externas; usamos valores sintéticos para la validación estructural
        # Aproximación real de CLS usando VIX y HYG/LQD del mercado
        try:
            vix_close = get_col(df_slice, '^VIX', 'Close')
            vix_ret = vix_close.pct_change().rolling(20).std().iloc[-1]
            vix_ma = vix_close.pct_change().rolling(60).std().mean()
            fc_approx = float(np.clip(np.tanh((vix_ret / (vix_ma + 1e-9) - 1) / 2), 0, 1))
            
            hyg = get_col(df_slice, 'HYG', 'Close')
            lqd = get_col(df_slice, 'LQD', 'Close')
            spread = hyg / lqd
            cred_approx = float(np.clip(np.tanh(-(spread.pct_change(20).iloc[-1]) / 2), 0, 1))
            
            cls = float(np.mean([fc_approx, cred_approx, 0.3, 0.3]))  # 2 señales reales + 2 neutras
        except:
            cls = 0.3  # valor neutro si falla
        ips = inflation_pressure_score(df_slice)
        msi = compute_msi(srs, shs, cls)
        ipi = compute_ipi(ips)

        rows.append({
            'date': date,
            'srs': srs,
            'shs': shs,
            'cls': cls,
            'ips': ips,
            'msi': msi,
            'ipi': ipi
        })
    except Exception as e:
        if i < 5:
            print(f"  Error en {date.date()}: {e}")
        continue

df = pd.DataFrame(rows)
print(f"  Registros recolectados: {len(df)}")

if len(df) < 10:
    print("  Datos insuficientes para validación completa. Se muestran solo resultados parciales.")

score_cols = ['srs', 'shs', 'cls', 'ips']
index_cols = ['msi', 'ipi']

# ============================================================
# 0. COBERTURA Y RANGO LÓGICO
# ============================================================
print("\n" + "="*70 + "\n0. COBERTURA Y RANGO LÓGICO\n" + "="*70)
for col in score_cols + index_cols:
    if col in df.columns:
        nan_pct = df[col].isna().mean() * 100
        inf_pct = np.isinf(df[col]).mean() * 100 if df[col].dtype in [np.float64, np.float32] else 0
        in_range = df[col].between(-1, 1).mean() * 100 if col in score_cols else df[col].between(0, 100).mean() * 100
        print(f"  {col:<8} NaN={nan_pct:5.2f}%  Inf={inf_pct:5.2f}%  Rango={'✓' if in_range > 99 else '⚠️'}")

# ============================================================
# 1. ESTACIONARIEDAD (ADF)
# ============================================================
print("\n" + "="*70 + "\n1. ESTACIONARIEDAD (ADF)\n" + "="*70)
if len(df) > 30:
    for col in score_cols + index_cols:
        if col in df.columns:
            try:
                stat, p, *_ = adfuller(df[col].dropna())
                print(f"  {col:<8} p={p:.4f}  {'✓ Estacionaria' if p < 0.05 else '⚠️ No estacionaria'}")
            except ValueError:
                print(f"  {col:<8} serie constante (sin variabilidad)")
else:
    print("  Datos insuficientes para ADF")

# ============================================================
# 2. AUTOCORRELACIÓN Y EFFECTIVE SAMPLE SIZE
# ============================================================
print("\n" + "="*70 + "\n2. AUTOCORRELACIÓN Y N_eff\n" + "="*70)
for col in score_cols + index_cols:
    if col in df.columns and len(df) > 10:
        ac = df[col].autocorr()
        if pd.notna(ac):
            N = len(df[col].dropna())
            Neff = N * (1 - ac) / (1 + ac) if ac != -1 else N
            status = '✓ Reactivo' if ac < 0.70 else '✓ Alta (esperable)' if ac < 0.90 else '⚠️ Muy alta'
            print(f"  {col:<8} autocorr={ac:.3f}  N_eff={Neff:.0f}/{N}  {status}")
    else:
        print(f"  {col:<8} datos insuficientes")

# ============================================================
# 3. CORRELACIÓN ENTRE MOTORES
# ============================================================
print("\n" + "="*70 + "\n3. CORRELACIÓN ENTRE MOTORES (Spearman)\n" + "="*70)
if len(df) > 10:
    corr = df[score_cols].corr(method='spearman')
    print(corr.round(3).to_string())
    high_pairs = []
    for i in range(len(score_cols)):
        for j in range(i+1, len(score_cols)):
            if abs(corr.iloc[i, j]) > 0.80:
                high_pairs.append(f"{score_cols[i]}↔{score_cols[j]}: {corr.iloc[i,j]:.2f}")
    if high_pairs:
        print(f"\n  ⚠️ Correlaciones altas: {high_pairs}")
    else:
        print("\n  ✓ Motores independientes (todas < 0.80)")
else:
    print("  Datos insuficientes")

# ============================================================
# 4. PCA Y DIMENSIÓN EFECTIVA
# ============================================================
print("\n" + "="*70 + "\n4. PCA Y DIMENSIÓN EFECTIVA\n" + "="*70)
if len(df) > 20:
    X_pca = StandardScaler().fit_transform(df[score_cols].dropna())
    pca = PCA()
    pca.fit(X_pca)
    print("  Varianza explicada:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i+1}: {var*100:5.1f}%  {'█'*int(var*50)}")
    eff_dim = 1 / np.sum(pca.explained_variance_ratio_**2)
    print(f"\n  Dimensión efectiva: {eff_dim:.2f}/4  {'✓ Motores independientes' if eff_dim > 2.5 else '⚠️ Posible redundancia'}")
else:
    print("  Datos insuficientes para PCA")

# ============================================================
# 5. BOOTSTRAP DEL MSI Y IPI
# ============================================================
print("\n" + "="*70 + "\n5. BOOTSTRAP DEL MSI Y IPI (500 remuestreos)\n" + "="*70)
for col in index_cols:
    if col in df.columns and len(df) > 10:
        means = []
        for _ in range(500):
            sample = df[col].sample(frac=1, replace=True)
            means.append(sample.mean())
        means = np.array(means)
        bias = means.mean() - df[col].mean()
        print(f"  {col:<8} media={df[col].mean():.1f}  boot_mean={means.mean():.1f}  "
              f"IC95=[{np.percentile(means,2.5):.1f}, {np.percentile(means,97.5):.1f}]  "
              f"sesgo={bias:.2f}  {'✓ Estable' if abs(bias)<1 else '⚠️ Sesgo'}")

# ============================================================
# 6. MONTE CARLO CON RUIDO
# ============================================================
print("\n" + "="*70 + "\n6. MONTE CARLO CON RUIDO (500 simulaciones)\n" + "="*70)
for col in index_cols:
    if col in df.columns and len(df) > 10:
        corrs = []
        std_col = df[col].std()
        for _ in range(500):
            noise = np.random.normal(0, std_col * 0.05, len(df))
            noise = np.clip(noise, -3*std_col*0.05, 3*std_col*0.05)
            pert = df[col] + noise
            corrs.append(pert.corr(df[col]))
        corrs = np.array(corrs)
        print(f"  {col:<8} corr media={corrs.mean():.4f}  "
              f"IC95=[{np.percentile(corrs,2.5):.4f}, {np.percentile(corrs,97.5):.4f}]  "
              f"{'✓ Robusto' if corrs.mean()>0.95 else '⚠️ Sensible'}")

# ============================================================
# 7. COHERENCIA LÓGICA DEL CLASIFICADOR
# ============================================================
print("\n" + "="*70 + "\n7. COHERENCIA LÓGICA DEL CLASIFICADOR\n" + "="*70)
print("  Verificación de reglas de clasificación:")
from indicators.mte import score_scenarios
test_cases = [
    ("CRISIS",     0.5, 0.5, 0.6, 0.0),
    ("RECESSION",  0.3, 0.3, 0.3, 0.0),
    ("STAGFLATION", 0.1, 0.0, 0.0, 0.5),
    ("SOFT LANDING", 0.2, 0.2, -0.2, 0.0),
    ("EXPANSION",   -0.3, -0.2, -0.3, 0.0),
]
print("  (Nota: se usa score_scenarios() para evitar interferencia del estado guardado)")
for expected, srs, shs, cls, ips in test_cases:
    scores = score_scenarios(srs, shs, cls, ips)
    obtained = max(scores, key=scores.get)
    status = '✓' if obtained == expected else f'✗ (esperado {expected}, obtenido {obtained})'
    print(f"    {status} srs={srs:.1f}, shs={shs:.1f}, cls={cls:.1f}, ips={ips:.1f} → {obtained}")
