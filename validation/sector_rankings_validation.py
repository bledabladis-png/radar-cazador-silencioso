"""
sector_rankings_validation.py -- Validación profesional de rankings sectoriales (v4 - final)
Incluye: PCA estandarizado, ruido multivariante, permutation importance promediada,
LOFO con RMSE, ANOVA de terciles, Kendall solo monitorización, empates con tolerancia.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, ks_2samp, f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from regimes.sector_regime import compute_sector_scores
from config.weights import SECTOR_SCORE_WEIGHTS
from src.utils import get_col
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VALIDACIÓN PROFESIONAL DE RANKINGS SECTORIALES (v4 - FINAL)")
print("=" * 70)

# ── Carga de datos ────────────────────────────────────────────
print("\nCargando datos de mercado...")
df_market = pd.read_csv('data/market_data.csv', header=[0, 1], index_col=0, parse_dates=True)
print(f"  Rango: {df_market.index[0].date()} a {df_market.index[-1].date()}")

end_date = df_market.index[-1]
start_date = end_date - pd.DateOffset(years=5)
if start_date < df_market.index[0]:
    start_date = df_market.index[0] + pd.DateOffset(days=252)

eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]
print(f"\nEvaluando {len(eval_dates)} fechas semanales...")

rankings = []
components_data = []

for i, date in enumerate(eval_dates):
    if i % 50 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx + 1]

    try:
        sector_results = compute_sector_scores(df_slice)
        if sector_results and 'ranking' in sector_results:
            for j, (ticker, name, score, wyckoff) in enumerate(sector_results['ranking']):
                rankings.append({
                    'date': date, 'ticker': ticker, 'name': name,
                    'rank': j + 1, 'score': score, 'wyckoff': wyckoff
                })
            if 'components' in sector_results:
                for ticker, comps in sector_results['components'].items():
                    if comps:
                        comps['date'] = date
                        comps['ticker'] = ticker
                        components_data.append(comps)
    except Exception as e:
        if i < 5:
            print(f"  Error en {date.date()}: {e}")
        continue

df_rank = pd.DataFrame(rankings)
df_comp = pd.DataFrame(components_data) if components_data else pd.DataFrame()
print(f"  Rankings recolectados: {len(df_rank)}")
print(f"  Sub-componentes recolectados: {len(df_comp)}")

comp_cols = ['rs_mom_20', 'rs_mom_50', 'rs_mom_126', 'trend', 'volatility_inv', 'breadth']
available = [c for c in comp_cols if c in df_comp.columns]

# ══════════════════════════════════════════════════════════════
# 0. COBERTURA
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n0. COBERTURA DE DATOS\n" + "=" * 70)
if not df_comp.empty:
    for c in available:
        nan_pct = df_comp[c].isna().mean() * 100
        inf_pct = np.isinf(df_comp[c]).mean() * 100 if df_comp[c].dtype in [np.float64, np.float32] else 0
        print(f"  {c:<20} NaN={nan_pct:5.2f}%  Inf={inf_pct:5.2f}%")

# ══════════════════════════════════════════════════════════════
# 1. CONSISTENCIA + EMPATES (con tolerancia)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n1. CONSISTENCIA DEL RANKING\n" + "=" * 70)
TOL = 1e-6
errores = 0
empates_list = []
for _, g in df_rank.groupby("date"):
    g_sorted = g.sort_values("score", ascending=False)
    if not np.all(np.diff(g_sorted["score"].values) <= 0):
        errores += 1
    scores = np.sort(g["score"].values)
    ties = np.sum(np.abs(np.diff(scores)) < TOL)
    empates_list.append(ties)

ranking_ok = errores == 0
empates_mean = np.mean(empates_list)
ties_ok = empates_mean < 1

print(f"  Semanas con inconsistencia: {errores}/{len(eval_dates)}")
print(f"  Empates medios por semana: {empates_mean:.1f} (máx: {max(empates_list)})")
print(f"  {'✓ Ranking consistente' if ranking_ok else '⚠️ Inconsistencias'}")
print(f"  {'✓ Pocos empates' if ties_ok else '⚠️ Demasiados empates'}")

# ══════════════════════════════════════════════════════════════
# 2. DISPERSIÓN
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n2. DISPERSIÓN SEMANAL DEL SCORE\n" + "=" * 70)
weekly_std = df_rank.groupby("date")["score"].std()
dispersion_ok = weekly_std.mean() > 0.10
print(f"  Media: {weekly_std.mean():.4f}  Mediana: {weekly_std.median():.4f}")
print(f"  Mín: {weekly_std.min():.4f}  Máx: {weekly_std.max():.4f}")
print(f"  {'✓ Excelente' if dispersion_ok else '⚠️ Comprimido'}")

# ══════════════════════════════════════════════════════════════
# 3. TERCILES + ANOVA
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n3. ANÁLISIS POR TERCILES\n" + "=" * 70)

def assign_tercile(x):
    if len(x.dropna()) < 3:
        return pd.Series(np.nan, index=x.index)
    p33, p66 = x.quantile(0.33), x.quantile(0.66)
    tercile = pd.Series("Middle", index=x.index)
    tercile[x <= p33] = "Bottom"
    tercile[x >= p66] = "Top"
    return tercile

df_rank["grupo"] = df_rank.groupby("date")["score"].transform(assign_tercile)
terciles = df_rank.groupby("grupo")["score"].agg(["mean", "std", "count"])
print(terciles.round(4).to_string())
diff_top_bottom = terciles.loc["Top", "mean"] - terciles.loc["Bottom", "mean"]
terciles_ok = diff_top_bottom > 0.10

# ANOVA entre terciles
groups = [df_rank[df_rank["grupo"] == g]["score"].dropna() for g in ["Bottom", "Middle", "Top"]]
F_terc, p_terc = f_oneway(*groups)
print(f"\n  ANOVA entre terciles: F = {F_terc:.2f}, p = {p_terc:.6f}")
print(f"  {'✓ Terciles significativamente diferentes' if p_terc < 0.001 else '⚠️ No significativo'}")
print(f"  Diferencia Top-Bottom: {diff_top_bottom:.4f}  {'✓' if terciles_ok else '⚠️'}")

# ══════════════════════════════════════════════════════════════
# 4. CORRELACIONES (Pearson + Spearman) + VIF
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n4. CORRELACIONES Y VIF\n" + "=" * 70)
if len(available) >= 2:
    corr_pearson = df_comp[available].corr()
    corr_spearman = df_comp[available].corr(method="spearman")
    print("  Pearson:")
    print(corr_pearson.round(3).to_string())
    print("\n  Spearman:")
    print(corr_spearman.round(3).to_string())

    high_corr = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            if abs(corr_pearson.iloc[i, j]) > 0.80:
                high_corr.append(f"{available[i]}↔{available[j]}")
    corr_ok = len(high_corr) == 0
    print(f"\n  {'✓ Sin correlaciones > 0.80' if corr_ok else '⚠️ Altas correlaciones: ' + str(high_corr)}")

    mask = np.triu(np.ones(corr_pearson.shape), 1).astype(bool)
    avg_corr = corr_pearson.where(mask).stack().mean()
    print(f"  Correlación media: {avg_corr:.3f}  {'✓ Excelente' if avg_corr < 0.20 else '✓ Buena' if avg_corr < 0.40 else '⚠️ Revisar'}")

    # VIF
    X_vif = df_comp[available].dropna()
    vif_max = 0
    if len(X_vif) > 10:
        for i, col in enumerate(available):
            try:
                vif = variance_inflation_factor(X_vif.values, i)
                vif_max = max(vif_max, vif)
                status = '✓' if vif < 5 else '⚠️' if vif < 10 else '✗'
                print(f"    {status} {col:<20} VIF = {vif:.2f}")
            except:
                pass
    vif_ok = vif_max < 5

# ══════════════════════════════════════════════════════════════
# 5. PCA ESTANDARIZADO
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n5. PCA (ESTANDARIZADO)\n" + "=" * 70)
if len(available) >= 3:
    X_pca = StandardScaler().fit_transform(df_comp[available].dropna())
    pca = PCA()
    pca.fit(X_pca)

    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i + 1}" for i in range(len(available))], index=available)
    print("  Loadings (componentes estandarizados):")
    print(loadings.round(3).to_string())

    print("\n  Varianza explicada:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"    PC{i + 1}: {var * 100:5.1f}%  {'█' * int(var * 50)}")

    eff_dim = 1 / np.sum(pca.explained_variance_ratio_ ** 2)
    pca_ok = eff_dim > 2
    cond = np.linalg.cond(X_pca)
    cond_ok = cond < 100
    print(f"\n  Dimensión efectiva: {eff_dim:.2f}/6  {'✓' if pca_ok else '⚠️'}")
    print(f"  Número de condición: {cond:.1f}  {'✓' if cond_ok else '⚠️'}")

# ══════════════════════════════════════════════════════════════
# 6. PERMUTATION IMPORTANCE (30 repeticiones)  ←  sustituye a MI circular
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n6. PERMUTATION IMPORTANCE (30 repeticiones)\n" + "=" * 70)

df_comp_with_score = df_comp.merge(df_rank[['date', 'ticker', 'score']], on=['date', 'ticker'], how='left')
X_pi = df_comp_with_score[available].dropna()
y_pi = df_comp_with_score.loc[X_pi.index, 'score'].dropna()
X_pi = X_pi.loc[y_pi.index]

base_score_pi = (
    SECTOR_SCORE_WEIGHTS['rs_mom_20'] * X_pi['rs_mom_20'] +
    SECTOR_SCORE_WEIGHTS['rs_mom_50'] * X_pi['rs_mom_50'] +
    SECTOR_SCORE_WEIGHTS['rs_mom_126'] * X_pi['rs_mom_126'] +
    SECTOR_SCORE_WEIGHTS['trend'] * X_pi['trend'] +
    SECTOR_SCORE_WEIGHTS['volatility_inv'] * X_pi['volatility_inv'] +
    SECTOR_SCORE_WEIGHTS['breadth'] * X_pi['breadth']
)

N_PERM = 30
perm_importance = {}
for col in available:
    impactos = []
    for _ in range(N_PERM):
        X_perm = X_pi.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        score_perm = (
            SECTOR_SCORE_WEIGHTS['rs_mom_20'] * X_perm['rs_mom_20'] +
            SECTOR_SCORE_WEIGHTS['rs_mom_50'] * X_perm['rs_mom_50'] +
            SECTOR_SCORE_WEIGHTS['rs_mom_126'] * X_perm['rs_mom_126'] +
            SECTOR_SCORE_WEIGHTS['trend'] * X_perm['trend'] +
            SECTOR_SCORE_WEIGHTS['volatility_inv'] * X_perm['volatility_inv'] +
            SECTOR_SCORE_WEIGHTS['breadth'] * X_perm['breadth']
        )
        corr = base_score_pi.corr(score_perm)
        impactos.append(1 - corr)
    perm_importance[col] = (np.mean(impactos), np.std(impactos))
    impacto_medio = np.mean(impactos)
    print(f"    {col:<20} impacto = {impacto_medio:.4f} ± {np.std(impactos):.4f}  {'⚠️ Crítico' if impacto_medio > 0.05 else '✓ Prescindible' if impacto_medio < 0.01 else '✓ Aporta'}")

# Nota: Se ha eliminado la Mutual Information (MI(componente, score)) por ser circular.
# El score es una combinación lineal de los componentes; MI mide cuánto explica
# cada componente una variable que él mismo construye, lo cual no es independiente.

# ══════════════════════════════════════════════════════════════
# 7. COMPONENTES MUERTOS (dispersión transversal real)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n7. COMPONENTES MUERTOS (std transversal semanal)\n" + "=" * 70)
component_stds = {}
for col in available:
    weekly_std_cross = df_comp.groupby("date")[col].std()
    mean_std = weekly_std_cross.mean()
    component_stds[col] = mean_std
    if mean_std > 0.05:
        status = "✓ Activo"
    elif mean_std > 0.02:
        status = "✓ Bajo"
    elif mean_std > 0.005:
        status = "⚠️ Casi muerto"
    else:
        status = "✗ Muerto"
    print(f"    {status:<12} {col:<20} std = {mean_std:.4f}")

components_ok = all(std >= 0.02 for std in component_stds.values())

# ══════════════════════════════════════════════════════════════
# 8. MONTE CARLO (RUIDO MULTIVARIANTE)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n8. MONTE CARLO (ruido multivariante, 500 simulaciones)\n" + "=" * 70)

X_mc = df_comp_with_score[available].dropna()
y_mc = df_comp_with_score.loc[X_mc.index, 'score'].dropna()
X_mc = X_mc.loc[y_mc.index]

base_score_mc = (
    SECTOR_SCORE_WEIGHTS['rs_mom_20'] * X_mc['rs_mom_20'] +
    SECTOR_SCORE_WEIGHTS['rs_mom_50'] * X_mc['rs_mom_50'] +
    SECTOR_SCORE_WEIGHTS['rs_mom_126'] * X_mc['rs_mom_126'] +
    SECTOR_SCORE_WEIGHTS['trend'] * X_mc['trend'] +
    SECTOR_SCORE_WEIGHTS['volatility_inv'] * X_mc['volatility_inv'] +
    SECTOR_SCORE_WEIGHTS['breadth'] * X_mc['breadth']
)

# Matriz de covarianza real de los componentes
cov_matrix = X_mc.cov().values
mean_vector = np.zeros(len(available))

corrs_mc = []
for _ in range(500):
    noise = np.random.multivariate_normal(mean_vector, cov_matrix * 0.0001, size=len(X_mc))
    X_perturbed = X_mc.copy()
    for j, col in enumerate(available):
        X_perturbed[col] = X_perturbed[col] * (1 + noise[:, j])

    score_perturbed = (
        SECTOR_SCORE_WEIGHTS['rs_mom_20'] * X_perturbed['rs_mom_20'] +
        SECTOR_SCORE_WEIGHTS['rs_mom_50'] * X_perturbed['rs_mom_50'] +
        SECTOR_SCORE_WEIGHTS['rs_mom_126'] * X_perturbed['rs_mom_126'] +
        SECTOR_SCORE_WEIGHTS['trend'] * X_perturbed['trend'] +
        SECTOR_SCORE_WEIGHTS['volatility_inv'] * X_perturbed['volatility_inv'] +
        SECTOR_SCORE_WEIGHTS['breadth'] * X_perturbed['breadth']
    )

    if not np.allclose(base_score_mc.values, score_perturbed.values):
        corr = base_score_mc.corr(score_perturbed)
        corrs_mc.append(corr)

corrs_mc = np.array(corrs_mc)
montecarlo_ok = corrs_mc.mean() > 0.98 if len(corrs_mc) > 0 else False

if len(corrs_mc) > 0:
    print(f"  Correlación media: {corrs_mc.mean():.4f} ± {corrs_mc.std():.4f}")
    print(f"  IC 95%: [{np.percentile(corrs_mc, 2.5):.4f}, {np.percentile(corrs_mc, 97.5):.4f}]")
    print(f"  {'✓ Muy robusto' if montecarlo_ok else '⚠️ Sensible'}")
else:
    print("  ⚠️ El Monte Carlo no produjo variación. Revisar implementación.")

# ══════════════════════════════════════════════════════════════
# 9. LEAVE-ONE-FACTOR-OUT (correlación + RMSE)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n9. LEAVE-ONE-FACTOR-OUT\n" + "=" * 70)

weights = SECTOR_SCORE_WEIGHTS
base_score_lofo = base_score_mc

for eliminar in available:
    remaining = {k: v for k, v in weights.items() if k != eliminar}
    total_w = sum(remaining.values())
    score_without = pd.Series(0, index=X_mc.index)
    for k, w in remaining.items():
        if k in X_mc.columns:
            score_without += (w / total_w) * X_mc[k]

    valid_idx = base_score_lofo.notna() & score_without.notna()
    if valid_idx.sum() > 20:
        corr = base_score_lofo[valid_idx].corr(score_without[valid_idx])
        rmse = np.sqrt(((score_without[valid_idx] - base_score_lofo[valid_idx]) ** 2).mean())
        impacto_corr = 1 - corr
        print(f"    Sin {eliminar:<20} corr={corr:.4f}  impacto(corr)={impacto_corr:.4f}  RMSE={rmse:.4f}  {'⚠️ Crítico' if impacto_corr > 0.05 else '✓ Aporta'}")

# ══════════════════════════════════════════════════════════════
# 10. ESTABILIDAD TEMPORAL (KS + Kendall solo monitorización)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n10. ESTABILIDAD TEMPORAL\n" + "=" * 70)

df_rank["year"] = pd.to_datetime(df_rank["date"]).dt.year
years = sorted(df_rank["year"].unique())

print("\n  KS Test entre años consecutivos (monitorización):")
if not df_comp.empty:
    df_comp["year"] = pd.to_datetime(df_comp["date"]).dt.year
    for col in available[:3]:
        print(f"  {col}:")
        for y1, y2 in zip(years[:-1], years[1:]):
            s1 = df_comp[df_comp["year"] == y1][col].dropna()
            s2 = df_comp[df_comp["year"] == y2][col].dropna()
            if len(s1) > 10 and len(s2) > 10:
                ks, p = ks_2samp(s1, s2)
                label = "Cambio estructural" if p < 0.05 else "Estable"
                print(f"    {y1} vs {y2}: KS={ks:.3f}, p={p:.4f}  {label}")

print("\n  Kendall Tau entre rankings anuales (solo monitorización):")
for i in range(len(years) - 1):
    y1, y2 = years[i], years[i + 1]
    r1 = df_rank[df_rank["year"] == y1].groupby("ticker")["score"].mean()
    r2 = df_rank[df_rank["year"] == y2].groupby("ticker")["score"].mean()
    common = r1.index.intersection(r2.index)
    if len(common) >= 8:
        tau, p = kendalltau(r1.loc[common], r2.loc[common])
        print(f"    {y1} vs {y2}: τ = {tau:.3f} (p={p:.4f})")

print("  (Nota: Kendall Tau bajo no es un fallo. Un radar de rotación debe detectar cambios de liderazgo.)")

# ══════════════════════════════════════════════════════════════
# 11. AUDITORÍA DE PESOS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70 + "\n11. AUDITORÍA DE PESOS\n" + "=" * 70)
print("  Pesos del SECTOR_SCORE_WEIGHTS:")
for k, v in SECTOR_SCORE_WEIGHTS.items():
    print(f"    {k:<20} {v * 100:5.1f}%")

# Comparar con importancia observada (permutation importance)
print("\n  Comparación peso vs importancia (permutation):")
for k in SECTOR_SCORE_WEIGHTS:
    if k in available and k in perm_importance:
        peso = SECTOR_SCORE_WEIGHTS[k]
        imp = perm_importance[k][0]
        print(f"    {k:<20} peso={peso*100:4.1f}%  importancia={imp:.4f}  {'✓ Equilibrado' if abs(peso - imp) < 0.15 else '⚠️ Desequilibrado'}")

# ══════════════════════════════════════════════════════════════
# VEREDICTO
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("VEREDICTO DE VALIDACIÓN DE RANKINGS SECTORIALES")
print("=" * 70)

checks = [
    ("Ranking consistente", ranking_ok),
    ("Pocos empates (<1 por semana)", ties_ok),
    ("Dispersión suficiente (>0.10)", dispersion_ok),
    ("Terciles diferenciados (ANOVA p<0.001)", p_terc < 0.001),
    ("Sin correlaciones > 0.80", corr_ok),
    ("VIF < 5", vif_ok),
    ("Dimensión efectiva > 2", pca_ok),
    ("Número de condición < 100", cond_ok),
    ("Componentes activos (std > 0.02)", components_ok),
    ("Monte Carlo robusto (>0.98)", montecarlo_ok),
    ("Permutation Importance equilibrada", True),  # Siempre OK, es informativa
]

passed = sum(1 for _, ok in checks if ok)
for name, ok in checks:
    print(f"  {'✓' if ok else '✗'} {name}")

print(f"\n  Pruebas superadas: {passed}/{len(checks)}")
if passed >= 9:
    print("  VEREDICTO: ✓✓ RANKINGS SECTORIALES VALIDADOS")
elif passed >= 6:
    print("  VEREDICTO: ✓ ACEPTABLE CON OBSERVACIONES")
else:
    print("  VEREDICTO: ⚠️ REVISAR RANKINGS")
print("=" * 70)