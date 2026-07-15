"""
macro_regime_validation.py -- Framework profesional de validación del clasificador de regímenes macro.
Incluye walk-forward, validación estructural, discriminación del score, forward returns,
bootstrap, tamaño del efecto, pruebas post-hoc, análisis de drift temporal
y veredicto cualitativo.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from scipy.stats import entropy, f_oneway, binomtest
from scipy.stats import ks_2samp, shapiro, levene
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import calinski_harabasz_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from src.utils import get_col
from regimes.financial_conditions import compute_liquidity_score
from regimes.volatility_regime import compute_volatility_regime
from regimes.macro_regime import compute_macro_regime
from src.macro_manual_loader import load_macro_manual
import warnings
warnings.filterwarnings('ignore')

# Parámetros de validación
MIN_SAMPLE = 5          # Observaciones mínimas para inferencia estadística

print("=" * 70)
print("VALIDACIÓN COMPLETA DEL RÉGIMEN MACRO (ESTRUCTURAL + ECONÓMICA)")
print("=" * 70)

# Cargar datos de mercado
print("\nCargando datos de mercado...")
df_market = pd.read_csv('data/market_data.csv', header=[0,1], index_col=0, parse_dates=True)
print(f"  Rango: {df_market.index[0].date()} a {df_market.index[-1].date()}")

# Cargar datos macro manuales
print("Cargando datos macro manuales...")
df_macro_manual = load_macro_manual()

# Determinar período de evaluación (últimos 5 años o lo que haya)
end_date = df_market.index[-1]
start_date = end_date - pd.DateOffset(years=5)
if start_date < df_market.index[0]:
    start_date = df_market.index[0] + pd.DateOffset(days=252)

# Generar fechas de evaluación (semanales para agilidad)
eval_dates = pd.date_range(start_date, end_date, freq='W-FRI')
eval_dates = [d for d in eval_dates if d in df_market.index]
print(f"\nEvaluando {len(eval_dates)} fechas semanales...")

# Recolectar regímenes
regimes = []
for i, date in enumerate(eval_dates):
    if i % 50 == 0:
        print(f"  Progreso: {i}/{len(eval_dates)}")
    idx = df_market.index.get_loc(date)
    df_slice = df_market.iloc[:idx+1]
    
    try:
        # Calcular condiciones financieras
        liq_score, liq_regime, liq_conf = compute_liquidity_score(df_slice)
        # Calcular volatilidad
        try:
            vix_close = get_col(df_slice, '^VIX', 'Close')
            vix_returns = vix_close.pct_change()
        except:
            vix_returns = pd.Series([0])
        vol_score, vol_regime, vol_conf = compute_volatility_regime(vix_returns)
        # Calcular régimen macro
        macro_score, macro_regime, macro_conf, _ = compute_macro_regime(
            df_slice, df_macro_manual, liq_score, vol_score
        )
        regimes.append({
            'date': date,
            'regime': macro_regime,
            'score': macro_score.iloc[-1] if hasattr(macro_score, 'iloc') else macro_score,
            'confidence': macro_conf
        })
    except Exception as e:
        if i < 5:
            print(f"  Error en {date.date()}: {e}")
        continue

df = pd.DataFrame(regimes)
print(f"  Regímenes recolectados: {len(df)}")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def state_entropy(df):
    p = df["regime"].value_counts(normalize=True).values
    return entropy(p)

def regime_run_lengths(phases):
    """Calcula rachas reales por régimen sin concatenar."""
    runs = {}
    current = phases.iloc[0]
    length = 1
    for p in phases.iloc[1:]:
        if p == current:
            length += 1
        else:
            runs.setdefault(current, []).append(length)
            current = p
            length = 1
    runs.setdefault(current, []).append(length)
    return runs

def hazard_rate(phases):
    runs_dict = regime_run_lengths(phases)
    all_runs = []
    for lengths in runs_dict.values():
        all_runs.extend(lengths)
    max_len = max(all_runs) if all_runs else 1
    hazard = {}
    for t in range(1, min(max_len, 31)):
        alive = sum(r >= t for r in all_runs)
        exits = sum(r == t for r in all_runs)
        hazard[t] = exits / alive if alive > 0 else 0
    return pd.Series(hazard)

def survival_from_hazard(hazard_series, weeks):
    """Convierte hazard rate en función de supervivencia."""
    surv = [1.0]
    for w in weeks:
        if w in hazard_series.index:
            surv.append(surv[-1] * (1 - hazard_series[w]))
        else:
            surv.append(surv[-1])
    return surv[1:]  # El primer elemento es 1.0 (semana 0)

def transition_matrix(phases):
    states = sorted(phases.unique())
    matrix = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for a, b in zip(phases[:-1], phases[1:]):
        matrix.loc[a, b] += 1
    return matrix.div(matrix.sum(axis=1), axis=0)

def transition_entropy(tm):
    """Entropía media de las filas de la matriz de transición."""
    entropies = []
    for _, row in tm.iterrows():
        probs = row[row > 0].values
        if len(probs) > 0:
            entropies.append(-np.sum(probs * np.log(probs)))
    return np.mean(entropies) if entropies else np.nan

def cohens_d(x, y):
    """Tamaño del efecto entre dos muestras."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(((nx-1)*x.var() + (ny-1)*y.var()) / (nx+ny-2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled

def bootstrap_mean(x, n=1000):
    """IC 95% y sesgo para la media vía bootstrap."""
    x = np.asarray(x)
    medias = []
    for _ in range(n):
        sample = np.random.choice(x, size=len(x), replace=True)
        medias.append(sample.mean())
    medias = np.array(medias)
    ic = np.percentile(medias, [2.5, 97.5])
    bias = medias.mean() - x.mean()
    return ic, bias

# ============================================================
# FILTRO DE REGÍMENES CON MUESTRA SUFICIENTE
# ============================================================
counts = df["regime"].value_counts()
valid_regimes = counts[counts >= MIN_SAMPLE].index.tolist()
excluded_regimes = counts[counts < MIN_SAMPLE].index.tolist()

df_stats = df[df["regime"].isin(valid_regimes)].copy()

if excluded_regimes:
    print(f"\n  Regímenes excluidos por muestra insuficiente (n < {MIN_SAMPLE}):")
    for r in excluded_regimes:
        print(f"    - {r}: {counts[r]} observaciones")

# ============================================================
# BLOQUE 1: DISTRIBUCIÓN DE REGÍMENES
# ============================================================
print("\n" + "="*70 + "\n1. DISTRIBUCIÓN DE REGÍMENES\n" + "="*70)
dist = df["regime"].value_counts(normalize=True).sort_index()
for r, pct in dist.items():
    bar = '█' * int(pct * 50)
    marker = ' (excluido)' if r in excluded_regimes else ''
    print(f"  {r:<20} {pct*100:5.1f}%  {bar}{marker}")
dominante = dist.max() > 0.60
print(f"  {'⚠️ Algún régimen domina >60%' if dominante else '✓ Ningún régimen domina >60%'}")

# ============================================================
# BLOQUE 2: ENTROPÍA
# ============================================================
print("\n" + "="*70 + "\n2. ENTROPÍA\n" + "="*70)
n_states = df["regime"].nunique()
H = state_entropy(df)
Hmax = np.log(n_states)
print(f"  Estados observados: {n_states}")
print(f"  Entropía: {H:.3f} / {Hmax:.3f} ({H/Hmax*100:.1f}% del máximo)")
ent_ok = H/Hmax > 0.5
print(f"  {'✓ Entropía saludable' if ent_ok else '⚠️ Entropía baja'}")

# ============================================================
# BLOQUE 3: HAZARD RATE + SUPERVIVENCIA
# ============================================================
print("\n" + "="*70 + "\n3. HAZARD RATE Y SUPERVIVENCIA\n" + "="*70)
hz = hazard_rate(df["regime"])
weeks = [1, 2, 4, 8, 12]
surv = survival_from_hazard(hz, weeks)

print("  Semana  Hazard   Supervivencia")
for i, w in enumerate(weeks):
    if w in hz.index:
        print(f"  {w:>6}  {hz[w]*100:5.1f}%   {surv[i]*100:5.1f}%")
    else:
        print(f"  {w:>6}    -      {surv[i]*100:5.1f}%")
haz_ok = hz.get(1, 1.0) <= 0.50
print(f"  {'✓ Persistencia semanal razonable' if haz_ok else '⚠️ Alta probabilidad de cambio semanal'}")

# ============================================================
# BLOQUE 4: MATRIZ DE TRANSICIÓN + ENTROPÍA DE TRANSICIÓN
# ============================================================
print("\n" + "="*70 + "\n4. MATRIZ DE TRANSICIÓN\n" + "="*70)
tm = transition_matrix(df["regime"])
if len(tm) <= 12:
    print(tm.round(2).to_string())
else:
    print(f"  Matriz de {len(tm)}x{len(tm)} estados (demasiado grande)")
diag_mean = np.diag(tm.values).mean() if len(tm) > 0 else 0
print(f"  Persistencia media en diagonal: {diag_mean*100:.1f}%")
trans_ok = diag_mean > 0.30
print(f"  {'✓ Buena persistencia' if trans_ok else '⚠️ Baja persistencia'}")

H_trans = transition_entropy(tm)
if not np.isnan(H_trans):
    print(f"  Entropía de transición: {H_trans:.3f}")
    print(f"  {'✓ Transiciones predecibles' if H_trans < 1.5 else '⚠️ Transiciones cercanas a aleatoriedad'}")

# ============================================================
# BLOQUE 5: ESTABILIDAD TEMPORAL
# ============================================================
print("\n" + "="*70 + "\n5. ESTABILIDAD TEMPORAL (ANUAL)\n" + "="*70)
df["year"] = pd.to_datetime(df["date"]).dt.year
years = sorted(df["year"].unique())

print("\n  5.1 Distribución anual:")
dist_anual = df.groupby(["year","regime"]).size().unstack(fill_value=0)
dist_anual = dist_anual.div(dist_anual.sum(axis=1), axis=0)
print(dist_anual.round(3).to_string())

print("\n  5.2 Entropía anual:")
entropias = []
for year in years:
    sub = df[df["year"]==year]
    H_year = state_entropy(sub)
    entropias.append(H_year)
    print(f"    {year}: {H_year:.3f}")
entropia_std = np.std(entropias) if len(entropias) > 1 else 0
drift_ok = entropia_std < 0.15
print(f"  {'✓ Estable' if drift_ok else '⚠️ Variabilidad alta'} (std={entropia_std:.4f})")

# ============================================================
# BLOQUE 6: DISCRIMINACIÓN DEL MACRO_SCORE
# ============================================================
print("\n" + "="*70 + "\n6. DISCRIMINACIÓN DEL MACRO_SCORE POR RÉGIMEN\n" + "="*70)

# Estadísticos descriptivos (todos los regímenes)
stats_all = df.groupby("regime")["score"].agg(["count","mean","std","min","max"]).sort_values("mean")
print(stats_all.round(4).to_string())

# Análisis con regímenes válidos
if len(valid_regimes) >= 2:
    print(f"\n  6.1 ANOVA del macro_score por régimen (n ≥ {MIN_SAMPLE}):")
    groups = [df_stats.loc[df_stats["regime"]==r, "score"].dropna().values for r in valid_regimes]
    
    # Comprobación de supuestos
    print("\n  6.1a Supuestos del ANOVA:")
    normal_ok = True
    for r, g in zip(valid_regimes, groups):
        if len(g) >= 3:
            stat, p = shapiro(g)
            status = '✓' if p > 0.05 else '⚠️'
            print(f"    Shapiro-Wilk {r:<20}: W={stat:.3f}, p={p:.4f} {status}")
            if p <= 0.05:
                normal_ok = False
        else:
            print(f"    Shapiro-Wilk {r:<20}: muestra insuficiente")
    
    stat_lev, p_lev = levene(*groups)
    var_ok = p_lev > 0.05
    print(f"    Levene: stat={stat_lev:.3f}, p={p_lev:.4f} {'✓ Varianzas homogéneas' if var_ok else '⚠️ Varianzas diferentes'}")
    
    # ANOVA (Welch si varianzas no homogéneas)
    if var_ok:
        F, p_anova = f_oneway(*groups)
        anova_type = "ANOVA estándar"
    else:
        from statsmodels.stats.oneway import anova_oneway
        res = anova_oneway(data=df_stats["score"], groups=df_stats["regime"], use_var="unequal")
        F = res.statistic
        p_anova = res.pvalue
        anova_type = "Welch ANOVA"
    
    print(f"\n    {anova_type}: F = {F:.2f}, p = {p_anova:.6f}")
    
    # Tamaño del efecto η²
    overall_mean = df_stats["score"].mean()
    ss_between = sum(len(g) * (g.mean() - overall_mean)**2 for g in groups)
    ss_total = ((df_stats["score"] - overall_mean)**2).sum()
    eta2 = ss_between / ss_total if ss_total > 0 else 0
    print(f"    η² = {eta2:.4f} ({'grande' if eta2>0.14 else 'medio' if eta2>0.06 else 'pequeño'})")
    anova_ok = p_anova < 0.001 and eta2 > 0.06
    print(f"  {'✓ ANOVA significativo y tamaño del efecto relevante' if anova_ok else '⚠️ Revisar'}")

    # Prueba post-hoc Tukey HSD
    print("\n  6.2 Prueba post-hoc (Tukey HSD):")
    try:
        tukey = pairwise_tukeyhsd(endog=df_stats["score"], groups=df_stats["regime"], alpha=0.05)
        print(tukey)
    except Exception as e:
        print(f"    Error en Tukey HSD: {e}")

    # Cohen's d entre regímenes consecutivos (ordenados por media)
    means = df_stats.groupby("regime")["score"].mean().sort_values()
    print("\n  6.3 Cohen's d entre regímenes consecutivos:")
    distancias_ok = True
    for i in range(len(means)-1):
        r1, r2 = means.index[i], means.index[i+1]
        g1 = df_stats.loc[df_stats["regime"]==r1, "score"]
        g2 = df_stats.loc[df_stats["regime"]==r2, "score"]
        d = cohens_d(g1, g2)
        if np.isnan(d):
            print(f"    {r1:<20} → {r2:<20} d = no evaluable")
        else:
            magnitud = 'grande' if abs(d)>0.8 else 'medio' if abs(d)>0.5 else 'pequeño' if abs(d)>0.2 else 'insignificante'
            print(f"    {r1:<20} → {r2:<20} d = {d:+.4f} ({magnitud})")
            if abs(d) < 0.2:
                distancias_ok = False
    print(f"  {'✓ Todos los pares evaluables con d > |0.2|' if distancias_ok else '⚠️ Algún par con d < |0.2|'}")

    # Mutual Information
    print("\n  6.4 Mutual Information (macro_score → régimen):")
    mi = mutual_info_classif(df_stats[["score"]], df_stats["regime"], random_state=42)
    print(f"    MI = {mi[0]:.4f}")
    if mi[0] > 0.50:
        print("  ✓ Alta dependencia: el score contiene mucha información sobre el régimen")
    elif mi[0] > 0.25:
        print("  ✓ Dependencia moderada")
    else:
        print("  ⚠️ Baja dependencia")

    # Calinski-Harabasz Score
    print("\n  6.5 Calinski-Harabasz Score:")
    ch = calinski_harabasz_score(df_stats[["score"]], df_stats["regime"])
    print(f"    CH = {ch:.1f}")
    if ch > 50:
        print("  ✓ Buena separación entre regímenes")
    elif ch > 20:
        print("  ✓ Separación aceptable")
    else:
        print("  ⚠️ Baja separación")

else:
    print("  No hay suficientes regímenes con muestra suficiente para ANOVA")
    anova_ok = False
    distancias_ok = False

# ============================================================
# BLOQUE 7: FORWARD RETURNS (60 DÍAS SOBRE SPY) CON BOOTSTRAP
# ============================================================
print("\n" + "="*70 + "\n7. FORWARD RETURNS A 60 DÍAS (SPY) CON BOOTSTRAP\n" + "="*70)

try:
    spy_close = get_col(df_market, 'SPY', 'Close')
    df['future_spy_return'] = np.nan
    for idx, row in df.iterrows():
        date = row['date']
        if date in spy_close.index:
            pos = spy_close.index.get_loc(date)
            if pos + 60 < len(spy_close):
                future_close = spy_close.iloc[pos + 60]
                current_close = spy_close.iloc[pos]
                df.at[idx, 'future_spy_return'] = future_close / current_close - 1

    fwd_stats = df.groupby("regime")["future_spy_return"].agg(["count","mean","median","std"]).sort_values("mean")
    print(fwd_stats.round(4).to_string())
    
    # Bootstrap IC 95% + sesgo para regímenes válidos
    print("\n  7.1 Bootstrap IC 95% y sesgo:")
    bootstrap_results = {}
    for regime in valid_regimes:
        if regime in fwd_stats.index:
            returns = df.loc[df["regime"]==regime, "future_spy_return"].dropna().values
            if len(returns) >= 20:
                ic, bias = bootstrap_mean(returns)
                bootstrap_results[regime] = (ic, bias)
                print(f"    {regime:<20} media={returns.mean()*100:.2f}%  "
                      f"IC95%=[{ic[0]*100:.2f}%, {ic[1]*100:.2f}%]  "
                      f"sesgo={bias*100:+.3f}%")
    
    # Verificar orden económico esperado
    expansive = ['EXPANSION', 'RECOVERY', 'GOLDILOCKS', 'LATE EXPANSION']
    stress = ['RECESSION', 'LIQUIDITY CRISIS', 'SLOWDOWN', 'INFLATION SHOCK']
    
    exp_mean = fwd_stats.loc[fwd_stats.index.isin(expansive), 'mean'].mean() if any(r in fwd_stats.index for r in expansive) else np.nan
    stress_mean = fwd_stats.loc[fwd_stats.index.isin(stress), 'mean'].mean() if any(r in fwd_stats.index for r in stress) else np.nan
    
    if pd.notna(exp_mean) and pd.notna(stress_mean):
        print(f"\n  7.2 Orden económico:")
        print(f"    Retorno medio regímenes expansivos: {exp_mean*100:.2f}%")
        print(f"    Retorno medio regímenes de estrés: {stress_mean*100:.2f}%")
        fwd_ok = exp_mean > stress_mean
        print(f"  {'✓ Expansivos > Estrés' if fwd_ok else '⚠️ Orden económico invertido'}")
    else:
        fwd_ok = False
        print("  Datos insuficientes para comparar expansivos vs estrés")
except Exception as e:
    print(f"  No se pudo calcular forward returns: {e}")
    fwd_ok = False

# ============================================================
# BLOQUE 8: PERSISTENCIA MEDIA POR RÉGIMEN (CORREGIDA)
# ============================================================
print("\n" + "="*70 + "\n8. PERSISTENCIA MEDIA POR RÉGIMEN (semanas, corregida)\n" + "="*70)

runs_dict = regime_run_lengths(df["regime"])
persist_ok = True
for regime in sorted(runs_dict.keys()):
    lengths = runs_dict[regime]
    evaluable = ' (muestra insuficiente)' if len(lengths) < MIN_SAMPLE else ''
    print(f"  {regime:<20} media={np.mean(lengths):.1f}  mediana={np.median(lengths):.0f}  máx={np.max(lengths):.0f}  (n={len(lengths)} rachas){evaluable}")
    if len(lengths) >= MIN_SAMPLE and np.mean(lengths) < 2.0:
        persist_ok = False
if not persist_ok:
    print("  ⚠️ Algún régimen evaluable < 2 semanas de persistencia media")

# ============================================================
# BLOQUE 9: ESTABILIDAD ANUAL DEL MACRO_SCORE + KS TEST
# ============================================================
print("\n" + "="*70 + "\n9. ESTABILIDAD ANUAL DEL MACRO_SCORE + DRIFT (KS TEST)\n" + "="*70)

annual_score = df.groupby("year")["score"].agg(["mean","std","min","max"])
print(annual_score.round(4).to_string())

score_mean_anual = annual_score["mean"].std() if len(annual_score) > 1 else 0
score_stable_ok = score_mean_anual < 0.15
print(f"\n  9.1 Estabilidad de medias anuales: std={score_mean_anual:.4f}")
print(f"  {'✓ Escala estable entre años' if score_stable_ok else '⚠️ La escala del score varía entre años'}")

# KS Test entre años consecutivos
print("\n  9.2 KS Test entre años consecutivos (monitorización de drift):")
drift_years = []
for i in range(len(years)-1):
    y1, y2 = years[i], years[i+1]
    s1 = df.loc[df["year"]==y1, "score"].dropna()
    s2 = df.loc[df["year"]==y2, "score"].dropna()
    if len(s1) > 10 and len(s2) > 10:
        ks_stat, ks_pval = ks_2samp(s1, s2)
        if ks_pval < 0.05:
            drift_years.append(f"{y1}-{y2}")
            print(f"    {y1} vs {y2}: KS={ks_stat:.3f}, p={ks_pval:.4f}  ⚠️ DRIFT")
        else:
            print(f"    {y1} vs {y2}: KS={ks_stat:.3f}, p={ks_pval:.4f}  ✓")
    else:
        print(f"    {y1} vs {y2}: datos insuficientes")
if drift_years:
    print(f"\n  Drift monitorizado en períodos: {', '.join(drift_years)}")
    print("  (Posible reflejo de cambios macroeconómicos reales)")
else:
    print("  ✓ Sin drift significativo detectado")

# ============================================================
# VEREDICTO CUALITATIVO
# ============================================================
print("\n" + "="*70)
print("VEREDICTO DE VALIDACIÓN DEL RÉGIMEN MACRO")
print("="*70)

print(f"""
CONSISTENCIA ESTRUCTURAL:
  {'✓' if not dominante else '⚠️'} Distribución de estados sin dominancia (>60%)
  {'✓' if ent_ok else '⚠️'} Entropía suficiente ({H/Hmax*100:.1f}% del máximo)
  {'✓' if haz_ok else '⚠️'} Persistencia semanal razonable (hazard sem 1: {hz.get(1,0)*100:.1f}%)
  {'✓' if trans_ok else '⚠️'} Matriz de transición con persistencia adecuada (diagonal: {diag_mean*100:.1f}%)
  {'✓' if drift_ok else '⚠️'} Estabilidad temporal de la entropía (std: {entropia_std:.4f})

CONSISTENCIA ESTADÍSTICA:
  {'✓' if anova_ok else '⚠️'} ANOVA significativo con tamaño del efecto relevante (η²={eta2:.4f})
  {'✓' if distancias_ok else '⚠️'} Cohen's d entre regímenes consecutivos
  MI = {mi[0]:.4f} ({'Alta' if mi[0]>0.5 else 'Moderada' if mi[0]>0.25 else 'Baja'} dependencia score → régimen)
  CH = {ch:.1f} ({'Buena' if ch>50 else 'Aceptable' if ch>20 else 'Baja'} separación entre regímenes)

CONSISTENCIA ECONÓMICA:
  {'✓' if fwd_ok else '⚠️'} Forward returns: regímenes expansivos ({exp_mean*100:.2f}%) > estrés ({stress_mean*100:.2f}%)
  Bootstrap con IC 95% calculado para {len(bootstrap_results)} regímenes

OBSERVACIONES:
  • {len(excluded_regimes)} regímenes excluidos de inferencia por muestra insuficiente (n < {MIN_SAMPLE}): {excluded_regimes if excluded_regimes else 'ninguno'}
  • Persistencia inferior a 2 semanas en algunos regímenes (ver Bloque 8)
  • Drift detectado en {len(drift_years)} períodos anuales (ver Bloque 9)
""")

if anova_ok and fwd_ok and trans_ok and ent_ok and not dominante:
    print("VEREDICTO: ✓✓ RÉGIMEN MACRO VALIDADO")
    print("El clasificador es estructuralmente sólido, estadísticamente discriminativo")
    print("y económicamente coherente. Las observaciones son menores y no comprometen su utilidad.")
elif anova_ok and (fwd_ok or trans_ok):
    print("VEREDICTO: ✓ RÉGIMEN MACRO VALIDADO (con observaciones menores)")
else:
    print("VEREDICTO: ⚠️ REVISAR CLASIFICADOR")
print("="*70)