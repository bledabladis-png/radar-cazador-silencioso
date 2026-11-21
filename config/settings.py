"""
Global configuration for Sector Rotation Radar v4.3.

This module centralizes shared constants: time windows,
data quality thresholds and cache parameters.
Model-specific weights remain in config/weights.py.
"""

import os

# ============================================================
# TIME WINDOWS — TRADING SESSIONS
# ============================================================

RS_MEDIUM_WINDOW = 63         # ~3 meses
MOMENTUM_LONG_WINDOW = 126    # ~6 meses
RS_STRUCTURAL_WINDOW = 252    # ~1 año

VOLATILITY_BASELINE_WINDOW = 756  # ~3 años (252 * 3)

# ============================================================
# DATA CACHE & DOWNLOADS
# ============================================================

CACHE_HOURS = 23              # regenerar caché tras 23 horas
CACHE_VALIDATE_TRADING_DATE = True  # verificar que la caché cubre el último día de mercado
CACHE_MIN_COVERAGE = 0.90  # cobertura minima de la ultima fila del cache para aceptar cache-hit

# ============================================================
# FRESHNESS THRESHOLDS (src/report/helpers.py)
# ============================================================
# (max_current, max_recent, max_stale) en dias. Ver helpers.py.
# Cada tupla corresponde a una fuente. Si un umbral debe cambiarse,
# cambiarlo aqui: helpers.py lee estas constantes.

FRESHNESS_DEFAULT = (3, 7, 14)    # resto de fuentes
FRESHNESS_CBOE = (3, 5, 10)       # CBOE (options)
FRESHNESS_YAHOO = (3, 7, 14)      # Yahoo Finance
# F6-02 (2026-09-28): umbrales ajustados. Antes (30, 45, 60) marcaba
# CURRENT a un retraso de 26d, fuera del rango regulatorio documentado
# (2-4 semanas = 14-28d). Ahora CURRENT <= 20d (dentro de lo normal),
# RECENT <= 28d (limite superior del retraso regulatorio).
FRESHNESS_FINRA = (20, 28, 45)    # FINRA (retraso regulatorio 2-4 sem.)
FRESHNESS_FRED = (30, 60, 90)     # FRED (series macro)
# H7 (2026-10-16, auditoria): SEC N-PORT es trimestral (latencia
# regulatoria ~45-60d). Un ciclo completo es 90d. Umbrales a 1/2/3
# trimestres.
FRESHNESS_SEC = (90, 180, 270)    # SEC N-PORT (trimestral)
# CFTC TFF: semanal. 45d ya es anomalia.
FRESHNESS_CFTC = (45, 90, 120)    # CFTC Traders in Financial Futures

# ============================================================
# CFTC POSITION FLOW (data/providers/cftc_data.py)
# ============================================================
# CFTC_HISTORY_DAYS: ventana historica de contratos incluidos en el CSV.
# CFTC_ACTIVE_CONTRACT_DAYS: un contrato se descarta si su ultimo dato
#   reportado es mas antiguo que este valor respecto al maximo global.
# Nota: la clasificacion de frescura de la fuente (indicators/data_quality.py)
#   usa umbrales distintos (45/90/120) por frecuencia declarada. No son el
#   mismo concepto: aqui se decide actividad operativa del contrato, alli
#   frescura de la fuente.

CFTC_HISTORY_DAYS = 365
CFTC_ACTIVE_CONTRACT_DAYS = 30

# ============================================================
# DATA QUALITY
# ============================================================

EXPECTED_SECTOR_COUNT = 11
MIN_SECTOR_COVERAGE = 0.70    # umbral minimo de cobertura por sector (se marca [BAJA] si < 70%)

# P2.2 (2026-10-13, auditor): umbral de cobertura para la fase
# Wyckoff por sector. Distinto de MIN_SECTOR_COVERAGE (ese aplica
# a breadth de precio; este a la distribucion de fases de los
# componentes). Bajo 50% la distribucion no es representativa
# del sector. Constante declarada para evitar hardcodeos en el
# render y permitir ajuste en un unico sitio.
WYCKOFF_MIN_COVERAGE_PCT = 50.0

# ============================================================
# DARK POOL / ATS HISTORY
# ============================================================

DARKPOOL_FULL_HISTORY_WEEKS = 104  # Z-Score completo (2 años)
MTE_STATE_FILE = "outputs/state/mte_state.json"  # archivo de estado del Market Transition Engine

# ============================================================
# M3 — CENTRALIZED TIME WINDOWS (v4.0)
# ============================================================

FLOW_ZSCORE_WINDOW = 60
ETF_PRIMARY_FLOW_ZSCORE_WINDOW = 120  # ventana para ETF Primary Flow (SSGA)
FLOW_EWM_SPAN = 10
FLOW_CMF_WINDOW = 20

MOMENTUM_SHARPE_WINDOW = 63
MOMENTUM_PRICE_WINDOW = 20

PERSISTENCE_LOOKBACK = 12

BREADTH_EMA_FAST = 20
BREADTH_EMA_MEDIUM = 50
BREADTH_EMA_SLOW = 200

# ============================================================
# WYCKOFF (v4.0 audit)
# ============================================================
WYCKOFF_VOLUME_WINDOW = 20
WYCKOFF_TREND_FAST_MA = 50
WYCKOFF_TREND_SLOW_MA = 200

# ============================================================
# WYCKOFF v4.1
# ============================================================
WYCKOFF_THRESHOLD_MARKUP = 0.30
WYCKOFF_THRESHOLD_ACCUMULATION = 0.00
WYCKOFF_THRESHOLD_DISTRIBUTION = -0.30
WYCKOFF_ATR_WINDOW = 20
WYCKOFF_VOLUME_ZSCORE_WINDOW = 60
# Pesos calibrados por ablacion (importancia empirica)

# v4.2: pesos para scores estructural y tactico
WYCKOFF_STRUCT_WEIGHT_TREND = 0.60
WYCKOFF_STRUCT_WEIGHT_COMPRESSION = 0.40
WYCKOFF_TACT_WEIGHT_VOLUME = 0.50
WYCKOFF_TACT_WEIGHT_EFFORT = 0.50
# WYCKOFF v1.3 (P1 critico, hallazgo t_norm)
WYCKOFF_T_NORM_K = 0.25  # CALIBRADO 2026-10-02. Ver 12_calibracion_K_5b2.md.
# WYCKOFF v1.9 (candidate/confirmed): SOW para confirmar DISTRIBUTION.
# CANDIDATA v1.9 FROZEN_FOR_VALIDATION / NOT_PRODUCTION.
# Valores calibrados en 5b.4-bis. Validacion out-of-sample 5b.X
# pendiente. Ver docs/auditoria/wyckoff/01_contrato_semantico_v1_9.md.
WYCKOFF_SOW_WINDOW_N = None   # CANDIDATA v1.9 = 60. NO ACTIVAR hasta 5b.X exitoso.
WYCKOFF_SOW_MAX_AGE_M = None  # CANDIDATA v1.9 = 30. NO ACTIVAR hasta 5b.X exitoso.
WYCKOFF_SOW_X_ATR = None      # CANDIDATA v1.9 = 0.25. NO ACTIVAR hasta 5b.X exitoso.
WYCKOFF_SOW_Y_VOL = None      # CANDIDATA v1.9 = 1.10. NO ACTIVAR hasta 5b.X exitoso.
WYCKOFF_COMBINED_STRUCT_WEIGHT = 0.70
WYCKOFF_COMBINED_TACT_WEIGHT = 0.30

# ============================================================
# FASE B v4.3 - CENTRALIZACION DE PARAMETROS
# ============================================================

FINANCIAL_CONDITIONS_WEIGHTS = {'vix': 0.40, 'credit': 0.30, 'dollar': 0.15, 'curve': 0.15}
FINANCIAL_CONDITIONS_THRESHOLDS = {
    'abundante': 0.3,
    'neutral': 0.0,
    'estrecha': -0.3,
    'high_stress': -0.6,
}

PCR_THRESHOLDS = {
    'panico': 2.0,
    'miedo': 1.0,
    'neutral': -1.0,
    'optimismo': -2.0,
}

IHR_THRESHOLDS = {
    'cobertura_extrema': 2.5,
    'cobertura_alta': 1.6,
    'equilibrado': 1.2,
    'especulacion_alta': 0.8,
}

DARKPOOL_THRESHOLDS = {
    'extremadamente_alta': 2.5,
    'muy_alta': 1.5,
    'alta': 0.5,
    'normal': -0.5,
    'baja': -1.5,
    'muy_baja': -2.5,
}

# ============================================================
# RUTAS DE CACHES LOCALES (single source of truth)
# ============================================================

CACHE_MARKET_PATH = 'data/market_data.parquet'
CACHE_STOCKS_PATH = 'data/stock_prices.parquet'

# FU-002 (2026-09-15): umbral heuristico de duplicacion para manifiestos.
# No es ley estadistica: si >50% tickers tienen Close[-1]==Close[-2] en
# sesion esperada, el artefacto se marca INVALID en su manifest.
MANIFEST_DUP_THRESHOLD = 0.5

# ============================================================
# SELECCION DE LIDERES (C18)
# ============================================================

TOP_N_CANDIDATES = 15   # pre-filtro por weight antes del WLS
TOP_N_SECTOR_COMPONENTS = 20  # componentes por sector descargados y considerados en breadth

# Margen de publicacion del dataset SEC 13F tras el CIERRE DE LA VENTANA
# de envio (no tras el cierre del trimestre ni el deadline de filing).
# SEC: "the quarterly data sets will be run for the prior three months
# following the end of the months: February, May, August, November"
# (sec.gov/data-research/sec-markets-data/form-13f-data-sets).
# Ejemplo Q3: cierre 30-sep, ventana 01sep-30nov; el dataset no existe
# antes del 30-nov. Bracket observado: jun-ago 2026 publicado entre
# 31-ago (update de pagina) y 24-sep (descarga real, run 36014804820).
# Margen 7d dentro del bracket. Coherencia: cron mensual dia 10
# (window-end + 9/10d) >= margen; centinela en
# tests/test_update_sec_13f.py lo fija con el calendario real.
# Cambiar este valor obliga a revisar test_latest_published_quarter
# y el cron de .github/workflows/update_sec_13f.yml.
# Sustituye a SEC_13F_QUARTER_LAG_DAYS=50 (ancla incorrecta: deadline
# 45d + 5d margen). Causa del 404 estructural del run 44585740091.
# Ver REGISTRO: SEC-13F-PUBLICATION-ANCHOR.
SEC_13F_PUBLICATION_MARGIN_DAYS = 7
TOP_N_LEADERS = 5       # cuantos se muestran en el reporte

# 2026-10-30 (O2 coherencia analisis=descarga): compute_all_stock_metrics
# usaba TOP_N_CANDIDATES=15 como cap de analisis por sector. Tras
# universe_expansions.yaml (c879df71) + get_stock_list(), el parquet
# tiene mas holdings del top-15 por sector (XLF 34, XLV 26, XLU 21,
# XLY 21). El cap 15 dejaba fuera tickers descargados: incoherencia
# "descarga vs analisis". Este guard reemplaza el cap rigido por uno
# defensivo, para no procesar ETFs gigantes si alguna vez se anade
# el universo completo (IWM 1977 holdings).
MAX_CANDIDATES_PER_SECTOR = 100

# ============================================================
# CONFIDENCE (C19)
# ============================================================
# Confidence = 1 - (max - min) / CONFIDENCE_RANGE_DIVISOR
# Divisor 2.0 = normalizacion teorica con componentes en [-1, +1].
# Politica conservadora: disagreement extremo penaliza fuerte.
CONFIDENCE_RANGE_DIVISOR = 2.0

# ============================================================
# FU-021-5 — CONTRATO TEMPORAL
# ============================================================

CURRENT_TEMPORAL_CONTRACT_VERSION = "FU-021-5-v2"

# ============================================================
# LSE SCRAPER (F-IAE-LSE-INTEGRATION)
# ============================================================
# Override parcial de Close para tickers .L (LSE) usando los JSON
# del scraper privado lse-close-scraper. Solo se envia via env var
# LSE_SCRAPER_DATOS_DIR. El repo/ref/commit los resuelve el workflow
# (dictamen auditor externo 2026-09-25, D3).
LSE_SCRAPER_DATOS_DIR = os.environ.get(
    "LSE_SCRAPER_DATOS_DIR",
    "data/external/lse_close/datos",
)
LSE_SCRAPER_PROVENANCE_PATH = os.environ.get(
    "LSE_SCRAPER_PROVENANCE_PATH",
    "data/lse_close_provenance.json",
)
LSE_SCRAPER_REPO = "bledabladis-png/lse-close-scraper"

# ============================================================
# H14 + H15 (REP-AUDIT-20261017-capa2): umbrales de modulacion
# del texto del reporte.
# ============================================================
# H14: banda neutral del Z-score HYG/LQD en la alerta de credito.
# Ampliacion de 0.5 a 0.75. Con |Z| < 0.75 la desviacion no confirma
# por si sola estres sistemico; el texto publica 'proximo a neutral'.
# Coherente con la critica del auditor externo (2026-10-17): con
# Z=-0.60 el reporte decia 'el credito lo confirma', lo cual exageraba
# la evidencia. La banda anterior |Z|<0.5 era demasiado estrecha.
NEUTRAL_BAND_HYG_LQD_ZSCORE = 0.75

# H15: umbral del Funding & Liquidity Stress (FLS) para modular el
# texto del Resumen de Regimenes. Por encima del umbral, el texto
# declara explicitamente que el modulo de financiacion SI senala
# estres de corto plazo, y separa esa dimension de la liquidez real
# (FRED). Por debajo, mantiene el texto historico. Coherente con
# la critica del auditor (2026-10-17): 'NEUTRAL de liquidez FRED'
# no contradice 'FLS 85/100', pero el reporte debe explicarlo.
FLS_STRESS_THRESHOLD = 70

# ============================================================
# H17 (REP-AUDIT-20261017-capa2b): umbral de materialidad
# para las etiquetas de divergencia precio-flujo.
# ============================================================
# Retornos |r| < umbral son ruido: no deben publicarse como
# 'Confirmacion de debilidad' o 'Absorcion potencial' sin
# cualificar. La etiqueta anade el sufijo ' (debil)'.
# Casos reales del reporte 2026-10-17: XLRE ret=-0.04%,
# XLC ret=-0.11%. Antes salian como etiquetas categoricas.
FLOW_DIVERGENCE_MATERIALITY_THRESHOLD = 0.01
# 13.5 (auditoria analista 2026-11-21): umbrales para cualificar "Absorcion potencial".
# Se aplican solo cuando price_ret < 0 y flow_sum > 0. Si el flujo es insignificante,
# poco persistente o revirtiendo a 5d, la etiqueta se matiza. Sin estos umbrales,
# sectores con flujo de ruido (ej. XLI con 4.9M y 0.00% AUM) recibian la misma
# etiqueta que sectores con flujo material (ej. XLK con 893M).
ABSORPTION_FLOW_PCT_AUM_THRESHOLD = 0.05   # 0.05% del AUM: por debajo, flujo no material
ABSORPTION_PERSISTENCE_THRESHOLD = 0.50    # 50% de dias positivos en ventana 20d
