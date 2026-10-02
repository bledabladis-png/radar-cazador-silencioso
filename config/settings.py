"""
Global configuration for Sector Rotation Radar v4.3.

This module centralizes shared constants: time windows,
data quality thresholds, cache parameters, and SLPM coverage.
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

# ============================================================
# DARK POOL / ATS HISTORY
# ============================================================

DARKPOOL_FULL_HISTORY_WEEKS = 104  # Z-Score completo (2 años)
MTE_STATE_FILE = "outputs/state/mte_state.json"  # archivo de estado del Market Transition Engine

# ============================================================
# SLPM COVERAGE
# ============================================================

SLPM_EXPECTED_LEADERS = 5  # número típico de líderes analizados por sector

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
# WYCKOFF v1.6 (candidate/confirmed): SOW para confirmar DISTRIBUTION.
# N y M son PROPUESTOS. Calibracion en 5b.3.
WYCKOFF_SOW_WINDOW_N = None   # PROPUESTO (5b.3). Grid N in {20,30,40,60}. Fail-closed.
WYCKOFF_SOW_MAX_AGE_M = None  # PROPUESTO (5b.3). Grid M in {5,10,15,20,30}. Fail-closed.
WYCKOFF_SOW_X_ATR = None  # PROPUESTO (5b.3). Grid X in {0.25,0.50,0.75,1.00}. Fail-closed.
WYCKOFF_SOW_Y_VOL = None  # PROPUESTO (5b.3). Grid Y in {1.10,1.20,1.50}. Fail-closed.
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

# SEC 13F trimestral: dias minimos desde el cierre del trimestre
# antes de intentar descargar el dataset.
# Politica oficial (2026-09-30): 50d. Latencia regulatoria ~45d
# (03_IAE.md); los 5d extra son margen. Alinea con cron
# 20-feb/may/ago/nov (=50d post cierre real) y con
# scripts/update_sec_13f.py --latest, unico consumidor de este valor.
# Cambiar 50 obliga a revisar
# tests/test_update_sec_13f.py::test_latest_published_quarter.
SEC_13F_QUARTER_LAG_DAYS = 50
TOP_N_LEADERS = 5       # cuantos se muestran en el reporte

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