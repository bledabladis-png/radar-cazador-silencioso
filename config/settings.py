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

MAX_RETRIES = 3

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