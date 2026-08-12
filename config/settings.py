"""
Global configuration for Sector Rotation Radar v4.2.

This module centralizes shared constants: time windows,
data quality thresholds, cache parameters, and SLPM coverage.
Model-specific weights remain in config/weights.py.
"""

# ============================================================
# TIME WINDOWS — TRADING SESSIONS
# ============================================================

MOMENTUM_WINDOW = 20          # sesiones bursátiles
RS_MEDIUM_WINDOW = 63         # ~3 meses
MOMENTUM_LONG_WINDOW = 126    # ~6 meses
RS_STRUCTURAL_WINDOW = 252    # ~1 año

VOLATILITY_WINDOW = 20        # sesiones para desviación estándar
VOLATILITY_BASELINE_WINDOW = 756  # ~3 años (252 * 3)

ZSCORE_SHORT_WINDOW = 60      # ~3 meses
ZSCORE_YEAR_WINDOW = 252      # ~1 año

TREND_EMA_WINDOW = 50         # EMA rápida para tendencia
BREADTH_WINDOW = 20           # ventana para breadth táctico

# ============================================================
# DATA CACHE & DOWNLOADS
# ============================================================

CACHE_HOURS = 23              # regenerar caché tras 23 horas
CACHE_VALIDATE_TRADING_DATE = True  # verificar que la caché cubre el último día de mercado

DOWNLOAD_BATCH_SIZE = 5

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
RETRY_BACKOFF_MULTIPLIER = 2

REQUEST_TIMEOUT = 30          # timeout genérico
FINRA_REQUEST_TIMEOUT = 60    # FINRA requiere más tiempo por paginación

# ============================================================
# DATA QUALITY
# ============================================================

MAX_NAN_RATIO = 0.10          # 10% máximo de NaN por ticker
MIN_VALID_TICKERS = 5         # mínimo global para operar

EXPECTED_SECTOR_COUNT = 11
MIN_VALID_SECTORS = 8         # al menos 8 sectores para ranking fiable
MIN_SECTOR_COVERAGE = 0.80    # 80% de los 11 sectores

# ============================================================
# PCR / OPTIONS HISTORY
# ============================================================

PCR_MIN_HISTORY_DAYS = 20
PCR_CONFIDENCE_HISTORY_DAYS = 60
PCR_FULL_HISTORY_DAYS = 252

# ============================================================
# DARK POOL / ATS HISTORY
# ============================================================

DARKPOOL_MIN_HISTORY_WEEKS = 13  # mínimo para mostrar señal
DARKPOOL_ZSCORE_WINDOWS = (13, 26, 52, 104)
DARKPOOL_FULL_HISTORY_WEEKS = 104  # Z-Score completo (2 años)
MTE_STATE_FILE = "outputs/mte_state.json"  # archivo de estado del Market Transition Engine

# ============================================================
# SLPM COVERAGE
# ============================================================

SLPM_TOTAL_SECTORS = 11
SLPM_MIN_COVERAGE_WARNING = 0.50
SLPM_FULL_COVERAGE = 1.00
SLPM_EXPECTED_LEADERS = 5  # número típico de líderes analizados por sector

# ============================================================
# DATA FRESHNESS — DAYS
# ============================================================

FRESHNESS_CURRENT_DAYS = 7
FRESHNESS_RECENT_DAYS = 14
FRESHNESS_STALE_DAYS = 21
# > 21 days = ARCHIVAL



# ============================================================
# CACHE TTL BY PROVIDER (horas)
# ============================================================

CACHE_TTL = {
    "yahoo": 23,
    "fred": 168,     # 7 días
    "cboe": 24,
    "finra": 168,    # 7 días
}

# ============================================================
# M3 — CENTRALIZED TIME WINDOWS (v4.0)
# ============================================================

FLOW_ZSCORE_WINDOW = 60
FLOW_EWM_SPAN = 10
FLOW_CMF_WINDOW = 20

MOMENTUM_SHARPE_WINDOW = 63
MOMENTUM_PRICE_WINDOW = 20

PERSISTENCE_LOOKBACK = 12

BREADTH_EMA_FAST = 20
BREADTH_EMA_MEDIUM = 50
BREADTH_EMA_SLOW = 200

WYCKOFF_MIN_PERIODS = 60

def validate_windows():
    warnings = []
    if FLOW_ZSCORE_WINDOW <= FLOW_EWM_SPAN:
        warnings.append("FLOW_ZSCORE_WINDOW debe ser mayor que FLOW_EWM_SPAN")
    if PERSISTENCE_LOOKBACK > MOMENTUM_PRICE_WINDOW:
        warnings.append("PERSISTENCE_LOOKBACK no deberia exceder MOMENTUM_PRICE_WINDOW")
    if WYCKOFF_MIN_PERIODS < 40:
        warnings.append("WYCKOFF_MIN_PERIODS puede ser insuficiente")
    if warnings:
        print("ADVERTENCIAS DE VENTANAS:")
        for w in warnings:
            print(f"  - {w}")
    return len(warnings) == 0

# ============================================================
# WYCKOFF (v4.0 audit)
# ============================================================
WYCKOFF_VOLUME_WINDOW = 20
WYCKOFF_TREND_FAST_MA = 50
WYCKOFF_TREND_SLOW_MA = 200
WYCKOFF_MIN_PERIODS = 60

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

