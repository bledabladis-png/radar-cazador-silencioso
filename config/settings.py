"""
Global configuration for Sector Rotation Radar v3.15.

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
