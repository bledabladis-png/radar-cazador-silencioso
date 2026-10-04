"""Constantes del SOW Protocol v4.1.

Referencia: docs/auditoria/wyckoff/SOW_PROTOCOL_V4_1.md (pendiente de
congelacion por el auditor). Este modulo NO contiene logica. Solo
constantes, para tener un unico punto de verdad.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# --- Datos ---
DATA_PARQUET = ROOT / "data" / "stock_prices_extended.parquet"
MIN_OBS = 200

# --- Universo temporal ---
DATASET_START = "2015-01-02"
DATASET_END = "2026-10-01"

# --- Outer folds (8). Tupla: (train_start, train_end, test_start, test_end) ---
OUTER_FOLDS = (
    ("2015-01-02", "2016-12-31", "2017-01-01", "2017-12-31"),
    ("2015-01-02", "2017-12-31", "2018-01-01", "2018-12-31"),
    ("2015-01-02", "2018-12-31", "2019-01-01", "2019-12-31"),
    ("2015-01-02", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2015-01-02", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2015-01-02", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2015-01-02", "2022-12-31", "2023-01-01", "2024-12-31"),
    ("2015-01-02", "2024-12-31", "2025-01-01", "2026-10-01"),
)

# --- Grid de 240 combinaciones ---
GRID_N = (20, 30, 40, 60)
GRID_M = (5, 10, 15, 20, 30)
GRID_X_ATR = (0.25, 0.50, 0.75, 1.00)
GRID_Y_VOL = (1.10, 1.20, 1.50)

# --- Regimen de referencia (seccion 3 del protocolo v4.1) ---
# Regimen = 223 tickers US con sector asignado en data/etf_holdings.csv.
# Experimento SOW = 316 tickers (todos los del parquet extendido).
SECTOR_MAP_CSV = ROOT / "data" / "etf_holdings.csv"
REGIME_REFERENCE_SECTORS = (
    "XLK", "XLF", "XLV", "XLE", "XLY", "XLP",
    "XLI", "XLB", "XLU", "XLRE", "XLC",
)

# --- Regimen de mercado (seccion 3 del protocolo) ---
REGIME_MIN_SECTORES_VOL = 6
REGIME_MIN_TICKERS_RET = 5
REGIME_VOL_WINDOW = 20
REGIME_VOL_DDOF = 1

# --- Baseline estructural en t0 (seccion 4.3) ---
BASELINE_DELTA_WINDOW = 20

# --- Outcome ---
HORIZON = 20

# --- Purga por M (seccion 8.2) ---
PURGE_BY_M = {5: 25, 10: 30, 15: 35, 20: 40, 30: 50}

# --- Soporte minimo (seccion 6) ---
SUPPORT_MIN_N_CONF = 20
SUPPORT_MIN_N_BASE = 20
SUPPORT_MIN_TICKERS_CONF = 5
SUPPORT_MIN_TICKERS_BASE = 5
SUPPORT_MIN_BLOCKS_B20 = 4

# --- Bootstrap MBB (seccion 7) ---
BOOT_B_PRIMARY = 20
BOOT_B_SENSITIVITY = 40
BOOT_B_DIAGNOSTIC = 50
BOOT_N_REPLICAS = 2000
BOOT_SEED = 20261004

# --- Placebos (seccion 11) ---
PLACEBO_MASTER_SEED = 20261004
PLACEBO_N_PERMUTACIONES = 1000
PLACEBO_LEAD_SESIONES = 40

# --- Decision (seccion 13) ---
DELTA_MIN = 0.05
DELTA_PLACEBO = 0.025
DECISION_PASS_FRACTION = 2.0 / 3.0
DECISION_MIN_EVAL = 6
DECISION_MIN_EVAL_STRESS = 4
DECISION_MIN_EVAL_NORMAL = 4
DECISION_ALPHA_95 = (2.5, 97.5)
DECISION_ALPHA_90 = (5.0, 95.0)

# --- Power analysis (seccion 12) ---
POWER_N_REPLICAS = 2000
POWER_SCENARIOS = (
    # (nombre, rd_normal, rd_stress)
    ("S0_nulo",              0.0,   0.0),
    ("S1_condicional",       0.0,   0.05),
    ("S2_universal",         0.05,  0.05),
    ("S3_universal_fuerte",  0.08,  0.08),
    ("S4_condicional_fuerte",0.0,   0.08),
    ("C_N0",                 0.0,   0.05),
    ("C_N1",                 0.02,  0.05),
    ("C_N2",                 0.04,  0.05),
    ("C_N3",                 0.049, 0.05),
    ("C_N4",                 0.05,  0.05),
)
POWER_MIN_POTENCIA = 0.80
POWER_MAX_FALSE_ACTIVATION = 0.05
POWER_DGP_TOLERANCIA = 1e-6
POWER_DGP_MAX_ITER = 200

# --- Inner selection (seccion 9) ---
INNER_N_SEGMENTOS = 4
INNER_LAMBDA_IQR = 0.5
INNER_TIEBREAK_BANDA = 0.001
INNER_MIN_FOLDS_ELEGIBLES = 2
INNER_MIN_FOLDS_PASAN = 2

# --- Modelo primario (seccion 5.7) ---
PRIMARY_MODEL_MAX_SEPARATION_FRAC = 0.05

# --- Rutas de salida ---
OUT_DIR = ROOT / "outputs" / "audit" / "sow_v4"

def session_union_dates(feats: dict):
    """Union ordenada de todos los calendarios de tickers.

    El parquet extendido tiene calendarios distintos por ticker
    (LSE, NYSE, Xetra, etc.). Cualquier operacion que necesite
    un indice de sesiones global (bootstrap, purge) debe usar la
    union, no el calendario del primer ticker.
    """
    if not feats:
        return []
    union = set()
    for f in feats.values():
        union.update(f["dates"])
    return sorted(union)

