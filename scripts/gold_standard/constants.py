# -*- coding: utf-8 -*-
"""Constantes congeladas del estudio Gold Standard SOW v1.9.

BORRADOR — pendiente firma auditor.

Toda constante que forme parte del contrato va aqui, congelada.
No se modifica tras la firma.
"""
from __future__ import annotations

# --- Parametros del detector congelado (candidata v1.9) ---
FROZEN_V19_PARAMS = {
    "window": 60,
    "x_atr": 0.25,
    "y_vol": 1.10,
    "max_age_m": 10,
}

# --- Sampling frame ---
WARMUP_MIN = 200
VISUAL_WINDOW = 240
FIELDS_REQUIRED = ("Open", "High", "Low", "Close", "Volume")

# --- Muestreo ---
SEED_GLOBAL = 20261006
N_POS_A = 200
N_NEG_A = 200
MIN_GAP_A_SESSIONS = 120

# Muestra B: 11 sectores x 3 periodos = 33 estratos
PERIODOS = [(2021, 2022), (2023, 2024), (2025, 2026)]
N_MIN_ESTRATO = 5          # colapso si N_h < 5
N_MIN_RAO_WU = 2          # condicion Rao-Wu-Yue
CAPACIDAD_MAX_NB = 800    # suspender si n_B simulado > 800

# --- Capa 2 ---
M_CAPA2 = 30
H_SOW = 20
R_BANDA = 0.12
N_POR_CELDA_CAPA2 = 75

# --- Metricas ---
E_SE = 0.07
E_SP = 0.07
E_PPV = 0.08
E_NPV = 0.08
KAPPA_GATE = 0.60
LCI_SE_GATE = 0.70
LCI_SP_GATE = 0.70
DELTA_MATERIAL = 0.10

# --- Bootstrap ---
B_BOOTSTRAP = 2000
B_BOOTSTRAP_CHECK = 5000

# --- Regimen VIX ---
VIX_WINDOW = 60
VIX_Q33 = 0.33
VIX_Q66 = 0.66