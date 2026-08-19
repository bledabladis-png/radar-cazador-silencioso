# -*- coding: utf-8 -*-
"""
Control de calidad de históricos antes del commit diario.

Verifica:
    - Existencia de archivos clave.
    - Tamaño mínimo.
    - Ausencia de NaN en columnas críticas.
    - Frescura de fuentes sensibles.

Salida:
    Código 0 si todo OK.
    Código 1 si hay fallos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "outputs/history/etf_primary_flow.csv",
    "outputs/history/blackrock_dax_primary_flow.csv",
    "outputs/history/blackrock_isf_primary_flow.csv",
    "outputs/history/blackrock_iwm_historical.csv",
    "outputs/history/amundi_lyxi_primary_flow.csv",
    "outputs/history/cftc_position_flow.csv",
    "outputs/history/macro_regime.csv",
    "outputs/history/qqq_sec_primary_flow.csv",
]

MIN_ROWS = {
    "outputs/history/etf_primary_flow.csv": 50,
    "outputs/history/blackrock_dax_primary_flow.csv": 20,
    "outputs/history/blackrock_isf_primary_flow.csv": 20,
    "outputs/history/blackrock_iwm_historical.csv": 20,
    "outputs/history/amundi_lyxi_primary_flow.csv": 20,
    "outputs/history/cftc_position_flow.csv": 5,
    "outputs/history/macro_regime.csv": 10,
    "outputs/history/qqq_sec_primary_flow.csv": 5,
}

CRITICAL_NAN_COLS = {
    "outputs/history/etf_primary_flow.csv": ["ticker", "nav", "shares_outstanding", "total_net_assets", "primary_flow_usd"],
    "outputs/history/blackrock_dax_primary_flow.csv": ["date", "nav", "shares_outstanding"],
    "outputs/history/blackrock_isf_primary_flow.csv": ["date", "nav", "shares_outstanding"],
    "outputs/history/blackrock_iwm_historical.csv": ["date", "nav", "shares_outstanding"],
    "outputs/history/amundi_lyxi_primary_flow.csv": ["date", "nav", "shares_outstanding"],
    "outputs/history/cftc_position_flow.csv": ["date", "contract", "net_position"],
    "outputs/history/macro_regime.csv": ["date"],
    "outputs/history/qqq_sec_primary_flow.csv": ["year", "shares_sold", "shares_repurchased"],
}

errors = []

for file in REQUIRED_FILES:
    path = PROJECT_ROOT / file

    if not path.exists():
        errors.append(f"{file}: no existe")
        continue

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:
        errors.append(f"{file}: error de lectura: {exc}")
        continue

    if df.empty:
        errors.append(f"{file}: está vacío")
        continue

    min_rows = MIN_ROWS.get(file, 5)
    if len(df) < min_rows:
        errors.append(f"{file}: solo {len(df)} filas (mínimo {min_rows})")

    nan_cols = CRITICAL_NAN_COLS.get(file, [])
    for col in nan_cols:
        if col not in df.columns:
            errors.append(f"{file}: falta columna {col}")
            continue

        # Caso especial: etf_primary_flow.csv primary_flow_usd
        # El primer NaN de cada ticker es esperado (diff inicial)
        if file == "outputs/history/etf_primary_flow.csv" and col == "primary_flow_usd":
            if 'Date' not in df.columns or 'ticker' not in df.columns:
                errors.append(f"{file}: faltan columnas Date/ticker para validar primary_flow_usd")
                continue

            df_sorted = df.sort_values(['ticker', 'Date'])
            later_nan = (
                df_sorted.assign(_idx=df_sorted.groupby('ticker').cumcount())
                .query('_idx > 0')['primary_flow_usd']
                .isna().any()
            )
            if later_nan:
                errors.append(f"{file}: NaN en primary_flow_usd más allá de la primera fila de cada ticker")
        else:
            if df[col].isna().any():
                errors.append(f"{file}: NaN en {col}")

# Validación de integridad QQQ SEC
try:
    qqq_sec_path = PROJECT_ROOT / "outputs/history/qqq_sec_primary_flow.csv"
    if qqq_sec_path.exists():
        df_qqq_sec = pd.read_csv(qqq_sec_path, encoding="utf-8-sig")
        required_qqq = [
            "shares_sold",
            "shares_repurchased",
            "shares_beginning",
            "shares_end",
            "proceeds_shares_sold",
            "value_shares_repurchased",
            "primary_flow_usd",
        ]
        missing_qqq = [c for c in required_qqq if c not in df_qqq_sec.columns]
        if missing_qqq:
            errors.append(f"outputs/history/qqq_sec_primary_flow.csv: faltan columnas {missing_qqq}")
        else:
            df_qqq_valid = df_qqq_sec.dropna(subset=required_qqq)
            for idx, row in df_qqq_valid.iterrows():
                net_shares = row["shares_sold"] + row["shares_repurchased"]
                change_shares = row["shares_end"] - row["shares_beginning"]
                if abs(net_shares - change_shares) > 0.5:
                    errors.append(f"outputs/history/qqq_sec_primary_flow.csv: incoherencia shares fila {idx}")

                flow_usd = row["proceeds_shares_sold"] + row["value_shares_repurchased"]
                if abs(flow_usd - row["primary_flow_usd"]) > 1.0:
                    errors.append(f"outputs/history/qqq_sec_primary_flow.csv: incoherencia USD fila {idx}")
except Exception as exc:
    errors.append(f"outputs/history/qqq_sec_primary_flow.csv: error en validación integridad: {exc}")

# Frescura de PCR y Dark Pool
for file, col, max_age in [
    ("outputs/history/pcr_history.csv", "date", 7),
    ("outputs/history/darkpool_history.csv", "week", 30),
]:
    path = PROJECT_ROOT / file
    if not path.exists():
        continue

    try:
        df = pd.read_csv(path, parse_dates=[col], encoding="utf-8-sig")
        if df.empty:
            errors.append(f"{file}: vacío")
            continue
        last = pd.Timestamp(df[col].max())
        age = (pd.Timestamp.now() - last).days
        if age > max_age:
            errors.append(f"{file}: dato obsoleto ({age} días, máximo {max_age})")
    except Exception as exc:
        errors.append(f"{file}: error de frescura: {exc}")

if errors:
    print("❌ Fallos en control de calidad de históricos:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

print("✅ Control de calidad de históricos superado")
sys.exit(0)
