# -*- coding: utf-8 -*-
"""
Proveedor QQQ Primary Flow desde SEC (N-30B-2 / N-CSRS).

Lee el CSV consolidado generado por:
    scripts/extract_sec_qqq_primary_flow_v4.py
    y el constructor de CSV semestral.

Devuelve la última fila disponible para el reporte.
"""

import pandas as pd
from pathlib import Path

CSV_PATH = Path("outputs/history/qqq_sec_primary_flow.csv")

def get_qqq_sec_primary_flow() -> pd.DataFrame:
    """Lee el CSV oficial y devuelve la última fila."""
    if not CSV_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("year", ascending=False)
    return df.head(1).reset_index(drop=True)
