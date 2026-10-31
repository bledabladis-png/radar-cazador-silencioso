# -*- coding: utf-8 -*-
"""Breadth de Mercado (11 sectores).

Extraido de src/report_generator.py (refactor C1, fase C1-6a2).
"""

import pandas as pd
from config.settings import EXPECTED_SECTOR_COUNT
from src.report.helpers import _fmt_num


def render_breadth_market(breadth_values):
    """Renderiza la seccion Breadth de Mercado (11 sectores).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if breadth_values:
        out.append(f"## Breadth de Mercado ({EXPECTED_SECTOR_COUNT} sectores)\n")
        out.append("| Metrica | Valor |\n")
        out.append("|---------|-------|\n")
        # H12-bis B6 (2026-10-31): default de .get se evalua SIEMPRE.
        # int(round(NaN*11)) crasheaba aunque el count estuviera en el dict.
        def _count(key, pct):
            if key in breadth_values:
                return breadth_values[key]
            if pct is None or not pd.notna(pct):
                return 0
            return int(round(pct * EXPECTED_SECTOR_COUNT))
        b20_pct = breadth_values.get('% sobre EMA20', 0)
        b50_pct = breadth_values.get('% sobre EMA50', 0)
        b200_pct = breadth_values.get('% sobre EMA200', 0)
        nh_pct = breadth_values.get('New Highs (%)', 0)
        nl_pct = breadth_values.get('New Lows (%)', 0)

        b20_count = _count('EMA20 count', b20_pct)
        b50_count = _count('EMA50 count', b50_pct)
        b200_count = _count('EMA200 count', b200_pct)
        nh_count = _count('New Highs count', nh_pct)
        nl_count = _count('New Lows count', nl_pct)

        out.append(f"| % sobre EMA20 | {b20_count}/{EXPECTED_SECTOR_COUNT} ({_fmt_num(b20_pct, '{:.2%}')}) |\n")
        out.append(f"| % sobre EMA50 | {b50_count}/{EXPECTED_SECTOR_COUNT} ({_fmt_num(b50_pct, '{:.2%}')}) |\n")
        out.append(f"| % sobre EMA200 | {b200_count}/{EXPECTED_SECTOR_COUNT} ({_fmt_num(b200_pct, '{:.2%}')}) |\n")
        out.append(f"| New Highs sectoriales | {nh_count}/{EXPECTED_SECTOR_COUNT} ({_fmt_num(nh_pct, '{:.2%}')}) |\n")
        out.append(f"| New Lows sectoriales | {nl_count}/{EXPECTED_SECTOR_COUNT} ({_fmt_num(nl_pct, '{:.2%}')}) |\n")
        out.append("\n")
        out.append("*Cada fila cuenta sectores (ETFs) cuyo precio esta "
                   "sobre la EMA correspondiente, no tickers individuales.*\n\n")
        # B2 (2026-10-13): aclarar diferencia con 'Sector Breadth & Health'.
        out.append("*Metrica distinta de 'Sector Breadth & Health': aqui se "
                   "cuenta el % de ETFs sectoriales por encima de su propia "
                   "EMA (breadth macro de los 11 sectores). En Sector Breadth "
                   "& Health se cuenta, para cada sector, el % de sus "
                   "componentes por encima de la EMA del sector (breadth "
                   "micro). Los dos numeros no tienen por que coincidir.*\n\n")
    return out
