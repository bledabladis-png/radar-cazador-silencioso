# -*- coding: utf-8 -*-
"""Breadth de Mercado (11 sectores).

Extraido de src/report_generator.py (refactor C1, fase C1-6a2).
"""

from config.settings import EXPECTED_SECTOR_COUNT


def render_breadth_market(breadth_values):
    """Renderiza la seccion Breadth de Mercado (11 sectores).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if breadth_values:
        out.append(f"## Breadth de Mercado ({EXPECTED_SECTOR_COUNT} sectores)\n")
        out.append("| Metrica | Valor |\n")
        out.append("|---------|-------|\n")
        b20_pct = breadth_values.get('% sobre EMA20', 0)
        b50_pct = breadth_values.get('% sobre EMA50', 0)
        b200_pct = breadth_values.get('% sobre EMA200', 0)
        nh_pct = breadth_values.get('New Highs (%)', 0)
        nl_pct = breadth_values.get('New Lows (%)', 0)

        b20_count = breadth_values.get('EMA20 count', int(round(b20_pct * EXPECTED_SECTOR_COUNT)))
        b50_count = breadth_values.get('EMA50 count', int(round(b50_pct * EXPECTED_SECTOR_COUNT)))
        b200_count = breadth_values.get('EMA200 count', int(round(b200_pct * EXPECTED_SECTOR_COUNT)))
        nh_count = breadth_values.get('New Highs count', int(round(nh_pct * EXPECTED_SECTOR_COUNT)))
        nl_count = breadth_values.get('New Lows count', int(round(nl_pct * EXPECTED_SECTOR_COUNT)))

        out.append(f"| % sobre EMA20 | {b20_count}/{EXPECTED_SECTOR_COUNT} ({b20_pct:.2%}) |\n")
        out.append(f"| % sobre EMA50 | {b50_count}/{EXPECTED_SECTOR_COUNT} ({b50_pct:.2%}) |\n")
        out.append(f"| % sobre EMA200 | {b200_count}/{EXPECTED_SECTOR_COUNT} ({b200_pct:.2%}) |\n")
        out.append(f"| New Highs sectoriales | {nh_count}/{EXPECTED_SECTOR_COUNT} ({nh_pct:.2%}) |\n")
        out.append(f"| New Lows sectoriales | {nl_count}/{EXPECTED_SECTOR_COUNT} ({nl_pct:.2%}) |\n")
        out.append("\n")
        out.append("*Cada fila cuenta sectores (ETFs) cuyo precio esta sobre la EMA correspondiente, no tickers individuales.*\n\n")
    return out
