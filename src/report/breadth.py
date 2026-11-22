# -*- coding: utf-8 -*-
"""Breadth de Mercado (11 sectores).

Extraido de src/report_generator.py (refactor C1, fase C1-6a2).
"""

import pandas as pd
from config.settings import EXPECTED_SECTOR_COUNT
from src.report.helpers import _fmt_num


def render_breadth_market(breadth_values):
    """Renderiza la seccion Breadth de Mercado.

    2026-11-21 (S39, 13.11): denominadores dinamicos. El header, los
    counts (X/Y) y la nota al pie reflejan el numero real de sectores
    con datos validos a la fecha. Cuando Y == expected (caso normal),
    el render es identico al anterior. Cuando Y < expected, se declara
    explicitamente el gap.

    Fuente de expected: breadth_values['expected_sector_count'] (derivado
    de len(MARKET_TICKERS['sectors']) en compute_breadth). Fallback:
    EXPECTED_SECTOR_COUNT (config) solo si el dict viene de un artifact
    antiguo sin el campo nuevo.
    """
    out = []
    if breadth_values:
        # Denominadores dinamicos con fallback a config (compat artifact antiguo).
        _expected = breadth_values.get('expected_sector_count', EXPECTED_SECTOR_COUNT)
        _denom20 = breadth_values.get('denominators_ema20', _expected)
        _denom50 = breadth_values.get('denominators_ema50', _expected)
        _denom200 = breadth_values.get('denominators_ema200', _expected)
        _denom_nhnl = breadth_values.get('denominators_nhnl', _expected)

        if _denom200 < _expected:
            out.append(f"## Breadth de Mercado ({_denom200} de {_expected} sectores con datos)\n")
        else:
            out.append(f"## Breadth de Mercado ({_expected} sectores)\n")

        out.append("| Metrica | Valor |\n")
        out.append("|---------|-------|\n")

        def _count(key, pct, denom):
            if key in breadth_values:
                return breadth_values[key]
            if pct is None or not pd.notna(pct) or denom is None or not pd.notna(denom) or denom <= 0:
                return 0
            return int(round(float(pct) * float(denom)))

        b20_pct = breadth_values.get('% sobre EMA20', 0)
        b50_pct = breadth_values.get('% sobre EMA50', 0)
        b200_pct = breadth_values.get('% sobre EMA200', 0)
        nh_pct = breadth_values.get('New Highs (%)', 0)
        nl_pct = breadth_values.get('New Lows (%)', 0)

        b20_count = _count('EMA20 count', b20_pct, _denom20)
        b50_count = _count('EMA50 count', b50_pct, _denom50)
        b200_count = _count('EMA200 count', b200_pct, _denom200)
        nh_count = _count('New Highs count', nh_pct, _denom_nhnl)
        nl_count = _count('New Lows count', nl_pct, _denom_nhnl)

        out.append(f"| % sobre EMA20 | {b20_count}/{_denom20} ({_fmt_num(b20_pct, '{:.2%}')}) |\n")
        out.append(f"| % sobre EMA50 | {b50_count}/{_denom50} ({_fmt_num(b50_pct, '{:.2%}')}) |\n")
        out.append(f"| % sobre EMA200 | {b200_count}/{_denom200} ({_fmt_num(b200_pct, '{:.2%}')}) |\n")
        out.append(f"| New Highs sectoriales | {nh_count}/{_denom_nhnl} ({_fmt_num(nh_pct, '{:.2%}')}) |\n")
        out.append(f"| New Lows sectoriales | {nl_count}/{_denom_nhnl} ({_fmt_num(nl_pct, '{:.2%}')}) |\n")
        out.append("\n")
        out.append("*Cada fila cuenta sectores (ETFs) cuyo precio esta "
                   "sobre la EMA correspondiente, no tickers individuales.*\n\n")
        if _denom200 < _expected:
            out.append(f"*Cobertura parcial: {_denom200} de {_expected} sectores "
                       "con datos suficientes para EMA200. Los porcentajes se "
                       "calculan sobre los sectores disponibles en la fecha.\n"
                       "Sectores sin historial suficiente no contribuyen al "
                       "denominador ni al numerador.*\n\n")
        out.append("*Metrica distinta de 'Sector Breadth & Health': aqui se "
                   "cuenta el % de ETFs sectoriales por encima de su propia "
                   "EMA (breadth macro de los sectores). En Sector Breadth "
                   "& Health se cuenta, para cada sector, el % de sus "
                   "componentes por encima de la EMA del sector (breadth "
                   "micro). Los dos numeros no tienen por que coincidir.*\n\n")
    return out
