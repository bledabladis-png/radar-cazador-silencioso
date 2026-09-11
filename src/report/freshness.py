# -*- coding: utf-8 -*-
"""Seccion Data Freshness del reporte diario.

Extraido de src/report_generator.py (refactor C1, fase C1-6a1).
"""

from datetime import datetime

import pandas as pd

from src.report.helpers import (
    _classify_freshness,
    _classify_finra_freshness,
    _classify_fred_freshness,
    _generate_coverage_table,
)


def render_data_freshness(pcr_data, darkpool_data, sector_results):
    """Renderiza la seccion 'Data Freshness' + tabla de cobertura.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("### Data Freshness\n")
    out.append("| Fuente | Ultimo dato | Antigüedad | Estado | Data Conf |\n")
    out.append("|--------|-------------|------------|--------|----------|\n")
    now = datetime.now()
    
    if pcr_data and pcr_data.get('last_date', 'N/A') != 'N/A':
        try:
            d = pd.Timestamp(pcr_data['last_date'])
            age = (now - d).days
            cboe_status = _classify_freshness(age, 3, 5, 10)
            cboe_conf = 'Alta' if cboe_status in ('CURRENT', 'RECENT') else 'Baja'
            out.append(f"| CBOE (Opciones) | {d.strftime('%Y-%m-%d')} | {age} dias | {cboe_status} | {cboe_conf} |\n")
        except:
            out.append(f"| CBOE (Opciones) | {pcr_data.get('last_date', 'N/A')} | N/D | N/D | N/D |\n")
    else:
        out.append("| CBOE (Opciones) | N/D | N/D | N/D | N/D |\n")
    
    if darkpool_data:
        week = darkpool_data.get('week', 'N/A')
        if week != 'N/A':
            try:
                d = pd.Timestamp(week)
                age = (now - d).days
                finra_status = _classify_finra_freshness(age)
                finra_conf = 'Alta' if finra_status in ('CURRENT', 'RECENT') else 'Baja'
                out.append(f"| FINRA (Dark Pools) | {d.strftime('%Y-%m-%d')} | {age} dias | {finra_status} | {finra_conf} |\n")
            except:
                out.append(f"| FINRA (Dark Pools) | {week} | N/D | N/D | N/D |\n")
        else:
            out.append("| FINRA (Dark Pools) | N/D | N/D | N/D | N/D |\n")
    
    # FRED: intentar obtener fecha real desde liquidity_state.json
    try:
        import json
        from pathlib import Path as _Path
        liq_state_path = _Path('outputs/state/liquidity_state.json')
        if liq_state_path.exists():
            with liq_state_path.open('r') as f:
                liq_state = json.load(f)
            last_fred = liq_state.get('date', 'N/A')
            if last_fred != 'N/A':
                d = pd.Timestamp(last_fred)
                age = (now - d).days
                fred_status = _classify_fred_freshness(age)
                fred_conf = 'Alta' if fred_status in ('CURRENT', 'RECENT') else 'Baja'
                out.append(f"| FRED (Macro) | {d.strftime('%Y-%m-%d')} | {age} dias | {fred_status} | {fred_conf} |\n")
            else:
                out.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
        else:
            out.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
    except Exception:
        out.append("| FRED (Macro) | N/D | N/D | N/D | N/D |\n")
    out.append("| Yahoo Finance (Precios) | Diario | < 1 dia | CURRENT | Alta |\n")
    out.append("\n")
    coverage_lines = _generate_coverage_table(pcr_data, darkpool_data, sector_results)
    for cl in coverage_lines:
        out.append(cl)

    return out
