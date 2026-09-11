# -*- coding: utf-8 -*-
"""Fase 10b del pipeline: indices internacionales (fases Wyckoff + lideres).

Extraido de run.py (refactor C2, fase C2-10b).
"""

import pandas as pd

from src.stock_data_loader import download_stock_prices
from indicators.index_phase import compute_index_phases
from indicators.index_leaders import select_index_leaders


def compute_indices_intl(df_market):
    """Calcula fases Wyckoff y lideres de indices internacionales.

    Returns:
        dict con keys:
            index_phases, index_data, index_leaders
    """
    print("Calculando fases Wyckoff para indices internacionales...")
    index_phases, index_data = compute_index_phases(df_market)
    indices_en_acumulacion = [nombre for nombre, fase in index_phases.items() if fase in ['ACCUMULATION', 'MARKUP']]
    if indices_en_acumulacion:
        print(f"  Indices en acumulacion: {', '.join(indices_en_acumulacion)}")
        df_index_stocks = download_stock_prices()
        index_leaders = {}
        for nombre in indices_en_acumulacion:
            try:
                leaders_single = select_index_leaders(None, df_index_stocks, [nombre])
                if nombre in leaders_single and not leaders_single[nombre].empty:
                    index_leaders[nombre] = leaders_single[nombre]
                    print(f"    {nombre}: {len(leaders_single[nombre])} empresas seleccionadas")
                else:
                    print(f"    {nombre}: sin lideres disponibles")
            except Exception as e:
                print(f"    {nombre}: error al calcular lideres - {e}")
    else:
        print("  Ningun indice en fase de acumulacion.")
        index_leaders = {}

    # Exportar CSV de lideres internacionales para revision manual
    if index_leaders:
        try:
            all_leaders = []
            for nombre, df in index_leaders.items():
                df_copy = df.copy()
                df_copy['indice'] = nombre
                all_leaders.append(df_copy)
            if all_leaders:
                pd.concat(all_leaders, ignore_index=True).to_csv(
                    'outputs/report/analisis_lideres_internacionales.csv', index=False
                )
                print("  CSV de lideres internacionales generado.")
        except Exception as e:
            print(f"  Error al generar CSV internacional: {e}")

    return {
        'index_phases': index_phases,
        'index_leaders': index_leaders,
    }
