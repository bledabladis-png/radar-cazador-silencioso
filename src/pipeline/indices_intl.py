# -*- coding: utf-8 -*-
"""Fase 10b del pipeline: indices internacionales (fases Wyckoff + lideres).

Extraido de run.py (refactor C2, fase C2-10b).
"""

import pandas as pd
from pathlib import Path

from src.stock_data_loader import download_stock_prices
from indicators.index_phase import compute_index_phases
from indicators.index_leaders import select_index_leaders


def compute_indices_intl(df_market, reference_date=None, run_id=None, temporal_meta=None):
    """Calcula fases Wyckoff y lideres de indices internacionales.

    Returns:
        dict con keys:
            index_phases, index_leaders
    """
    print("Calculando fases Wyckoff para indices internacionales...")
    # A5-09 (2026-09-28): compute_index_phases sin guard tumbaba el
    # pipeline entero si fallaba. Contrato del fichero: cada bloque
    # degrada. Fix: try/except que devuelve ({}, {}) si falla.
    try:
        index_phases, _index_data = compute_index_phases(df_market, temporal_meta=temporal_meta)
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
        print(f"  Fases Wyckoff indices omitidas: {e}")
        index_phases = {}
    indices_en_acumulacion = [nombre for nombre, fase in index_phases.items() if fase in ['ACCUMULATION', 'MARKUP']]
    if indices_en_acumulacion:
        print(f"  Indices en acumulacion: {', '.join(indices_en_acumulacion)}")
        # A5-10 (2026-09-28): download_stock_prices sin guard.
        # Fix: try/except con degradacion a None -> bloque lideres omite.
        try:
            df_index_stocks = download_stock_prices(reference_date=reference_date, run_id=run_id)
        except (ValueError, TypeError, OSError, RuntimeError, KeyError) as e:
            print(f"  Descarga indices internacionales omitida: {e}")
            df_index_stocks = None
        index_leaders = {}
        if df_index_stocks is not None:
            for nombre in indices_en_acumulacion:
                try:
                    leaders_single = select_index_leaders(None, df_index_stocks, [nombre], temporal_meta=temporal_meta)
                    if nombre in leaders_single and not leaders_single[nombre].empty:
                        index_leaders[nombre] = leaders_single[nombre]
                        print(f"    {nombre}: {len(leaders_single[nombre])} empresas seleccionadas")
                    else:
                        print(f"    {nombre}: sin lideres disponibles")
                except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError) as e:
                    print(f"    {nombre}: error al calcular lideres - {e}")
        else:
            print("  Sin df_stocks de indices: lideres internacionales omitidos.")
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
                csv_path = Path('outputs/report/analisis_lideres_internacionales.csv')
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                _tmp_csv = csv_path.with_suffix(csv_path.suffix + '.tmp')
                pd.concat(all_leaders, ignore_index=True).to_csv(_tmp_csv, index=False, lineterminator='\n')
                _tmp_csv.replace(csv_path)
                print("  CSV de lideres internacionales generado.")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, pd.errors.ParserError) as e:
            print(f"  Error al generar CSV internacional: {e}")

    return {
        'index_phases': index_phases,
        'index_leaders': index_leaders,
    }
