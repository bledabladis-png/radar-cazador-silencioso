# -*- coding: utf-8 -*-
"""Fase 8b del pipeline: SLPM v1.2 (State Machine centralizada).

Extraido de run.py (refactor C2, fase C2-8b).
"""


def compute_slpm_v12(df_market, sector_results, leader_metrics_for_slpm,
                     top_sector_flow, tactical_scores, structural_scores,
                     sector_persistence):
    """Ejecuta SLPM v1.2 State Machine.

    Returns:
        dict con la salida de evaluate_slpm_v12, o None si falla.
    """
    slpm_v12_data = None
    try:
        from indicators.slpm_v12 import evaluate_slpm_v12
        slpm_v12_data = evaluate_slpm_v12(
            df_market, sector_results, leader_metrics_for_slpm, top_sector_flow,
            tactical_scores=tactical_scores,
            structural_scores=structural_scores,
            sector_persistence=sector_persistence
        )
        if slpm_v12_data:
            state = slpm_v12_data.get('state', '?')
            lis = slpm_v12_data.get('leader_integrity', {}).get('lis', 0)
            breadth = slpm_v12_data.get('leader_breadth_v2', {}).get('composite', 0)
            t_score = slpm_v12_data.get('tactical_score', 0)
            s_score = slpm_v12_data.get('structural_score', 0)
            errors = slpm_v12_data.get('validation_errors', [])
            error_msg = f" ({len(errors)} errores)" if errors else ""
            print(f"    SLPM v1.2: {state} | T={t_score:+.2f} S={s_score:+.2f} LIS={lis:.2f} Breadth={breadth:.2f}{error_msg}")
    except Exception as e:
        print(f"    SLPM v1.2 omitido: {e}")
    return slpm_v12_data
