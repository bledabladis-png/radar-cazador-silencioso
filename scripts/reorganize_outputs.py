# -*- coding: utf-8 -*-
# reorganize_outputs.py
# Reorganiza automaticamente las rutas de outputs/ en los archivos .py
# y mueve los archivos existentes a las subcarpetas correspondientes.
import os
import shutil
from pathlib import Path

# Mapeo de rutas exactas
MAPPING = {
    # report
    'outputs/report/analisis_lideres.csv': 'outputs/report/analisis_lideres.csv',
    'outputs/report/analisis_lideres_internacionales.csv': 'outputs/report/analisis_lideres_internacionales.csv',
    'outputs/report/reporte_diario.md': 'outputs/report/reporte_diario.md',
    'outputs/report/sector_rankings.csv': 'outputs/report/sector_rankings.csv',
    # history
    'outputs/history/pcr_history.csv': 'outputs/history/pcr_history.csv',
    'outputs/history/darkpool_history.csv': 'outputs/history/darkpool_history.csv',
    'outputs/history/slpm_history.csv': 'outputs/history/slpm_history.csv',
    'outputs/history/macro_regime.csv': 'outputs/history/macro_regime.csv',
    'outputs/history/forward_test_historico.csv': 'outputs/history/forward_test_historico.csv',
    # state
    'outputs/state/slpm_state.json': 'outputs/state/slpm_state.json',
    'outputs/state/mte_state.json': 'outputs/state/mte_state.json',
    'outputs/state/liquidity_state.json': 'outputs/state/liquidity_state.json',
    # holdings
    'outputs/holdings/FEZ_final_holdings.csv': 'outputs/holdings/FEZ_final_holdings.csv',
    'outputs/holdings/DAXEX_final_holdings.csv': 'outputs/holdings/DAXEX_final_holdings.csv',
    'outputs/holdings/ISF.L_final_holdings.csv': 'outputs/holdings/ISF.L_final_holdings.csv',
    'outputs/holdings/amundi_lyxi_holdings.csv': 'outputs/holdings/amundi_lyxi_holdings.csv',
    'outputs/holdings/amundi_lyxi_validation.csv': 'outputs/holdings/amundi_lyxi_validation.csv',
    # audit
    'outputs/audit/auditoria_flow_comparison.md': 'outputs/audit/auditoria_flow_comparison.md',
    'outputs/audit/auditoria_frescura.md': 'outputs/audit/auditoria_frescura.md',
    'outputs/audit/auditoria_holdings.md': 'outputs/audit/auditoria_holdings.md',
    'outputs/audit/auditoria_lideres.md': 'outputs/audit/auditoria_lideres.md',
    'outputs/audit/auditoria_redundancia.md': 'outputs/audit/auditoria_redundancia.md',
    'outputs/audit/auditoria_validacion.md': 'outputs/audit/auditoria_validacion.md',
    'outputs/audit/backtest_signals_results.csv': 'outputs/audit/backtest_signals_results.csv',
    'outputs/audit/backtest_v2_results.csv': 'outputs/audit/backtest_v2_results.csv',
    'outputs/audit/backtest_v3_results.csv': 'outputs/audit/backtest_v3_results.csv',
    'outputs/audit/flow_comparison_results.csv': 'outputs/audit/flow_comparison_results.csv',
    'outputs/audit/forward_test_6m.csv': 'outputs/audit/forward_test_6m.csv',
    'outputs/audit/forward_test_6m.md': 'outputs/audit/forward_test_6m.md',
    'outputs/audit/informe_monitorizacion.md': 'outputs/audit/informe_monitorizacion.md',
    'outputs/audit/sensitivity_results.csv': 'outputs/audit/sensitivity_results.csv',
    'outputs/audit/threshold_stability_summary.md': 'outputs/audit/threshold_stability_summary.md',
    'outputs/audit/validacion_indices.md': 'outputs/audit/validacion_indices.md',
    'outputs/audit/validacion_indices_v2.md': 'outputs/audit/validacion_indices_v2.md',
    'outputs/audit/walk_forward_ics.csv': 'outputs/audit/walk_forward_ics.csv',
    'outputs/audit/walk_forward_results.md': 'outputs/audit/walk_forward_results.md',
    'outputs/audit/wyckoff_ablation_results.csv': 'outputs/audit/wyckoff_ablation_results.csv',
    'outputs/audit/wyckoff_montecarlo_results.csv': 'outputs/audit/wyckoff_montecarlo_results.csv',
    'outputs/audit/wyckoff_score_boxplot.png': 'outputs/audit/wyckoff_score_boxplot.png',
}

# Crear subcarpetas
for subdir in ['report', 'history', 'state', 'holdings', 'audit']:
    Path(f'outputs/{subdir}').mkdir(parents=True, exist_ok=True)

# 1. Actualizar rutas en archivos .py
print('Actualizando rutas en archivos .py...')
for root, dirs, files in os.walk('.'):
    for file in files:
        if not file.endswith('.py'):
            continue
        path = Path(root) / file
        content = path.read_text(encoding='utf-8-sig')
        original = content
        for old, new in MAPPING.items():
            content = content.replace(old, new)
        # Reemplazar patron dinámico en parse_blackrock_final.py
        content = content.replace(
            "f'outputs/holdings/{etf}_final_holdings.csv'",
            "f'outputs/holdings/{etf}_final_holdings.csv'"
        )
        if content != original:
            path.write_text(content, encoding='utf-8')
            print(f'  {path}')

# 2. Mover archivos existentes
print('\nMoviendo archivos existentes...')
for old, new in MAPPING.items():
    if os.path.exists(old):
        os.makedirs(os.path.dirname(new), exist_ok=True)
        # Si destino ya existe, usar mover con overwrite
        if os.path.exists(new):
            os.remove(new)
        shutil.move(old, new)
        print(f'  {old} -> {new}')

print('\nReorganización automática completada.')
