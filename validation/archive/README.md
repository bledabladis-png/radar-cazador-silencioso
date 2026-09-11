# Scripts de validación archivados

Auditorías históricas one-shot conservadas por trazabilidad. **No se invocan desde workflows, `run.py` ni `tests/`.**

## Clasificación temática

### MTE (Market Transition Engine)
- `audit_mte_oos.py` — Out-of-sample MTE
- `mte_validation.py` — Validación completa MTE
- `redundancia_mte_fc.py` — Redundancia MTE vs Financial Conditions

### Breadth
- `breadth_validation.py` — Validación breadth
- `breadth_equity_validation.py` — Validación breadth equity
- `backtest_pesos_historicos.py` — Backtest pesos

### Sector rankings
- `sector_rankings_validation.py` — Validación profesional rankings sectoriales
- `phase2_validate_new_assets.py` — Validación de nuevos activos

### Wyckoff
- `wyckoff_ablation_components.py`
- `wyckoff_correlation_audit.py`
- `wyckoff_out_of_sample.py`
- `wyckoff_weight_sensitivity.py`

### Out-of-Sample (OOS)
- `oos_cftc_validation.py`
- `oos_double_counting_v1.py`
- `oos_flow_validation.py`
- `oos_nport_validation.py`
- `audit_persistence_oos.py`

### Monte Carlo / Bootstrap
- `bootstrap_montecarlo.py`
- `montecarlo_perturbacion_ranking.py`
- `montecarlo_ranking_global.py`
- `jackknife.py`
- `deflated_sharpe.py`

### Sensitivity
- `sensitivity_coverage.py`
- `sensitivity_heatmap.py`
- `sensitivity_noise.py`
- `sensitivity_persistence.py`
- `threshold_stability.py`
- `temporal_stability.py`
- `feature_corr.py`
- `feature_importance.py`
- `vif_analysis.py`

### Walk-forward / CV
- `rolling_walkforward.py`
- `walk_forward.py`
- `purged_cv.py`
- `forward_test_auto.py`
- `transition_matrix.py`

### Slots / holdings
- `holdings_audit.py`
- `leader_selection_audit.py`
- `validacion_indices_v2.py`

### Flujos
- `flow_comparison.py`
- `obv_method_comparison.py`
- `solapamiento_fls_liquidity.py`

### Otros
- `cls_comparison.py`
- `data_audit.py`
- `data_freshness_audit.py`
- `dependency_graph.py`
- `event_analysis.py`
- `information_coef.py`
- `integration_check.py`
- `module_correlation.py`
- `oms_darkpool_validation.py`
- `regresion_base_vs_lis.py`
- `signal_dependency_matrix.py`
- `signal_dependency_matrix_full.py`
- `slpm_ablation.py`
- `tactical_incremental_info.py`
- `turnover.py`

## Reactivación

Si en el futuro se necesita una:

1. Mover de vuelta a `validation/`.
2. Verificar dependencias (indicadores, paths, formatos).
3. Ejecutar como one-shot (`py validation/script.py`).

## Verificación previa al archivado

- Ningún workflow los invoca.
- `run.py` no los menciona.
- `tests/` no los importa.
- Los 6 scripts activos no los importan.
