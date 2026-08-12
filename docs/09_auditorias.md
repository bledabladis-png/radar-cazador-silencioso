# Scripts de Auditoria

## Proposito
Scripts independientes que validan la estabilidad y robustez del sistema. No modifican el codigo productivo.

| Script | Descripcion |
|---|---|
| wyckoff_correlation_audit.py | Matriz de correlaciones entre componentes Wyckoff con bootstrap |
| wyckoff_weight_sensitivity.py | Monte Carlo de sensibilidad de pesos (Kendall Tau) |
| wyckoff_ablation_components.py | Ablacion por componentes del modulo Wyckoff |
| wyckoff_out_of_sample.py | Validacion out-of-sample por periodos historicos |
| montecarlo_perturbacion_ranking.py | Perturbacion del ranking global con ruido gaussiano |
| montecarlo_ranking_global.py | Monte Carlo del ranking sectorial global |
| backtest_pesos_historicos.py | Backtest historico de estabilidad temporal del ranking |
| slpm_ablation.py | Ablacion de componentes del SLPM |
| forward_test_auto.py | Registro semanal del forward test |
