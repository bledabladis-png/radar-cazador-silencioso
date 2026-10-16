# Indice consolidado H1-H18 (auditoria externa del 2026-10-15)

**Generado automaticamente desde `docs/gobernanza/REGISTRO.yaml`.**
**NO editar a mano.** Regenerar con:

    py scripts/genera_indice_auditorias.py --apply

Puerta de entrada a los informes originales y a los tests de regresion
de cada hallazgo. Ver `docs/auditoria/radar/audits/` y los commits
referenciados.

| ID | Descripcion | Estado | Commit | Test | Razon |
|---|---|---|---|---|---|

| H1 | Alerta HYG/LQD usaba texto literal fijo | FIXED | edabdb52 | tests/test_report_alerts_header.py | - |
| H2 | Se creyo bug de rs nominal en WLS | FALSE_POSITIVE | 804968b1 | tests/test_h2_rs_mom_scale_invariance.py | Verificado empiricamente con split 10:1 simulado, resultado identico. Solo afecta a LIS (N2), no a WLS. |
| H3 | Tabla declaraba 1 fuente SEC, hay 2 (N-PORT + N-PORT-P QQQ) | FIXED | db0462c2 | tests/test_h3_sec_freshness_sources.py | - |
| H4 | Volatilidad (velocidad) vs Lectura (nivel) homonimas | FIXED | 3bd5f68a | tests/test_h4_volatility_level_vs_speed.py | - |
| H5 | Data Conf se deriva solo de frescura, no de cobertura | FIXED | 47b34d82 | tests/test_h5_data_conf_frescura.py | - |
| H6 | Z=5.00 era clip, no valor bruto | FIXED | 4d905104 | tests/test_h6_h10_flow_z_saturation.py | - |
| H7 | rolling(N, min_periods=N) sobre NaN dispersos devuelve NaN | FIXED | 1189b7f8 | tests/test_audit_contracts_rolling_nan.py | - |
| H8 | Sector Lider publicado cuando state=UNRESOLVED | FIXED | 0710db68 | - | - |
| H9 | Correlacion entre evidencias no cuantificada | FIXED | 3540218b | tests/test_h9_evidence_correlation.py | - |
| H10 | Exponer z bruto sin clip cuando saturado | FIXED | 4d905104 | tests/test_h6_h10_flow_z_saturation.py | - |
| H11 | Frescura SEC unificada | SUPERSEDED | db0462c2 | tests/test_h3_sec_freshness_sources.py | Absorbido por H3, que ya separaba las dos fuentes SEC. No requiere accion propia. |
| H12 | ETF Primary Flow es estimacion operativa, no flujo secundario | FIXED | 0f990fdf | tests/test_h12_primary_flow_disclaimer.py | - |
| H13 | Benchmark de rs_abs_20d no declarado (SPY vs ^GSPC) | FIXED | 7e621ea4 | tests/test_h13_rs_abs_benchmark.py | - |
| H14 | NIPC 13F sin unidad, formula ni tratamiento | FIXED | e9cefaeb | tests/test_h14_nipc_documentacion.py | - |
| H15 | MTE no publicaba pesos ni umbrales | FIXED | bb7fd03b | tests/test_h15_mte_weights.py | - |
| H16 | Momentum amplitud sin unidad ni reglas de flags | FIXED | 16e064f4 | tests/test_h16_breadth_momentum_units.py | - |
| H17 | 3 persistencias homonimas sin desambiguar | FIXED | fdbc7ccf | tests/test_h17_persistence_disambiguation.py | - |
| H18 | QQQ/SPY Desde inicio sin fechas reales ni aviso | FIXED | fbcfe4cd | tests/test_h18_qqq_spy_inception.py | - |

**Fin del indice.**
