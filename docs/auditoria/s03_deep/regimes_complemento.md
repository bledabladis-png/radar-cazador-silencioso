# S-03-deep - complemento regimes/ (6 ficheros menores)

Cierre de regimes/ (8/8 auditados). Este documento cubre los 6
ficheros restantes, todos <= 80 LOC.

- **Auditoria:** 2026-10-03
- **HEAD inicial:** c124674
- **Ficheros:**
  - `regimes/liquidity.py` (77 LOC)
  - `regimes/financial_conditions.py` (70)
  - `regimes/tactical_engine.py` (59)
  - `regimes/structural_engine.py` (40)
  - `regimes/volatility_regime.py` (27)
  - `regimes/__init__.py` (1)
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. liquidity.py

77 LOC. `compute_liquidity_score()` sin parametros. Consume
`DataRouter().get_fed_data()`, calcula score agregado con 4
componentes (fed_balance, reverse_repo, sofr, fed_funds), persiste
estado a `outputs/state/liquidity_state.json`, devuelve
`(score_series, regime, confidence, previous_score)`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| LQ-1 | INFO | conf | `confidence = 1 - safe_std/2`. `sig_vals` en [-1,1] -> std max 1.0 -> conf min 0.5. Sin clip necesario. | WONT FIX. |
| LQ-2 | INFO | state write | `json.dump` / `open` sin `encoding='utf-8'`. Contenido ASCII. | WONT FIX. |
| LQ-3 | INFO | import | `import json, os` dentro del try. | WONT FIX. |

A3.1-01 (state no actualizado no tumba el pipeline, pero es
visible con WARN) y A3.1-03 (4-tuple uniforme) aplicados.

## 2. financial_conditions.py

70 LOC. `compute_financial_conditions(df)`. Componentes: VIX, credito
(HYG/LQD), dolar (DXY), curva (TNX-FVX). Pesos en
`FINANCIAL_CONDITIONS_WEIGHTS`. Conf = `confidence_from_range`.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| FC-1 | BAJA | w_sum == 0 | Sin ninguna de las 4 columnas -> `('NEUTRAL', 1.0)`. Conf 1.0 con evidencia nula. | WONT FIX condicional: sin consumidor afectado (header solo chequea macro_conf). |
| FC-2 | INFO | __doc__ | Asignacion manual de `__doc__` tras la funcion. | WONT FIX. |

Docstring original documenta correcciones de auditoria 24/07/2026:
signo de HYG/LQD, ventana 120->60, peso DXY 0.20->0.15, rename.

## 3. tactical_engine.py

59 LOC. `compute_tactical_score(df_market, sector_etf, benchmark='^GSPC',
temporal_meta=None)`. 5 componentes: RS20, Momentum20, Flow, Breadth20,
Aceleracion. Clip [-1, 1].

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| TE-1 | INFO | firma | `temporal_meta` no usado. | WONT FIX: contrato. |
| TE-2 | INFO | flow | `volume_sector.iloc[-10:]` con `len >= 6`. Con 6-9 filas, toma las que haya. | WONT FIX: tolerancia. |

## 4. structural_engine.py

40 LOC. `compute_structural_score(df_market, sector_etf,
leader_breadth=0.5, flow_structure=0.0, persistence=0.5,
benchmark='^GSPC', temporal_meta=None)`. 3 componentes: RS
multi-ventana (63/126/252), Flow Structure, Persistence. Clip [-1, 1].

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SE-1 | INFO | firma | `temporal_meta` no usado. | WONT FIX. |
| SE-2 | INFO | rs_structural | `safe_mean([]) -> 0.0`. Correcto. | WONT FIX. |
| SE-3 | INFO | firma | `leader_breadth=0.5` no usado. Residuo. Test `test_double_counting_regression.py:36` prohibe `leader_breadth` en `STRUCTURAL_WEIGHTS`. | WONT FIX: compatibilidad de firma. |

## 5. volatility_regime.py

27 LOC. `compute_volatility_regime(returns)`. Envuelve
`indicators.volatility.volatility_regime` y mapea z a LOW/NORMAL/
ELEVATED/STRESS.

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| VR-1 | INFO | C-08 | `z` vacio o ultimo NaN -> "N/D", conf 0.0. Documentado (C-08, 1803/1884 STRESS falsos positivos por NaN de huecos VIX). | WONT FIX: con test. |

## 6. __init__.py

1 LOC. Solo docstring: "Regimenes macro: Financial Conditions,
Liquidity, Volatility, Macro.". Nada que auditar.

## 7. Contraste con tests

- `tests/test_regimes.py`: `compute_financial_conditions` con
  escenario HIGH_STRESS realista. C19 (conf = formula).
- `tests/test_regimes_low_coverage_d30.py` (D30): 6 tests de
  `compute_volatility_regime` (mapeo LOW/NORMAL/ELEVATED/STRESS +
  rango conf), 5 de `compute_tactical_score` (ticker/benchmark
  ausente, normal, serie corta, benchmark custom), 6 de
  `compute_structural_score` (analogos + persistence/flow).
- `tests/test_volatility_regime.py` (C-08): serie vacia, ultimo NaN,
  todo NaN -> N/D con conf 0.0.
- `tests/test_double_counting_regression.py:36`: prohibe
  `leader_breadth` en STRUCTURAL_WEIGHTS.

No cubren:
- FC-1: `w_sum == 0` con conf 1.0 no se testea.
- `compute_liquidity_score` con todos los componentes ausentes
  (A3.1-03 cubierto indirectamente).

## 8. Estado del cierre de regimes/

8/8 ficheros auditados:
- `sector_regime.py`: CERRADO CON FIX (SC-2). 1 BAJA WONT FIX (SC-1).
- `macro_regime.py`: CERRADO SIN CAMBIOS. 1 BAJA WONT FIX (MR-2).
- 6 menores: CERRADO SIN CAMBIOS. 1 BAJA WONT FIX (FC-1).

Totales en regimes/: 0 ALTA, 0 MEDIA, 2 BAJA corregidas (SC-2),
4 BAJA WONT FIX, 30+ INFO WONT FIX.

## 9. Proximo

`src/report/` (19 ficheros restantes, ~1700 LOC). Ya auditado:
`flows_international.py`. Siguientes por LOC:
- `helpers.py` (157), `synthesis.py` (149), `slpm.py` (137),
  `leaders.py` (128), `rankings.py` (116).
