# S-03-deep - regimes/sector_regime.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 1436728
- **Fichero:** `regimes/sector_regime.py` (210 LOC, 8.7 KB)
- **mtime:** 2026-09-30
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO CON FIX (SC-2). Cero ALTA, cero MEDIA.

## 1. Alcance

210 LOC. Regimenes sectoriales. 2 funciones publicas:
- `compute_sector_scores(df, benchmark='^GSPC', df_stocks=None, holdings_df=None)`.
- `compute_price_flow_rankings(df)`.

Funciones privadas: `_dispersion(row)`.

Consumidor: `src/pipeline/sectors_base.py:13`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| SC-1 | BAJA | compute_sector_scores (regime) | Ramas `CYCLICAL LEADERSHIP`, `DEFENSIVE LEADERSHIP`, `MIXED` inalcanzables: `positive_count` en [0, 11] y las 3 primeras ramas (>=8, <=3, 4-7) cubren todo el rango. Golden fija `contract.regime` y test `test_regime_es_uno_de_los_conocidos` solo verifica pertenencia. Cambiar el orden rompe el golden. | WONT FIX: golden + tests acotan. |
| SC-2 | BAJA | compute_price_flow_rankings | `sectors = ['XLK','XLF',...]` hardcoded, duplicando `MARKET_TICKERS['sectors']` (ya importado). | **FIX APLICADO** (commit 982f135). Precedente A5-11. |
| SC-3 | INFO | compute_sector_scores | `df_stocks`, `holdings_df` recibidos pero no usados. Comentarios obsoletos mencionan amplitud interna por stocks. | WONT FIX: compatibilidad tras refactor. |
| SC-4 | INFO | compute_sector_scores | No recibe `temporal_meta`; resuelve `_eff_date` internamente via `resolve_effective_date` (FU-020/R2). Autogestion. | WONT FIX: cierra SS-3 de S-03-deep (diferido desde sectors_base). |
| SC-5 | INFO | FU-020/R2 except | `except Exception` amplio, solo para log de fallo de resolver effective_date. | WONT FIX: red de seguridad, no silencia bug. |

## 3. Contraste con tests

- `test_sector_regime_characterization.py`: contrato golden
  (ranking, regime, top3, last_scores diagnosticos). Datos
  sinteticos reproducibles via `RandomState(seed=42)`, n_days=300.
- `test_sector_regime_edge_cases.py`: df vacio -> None, sin
  benchmark -> None, sin sectores suficientes -> None o ranking
  vacio, sector faltante no rompe, regime en set conocido, ranking
  descendente, scores no vacio, top3 = prefijo de ranking.
- `test_sector_regime_dispersion.py`: D1 (dispersion std/mean ->
  std ddof=0).
- `test_sector_regime_effective_date.py`: FU-020/R2.
- `test_sector_regime_min_periods.py`: F3-05-quinquies.
- `test_sector_regime_n_valid_threshold.py`: H1 (n_valid >= 4).
- `test_atr_nan_tolerance.py`: unico consumidor de atr en sector_regime.

No cubren:
- SC-1: el golden fija un solo `regime`, no recorre las ramas.
  `test_regime_es_uno_de_los_conocidos` solo verifica pertenencia
  al set de 6, no alcanzabilidad.

## 4. Falsos positivos descartados

- F3-05-quinquies (min_periods 20/50/126): documentado en el
  codigo. Evita NaN por festivos NYSE. Correcto.
- D1 (dispersion): std ddof=0 acota en [0, 1]. Correcto.
- H1 (n_valid >= 4): documentado. Bajo el umbral, score no
  publicable. Correcto.
- FU-020/R2 (effective_date): resuelto internamente con
  min_coverage=0.90. Correcto.
- `_dispersion` con `len(vals) < 2` -> 0.0: correcto.

## 5. Cierre de SS-3

SS-3 (diferido desde `sectors_base.py`): `compute_sector_scores`
no recibe `temporal_meta`. **Resuelto como INFO/WONT FIX:**
`compute_sector_scores` resuelve internamente su `_eff_date` via
`resolve_effective_date` sobre las columnas Close del universo
sectorial, cumpliendo R2 (FU-020) sin necesitar propagacion de
`temporal_meta` desde el caller.

SS-3 cerrado.

## 6. Patron: rama inalcanzable con golden test

SC-1 es un caso poco comun: codigo con ramas inalcanzables pero
protegido por un golden que fija el valor real. El patron es
intencional hasta cierto punto (la clasificacion en 6 categorias
existe por legado o por planificacion futura). El golden bloquea
cualquier cambio sin decision explicita. WONT FIX razonado.

## 7. Dictamen

CERRADO CON FIX. Cero ALTA, cero MEDIA. 1 BAJA corregida
(SC-2, commit 982f135) + 1 BAJA WONT FIX (SC-1). 3 INFO WONT FIX.
SS-3 cerrado.

## 8. Proximo

Continuar `regimes/` (1/8 auditado):
- `macro_regime.py` (205), `liquidity.py` (77),
  `financial_conditions.py` (70), `tactical_engine.py` (59),
  `structural_engine.py` (40), `volatility_regime.py` (27),
  `__init__.py` (1).
