# S-03-deep - regimes/macro_regime.py

- **Auditoria:** 2026-10-03
- **HEAD inicial:** 325c714
- **Fichero:** `regimes/macro_regime.py` (205 LOC, 8.7 KB)
- **mtime:** 2026-09-30
- **Contrato:** Fase 0 + revision linea a linea + contraste con tests.
- **Dictamen:** CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA.

## 1. Alcance

205 LOC. 3 funciones publicas:
- `compute_macro_signals(df_market, df_macro_manual=None, liquidity_score=None, vol_regime_score=None, real_liquidity_score=None, temporal_meta=None, reference_date=None)`.
- `compute_macro_score(all_signals)`.
- `compute_macro_regime(df_market, df_macro_manual, liquidity_score, vol_score, temporal_meta=None, reference_date=None)`.

Consumidor: `src/pipeline/regimes.py:12`.

## 2. Hallazgos

| # | Sev | Ubicacion | Descripcion | Accion |
|---|---|---|---|---|
| MR-1 | INFO | firma | `temporal_meta` recibido y no usado en el cuerpo. | WONT FIX: contrato de API. |
| MR-2 | BAJA | compute_macro_regime | `all_signals['volatility' / 'curve' / 'market_strength']` pueden no existir. Requiere ausencia simultanea de ^VIX, ^VIX3M, vol_regime_score, BIL+IEF+TLT, y todos los equity. Fallo aguas arriba mas grave. | WONT FIX condicional. |
| MR-3 | INFO | regime rules | 12 ramas, todas alcanzables (verificado a mano). A diferencia de SC-1 (sector_regime.py). | WONT FIX. |
| MR-4 | INFO | conf | `all_signals.iloc[-1].dropna()` incluye senales derivadas entre si (curve, credit, volatility, liquidity, real_liquidity). | WONT FIX: D13 intencional. |
| MR-5 | INFO | weighted_score | Renormalizacion por fila con fillna(0) + den divisor. Correcto. | WONT FIX. |
| MR-6 | INFO | L~130 | `from indicators.breadth import compute_breadth` en cuerpo, no cabecera. | WONT FIX: cosmetico. |
| MR-7 | INFO | L~186 | `all_signals.get('inflation', pd.Series(0))`. DataFrame.get con default. | WONT FIX. |

## 3. Contraste con tests

- `tests/test_macro_regime_confidence.py` (D13): 5 tests.
  Regresion estatica (no `conf = 0.5` en codigo), calculo de conf
  sobre all_signals sintetico (acordes, opuestas, una sola senal,
  rango valido).
- `tests/test_indicators_credit_breadth_macro.py` (D22): 12 tests
  de los 3 helpers consumidos (credit_risk_signal, compute_breadth,
  fundamental_signals). Incluye D23 (reference_date propagado).

No cubren:
- Las 12 reglas de precedencia de regime (ninguna tiene test).
- Los accesos a all_signals['volatility' / 'curve' / 'market_strength'].
- weighted_score y compute_macro_score.

Los tests existentes cubren la confianza (D13) y los helpers
(D22/D23), pero la logica central del regime no tiene cobertura
directa.

## 4. Falsos positivos descartados

- D13 (confianza desde confidence_from_range): correcto, con
  regresion estatica que prohibe `conf = 0.5`.
- D23 (reference_date propagado a fundamental_signals): correcto,
  test lo verifica con fecha distinta de hoy.
- `weighted_score` con fillna(0) + renormalizacion: correcto (no
  invalida el score completo por un componente ausente).
- Ramas de regime: todas alcanzables.
- `macro_score.rolling(2, min_periods=1).mean()`: suavizado
  intencional. INFO.

## 5. Patron: logica de negocio sin cobertura directa

`macro_regime.py` tiene 205 LOC y una docena de reglas de decision
(las 12 ramas del regime), pero **ninguna tiene test directo**. Los
tests existentes cubren los helpers y la confianza.

El patron contrasta con `sector_regime.py`, que tiene golden +
edge_cases + dispersion + effective_date + min_periods +
n_valid_threshold. `macro_regime.py` solo tiene D13 (confianza)
y D22 (helpers).

No es un hallazgo (no hay bug observado), pero es una asimetria de
cobertura. Candidato a frente futuro ("caracterizacion de
macro_regime" en la linea de DT1 para sector_regime).

## 6. Dictamen

CERRADO SIN CAMBIOS. Cero ALTA, cero MEDIA. 1 BAJA (MR-2) WONT
FIX condicional; 6 INFO WONT FIX razonado.

## 7. Proximo

Continuar `regimes/` (2/8 auditados):
- `liquidity.py` (77), `financial_conditions.py` (70),
  `tactical_engine.py` (59), `structural_engine.py` (40),
  `volatility_regime.py` (27), `__init__.py` (1).
