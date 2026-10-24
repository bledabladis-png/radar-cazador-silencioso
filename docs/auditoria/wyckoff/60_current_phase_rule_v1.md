# Regla de vigencia de fase — pre-registro v1

**Estado:** CONGELADO antes de correr holdout. No se ajusta tras ver resultados.
**Fecha:** 2026-10-24
**Sesgo declarado:** escrita tras observar 12 casos dist/acc contradictorios
(10/12 con precio por encima del soporte de distribucion). El pre-registro
existe para que ese sesgo no se convierta en overfitting: la regla se
congela antes de validar, y si falla, se descarta sin retocar.

## Definicion

Sea `C(t)` el cierre en `t = as_of`. Niveles del rango: `support`,
`resistance`, `tol_abs` (tolerancia del propio rango).

- **DIST vigente** si `C(t) <= resistance + tol_abs`
- **ACC vigente** si `C(t) >= support - tol_abs`

Las clasificaciones `DISTRIBUTION_CONFIRMED` / `ACCUMULATION_CONFIRMED`
solo cuentan como estado actual si la regla se cumple. Si no, el
orquestador las trata como episodio historico terminado (`RANGE`).

El resto del orquestador (tabla CONFLICT, "gana la mas reciente") no
cambia.

## Holdout

`data/stock_prices_reserved_cohort.parquet` (268 tickers S&P 500).
Nunca usados con la nueva cascada. No se corre sobre `stock_prices.parquet`
hasta terminar esta validacion.

## Criterios de aceptacion (2, simples)

- **C1 — No extremo:** el % de DIST invalidados debe estar entre 20% y
  80% sobre los DIST originalmente confirmados. Fuera de ese rango, la
  regla esta o demasiado permisiva (0-20%) o demasiado agresiva (80-100%)
  y no se puede confiar en ella.
- **C2 — Simetria:** |% invalidados DIST − % invalidados ACC| <= 40pp.
  Si la regla invalida sistematicamente mas de un lado, hay sesgo.

Aceptar ⇔ C1 ∧ C2. En cualquier otro caso → se descarta, se documenta,
el sistema queda como estaba (CONFLICT cuando ambas confirmadas).

## Compromiso

Tras correr holdout: no se ajusta la regla, no se prueba con otro
dataset. Si falla, falla. Si pasa, se aplica al pipeline.

---

## RESULTADO — 2026-10-24

Validacion corrida sobre `data/stock_prices_reserved_cohort.parquet`
(268 tickers S&P 500). Resultado:

- DIST confirmadas: 43. Invalidadas por precio: 8 (18.6%).
- ACC confirmadas: 40. Invalidadas por precio: 1 (2.5%).

**C1** (dist invalidados en [20%, 80%]): FALLA (18.6% < 20%).
**C2** (|dist - acc| <= 40pp): PASA (|18.6% - 2.5%| = 16.1pp).

**VEREDICTO: REGLA DESCARTADA.**

No se ajusta el umbral. No se re-prueba con otro dataset. El sistema
vuelve al estado previo al pre-registro: `CONFLICT` cuando ambas
cascadas devuelven `_CONFIRMED`, sin chequear vigencia por precio.

## Lo que esto ensena

- La regla estaba influida por 12 casos observados (10/12 con precio
  recuperado). En holdout, el 18.6% es un poco menor del umbral inferior
  que fije, pero esta en el mismo orden de magnitud (20% esperado).
- La eleccion de [20%, 80%] como umbral C1 fue arbitraria, pero
  descubrirlo despues del test no autoriza a reescribir el umbral.
- Asimetria observada en holdout: distribucion se invalida 7.5x mas
  que acumulacion. Esto es informacion de mercado (los patrones de
  distribucion se recuperan mas en S&P 500 alcista), no de la regla.
- La funcion `_is_dist_current` / `_is_acc_current` se elimino del
  codigo. Los campos `range_support`, `range_resistance`, `range_tol_abs`
  se mantienen en `ClassifiedBreak` como diagnostico, aunque ya no se
  usan para arbitrar.

## Opciones abiertas (no se decide aqui)

El problema original (falsos positivos por deteccion historica) NO esta
resuelto. Se deja documentado. Cualquier intento de resolverlo requiere
nuevo pre-registro con umbrales fijados antes de ver holdout, y
aceptacion del resultado sin retoques.
