## Proposito
Modulos que evaluan el contexto macroeconomico, las condiciones financieras, la liquidez real, la volatilidad y la amplitud sectorial.

## Arquitectura

- inancial_conditions.py: score basado en VIX, credito, dolar y curva (0.40/0.30/0.15/0.15).
- liquidity.py: liquidez real a partir de WALCL, SOFR, RRP y Fed Funds.
- olatility_regime.py: regimen de volatilidad basado en VIX.
- macro_regime.py: clasificacion en 11 categorias macro.
- sector_regime.py: ranking sectorial combinando momentum, tendencia, volatilidad, breadth y Wyckoff.


## Formulas

- **Financial Score:** 0.40*VIX_norm + 0.30*Credito_norm + 0.15*Dolar_norm + 0.15*Curva_norm.
- **Liquidity Score:** media ponderada de senhales normalizadas (0.35*Fed Balance + 0.25*RRP + 0.20*SOFR + 0.20*Fed Funds).


## Salidas

- Estados: ABUNDANTE, NEUTRAL, ESTRECHA, HIGH_STRESS, EXTREME_STRESS, CRISIS, LIQUIDITY CRISIS, RECESSION, INFLATION SHOCK, STAGFLATION, GOLDILOCKS, EXPANSION, LATE EXPANSION, RECOVERY, DEFLATION, SLOWDOWN, MIXED.
- Regimen sectorial: BROAD PARTICIPATION, ROTATIONAL, NARROW RALLY, CYCLICAL LEADERSHIP, DEFENSIVE LEADERSHIP, MIXED.

