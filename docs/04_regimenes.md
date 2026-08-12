# Regimenes

## Financial Conditions
```
Calcula el score de condiciones financieras agregando VIX, credito, dolar y curva.
    Formula: 0.40*VIX_norm + 0.30*Credito_norm + 0.15*Dolar_norm + 0.15*Curva_norm.
    Cada componente se normaliza con z-score robusto y tanh. Retorna puntuacion [-1, +1].
```

## Liquidity (FRED)
Calcula la liquidez real a partir del balance de la Fed (WALCL), SOFR, Reverse Repo y Fed Funds.

## Volatility
Basado en VIX. Z-Score robusto de la volatilidad realizada a 20 dias vs mediana de 3 anios.

## Macro Regime
```
Clasifica el regimen macro en 11 categorias.
```

## Sector Regime
Ranking sectorial combinando momentum, tendencia, volatilidad, breadth y Wyckoff.
