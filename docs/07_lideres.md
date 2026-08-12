# Lideres Sectoriales e Internacionales

## Proposito
Selecciona las mejores empresas de cada sector/indice en fase favorable (ACCUMULATION o MARKUP) usando el Wyckoff Leadership Score (WLS).

## WLS (Wyckoff Leadership Score)
Combina:
- RS (Relative Strength) normalizado: 35%
- Flujo (Flow Proxy) normalizado: 25%
- RWS (Relative Wyckoff Score) normalizado: 25%
- Estabilidad: 10%
Bonus por persistencia: +5% * min(persistence_10d/10, 1.0).

## Lideres Sectoriales
- Archivo: indicators/stock_leader.py
- Fuente de holdings: data/etf_holdings.csv (actualizacion trimestral automatica desde State Street).

## Lideres Internacionales
- Archivo: indicators/index_leaders.py
- Fuente de holdings: data/index_holdings.csv
- Indices cubiertos: S&P 500, Dow Jones, Nasdaq-100, Russell 2000, Euro Stoxx 50, Ibex 35, DAX 40, FTSE 100.
