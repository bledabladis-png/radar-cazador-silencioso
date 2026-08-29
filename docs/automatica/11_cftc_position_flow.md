## Proposito
Documenta el módulo de posicionamiento semanal de futuros financieros (CFTC TFF), que mide cambios en posiciones reportadas por tipo de participante.

## Arquitectura

- data/providers/cftc_data.py: descarga CSV de CFTC TFF (Futures Only).
- Selecciona contratos: E-MINI S&P 500, Nasdaq-100, Russell, DJIA, VIX, UST 10Y.
- Calcula net position, position change y flow z-score por participante (dealer, asset_mgr, lev_money).


## Formulas

- **Net Position:** Long - Short.
- **Position Change:** NetPosition(t) - NetPosition(t-1).
- **Flow Z-Score:** rolling 52 semanas del position change.


## Salidas

- outputs/history/cftc_position_flow.csv
- Sección en el reporte: '## Posicionamiento CFTC (TFF, Semanal)'.


## Limitaciones Conocidas
Frecuencia semanal. No representa flujo de capital al contado, sino posicionamiento en futuros.
