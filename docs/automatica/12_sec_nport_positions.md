## Proposito
Documenta los módulos de extracción de posiciones institucionales desde SEC N-PORT, con granularidad fondo + activo + fecha de reporte.

## Arquitectura

- data/providers/sec_nport_positioning.py: extrae posiciones de N-PORT (REGISTRANT, FUND_REPORTED_HOLDING, IDENTIFIERS).
- data/providers/sec_nport_position_change.py: calcula cambios de balance por fondo y activo.
- data/providers/sec_nport_quarters_position_change.py: compara trimestres Q1 y Q2.
- data/providers/sec_fund_flow.py: extrae flujos de fondos desde FUND_REPORTED_INFO.
- data/providers/sec_nport_international_leader_flows.py: cruza N-PORT de FEZ con líderes internacionales.


## Formulas

- **Position Change:** BALANCE(t) - BALANCE(t-1).
- **Position Change %:** PositionChange / BALANCE(t-1) × 100.
- **Net Fund Flow:** Sales + Reinvestment - Redemption.


## Salidas

- outputs/history/sec_nport_positions.csv
- outputs/history/sec_nport_position_change.csv
- outputs/history/sec_nport_position_change_quarterly.csv
- outputs/history/sec_fund_flow.csv
- outputs/report/sec_nport_international_leader_flows.csv


## Limitaciones Conocidas
Los datasets N-PORT se publican trimestralmente aunque los datos son mensuales. No se integra en run.py diario.
