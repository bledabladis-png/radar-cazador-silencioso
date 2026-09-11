## Proposito
Selecciona las mejores empresas de cada sector/indice en fase favorable (ACCUMULATION o MARKUP) usando el Wyckoff Leadership Score (WLS).

## Arquitectura

- stock_leader.py: compute_stock_metrics(), compute_wls(), generate_leader_section().
- index_leaders.py: analogo para indices internacionales.
- Fuente de holdings: data/etf_holdings.csv (sectores) y data/index_holdings.csv (indices).
- Actualizacion trimestral automatica via GitHub Actions.


## Formulas

- **WLS:** 0.35*rs_z + 0.25*flow_proxy_z_norm + 0.25*rws_z + 0.10*stab_z, con bonus por persistencia.
- **RWS:** Relative Wyckoff Score (normalizacion intra-sector/indice).


## Salidas

- Tablas 'Acciones Seleccionadas por el Modelo de Liderazgo Sectorial' en el reporte.
- Tablas 'Indices Internacionales - Oportunidades de Acumulacion' en el reporte.
- Archivos CSV: nalisis_lideres.csv y nalisis_lideres_internacionales.csv.


## Limitaciones Conocidas
Solo se muestran sectores/indices en fase ACCUMULATION o MARKUP. El resto se omiten por no cumplir criterios de liderazgo estructural.
