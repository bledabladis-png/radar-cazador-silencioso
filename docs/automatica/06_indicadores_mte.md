## Proposito
Infiere el escenario macro que el mercado parece estar descontando, basado en 4 motores (SRS, SHS, CLS, IPS).

## Arquitectura

- compute_srs(): Sector Rotation Score.
- compute_shs(): Safe Haven Score.
- compute_cls(): Credit/Liquidity Stress Score.
- compute_ips(): Inflation Pressure Score.
- compute_msi(): Market Stress Index (SRS + SHS + CLS).
- compute_ipi(): Inflation Pressure Index.
- classify_mte(): clasifica en CRISIS, RECESSION, STAGFLATION, SOFT LANDING, EXPANSION, MIXED.
- compute_confidence(): confianza del escenario (distancia a umbrales + consenso).


## Formulas

- **MSI:** agregacion de SRS, SHS y CLS (0-100).
- **IPI:** basado en IPS (0-100).
- **Confianza:** 0.6 * distancia_umbrales + 0.4 * consenso_motores.


## Salidas

- Seccion 'Market Transition Engine (MTE v1.0)' en el reporte.
- Escenario candidato, MSI, IPI, scores de los 4 motores, Signal Consistency.


## Limitaciones Conocidas
El escenario se marca como (UNCONFIRMED) si la confianza es inferior al 50%. La confianza no esta calibrada historicamente.
