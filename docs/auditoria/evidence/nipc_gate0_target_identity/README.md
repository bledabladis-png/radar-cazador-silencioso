# Evidencia: identidad radar + piloto TARGET_UNIVERSE Q1 2026

Ciclo: IAE NIPC / F2.3-bis (post NO GO v1.2).
Fecha: 2026-09-19.
Estado: BORRADOR para revision externa. No push.

## Proposito

Evidenciar empiricamente que el bloqueo arquitectonico H1 del
dictamen F2.3 (OpenFIGI no puede ser autoridad unica del
catalogo radar) queda resuelto por diseno, mediante:

  1. Construccion del RADAR_TARGET_CATALOG desde OpenFIGI
     TICKER/US -> shareClassFIGI (independiente del crosswalk).
  2. Construccion del TARGET_UNIVERSE desde CUSIP_13F -> OpenFIGI
     ID_CUSIP -> shareClassFIGI, cruzando contra el catalogo.

El cruce por shareClassFIGI es la identidad estable. El catalogo
es INPUT del resolver, no output. No hay circularidad.

## Ficheros

  result_radar.json           resultado crudo OpenFIGI para 242
                              tickers radar (TICKER, exchCode=US).
  run_radar_openfigi.py       probe deterministico que genero
                              result_radar.json.
  match_report.json           agregados del probe radar + primeros
                              48 matches por shareClassFIGI contra
                              los 113 CUSIPs del probe previo.
  pilot_cusips_sample.csv     500 CUSIPs del 13F 2026Q1 (top por
                              SSHPRNAMT, SH + PUTCALL NULL).
  run_pilot_13f.py            probe deterministico que genero la
                              muestra + resolvio target_membership.
  pilot_13f_top500.csv        resultado del piloto: 500 filas con
                              target_membership + status.
  pilot_13f_summary.json      resumen agregado del piloto.
  HASHES.txt                  SHA-256 de los 7 ficheros.

## Metodologia

Fuente de identidad: OpenFIGI /v3/mapping.
  - Radar: 242 tickers de stock_prices.parquet -> OpenFIGI
    (TICKER, exchCode=US) -> shareClassFIGI.
  - 13F: 500 CUSIPs top del periodo 2026Q1 -> OpenFIGI
    (ID_CUSIP, exchCode=US) -> shareClassFIGI + ticker.
  - Cruce por shareClassFIGI contra el catalogo radar.

Determinismo: ambos probes sin datetime.now(). source_date
explicito, aportado por el caller.

## Metricas

Radar (242 tickers):
  - 240/242 con FIGI (99.17%).
  - 240/242 con shareClassFIGI.
  - 2 MISS: BRK-B, MOG-A (nomenclatura Yahoo vs OpenFIGI BRK/B,
    MOG/A).

Piloto (top 500 CUSIPs 13F 2026Q1):
  - 142/500 target_membership=True (28.4%).
  - 309 NOT_IN_RADAR.
  - 47 NO_ID (sin match OpenFIGI).
  - 2 ERROR (incluye CUSIP 329882225, 2do por SSHPRNAMT con
    10.59B y n_lines=1).

## Reglas respetadas

  - Sin OpenFIGI masivo (24.838 CUSIPs). Solo top 500.
  - Sin thresholds fijados (THRESHOLD_1/2 UNDEFINED).
  - Policy v1.0 intacta (hash 57f2d01f...).
  - Sin tocar motor NIPC, C2, spec v1.4, baseline evidencia.
  - Sin usar crosswalk interno como fuente del catalogo.
