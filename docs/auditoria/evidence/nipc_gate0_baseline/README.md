# Evidence - Coverage baseline NIPC (Gate 0 baseline)

Ciclo: IAE NIPC / Q-T50-5 (post dictamen TOP 50).
Fase: coverage baseline (sin OpenFIGI, sin thresholds).
Informe asociado: pendiente (INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_COVERAGE_BASELINE_INFORME.md).
Dictamen habilitante: INSTITUTIONAL_ACCUMULATION_NIPC_GATE0_TOP50_DICTAMEN.md (Q-T50-5).

## Proposito

Medir las 6 metricas pairwise obligatorias (NIPC_COVERAGE_POLICY.md
seccion 2) sobre el operational_universe real y sobre tres universos
anidados previos, para cuantificar donde se pierde cobertura:

    RAW_13F_SH_NULL      SH + PUTCALL NULL
       |
       v  + SEC Official List {ACTIVE, ADDED}
    ELIGIBLE_SEC
       |
       v  + ticker in radar_equities
    TECHNICAL_RADAR       (cierre del UNIVERSO_TECNICO aprobado)
       |
       v  + get_instrument_class(ticker) == EQUITY
    OPERATIONAL_EQUITY    (universo de NIPC_COVERAGE_POLICY seccion 10)

Los tres primeros niveles son diagnostico. Solo OPERATIONAL_EQUITY
es el universo operativo de publicacion.
## Nomenclatura (correccion obligatoria del auditor)

El nivel 1 NO se llama "ALL_TECHNICAL". El UNIVERSO_TECNICO aprobado
cierra en TECHNICAL_RADAR (nivel 3), no en el nivel 1. Llamar "tecnico"
al nivel 1 mezcla el universo 13F completo con el universo radar.

## Fuentes

  - D:/13f_probe/processed/2025Q4/*.parquet     (7 TSVs, Q4 2025)
  - D:/13f_probe/processed/2026Q1/*.parquet     (7 TSVs, Q1 2026)
  - D:/13f_probe/official_list_13f/13flist_2025Q4.txt
  - D:/13f_probe/official_list_13f/13flist_2026Q1.txt
  - data/stock_prices.parquet                   (radar_equities)
  - data/etf_holdings.csv                       (crosswalk interno)
  - data/mappings/cusip_ticker_exceptions.csv   (crosswalk interno)
  - data/mappings/cusip_equivalence.csv         (equivalence)

## Funciones productivas utilizadas

  - temporal_filter::filter_by_period
  - amendments::apply_amendments
  - sec13f_list::load_official_list
  - sec13f_list::resolve_eligibility
  - security_identity::load_crosswalk_internal
  - security_identity::load_cusip_equivalence
  - security_identity::resolve_batch_identities
  - delta_shares::compute_reported_position_units
  - delta_shares::compute_delta_shares
  - nipc::compute_nipc_and_coverage
  - src.instrument_registry::get_instrument_class
No se duplica logica. No se modifican modulos.

## Definicion de radar_equities

    stock_prices.parquet
      -> columnas MultiIndex, nivel 1 = ticker
      -> excluir sufijos europeos:
         .L .DE .MC .PA .AS .MI .BR .ST .HE .CO .OL .VI .LS .IR

Convencion USA del micro-gate. Excluye los 20 tickers .L sin
provider oficial (limitacion conocida).

El probe imprime explicitamente:

    radar_equities = N
    excluidos sufijo EU = [...]

para dejar trazabilidad del snapshot utilizado.

## Reglas congeladas aplicadas

  - canonical_snapshot (apply_amendments) como base de agregados
    SSHPRNAMT. PROHIBIDO sumar desde INFOTABLE.parquet crudo.
  - Match key C2 sin report_period (spec 4.7).
  - canonical_security != ticker. Solo 'equity:<X>' produce ticker.
    'figi:<X>' NO es ticker (C1-revisada).
  - SSHPRNAMT entra UNA VEZ por source line.
## Lo que NO hace este probe

  - NO llama a OpenFIGI. NO usa OPENFIGI_API_KEY.
  - NO fija THRESHOLD_1 / THRESHOLD_2.
  - NO modifica el motor NIPC.
  - NO escribe parquet ni artefactos del pipeline.
  - NO pushea nada.
  - NO usa datetime.now() como fecha de observacion.

## Determinismo

  - Fechas literales: "2025-12-31", "2026-03-31".
  - Sin random. Sin estado externo.
  - Reproducible:
    py docs/auditoria/evidence/nipc_gate0_baseline/probe_coverage_baseline.py

## Orden metodologico

  1. README congelado (este fichero).
  2. probe_coverage_baseline.py escrito.
  3. pyflakes.
  4. Ejecucion.
  5. Resultados -> informe + hashes.

Los resultados NO se producen antes de congelar la metodologia.

## Estado del gate

  Gate-NIPC.2                  SIGUE BLOQUEADO
  OpenFIGI                     fuera de esta fase
  Modificacion motor NIPC      NO
  Thresholds                   NO