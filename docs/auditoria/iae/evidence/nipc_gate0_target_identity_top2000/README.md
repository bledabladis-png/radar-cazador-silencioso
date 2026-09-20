# Evidencia: piloto TARGET_UNIVERSE Q1 2026 - TOP 2000 CUSIPs

Ciclo: IAE NIPC / F2.3-bis -> validacion cuantitativa ponderada.
Fecha: 2026-09-19.
Estado: BORRADOR para dictamen externo. No push.
Autoridad: dictamen F2.3-bis + dictamen de autorizacion TOP 2000.

## Nomenclatura obligatoria (fijada por el auditor)

    RADAR_SNAPSHOT_DATE = 2026-09-19
    13F_OBSERVATION_PERIOD = 2026-03-31
    RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

## Proposito

Este piloto aplica el radar actual (2026-09-19) retrospectivamente a
las observaciones 13F del 2026Q1, para caracterizar que fraccion del
universo 13F, por numero de securities y por peso SSHPRNAMT, esta
cubierta por el radar actual.

NO se modela historical membership del radar. Esta convencion responde
al issue de temporalidad identificado por el auditor en F2.3-bis, sin
introducir una decision estructural sobre la pertenencia historica.

---
## Limites explicitos

  - NO afirma que los 242 tickers pertenecieran al radar en Q1 2026.
  - NO reconstruye el radar historico de Q1 2026.
  - NO sustituye la definicion futura de historical membership.
  - NO permite llamar a la cobertura obtenida "cobertura historica del
    radar Q1 2026".
  - SI permite medir retrospectivamente la interseccion entre el
    universo 13F Q1 2026 y el radar vigente a 2026-09-19.

## Metricas obligatorias

Por count y por peso SSHPRNAMT:

    target_true_count      target_true_shares
    target_false_count     target_false_shares
    no_id_count            no_id_shares
    error_count            error_shares

Y porcentajes respectivos: pct_count y pct_weight.

Diagnostico del catalogo (requerido por el auditor):

    duplicate_shareClassFIGI
    multiple_openfigi_hits
    conflicting_candidates

Separacion obligatoria: COUNT IMPACT vs WEIGHT IMPACT.

---
## Metodologia

Fuente de observaciones 13F:
    D:\13f_probe\processed\2026Q1\INFOTABLE.parquet
    D:\13f_probe\processed\2026Q1\SUBMISSION.parquet

Filtros sobre 13F:
    - PERIODOFREPORT = 2026-03-31
    - SSHPRNAMTTYPE = SH
    - PUTCALL IS NULL

Muestra: top 2000 CUSIPs por suma SSHPRNAMT.

Resolucion de identidad:
    CUSIP_13F
      -> OpenFIGI /v3/mapping (ID_CUSIP, exchCode=US)
      -> shareClassFIGI + ticker
      -> cruce con RADAR_TARGET_CATALOG por shareClassFIGI
      -> target_membership=True/False

El RADAR_TARGET_CATALOG (242 filas, 240 OK) es INPUT del resolver,
no output. No depende del crosswalk interno.

Determinismo: sin datetime.now(). source_date explicito.

---
## Ficheros generados

  run_pilot_13f_top2000.py    probe deterministico.
  pilot_cusips_sample_top2000.csv   2000 CUSIPs + sshprnamt_sum.
  pilot_13f_top2000.csv       2000 filas + target_membership + status.
  pilot_13f_summary_2000.json resumen ponderado (count + weight).
  catalog_diagnostics.json    diagnosticos de unicidad de shareClassFIGI.
  HASHES.txt                  SHA-256 de los 5 ficheros anteriores.

## Reglas respetadas

  - Sin OpenFIGI masivo (24.838 CUSIPs). Solo top 2000.
  - Sin thresholds fijados. THRESHOLD_1/2 UNDEFINED.
  - Policy v1.0 intacta (hash 57f2d01f...).
  - Sin tocar motor NIPC, C2, spec v1.4.
  - Sin usar crosswalk interno como fuente del catalogo.
  - MISS BRK-B / MOG-A NO convertidos a equivalencia.
  - NO_ID / ERROR conservados y ponderados por SSHPRNAMT.

---
## Expectativas

El objetivo del TOP 2000 no es "ver mas ejemplos", sino comprobar si la
relacion observada en el TOP 500 es estable al ampliar el universo
ponderado.

Comparacion TOP 500 vs TOP 2000:

    - pct_target_count
    - pct_target_weight
    - pct_no_id_weight
    - pct_error_weight
    - distribucion de errores entre CUSIPs de alto y bajo peso
    - conflictos de shareClassFIGI

Si el TOP 2000 muestra cobertura estable, pocos errores, errores de
bajo peso, conflictos practicamente nulos y comportamiento consistente,
entonces el full run queda mucho mejor fundamentado.

## Estado

    F2.3-bis H1                    CERRADO
    TOP 500                        CARACTERIZACION PASS
    TOP 2000 (este piloto)         EN CURSO
    FULL OpenFIGI                  NO AUTORIZADO
    THRESHOLD_1/2                  UNDEFINED
    F2.4                           NO AUTORIZADA
    Gate-NIPC.2                    BLOQUEADO
    Gate-NIPC.3                    NO AUTORIZADO

---

Fin del README. Ciclo F2.3-bis. HEAD c6d3e0a.