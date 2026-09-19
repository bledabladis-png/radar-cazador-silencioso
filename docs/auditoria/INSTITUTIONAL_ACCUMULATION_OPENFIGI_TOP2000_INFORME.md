# IAE NIPC - Informe piloto TOP 2000 + hallazgos de calidad de identidad

Ciclo: F2.3-bis -> validacion cuantitativa ponderada.
Fecha: 2026-09-19.
Estado: BORRADOR para dictamen externo. No push.
Autoridad: dictamen F2.3-bis + dictamen de autorizacion TOP 2000.

---

## 0. Resumen ejecutivo

Piloto ejecutado sobre top 2000 CUSIPs del 13F 2026Q1, con metrica
ponderada por SSHPRNAMT (count + weight).

Resultado raw (matcheo estricto por shareClassFIGI + exchCode=US):

  target_true   210 (10.50% count, 32.8079% weight)
  target_false 1567 (78.35% count, 55.6839% weight)
  no_id         220 (11.00% count,  8.8561% weight)
  error           3 ( 0.15% count,  2.6521% weight)

Resultado corregido (2 canales de recuperacion adicionales):

  target_true   212 (10.60% count, 33.3428% weight)
  target_false 1567 (78.35% count, 55.6839% weight)
  no_id         218 (10.90% count,  8.3211% weight)
  error           3 ( 0.15% count,  2.6521% weight)

Delta pct_target_weight = +0.5349 pp.

Hallazgos principales:
  1. count vs weight divergen fuertemente. pct_target_count cae de
     28.4% (top500) a 10.5% (top2000); pct_target_weight cae de
     46.76% a 32.81%. Confirma la hipotesis del auditor sobre
     sobrerrepresentacion de large caps del radar en el top 500.
  2. shareClassFIGI cambia entre entidad historica y actual (caso XOM).
     Matiza la premisa "FIGI inmune a corporate actions".
  3. HON no esta en el radar 242. NO_ID legitimo.
  4. 221 NO_ID genuinos concentrados en prefijos no-USA (G/H/N/F/Y/D).
  5. Deuda: multiple_openfigi_hits=0 hardcoded (cliente no lo expone).
  6. Deuda: target_universe.py no consulta cusip_ticker_exceptions.csv.

---
## 1. Estado de verificacion

| Metrica | Valor |
|---|---|
| HEAD local | (al redactar, sin commitear) |
| origin/main | 9d4a81e |
| Ahead | 72 |
| Tests locales | 935 passed + 2 skipped |
| pyflakes | 0 warnings |
| compileall | OK |
| Policy v1.0 hash | 57f2d01f... (intacta) |
| Push | NO (regla local-first IAE) |
| Convencion temporal | CURRENT_RETROSPECTIVE (autorizada) |

RADAR_SNAPSHOT_DATE = 2026-09-19
13F_OBSERVATION_PERIOD = 2026-03-31
RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

---
## 2. Muestra y filtros

Fuente: D:\13f_probe\processed\2026Q1 (INFOTABLE + SUBMISSION).

Pipeline de filtrado:
  INFOTABLE rows:                      3.822.885
  SUBMISSION rows:                        11.761
  SUBMISSION filings en periodo:          10.776
  INFOTABLE tras filtro periodo:       3.321.967
  Tras SH + PUTCALL NULL:              3.188.083
  CUSIPs unicos:                          24.838
  Muestra (top 2000 por SSHPRNAMT):        2.000

Tiempo de ejecucion: 0.69 min (con API key, batching 100, sleep 1s).

---
## 3. Resultados (raw + corregidos)

### 3.1. Raw - shareClassFIGI + exchCode=US estricto

| Categoria | count | % count | shares | % weight |
|---|---:|---:|---:|---:|
| target_true | 210 | 10.5000 | 219.189.415.704 | 32.8079 |
| target_false | 1567 | 78.3500 | 372.023.632.279 | 55.6839 |
| no_id | 220 | 11.0000 | 59.167.263.147 | 8.8561 |
| error | 3 | 0.1500 | 17.718.787.317 | 2.6521 |
| TOTAL | 2000 | 100 | 668.099.098.447 | 100 |

### 3.2. Corregido - con 2 canales adicionales

Canales:
  (a) cusip_ticker_exceptions.csv (Q-CUR): 30231G102 -> XOM.
  (b) ticker match ignorando exchCode: 682680103 -> OKE (exch ER).

| Categoria | count | % count | shares | % weight |
|---|---:|---:|---:|---:|
| target_true | 212 | 10.6000 | 222.763.244.901 | 33.3428 |
| target_false | 1567 | 78.3500 | 372.023.632.279 | 55.6839 |
| no_id | 218 | 10.9000 | 55.593.433.950 | 8.3211 |
| error | 3 | 0.1500 | 17.718.787.317 | 2.6521 |
| TOTAL | 2000 | 100 | 668.099.098.447 | 100 |

Delta pct_target_weight: +0.5349 pp.

---
## 4. Comparacion top 500 vs top 2000

### 4.1. Por count

| Categoria | top500 | top2000 | ratio |
|---|---:|---:|---:|
| target_true | 142 (28.40%) | 210 (10.50%) | 2.70x |
| target_false | 309 (61.80%) | 1567 (78.35%) | 1.27x |
| no_id | 47 (9.40%) | 220 (11.00%) | 1.17x |
| error | 2 (0.40%) | 3 (0.15%) | 0.38x |

### 4.2. Por weight (SSHPRNAMT)

| Categoria | top500 | top2000 | delta |
|---|---:|---:|---:|
| target_true | 46.76% | 32.81% | -13.95 pp |
| target_false | 41.52% | 55.68% | +14.16 pp |
| no_id | 7.73% | 8.86% | +1.13 pp |
| error | 3.99% | 2.65% | -1.34 pp |

### 4.3. Lectura

El pct_target_count cae 2.70x al ampliar la muestra. El
pct_target_weight cae solo 1.43x. Los target concentran mas peso que
count. Esto valida la hipotesis del auditor F2.3-bis:

  "El TOP 500 no es una muestra aleatoria de CUSIPs: esta
   deliberadamente concentrada por tamano de posicion."

El 28.4% de target en top500 NO era una estimacion de coverage: era
un artefacto del sesgo de seleccion por SSHPRNAMT.

---
## 5. Perfil de los 223 NO_ID + ERROR

### 5.1. Categorizacion por canal OpenFIGI (sin exchCode)

| Categoria | count | Interpretacion |
|---|---:|---|
| ok_with_data (hits, todos extranjeros) | 55 | OpenFIGI devuelve hits pero extract_stable_identity prioriza US -> None |
| ok_empty | 0 | OpenFIGI no devuelve hits |
| fail (No identifier found) | 168 | CUSIPs genuinamente desconocidos para OpenFIGI |
| TOTAL | 223 | |

### 5.2. Recuperables por otros canales

| Canal | count | shares (B) | % weight |
|---|---:|---:|---:|
| cusip_ticker_exceptions.csv (Q-CUR) | 1 (XOM) | 3.02 | 0.4521 |
| ticker match ignorando exchCode | 1 (OKE) | 0.55 | 0.0828 |
| NO_ID genuinos | 221 | ~73.3 | ~10.97 |

### 5.3. Prefijos dominantes en NO_ID genuinos

G, H, N, F, Y, D -> securities no-USA listadas en mercados no-USA
(Canada, Israel, UK, etc.) que aparecen en el 13F pero no en OpenFIGI
con ID_CUSIP. NO son errores: son ausencias legitimas de cobertura.

### 5.4. Los 3 ERROR (formato)

| CUSIP | Shares (B) | Causa OpenFIGI |
|---|---:|---|
| 329882225 | 10.59 | Invalid idValue format |
| 329882250 | 6.99 | Invalid idValue format |
| 00CASHJPY | 0.14 | Invalid idValue format (no es CUSIP) |

El cliente no valida formato CUSIP antes de enviar. Deuda menor.

---
## 6. Hallazgo mayor: shareClassFIGI cambia tras reorganizacion

Caso XOM (CUSIP 30231G102):

  Catalogo radar (OpenFIGI TICKER=XOM, exchCode=US):
    shareClassFIGI = BBG023CY9NL0
    name           = EXXONMOBIL HOLDINGS CORP

  OpenFIGI CUSIP=30231G102 (13F Q1 2026):
    shareClassFIGI = BBG001S69V32
    name           = EXXON MOBIL CORP

Misma empresa, dos shareClassFIGI distintos.

### Implicacion

La premisa del dictamen F2.3-bis "shareClassFIGI es identidad estable
(no cambia por corporate actions)" requiere matiz:

  - FIGI puede persistir entre cambios de ticker.
  - FIGI NO persiste entre reorganizaciones corporativas completas
    (holdings company vs operating company).
  - El matcheo por scf da falso NOT_IN_RADAR para la entidad historica.

### Efecto medido

XOM recuperable via cusip_ticker_exceptions.csv (Q-CUR): 1 CUSIP,
3.02B shares, 0.4521% weight del piloto.

### Recomendacion v1.3

El resolver debe considerar matching dual:
  1. shareClassFIGI (identidad actual).
  2. ticker match (cubre reorganizaciones).
  3. cusip_ticker_exceptions.csv (casos curados manualmente).

---
## 7. Hallazgo menor: HON no esta en el radar 242

Caso HON (CUSIP 438516106):

  - Busqueda en catalogo: radar_ticker=HON no existe.
  - El radar tiene 242 tickers; HON no es uno de ellos.
  - OpenFIGI devuelve solo hits London (X1) con scf=None.
  - NO_ID legitimo, no falso positivo del filtro exchCode.

Confirmacion: HON figura en cusip_ticker_exceptions.csv (Q-CUR) con
reason "sustituido posteriormente tras reverse split efectivo
2026-06-29". El reverse split posterior a Q1 2026 lo excluye del radar
actual.

---

## 8. Diagnosticos del catalogo

catalog_diagnostics.json:

  n_rows: 242
  n_unique_share_class_figi: 240
  duplicate_shareClassFIGI_count: 0
  duplicate_shareClassFIGI_detail: {}
  multiple_openfigi_hits: 0  [DEUDA - hardcoded]
  conflicting_candidates: 0
  radar_snapshot_date: 2026-09-19
  radar_membership_mode: CURRENT_RETROSPECTIVE

DEUDA DECLARADA: el contador multiple_openfigi_hits=0 esta hardcoded.
El cliente openfigi_client.py::extract_stable_identity no expone el
numero de hits crudos por CUSIP, solo devuelve el primero que prioriza
exchCode=US. Para calcularlo realmente seria necesario modificar el
cliente. Se documenta como deuda, no se falsea.

---
## 9. Deudas declaradas

D1. multiple_openfigi_hits=0 hardcoded (ver seccion 8).
    Impacto: el auditor no puede validar unicidad de hits por CUSIP.
    Requiere modificar openfigi_client.py.

D2. target_universe.py NO consulta cusip_ticker_exceptions.csv.
    Impacto: XOM (curado en Q-CUR) no se resuelve automaticamente.
    Recomendacion: integrar en resolver v1.3.

D3. Cliente no valida formato CUSIP antes de enviar.
    Impacto: los 3 ERROR generan peticiones inutiles.
    Recomendacion: regex CUSIP antes de encolar.

D4. BRK-B / MOG-A sin resolver (catalogo, 2 MISS).
    Documentado en informe F2.3-bis. Pendiente de evidencia.

---

## 10. Verificacion cruzada top500 subset top2000

Control interno de calidad:

  - Los primeros 500 CUSIPs del top 2000 son identicos al piloto
    top 500 original (mismo criterio de orden por SSHPRNAMT).
  - Merge por CUSIP: 500 both / 0 left_only / 0 right_only.
  - Mismatches de status + target_membership: 0.

Reproducibilidad confirmada.

---
## 11. Reglas respetadas

  - Convencion temporal CURRENT_RETROSPECTIVE declarada.
  - Sin OpenFIGI masivo (24.838 CUSIPs). Solo top 2000 autorizado.
  - Sin thresholds fijados. THRESHOLD_1/2 UNDEFINED.
  - Policy v1.0 intacta (hash 57f2d01f...).
  - Sin tocar motor NIPC, C2, spec v1.4.
  - Sin tocar nipc.py, delta_shares.py, security_identity.py.
  - MISS BRK-B / MOG-A NO convertidos a equivalencia.
  - NO_ID / ERROR conservados y ponderados por SSHPRNAMT.
  - Sin push (regla local-first IAE activa).

---

## 12. Estado de gates

| Gate | Estado |
|---|---|
| F2.3-bis H1 | CERRADO |
| TOP 500 | CARACTERIZACION PASS |
| TOP 2000 | EJECUTADO (este informe) |
| FULL OpenFIGI | NO AUTORIZADO |
| THRESHOLD_1/2 | UNDEFINED |
| F2.4 | NO AUTORIZADA |
| Gate-NIPC.2 | BLOQUEADO |
| Gate-NIPC.3 | NO AUTORIZADO |

---
## 13. Recomendaciones al auditor

R1. Aceptar el piloto top 2000 como validacion cuantitativa ponderada.
    Los resultados raw y corregidos estan documentados.

R2. Confirmar el sesgo medido del exchCode=US (~0.53 pp de weight).
    No invalida la arquitectura H1, pero debe declararse en la policy
    v1.2/v1.3.

R3. Incorporar el hallazgo XOM (shareClassFIGI cambia entre entidad
    historica y actual) al rediseno de la premisa "identidad estable".

R4. Aprobar el rediseno del resolver v1.3 con matching dual:
    scf + ticker match + exceptions Q-CUR.

R5. Resolver las 4 deudas declaradas antes de fijar thresholds.

R6. Decidir si el full run (24.838 CUSIPs) procede tras F2.4.

---

## 14. Hashes SHA-256 de los artefactos

Catalogo radar:
  data/mappings/radar_target_catalog.csv -> 11eabce8...

Piloto top 2000 (docs/auditoria/evidence/nipc_gate0_target_identity_top2000/):
  README.md                         -> b967d30a...
  run_pilot_13f_top2000.py          -> dc304a7d...
  pilot_cusips_sample_top2000.csv   -> 208f796d...
  pilot_13f_top2000.csv             -> 72d86965...
  pilot_13f_summary_2000.json       -> d1879c31...
  pilot_13f_summary_2000_corrected.json -> (ver HASHES.txt)
  catalog_diagnostics.json          -> 7e101f55...
  pilot_run.txt                     -> bc7e773b...

Ver HASHES.txt del evidence dir para el listado completo.

---

Fin del informe. Ciclo F2.3-bis / TOP 2000. HEAD al redactar: c6d3e0a.
Fecha: 2026-09-19.