# IAE NIPC - Informe OpenFIGI + RADAR_TARGET_CATALOG + TARGET_UNIVERSE

Ciclo: F2.3-bis -> entrega de evidencia empirica al auditor.
Fecha: 2026-09-19.
Estado: BORRADOR para dictamen externo. No push.
Autoridad: dictamen F2.3 NO GO con cambios obligatorios +
           dictamen F2.3-bis GO CONDICIONADO (Q1-Q5).

---

## 0. Resumen ejecutivo

El bloqueo arquitectonico H1 del dictamen F2.3 (OpenFIGI no puede
ser autoridad unica del catalogo radar) queda resuelto por diseno:

  - RADAR_TARGET_CATALOG construido desde OpenFIGI TICKER/US ->
    shareClassFIGI. Independiente del crosswalk interno.
  - TARGET_UNIVERSE resuelto desde CUSIP_13F -> OpenFIGI ID_CUSIP
    -> cruce por shareClassFIGI contra el catalogo.
  - OpenFIGI aporta identidad FIGI, no autoridad del catalogo.

Evidencia empirica: piloto top 500 CUSIPs del 13F 2026Q1.
  - 142/500 target_membership=True (28.4%).
  - 309 NOT_IN_RADAR, 47 NO_ID, 2 ERROR.

Estado gates: Gate-NIPC.2 BLOQUEADO por THRESHOLD_1/2 UNDEFINED.
F2.4 NO AUTORIZADA. OpenFIGI masivo NO AUTORIZADO.

---
## 1. Estado de verificacion local

| Metrica | Valor |
|---|---|
| HEAD local | c6d3e0a (transfer commiteado) |
| origin/main | 9d4a81e |
| Ahead | 70 |
| Working tree | limpio |
| Tests locales | 935 passed + 2 skipped |
| Tests OpenFIGI (24) | 24 passed |
| pyflakes | 0 warnings |
| compileall | OK |
| Policy v1.0 hash | 57f2d01f... (intacto) |
| Push | NO (regla local-first IAE) |

---
## 2. Componentes implementados

### 2.1. openfigi_client.py (180 lineas, 8 tests)

Cliente HTTP para OpenFIGI /v3/mapping.

API publica:

  URL = https://api.openfigi.com/v3/mapping
  VALID_ID_TYPES = [ID_CUSIP, ID_ISIN, ID_EXCH_SYMBOL, TICKER]
  RETRYABLE_HTTP = [429, 500, 503]

  def map_identifiers(id_type, values, *, exch_code, api_key, max_retry)
  def extract_stable_identity(hit)

Batching: 5 sin API key, 100 con OPENFIGI_API_KEY.
Reintentos SOLO en 429/500/503. Sin datetime.now().
### 2.2. radar_target_catalog.py (145 lineas, 8 tests)

Builder del catalogo desde el probe OpenFIGI del radar.

API publica:

  SOURCE_NAME = openfigi:/v3/mapping
  COLUMNS = [radar_ticker, figi, share_class_figi, composite_figi,
             ticker_from_openfigi, name, security_type,
             market_sector, exch_code, source, source_date, status]

  def load_radar_tickers(stock_prices_path)
  def build_from_probe_result(result_json_path, *, source_date)
  def write_catalog(df, out_path)
  def coverage_summary(df)

### 2.3. target_universe.py (171 lineas, 8 tests)

Resolver CUSIP_13F -> target_membership via OpenFIGI + cruce por
shareClassFIGI contra el catalogo.

API publica:

  COLUMNS = [cusip, share_class_figi, radar_ticker,
             target_membership, security_type, market_sector,
             exch_code, source, source_date, status]

  def load_catalog(path)
  def build_scf_index(catalog_df)
  def resolve_cusips(cusips, catalog_df, *, source_date, api_key)
  def membership_summary(df)

---
## 3. RADAR_TARGET_CATALOG - metricas

Fichero: data/mappings/radar_target_catalog.csv
  - 242 filas, 30092 bytes.
  - SHA-256: 11eabce8f8aaed1be6aa3c3557b5e392ad305757230333f84f37285c401b62b7

Distribucion de status:
  - OK: 240 (99.17%)
  - MISS: 2 (BRK-B, MOG-A)

Causa de los 2 MISS: divergencia de nomenclatura Yahoo vs OpenFIGI.
  - BRK-B (Yahoo) vs BRK/B (OpenFIGI).
  - MOG-A (Yahoo) vs MOG/A (OpenFIGI).
No es bug del cliente: es divergencia de convenciones de ticker.
Deuda aceptada del catalogo (documentada).

Cobertura shareClassFIGI: 240/242 (99.17%).

---
## 4. Piloto TARGET_UNIVERSE_Q1 - top 500 CUSIPs

Periodo: PERIODOFREPORT=2026-03-31 (2026Q1).
Muestra: top 500 CUSIPs por suma SSHPRNAMT (SH + PUTCALL NULL).
Fuente: D:\13f_probe\processed\2026Q1\INFOTABLE.parquet +
        SUBMISSION.parquet.

Resultados (pilot_13f_summary.json):

| Categoria | n | pct |
|---|---:|---:|
| n_total | 500 | 100.00 |
| n_target (True) | 142 | 28.40 |
| n_not_in_radar | 309 | 61.80 |
| n_no_id (sin match OpenFIGI) | 47 | 9.40 |
| n_error | 2 | 0.40 |

Verificacion: 142 + 309 + 47 + 2 = 500.

Top 15 target (por SSHPRNAMT):
  NVDA, AAPL, AMZN, MSFT, BAC, T, GOOGL, PFE, AVGO,
  NFLX, INTC, GOOG, KO, CMCSA, CSCO.
Hallazgos:

H1. 28.4% de target entre top 500 CUSIPs es consistente con la
    distribucion esperada. Universo 13F: ~24.838 CUSIPs unicos
    en 2026Q1. Radar: 242 tickers. Ratio bruto uniforme ~1%.
    El 28.4% refleja sobrerrepresentacion natural de large caps
    del radar en el top 500 por SSHPRNAMT.

H2. CUSIP 329882225 (2do por SSHPRNAMT, 10.59B, n_lines=1)
    devuelve status=ERROR en OpenFIGI. Requiere diagnostico
    manual en ciclo posterior. No bloquea el ciclo actual.

H3. 309 NOT_IN_RADAR son CUSIPs resueltos por OpenFIGI a
    FIGI+ticker pero que no cruzan por shareClassFIGI contra
    el catalogo radar. CUSIPs legitimamente fuera del radar.

H4. 47 NO_ID son CUSIPs sin match OpenFIGI. Coherente con el
    probe previo (152 CUSIPs, 39 errores ~ 25.6%).

---
## 5. Resolucion del bloqueo arquitectonico H1

Bloqueo original (dictamen F2.3):
  OpenFIGI no puede ser autoridad unica del catalogo radar.

Resolucion por diseno:

  Paso 1 - Construccion del catalogo radar:
    radar 242 tickers (stock_prices.parquet)
      -> OpenFIGI /v3/mapping (TICKER, exchCode=US)
      -> shareClassFIGI
    El catalogo es INPUT del resolver, no output.

  Paso 2 - Construccion del target membership:
    CUSIP_13F observado
      -> OpenFIGI /v3/mapping (ID_CUSIP, exchCode=US)
      -> shareClassFIGI + ticker
      -> cruce con catalogo por shareClassFIGI
      -> target_membership=True/False

  No circularidad:
    - El catalogo no depende del crosswalk interno.
    - La decision de pertenencia al radar la toma el universo
      242 tickers (fuente: stock_prices.parquet).
    - OpenFIGI aporta identidad FIGI, no autoridad.
    - shareClassFIGI es identidad estable (no cambia por
      corporate actions, documentado por OpenFIGI).

Evidencia estructural: los 3 modulos nuevos (openfigi_client,
radar_target_catalog, target_universe) NO referencian
etf_holdings.csv, cusip_ticker_exceptions.csv, isin_ticker_map.csv
ni cusip_equivalence.csv. Verificado por grep (0 matches).

---
## 6. Reglas respetadas

  - Sin OpenFIGI masivo (24.838 CUSIPs). Solo piloto top 500.
    Autorizado por Q2 del dictamen F2.3-bis.
  - Sin thresholds fijados. THRESHOLD_1/2 UNDEFINED.
  - Policy v1.0 intacta (hash 57f2d01f...).
  - Sin tocar motor NIPC (delta_shares, nipc, security_identity).
  - Sin tocar C2, spec v1.4, baseline evidencia.
  - Sin usar crosswalk interno como fuente del catalogo.
  - Sin datetime.now() en los 3 modulos nuevos.
  - Sin push (regla local-first IAE activa).

---

## 7. Estado de gates

| Gate | Estado |
|---|---|
| Gate-NIPC.2 | BLOQUEADO (THRESHOLD_1/2 UNDEFINED) |
| Gate-NIPC.3 | NO AUTORIZADO |
| F2.3 | NO GO v1.2 (cerrado) |
| F2.3-bis | GO CONDICIONADO (este ciclo) |
| F2.4 | NO AUTORIZADA |
| OpenFIGI masivo | NO AUTORIZADO |

---
## 8. Hashes SHA-256 de los artefactos clave

| Fichero | SHA-256 (prefijo) |
|---|---|
| NIPC_COVERAGE_POLICY.md | 57f2d01f... |
| NIPC_COVERAGE_POLICY_V12_PROPUESTA.md | b4e8b943... |
| data/mappings/radar_target_catalog.csv | 11eabce8... |
| src/.../openfigi_client.py | aa4891f5... |
| src/.../radar_target_catalog.py | 6c3ef0b9... |
| src/.../target_universe.py | b845d634... |

Evidencia del piloto (docs/auditoria/evidence/nipc_gate0_target_identity/):
  - README.md + HASHES.txt anadidos en este ciclo (antes: asimetria
    respecto a los otros 2 evidence dirs).
  - 7 ficheros con hash registrado en HASHES.txt.

---
## 9. Proximos pasos

Pendientes de dictamen externo (F2.3-bis final):

  1. Confirmacion del auditor de que el bloqueo H1 queda cerrado.
  2. Autorizacion para aplicar v1.2 sobre la policy (F2.4).
  3. Decision sobre escalado del piloto (top 2000 CUSIPs).
  4. Decision sobre fijacion de THRESHOLD_1/2 (Gate-NIPC.2).

Nada de lo anterior se ejecuta sin dictamen explicito.

---

Fin del informe. Ciclo F2.3-bis. HEAD c6d3e0a.