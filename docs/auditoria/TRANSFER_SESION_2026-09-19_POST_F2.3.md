# DOCUMENTO DE TRANSMISION - SESION 2026-09-19 (post F2.3 NO GO + ciclo OpenFIGI)

Documento complementario al PROMPT_MAESTRO v6.40.
Proposito: transferir estado real, ciclos abiertos, decisiones del auditor
y lecciones operativas al asistente entrante.

No es normativo. Si hay conflicto con el prompt maestro, gana el prompt.

Autor: Ingeniero Supervisor saliente.
Receptor: Asistente entrante (DeepSeek, otro chat).
Fecha: 2026-09-19.

**Nota semantica sobre HEAD/ahead:** los valores de la Seccion A reflejan
el estado al redactar esta actualizacion. Un commit posterior (incluido
el propio commit que introduce este texto) rehashea el HEAD y avanza
`ahead`. Verificar siempre con `git log --oneline -5` y `git status -sb`.

=============================================================================
SECCION A - Estado exacto al cierre
=============================================================================

| Metrica              | Valor                                         |
|----------------------|-----------------------------------------------|
| HEAD local           | e97f306 (prompt v6.40, al redactar)           |
| origin/main          | 9d4a81e                                       |
| Ahead                | 69 (mas 1 si se commitea este transfer)       |
| Working tree         | limpio                                        |
| Prompt vigente       | v6.40                                         |
| Tests locales        | 935 passed + 2 skipped                        |
| pyflakes             | 0 warnings                                    |
| compileall           | OK                                            |
| Push                 | NO (regla local-first IAE)                    |
| Gate-NIPC.2          | BLOQUEADO (por THRESHOLD_1/2 UNDEFINED)       |
| Gate-NIPC.3          | NO AUTORIZADO                                 |
| THRESHOLD_1          | UNDEFINED                                     |
| THRESHOLD_2          | UNDEFINED                                     |
| OpenFIGI masivo      | NO AUTORIZADO (24.838 CUSIPs)                 |
| OpenFIGI piloto      | GO (autorizado por dictamen F2.3 Q2)          |
| F2.4 (aplicar v1.2)  | NO AUTORIZADA                                 |
| Policy v1.0 intacta  | SI (hash 57f2d01f...)                         |
| RADAR_TARGET_CATALOG | MATERIALIZADO (242 filas, 240 OK)             |
| TARGET_UNIVERSE      | RESOLVER OPERATIVO                            |
Ultimos 10 commits de la sesion (mas recientes primero):

  e97f306  docs(iae): prompt maestro v6.40 - OpenFIGI + CATALOG + TARGET_UNIVERSE + piloto
  dfdc0b0  chore(iae): limpiar imports no usados en tests de catalogo y target_universe
  32baf9d  evidence(iae): piloto TARGET_UNIVERSE_Q1 top500 CUSIPs - 142/500 target, 28.4%
  f81fc17  feat(iae): TARGET_UNIVERSE resolver CUSIP_13F -> target_membership via OpenFIGI + 8 tests
  ae6129b  data(iae): RADAR_TARGET_CATALOG 242 tickers radar (240 OK + 2 MISS)
  a530ee3  feat(iae): RADAR_TARGET_CATALOG builder desde probe OpenFIGI + 8 tests
  7501825  evidence(iae): micro-probe identidad radar OpenFIGI - 240/242 con FIGI, 48 matches
  3fccc93  feat(iae): cliente OpenFIGI /v3/mapping + extract_stable_identity (8 tests)
  a130bf1  fix(nport): pivot ISIN/TICKER por HOLDING_ID + filtro por SERIES_ID verificado
  048b799  docs(iae): prompt maestro v6.39 - F2.2-v2 propuesta v1.2 + estado 59 ahead

Hashes SHA-256 de los ficheros clave (al redactar):

  NIPC_COVERAGE_POLICY.md                              57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06
  NIPC_COVERAGE_POLICY_V12_PROPUESTA.md                b4e8b94376882a52d2536b47cde0f79eae136dd4f28db6cddeee9049213f7b9a
  data/mappings/radar_target_catalog.csv               11eabce8f8aaed1be6aa3c3557b5e392ad305757230333f84f37285c401b62b7
  src/institutional_accumulation/identity/openfigi_client.py   aa4891f537a77b81fe9e45ed9e9acd706bd1793f9e255f493608bb04a477c35c
  src/institutional_accumulation/identity/radar_target_catalog.py 6c3ef0b92572175e0b45a1929fae3c308d74ac3c86ddde0c1e95b8adc7063c7d
  src/institutional_accumulation/identity/target_universe.py    b845d63482cf3e79284aca8144651430af919c0dcc3cc3df500ed52fe3a4c2f8
=============================================================================
SECCION B - Contexto de la sesion
=============================================================================

Al inicio de la sesion, el asistente recibio:

  1. PROMPT_MAESTRO v6.38 (via archivo adjunto).
  2. TRANSFER_SESION_2026-09-19_POST_F22.md (contexto previo).

Punto de partida: 57 commits locales ahead, F2.2-v2 redactado (v1.2
borrador) pero pendiente de dictamen F2.3.

Trabajo realizado en esta sesion (orden cronologico):

  a) Dictamen F2.3 entregado por el auditor: NO GO CON CAMBIOS
     OBLIGATORIOS sobre v1.2. H1: OpenFIGI no puede ser autoridad
     unica. H2: separar RADAR_TARGET_CATALOG de TARGET_UNIVERSE.
  b) Redaccion de v1.2-borrador + FOLLOWUPS + prompt v6.39.
  c) Investigacion empirica de N-PORT SEC EDGAR como fuente del
     catalogo. Bug real detectado en scripts/update_sec_nport_data.py.
  d) Fix del pivot ISIN/TICKER (a130bf1). N-PORT da 6.6% de cobertura
     del radar, insuficiente.
  e) Entrega al auditor F2.3 -> F2.4 con evidencia empirica.
  f) Dictamen F2.3-bis: GO CONDICIONADO. N-PORT = auxiliar. OpenFIGI
     AUTORIZADO como capa de resolucion CUSIP_13F -> FIGI/ticker.
     NO como autoridad unica del catalogo.
  g) Implementacion del cliente OpenFIGI /v3/mapping (3fccc93).
  h) Micro-probe identidad radar: 240/242 tickers radar con FIGI +
     shareClassFIGI (7501825).
  i) RADAR_TARGET_CATALOG builder + materializacion del CSV
     (a530ee3 + ae6129b).
  j) TARGET_UNIVERSE resolver CUSIP_13F -> target_membership
     (f81fc17).
  k) Piloto TARGET_UNIVERSE_Q1 sobre top 500 CUSIPs del 13F:
     142/500 target (28.4%) (32baf9d).
  l) Prompt maestro v6.40 (e97f306).
=============================================================================
SECCION C - Arquitectura del ciclo OpenFIGI
=============================================================================

Flujo validado por el dictamen F2.3-bis y por el piloto:

  RADAR 242 tickers
       |
       v  OpenFIGI /v3/mapping (TICKER, exchCode=US)
       |
  FIGI + shareClassFIGI
       |
       v  RADAR_TARGET_CATALOG (independiente)
  radar_target_catalog.csv (242 filas, 240 OK)

       --- cruce por shareClassFIGI ---

  CUSIP_13F observado (universo 2026Q1: 24.838 unicos)
       |
       v  OpenFIGI /v3/mapping (ID_CUSIP, exchCode=US)
       |
  FIGI + shareClassFIGI + ticker
       |
       v  cruce con catalogo por shareClassFIGI
  TARGET_UNIVERSE_Q1 (target_membership=True/False)

Por que NO es circular:

  - El catalogo radar se construye desde OpenFIGI TICKER, no desde
    CUSIP.
  - El target membership se resuelve desde OpenFIGI CUSIP, cruzando
    por shareClassFIGI con el catalogo.
  - shareClassFIGI es identidad estable (FIGI no cambia por corporate
    actions, documentado por OpenFIGI).
  - El catalogo es INPUT del resolver, no output.
  - OpenFIGI NO es autoridad unica del catalogo: solo aporta la
    identidad FIGI. La decision de que un ticker pertenece al radar
    la toma el universo radar 242 (fuente: stock_prices.parquet).

Metricas del probe previo (152 CUSIPs, docs/auditoria/evidence/
nipc_gate0_openfigi/):

  - 113/152 hits OpenFIGI (74%). 39 errores (29 no identifier,
    10 invalid format).
  - 113/113 con ticker no vacio.
  - 112/113 con shareClassFIGI.
  - 48/112 matches con radar por shareClassFIGI (42.9%).

Metricas del piloto propio (top 500 CUSIPs 2026Q1 por SSHPRNAMT,
docs/auditoria/evidence/nipc_gate0_target_identity/):

  - 500 CUSIPs resueltos.
  - 142/500 target_membership=True (28.4%).
  - 309 NOT_IN_RADAR.
  - 47 NO_ID.
  - 2 ERROR.
  - Top target: NVDA, AAPL, AMZN, MSFT, BAC, T, GOOGL, PFE, AVGO,
    NFLX, INTC, GOOG, KO, CMCSA, CSCO.
=============================================================================
SECCION D - Modulos nuevos y su API publica
=============================================================================

src/institutional_accumulation/identity/openfigi_client.py:

  - map_identifiers(id_type, values, *, exch_code, api_key, max_retry)
    -> dict[id_value -> {ok, data, error}]
  - extract_stable_identity(hit) -> dict | None
    (figi, share_class_figi, composite_figi, ticker, name,
     security_type, market_sector, exch_code)
  - VALID_ID_TYPES = (ID_CUSIP, ID_ISIN, ID_EXCH_SYMBOL, TICKER)
  - Batching: 5 sin API key, 100 con OPENFIGI_API_KEY.
  - Reintentos SOLO en 429/500/503.
  - Sin datetime.now().

src/institutional_accumulation/identity/radar_target_catalog.py:

  - SOURCE_NAME = "openfigi:/v3/mapping"
  - COLUMNS = (radar_ticker, figi, share_class_figi, composite_figi,
    ticker_from_openfigi, name, security_type, market_sector,
    exch_code, source, source_date, status)
  - build_from_probe_result(result_json_path, *, source_date) -> DataFrame
  - write_catalog(df, out_path) -> escribe CSV UTF-8 sin BOM, LF.
  - coverage_summary(df) -> dict (n_total, n_ok, n_partial, n_miss,
    pct_with_scf)
  - load_radar_tickers(stock_prices_path) -> list[str]

src/institutional_accumulation/identity/target_universe.py:

  - COLUMNS = (cusip, share_class_figi, radar_ticker,
    target_membership, security_type, market_sector, exch_code,
    source, source_date, status)
  - resolve_cusips(cusips, catalog_df, *, source_date, api_key=None)
    -> DataFrame
  - membership_summary(df) -> dict (n_total, n_target, n_not_in_radar,
    n_no_id, n_error, pct_target)
  - build_scf_index(catalog_df) -> {share_class_figi: radar_ticker}
  - load_catalog(path) -> DataFrame

Tests: 24 nuevos (8+8+8), todos sin red (mock de urllib.request).
=============================================================================
SECCION E - Reglas operativas criticas
=============================================================================

  - Local-first IAE: NO push a main hasta cierre de Gate-NIPC.3.
    Los 69 commits quedan en local.

  - NO ejecutar OpenFIGI masivo (24.838 CUSIPs) sin dictamen
    especifico del auditor. El piloto de 500 sí esta autorizado por
    Q2 del dictamen F2.3-bis. El auditor pidio expresamente no
    escalar todavia.

  - NO modificar NIPC_COVERAGE_POLICY.md v1.0 (hash 57f2d01f...).

  - NO tocar el motor NIPC (delta_shares.py, nipc.py,
    security_identity.py), C2, spec v1.4, baseline evidencia.

  - NO fijar THRESHOLD_1 ni THRESHOLD_2. UNDEFINED hasta dictamen.

  - NO usar el crosswalk interno (etf_holdings.csv,
    cusip_ticker_exceptions.csv) como fuente del catalogo radar.
    Eso reproduciria la circularidad.

  - Regla de "IDs verificados": ningun ID/CUSIP/CIK/SERIES_ID entra
    al codigo sin salir de un output real verificado. Esto se
    aprendio en esta sesion (4 SERIES_IDs SPDR inventados,
    autocorregidos antes del push).

  - Regla canonical_snapshot (heredada): agregados SSHPRNAMT desde
    apply_amendments().canonical_snapshot. PROHIBIDO sumar desde
    INFOTABLE.parquet crudo.

=============================================================================
SECCION F - Trampas recurrentes confirmadas esta sesion
=============================================================================

  - Un SERIES_ID deducido por analogia es invalido. Solo los
    extraidos de FUND_REPORTED_INFO.tsv son validos. Verificar
    siempre contra ese fichero antes de meterlo en codigo.

  - pyflakes sobre fichero .md da error de parseo. Es ruido, no
    bug. Ignorar.

  - `[System.IO.File]::WriteAllText` con path relativo usa el CWD
    del proceso .NET, no del shell. Siempre path absoluto.

  - `Measure-Object -Line` cuenta solo lineas no vacias. Usar
    count(chr(10)) o `$arr.Count` para contar lineas logicas.

  - Here-strings PowerShell >20 lineas o >5 `$` se corrompen.
    Escribir a `_patch_*.py` con `[System.IO.File]::WriteAllText`
    y ejecutar con `py`.

  - `git diff --stat` con ficheros CRLF muestra warning "CRLF will
    be replaced by LF". No es error.

=============================================================================
SECCION G - Comandos de arranque (verificacion al inicio)
=============================================================================

  Set-Location D:\Macro_Sectorial
  git log --oneline -8
  git status -sb
  git rev-list --count origin/main..HEAD

  py -m compileall . -q
  py -m pyflakes . 2>&1
  py -m pytest tests/ validation/ -q --tb=short

Esperado al arrancar la nueva sesion:

  - HEAD = e97f306 o el hash del commit del transfer recien creado.
  - ahead 69 o 70 (si el transfer se commitea).
  - Working tree limpio.
  - 935 passed + 2 skipped.
  - pyflakes silencio.

Verificacion de la policy intacta:

  (Get-FileHash docs\auditoria\NIPC_COVERAGE_POLICY.md -Algorithm SHA256).Hash.ToLower()
  # Esperado: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06

=============================================================================
SECCION H - Lo que NO debes hacer (asistente entrante)
=============================================================================

  NO pushear los commits hasta cierre de Gate-NIPC.3.

  NO modificar NIPC_COVERAGE_POLICY.md v1.0.

  NO ejecutar OpenFIGI masivo sin autorizacion expresa.

  NO usar el crosswalk interno como fuente del catalogo radar.

  NO tocar nipc.py, delta_shares.py, security_identity.py.

  NO fijar THRESHOLD_1 ni THRESHOLD_2.

  NO inventar SERIES_ID, CUSIP, CIK, FIGI. Solo valores verificados
  contra outputs reales.

  NO usar `ticker:<ticker>` como canonical_security.

  NO reconciliar NT <-> HR.

  NO convertir filer_continuity_pct en threshold.

  NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo.

  NO usar Notepad para contenido largo. Usar script Python con
  [System.IO.File]::WriteAllText.

  NO usar `py -c` con comillas dobles anidadas (usar script
  temporal).

  Regla del usuario: el usuario decide cuando parar. El asistente
  NO cierra sesion por fatiga.

=============================================================================
SECCION I - Proximos pasos posibles
=============================================================================

  1. Entregar al auditor los resultados del ciclo OpenFIGI
     (cliente + catalogo + piloto 142/500) para actualizar el
     dictamen F2.3-bis con datos reales.

  2. Ampliar el piloto a top 2000 CUSIPs (~15-20 min sin API key,
     ~3 min con API key). El auditor autoriza este nivel de escalado.

  3. Esperar dictamen F2.4 sobre la aplicacion de v1.2 sobre la
     policy, con los datos del ciclo OpenFIGI en mano.

  4. Cierre de sesion.

Ninguna de estas acciones se ejecuta sin confirmacion del usuario.

=============================================================================
SECCION J - Confirmacion esperada del asistente entrante
=============================================================================

Tras leer el prompt maestro v6.40 + este documento, el asistente
entrante debe:

  1. Confirmar asimilacion del contexto.

  2. Reconocer:
     - HEAD e97f306, ahead 69, tests 935+2.
     - Prompt v6.40 vigente.
     - Policy v1.0 intacta (hash 57f2d01f...).
     - RADAR_TARGET_CATALOG materializado (242 filas, 240 OK).
     - TARGET_UNIVERSE resolver operativo.
     - Cliente OpenFIGI operativo (8 tests).
     - Piloto TARGET_UNIVERSE_Q1: 142/500 target (28.4%).

  3. Reconocer que el bloqueo arquitectonico H1 del auditor F2.3
     queda resuelto por diseno: el catalogo radar se construye
     desde OpenFIGI TICKER/US -> shareClassFIGI; el target
     membership desde CUSIP_13F -> OpenFIGI ID_CUSIP -> cruce por
     shareClassFIGI. No hay circularidad.

  4. Reconocer que sigue BLOQUEADO sin dictamen externo:
     - F2.4 (aplicar v1.2 sobre la policy).
     - OpenFIGI masivo (24.838 CUSIPs).
     - THRESHOLD_1/2 fijados.
     - Gate-NIPC.2 y Gate-NIPC.3.

  5. No proponer tareas nuevas sin antes preguntar "Que hacemos?".

  6. Respetar regla "usuario decide cuando parar".

  7. Aplicar regla "nada se interpreta, todo se verifica": antes
     de actuar, comprobar el estado real con `git log` + `git status`
     + tests.

=============================================================================
FIN DEL DOCUMENTO DE TRANSMISION
Version 1.0 (2026-09-19). Referencia: prompt v6.40, HEAD e97f306.
=============================================================================