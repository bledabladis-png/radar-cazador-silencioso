# DOCUMENTO DE TRANSMISION - SESION 2026-09-19 (post F2.3-bis + TOP 2000)

Documento complementario al PROMPT_MAESTRO v6.41.
Proposito: transferir estado real, ciclos abiertos, decisiones del
auditor y lecciones operativas al asistente entrante.

No es normativo. Si hay conflicto con el prompt maestro, gana el prompt.

Autor: Ingeniero Supervisor saliente.
Receptor: Asistente entrante.
Fecha: 2026-09-19.

Nota semantica sobre HEAD/ahead: los valores de la Seccion A reflejan
el estado al redactar esta actualizacion. Un commit posterior (incluido
el propio commit que introduce este texto) rehashea HEAD y avanza
ahead. Verificar siempre con git log --oneline -5 y git status -sb.

=============================================================================
SECCION A - Estado exacto al cierre
=============================================================================

| Metrica              | Valor                                         |
|----------------------|-----------------------------------------------|
| HEAD local           | 06c49ad (prompt v6.41 al redactar)            |
| origin/main          | 9d4a81e                                       |
| Ahead                | 74 (mas 1 si se commitea este transfer)       |
| Working tree         | limpio                                        |
| Prompt vigente       | v6.41                                         |
| Tests locales        | 935 passed + 2 skipped                        |
| pyflakes             | 0 warnings                                    |
| compileall           | OK                                            |
| Push                 | NO (regla local-first IAE)                    |
| Gate-NIPC.2          | BLOQUEADO (por THRESHOLD_1/2 UNDEFINED)       |
| Gate-NIPC.3          | NO AUTORIZADO                                 |
| THRESHOLD_1          | UNDEFINED                                     |
| THRESHOLD_2          | UNDEFINED                                     |
| OpenFIGI masivo      | NO AUTORIZADO (24.838 CUSIPs)                 |
| OpenFIGI piloto      | GO (top 500 + top 2000 ejecutados)            |
| F2.4 (aplicar v1.2)  | NO AUTORIZADA                                 |
| Policy v1.0 intacta  | SI (hash 57f2d01f...)                         |
| RADAR_TARGET_CATALOG | MATERIALIZADO (242 filas, 240 OK)             |
| TARGET_UNIVERSE      | RESOLVER OPERATIVO                            |

Ultimos 8 commits de la sesion (mas recientes primero):

  06c49ad  chore: normalizar EOL segun .gitattributes (git add --renormalize .)
  9c67ac6  chore(iae): normalizar EOL en evidence historicos + recalcular HASHES (patron baseline)
  3ef4d2d  docs(iae): F2.3-bis - dictamen + autorizacion + informe TOP 2000 + evidence
  4593513  docs(iae): informe OpenFIGI + catalogo + target_universe + evidence README/HASHES
  c6d3e0a  docs(iae): transfer sesion 2026-09-19 post F2.3 + ciclo OpenFIGI
  e97f306  docs(iae): prompt maestro v6.40 - OpenFIGI + CATALOG + TARGET_UNIVERSE + piloto
  dfdc0b0  chore(iae): limpiar imports no usados en tests de catalogo y target_universe
  32baf9d  evidence(iae): piloto TARGET_UNIVERSE_Q1 top500 CUSIPs - 142/500 target, 28.4%

---
=============================================================================
SECCION B - Contexto de la sesion
=============================================================================

Al inicio de la sesion, el asistente recibio:

  1. PROMPT_MAESTRO v6.40 (via archivo adjunto).
  2. TRANSFER_SESION_2026-09-19_POST_F2.3.md (contexto previo).

Punto de partida: 70 commits locales ahead, ciclo OpenFIGI completo
(3 modulos + catalogo + piloto top 500) pero pendiente de dictamen F2.3.

Trabajo realizado en esta sesion (orden cronologico):

  a) Verificacion exhaustiva del estado (git log, tests, pyflakes,
     policy hash, 6 hashes clave).
  b) Redaccion del informe de entrega al auditor:
     INSTITUTIONAL_ACCUMULATION_OPENFIGI_CATALOG_TARGET_UNIVERSE_INFORME.md.
  c) Commit 4593513.
  d) Dictamen F2.3-bis: PASS CONDICIONADO.
     - H1 (circularidad crosswalk interno): CERRADO.
     - GO TOP 2000 con condiciones (medir count + weight).
     - 3 issues abiertos: temporalidad catalogo, peso de NO_ID/ERROR,
       dependencia de proveedor comun.
  e) Autorizacion formal TOP 2000 (convencion CURRENT_RETROSPECTIVE).
  f) Preservacion de dictamen + autorizacion (commits 3ef4d2d).
  g) Piloto TOP 2000 ejecutado (0.69 min con API key):
     - 210 target / 32.81% weight (raw).
     - 212 target / 33.34% weight (corregido con 2 canales).
  h) Verificacion cruzada top 500 subset top 2000: 0 mismatches.
  i) Investigacion de los 223 NO_ID/ERROR:
     - 55 con hits extranjeros (shareClassFIGI no matchea).
     - 168 sin hits.
     - XOM (30231G102) resuelve 11 hits extranjeros, 0 US.
     - Hallazgo: shareClassFIGI cambia tras reorganizacion (XOM).
  j) Deteccion de bug EOL latente en 5 ficheros de evidence historicos
     (CRLF en working tree, LF en index). Fix en commit 9c67ac6.
  k) Prompt maestro v6.41 + transfer doc.

=============================================================================
SECCION C - Decisiones del auditor F2.3-bis
=============================================================================

Estado formal:

  F2.1 Inventory                PASS / CERRADO
  F2.1-bis Identity sources     PASS / CERRADO
  F2.2 v1.1                     NO GO / HISTORICO
  F2.3 v1.2                     RECHAZADA
  F2.3-bis H1                   CERRADO

  RADAR_TARGET_CATALOG          GO CONDICIONADO
  Target CUSIP -> SCF            GO
  OpenFIGI como resolver        GO
  OpenFIGI como autoridad       NO

  N-PORT fuente auxiliar        GO
  N-PORT catalogo completo      NO

  TOP 500                       CARACTERIZACION PASS
  TOP 2000                      GO
  FULL OpenFIGI                 NO AUTORIZADO

  Temporalidad catalog          ISSUE ABIERTO
  NO_ID/ERROR weighted impact   ISSUE ABIERTO (parcialmente cerrado con TOP 2000)

  THRESHOLD_1/2                 UNDEFINED
  F2.4                          NO AUTORIZADA
  Gate-NIPC.2                   BLOQUEADO
  Gate-NIPC.3                   NO AUTORIZADO

---
=============================================================================
SECCION D - Hallazgos materiales del ciclo TOP 2000
=============================================================================

H1. count vs weight divergen fuertemente:

    Categoria       top500    top2000
    target count    28.40%    10.50%   (caida 2.70x)
    target weight   46.76%    32.81%   (caida 1.43x)

    Confirma la hipotesis del auditor: el top 500 sobrerrepresenta
    large caps del radar. El 28.4% del piloto top 500 NO era una
    estimacion de coverage, era un artefacto de seleccion por
    SSHPRNAMT.

H2. shareClassFIGI cambia tras reorganizacion corporativa (XOM):

    Catalogo radar (OpenFIGI TICKER=XOM): BBG023CY9NL0 (EXXONMOBIL HOLDINGS)
    OpenFIGI CUSIP 30231G102:             BBG001S69V32 (EXXON MOBIL CORP)

    Matiza la premisa del dictamen F2.3-bis "FIGI inmune a corporate
    actions". La FIGI persiste ante cambios de ticker, pero NO ante
    reorganizaciones corporativas completas.

H3. HON no esta en el radar 242. Su reverse split post-Q1 (2026-06-29)
    lo excluyo. NO_ID legitimo, no sesgo del filtro exchCode.

H4. 221 NO_ID genuinos concentrados en prefijos no-USA (G/H/N/F/Y/D).
    Securities listadas en mercados no-USA que aparecen en el 13F pero
    no resuelven en OpenFIGI con ID_CUSIP.

H5. exchCode=US causa ~0.53 pp de peso en falsos NO_ID recuperables
    (2 canales: cusip_ticker_exceptions + ticker match).

=============================================================================
SECCION E - Deudas declaradas
=============================================================================

D1. multiple_openfigi_hits=0 hardcoded en catalog_diagnostics. El
    cliente openfigi_client.py::extract_stable_identity no expone hits
    crudos. Requiere modificar el cliente.

D2. target_universe.py NO consulta cusip_ticker_exceptions.csv (Q-CUR).
    XOM curado manualmente no se resuelve automaticamente.

D3. Cliente no valida formato CUSIP antes de enviar. Los 3 ERROR del
    piloto son por formato invalido. Recomendacion: regex CUSIP.

D4. BRK-B / MOG-A sin resolver en el catalogo (2 MISS). Divergencia
    de nomenclatura Yahoo vs OpenFIGI. Pendiente evidencia.

D5. EOL hygiene: 130 ficheros con CRLF en working tree local. Blobs
    del repo en LF (correcto). No persiste en clone limpio. Cosmetico.

=============================================================================
SECCION F - Reglas operativas criticas
=============================================================================

  - Local-first IAE: NO push a main hasta cierre de Gate-NIPC.3.
    Los 74 commits quedan en local.

  - NO ejecutar OpenFIGI masivo (24.838 CUSIPs) sin dictamen
    especifico. El piloto top 2000 si esta autorizado.

  - NO modificar NIPC_COVERAGE_POLICY.md v1.0 (hash 57f2d01f...).

  - NO tocar el motor NIPC (delta_shares.py, nipc.py,
    security_identity.py), C2, spec v1.4, baseline evidencia.

  - NO fijar THRESHOLD_1 ni THRESHOLD_2. UNDEFINED hasta dictamen.

  - NO usar el crosswalk interno como fuente del catalogo radar.

  - Regla de "IDs verificados": ningun ID/CUSIP/CIK/SERIES_ID/FIGI
    entra al codigo sin salir de un output real verificado.

  - Regla canonical_snapshot (heredada): agregados SSHPRNAMT desde
    apply_amendments().canonical_snapshot. PROHIBIDO sumar desde
    INFOTABLE.parquet crudo.

  - Convencion temporal vigente para el piloto:
    RADAR_SNAPSHOT_DATE = 2026-09-19
    13F_OBSERVATION_PERIOD = 2026-03-31
    RADAR_MEMBERSHIP_MODE = CURRENT_RETROSPECTIVE

---
=============================================================================
SECCION G - Trampas recurrentes confirmadas esta sesion
=============================================================================

  - Un here-string PowerShell >20 lineas o >5 $ se corrompe silenciosamente.
    Leccion: escribir a _patch_*.py con [System.IO.File]::WriteAllText
    y ejecutar con py. Tropiezo confirmado 1x en esta sesion
    (informe TOP 2000 con 5 chunks grandes fallo; resolucion con
    11 chunks de ~15 lineas cada uno).

  - Los CSV y JSON generados por pandas en Windows salen con CRLF.
    .gitattributes los normaliza en index pero el working tree los
    mantiene en CRLF hasta el siguiente checkout. Los HASHES.txt
    calculados contra bytes CRLF son invalidables en clone limpio.
    Verificar siempre: git ls-files --eol <path> -> i/lf w/lf.

  - git add --renormalize . NO reescribe el working tree. Solo
    normaliza el index. Para reescribir el working tree hay que
    hacer checkout o rm --cached + reset --hard.

  - extract_stable_identity prioriza exchCode=US y descarta hits
    extranjeros, devolviendo None incluso cuando OpenFIGI da hits.
    No es bug: es comportamiento por diseno.

  - pyflakes sobre fichero .md da error de parseo. Ruido, no bug.

=============================================================================
SECCION H - Comandos de arranque (verificacion al inicio)
=============================================================================

  Set-Location D:\Macro_Sectorial
  git log --oneline -8
  git status -sb
  git rev-list --count origin/main..HEAD

  py -m compileall . -q
  py -m pyflakes . 2>&1
  py -m pytest tests/ validation/ -q --tb=short

Esperado al arrancar la nueva sesion:

  - HEAD = 06c49ad o el hash del commit del transfer recien creado.
  - ahead 74 o 75 (si el transfer se commitea).
  - Working tree limpio.
  - 935 passed + 2 skipped.
  - pyflakes silencio.

Verificacion de la policy intacta:

  (Get-FileHash docs\auditoria\NIPC_COVERAGE_POLICY.md -Algorithm SHA256).Hash.ToLower()
  # Esperado: 57f2d01feb68d8916dfa4fa0451224c2ee3dda7e4631a7863e671257b3e91f06

---
=============================================================================
SECCION I - Lo que NO debes hacer (asistente entrante)
=============================================================================

  NO pushear los commits hasta cierre de Gate-NIPC.3.

  NO modificar NIPC_COVERAGE_POLICY.md v1.0.

  NO ejecutar OpenFIGI masivo sin autorizacion expresa.

  NO usar el crosswalk interno como fuente del catalogo radar.

  NO tocar nipc.py, delta_shares.py, security_identity.py.

  NO fijar THRESHOLD_1 ni THRESHOLD_2.

  NO inventar SERIES_ID, CUSIP, CIK, FIGI. Solo valores verificados
  contra outputs reales.

  NO usar ticker:<ticker> como canonical_security.

  NO reconciliar NT <-> HR.

  NO convertir filer_continuity_pct en threshold.

  NO sumar SSHPRNAMT desde INFOTABLE.parquet crudo.

  NO usar Notepad para contenido largo. Usar script Python con
  [System.IO.File]::WriteAllText.

  NO usar py -c con comillas dobles anidadas (usar script temporal).

  NO confundir shareClassFIGI como identidad inmutable: cambia
  tras reorganizacion corporativa (caso XOM).

  Regla del usuario: el usuario decide cuando parar. El asistente
  NO cierra sesion por fatiga.

  ROTAR la API key de OpenFIGI si fue expuesta en algun log. La key
  actual se uso en el piloto top 2000 y quedo visible en el chat.

=============================================================================
SECCION J - Proximos pasos posibles
=============================================================================

  1. Esperar dictamen F2.4 del auditor sobre el informe TOP 2000
     (entregable entregado en commit 3ef4d2d).

  2. Si F2.4 aprueba: aplicar v1.2 sobre la policy (con los 2 cambios
     obligatorios del dictamen F2.3-bis: identity vs membership,
     weighted diagnostics para NO_ID/ERROR).

  3. Decidir sobre full OpenFIGI (24.838 CUSIPs) o quedarse en top 2000.

  4. Resolver las 5 deudas declaradas (D1-D5) antes de thresholds.

  5. Cierre de sesion.

Ninguna de estas acciones se ejecuta sin confirmacion del usuario.

=============================================================================
SECCION K - Confirmacion esperada del asistente entrante
=============================================================================

Tras leer el prompt maestro v6.41 + este documento, el asistente
entrante debe:

  1. Confirmar asimilacion del contexto.

  2. Reconocer:
     - HEAD 06c49ad, ahead 74, tests 935+2.
     - Prompt v6.41 vigente.
     - Policy v1.0 intacta (hash 57f2d01f...).
     - RADAR_TARGET_CATALOG materializado (242 filas, 240 OK).
     - TARGET_UNIVERSE resolver operativo.
     - Piloto TOP 500 + TOP 2000 ejecutados.
     - Informe TOP 2000 entregado al auditor.

  3. Reconocer F2.3-bis como PASS CONDICIONADO con H1 CERRADO.

  4. Reconocer que sigue BLOQUEADO sin dictamen externo:
     - F2.4 (aplicar v1.2 sobre la policy).
     - OpenFIGI masivo (24.838 CUSIPs).
     - THRESHOLD_1/2 fijados.
     - Gate-NIPC.2 y Gate-NIPC.3.

  5. No proponer tareas nuevas sin antes preguntar "Que hacemos?".

  6. Respetar regla "usuario decide cuando parar".

  7. Aplicar regla "nada se interpreta, todo se verifica".

=============================================================================
FIN DEL DOCUMENTO DE TRANSMISION
Version 1.0 (2026-09-19). Referencia: prompt v6.41, HEAD 06c49ad.
=============================================================================